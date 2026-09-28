"""Tests for retrying model calls.

The finding this covers was an omission rather than a bug: the classification
table, the backoff curve, and the retry loop were all written, all tested, and
never called by the application. Every test of each part passed. So the test that
matters most here is the one asserting the decorator is actually on the path —
that a transient failure earns another attempt.

Transient failures are injected directly rather than through a mock transport, so
the test describes the policy rather than an HTTP detail.
"""

from __future__ import annotations

import pytest

from app.core.constants import FailureKind
from app.core.exceptions import (
    AuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    UsageLimitExceededError,
)
from app.services.budget import BudgetedProvider, TokenBudget, token_budget
from app.services.llm import FakeLLMProvider, LLMResponse, Message, Role, TokenUsage
from app.services.resilience import RetryingProvider

MESSAGES = [Message(role=Role.USER, content="a question worth answering")]


class _Flaky:
    """A provider that fails a scripted number of times before succeeding."""

    name = "flaky"

    def __init__(self, failures: list[BaseException], *, content: str = "the answer") -> None:
        """Build a provider that raises ``failures`` in order, then answers."""
        self._failures = list(failures)
        self._content = content
        self.calls = 0
        self.default_model = "test-model"

    async def ainvoke(self, messages: object, **kwargs: object) -> LLMResponse:
        """Fail or answer, depending on what is left in the script."""
        self.calls += 1
        if self._failures:
            raise self._failures.pop(0)
        return LLMResponse(
            content=self._content,
            model=self.default_model,
            usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )


class _Recorder:
    """Collects delays and retry observations."""

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.delays: list[float] = []
        self.observed: list[tuple[int, FailureKind]] = []

    async def sleep(self, seconds: float) -> None:
        """Record a delay instead of waiting it out."""
        self.delays.append(seconds)

    def observe(self, attempt: int, kind: FailureKind, exc: BaseException) -> None:
        """Record one failed attempt."""
        del exc
        self.observed.append((attempt, kind))


def _wrap(
    inner: object, recorder: _Recorder | None = None, *, max_retries: int = 3
) -> RetryingProvider:
    """Wrap ``inner`` with a recorder, without a real sleep."""
    recorder = recorder or _Recorder()
    return RetryingProvider(
        inner,  # type: ignore[arg-type]
        max_retries=max_retries,
        sleep=recorder.sleep,
        on_retry=recorder.observe,
    )


# --------------------------------------------------------------------------- #
# Retrying
# --------------------------------------------------------------------------- #


async def test_a_transient_failure_earns_another_attempt() -> None:
    """The core claim: the retry loop is on the path, not merely available."""
    inner = _Flaky([LLMTimeoutError("too slow")])
    provider = _wrap(inner)

    response = await provider.ainvoke(MESSAGES)

    assert response.content == "the answer"
    assert inner.calls == 2


async def test_a_throttle_earns_another_attempt() -> None:
    """Rate limits are the failure this most needs to survive."""
    inner = _Flaky([LLMRateLimitError("slow down")])
    provider = _wrap(inner)

    await provider.ainvoke(MESSAGES)

    assert inner.calls == 2


async def test_a_successful_call_is_not_repeated() -> None:
    """Retrying a call that worked would double the bill and the side effects."""
    inner = _Flaky([])
    provider = _wrap(inner)

    await provider.ainvoke(MESSAGES)

    assert inner.calls == 1


async def test_retries_are_bounded() -> None:
    """A provider that is simply down must not be hammered indefinitely."""
    inner = _Flaky([LLMTimeoutError("down")] * 10)
    provider = _wrap(inner, max_retries=2)

    with pytest.raises(LLMTimeoutError):
        await provider.ainvoke(MESSAGES)

    assert inner.calls == 3


async def test_the_original_failure_is_what_escapes() -> None:
    """Callers classify what went wrong, so "exhausted" must not replace it."""
    inner = _Flaky([LLMTimeoutError("down")] * 10)
    provider = _wrap(inner)

    with pytest.raises(LLMTimeoutError) as captured:
        await provider.ainvoke(MESSAGES)

    assert isinstance(captured.value.__cause__, Exception)
    assert "attempt" in str(captured.value.__cause__)


async def test_the_observer_sees_every_failed_attempt() -> None:
    """Retry counts have to be observable, or nobody can tell they happen."""
    recorder = _Recorder()
    inner = _Flaky([LLMTimeoutError("down"), LLMRateLimitError("slow")])
    provider = _wrap(inner, recorder)

    await provider.ainvoke(MESSAGES)

    assert recorder.observed == [(1, FailureKind.TIMEOUT), (2, FailureKind.RATE_LIMIT)]
    assert len(recorder.delays) == 2


# --------------------------------------------------------------------------- #
# Failing fast
# --------------------------------------------------------------------------- #


async def test_a_permanent_failure_is_not_retried() -> None:
    """A bad credential is still bad on the second attempt."""
    inner = _Flaky([AuthenticationError("invalid key")] * 5)
    provider = _wrap(inner)

    with pytest.raises(AuthenticationError):
        await provider.ainvoke(MESSAGES)

    assert inner.calls == 1


async def test_a_spent_budget_is_not_retried() -> None:
    """This is the composition that matters: retry outside the budget.

    The budget refuses before the provider is called, and a refusal is permanent
    for that run — retrying it would be three refusals for the same answer.
    """
    recorder = _Recorder()
    inner = _Flaky([])
    provider = _wrap(BudgetedProvider(inner), recorder, max_retries=3)
    budget = TokenBudget(limit=1)

    with token_budget(budget):
        await provider.ainvoke(MESSAGES)
        with pytest.raises(UsageLimitExceededError):
            await provider.ainvoke(MESSAGES)

    assert inner.calls == 1
    assert recorder.observed == []


# --------------------------------------------------------------------------- #
# Composition with the budget
# --------------------------------------------------------------------------- #


async def test_a_failed_attempt_costs_nothing_to_account_for() -> None:
    """Only a request that came back carries a count.

    This pins down what the decorator ordering does and does not change. An
    attempt that raises yields no usage, so it is not billed; the attempt that
    succeeded is charged exactly once. What the ordering buys is that the
    allowance is checked between attempts, which the test above covers.
    """
    inner = _Flaky([LLMTimeoutError("down"), LLMTimeoutError("down")])
    provider = _wrap(BudgetedProvider(inner))
    budget = TokenBudget(limit=1_000_000)

    with token_budget(budget):
        await provider.ainvoke(MESSAGES)

    assert inner.calls == 3
    assert budget.calls == 1
    assert budget.spent == 10


async def test_a_retry_after_instruction_is_honoured_through_the_decorator() -> None:
    """The wrapper waits for the service, not merely for its own curve."""
    recorder = _Recorder()
    inner = _Flaky([LLMRateLimitError("slow down", retry_after=30.0)])
    provider = _wrap(inner, recorder)

    await provider.ainvoke(MESSAGES)

    assert recorder.delays == [30.0]


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #


def test_the_wrapper_names_what_it_wrapped() -> None:
    """A log line should still say which provider ran."""
    inner = FakeLLMProvider()

    provider = _wrap(inner)

    assert provider.name == f"retrying:{inner.name}"
    assert provider.default_model == inner.default_model
    assert provider.policy.max_retries == 3


async def test_an_empty_provider_list_still_works_end_to_end() -> None:
    """A sanity check that the decorators compose with a real provider object."""
    provider = RetryingProvider(BudgetedProvider(FakeLLMProvider(default="ok")), max_retries=1)

    response = await provider.ainvoke(MESSAGES)

    assert response.content
