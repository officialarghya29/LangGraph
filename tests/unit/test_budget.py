"""Tests for token accounting and budget enforcement.

The bug these cover was silent: token counts were parsed off every response and
then dropped, so the persisted columns were always zero and the configured budget
bounded nothing. A test that only checked "the run succeeded" would have passed
throughout. So these assert the numbers, not just the absence of a crash.
"""

from __future__ import annotations

import asyncio

import pytest

from app.core.exceptions import StructuredOutputError, UsageLimitExceededError
from app.models.execution import TokenUsageSummary
from app.services.budget import BudgetedProvider, TokenBudget, current_budget, token_budget
from app.services.llm import FakeLLMProvider, Message, Role

MESSAGES = [Message(role=Role.USER, content="Say something reasonably long, please.")]


def _provider() -> FakeLLMProvider:
    """Return a deterministic provider that answers in prose."""
    return FakeLLMProvider(responder=lambda messages: "a perfectly ordinary answer")


# --------------------------------------------------------------------------- #
# Accounting
# --------------------------------------------------------------------------- #


def test_a_budget_records_more_than_one_call_to_the_same_total() -> None:
    """Accumulation, not replacement — two calls are two calls' worth."""
    budget = TokenBudget(limit=1000)

    budget.record(prompt=10, completion=5)
    budget.record(prompt=20, completion=5)

    assert budget.calls == 2
    assert budget.usage.prompt_tokens == 30
    assert budget.usage.completion_tokens == 10
    assert budget.usage.total_tokens == 40
    assert budget.spent == 40
    assert budget.remaining == 960


def test_a_budget_writes_into_the_usage_object_it_was_given() -> None:
    """This aliasing is how the totals reach the task record.

    The budget is handed the run's own ``ExecutionMetadata.usage``, so recording
    into it is what makes the persisted columns non-zero. A copy would leave them
    at zero while every unit test of the accumulator still passed.
    """
    usage = TokenUsageSummary()
    budget = TokenBudget(limit=100, usage=usage)

    budget.record(prompt=7, completion=3)

    assert usage.total_tokens == 10


def test_negative_counts_are_clamped_rather_than_subtracted() -> None:
    """A provider that reports nonsense must not be able to refund a budget."""
    budget = TokenBudget(limit=100)
    budget.record(prompt=50, completion=50)

    budget.record(prompt=-500, completion=-500)

    assert budget.spent == 100


def test_a_budget_of_zero_is_rejected() -> None:
    """A zero budget refuses the first call, which looks like a broken provider."""
    with pytest.raises(ValueError):
        TokenBudget(limit=0)


def test_the_budget_is_exhausted_at_the_limit_not_past_it() -> None:
    """The boundary is inclusive, so a limit of one token allows one token."""
    budget = TokenBudget(limit=10)
    budget.record(prompt=10, completion=0)

    assert budget.exhausted is True
    assert budget.remaining == 0
    with pytest.raises(UsageLimitExceededError) as captured:
        budget.check()

    assert "8 of 10" not in str(captured.value)
    assert "10 of 10" in str(captured.value)


def test_remaining_never_goes_negative() -> None:
    """Overshooting by one call is allowed, and must not report a negative."""
    budget = TokenBudget(limit=10)
    budget.record(prompt=400, completion=0)

    assert budget.remaining == 0
    assert budget.exhausted is True


# --------------------------------------------------------------------------- #
# Scoping
# --------------------------------------------------------------------------- #


def test_there_is_no_budget_outside_a_scope() -> None:
    """A script calling a provider directly should not have to arrange one."""
    assert current_budget() is None


def test_the_scope_is_restored_afterwards() -> None:
    """Leaving the scope must not leave the budget in place for the next run."""
    budget = TokenBudget(limit=10)
    with token_budget(budget) as scoped:
        assert scoped is budget
        assert current_budget() is budget

    assert current_budget() is None


async def test_concurrent_tasks_do_not_pool_their_tokens() -> None:
    """The reason the budget is a ContextVar rather than provider state.

    Two runs share one provider object. If the budget lived on the provider, both
    runs' calls would land in one total and the first run to finish would decide
    whether the other was over budget.
    """
    provider = BudgetedProvider(_provider())
    seen: dict[str, int] = {}

    async def run(name: str, calls: int) -> None:
        budget = TokenBudget(limit=1_000_000)
        with token_budget(budget):
            for _ in range(calls):
                await provider.ainvoke(MESSAGES)
            seen[name] = budget.calls

    await asyncio.gather(run("left", 3), run("right", 5))

    assert seen == {"left": 3, "right": 5}


# --------------------------------------------------------------------------- #
# Enforcement
# --------------------------------------------------------------------------- #


async def test_every_call_is_charged() -> None:
    """The decorator counts, so a provider author cannot forget to."""
    provider = BudgetedProvider(_provider())
    budget = TokenBudget(limit=1_000_000)

    with token_budget(budget):
        response = await provider.ainvoke(MESSAGES)

    assert budget.calls == 1
    assert budget.spent == response.usage.total_tokens
    assert budget.spent > 0


async def test_a_spent_budget_refuses_the_next_call() -> None:
    """Refused before the call, so nothing new is paid for."""
    inner = _provider()
    provider = BudgetedProvider(inner)
    budget = TokenBudget(limit=1)

    with token_budget(budget):
        await provider.ainvoke(MESSAGES)
        with pytest.raises(UsageLimitExceededError):
            await provider.ainvoke(MESSAGES)

    # The refusal happened before delegating: the provider made one call, not two.
    assert inner.call_count == 1


async def test_the_refusal_names_the_arithmetic() -> None:
    """Being told the run is over budget is not actionable; the counts are."""
    provider = BudgetedProvider(_provider())
    budget = TokenBudget(limit=1)

    with token_budget(budget):
        await provider.ainvoke(MESSAGES)
        with pytest.raises(UsageLimitExceededError) as captured:
            await provider.ainvoke(MESSAGES)

    # The first call was allowed to complete and overshot, so the reported figure
    # is what was actually spent — not the limit, which would understate it.
    assert f"{budget.spent} of {budget.limit}" in str(captured.value)
    assert f"after {budget.calls} call(s)" in str(captured.value)
    assert budget.summary().startswith(f"{budget.spent}/{budget.limit} tokens")


async def test_an_unbudgeted_call_is_not_charged_to_anything() -> None:
    """Outside a scope the provider still works; it simply has no ledger."""
    provider = BudgetedProvider(_provider())

    response = await provider.ainvoke(MESSAGES)

    assert response.content
    assert current_budget() is None


async def test_structured_output_charges_every_attempt() -> None:
    """A retried structured call is several calls, and is billed as several.

    The wrapper only overrides ``ainvoke``, and structured output is implemented
    on top of it, so the correction attempts are counted without this module
    knowing that structured output retries at all.
    """
    from pydantic import BaseModel

    class _Schema(BaseModel):
        """A schema the provider will never satisfy."""

        answer: str

    provider = BudgetedProvider(FakeLLMProvider(default="not valid json"))
    budget = TokenBudget(limit=1_000_000)

    with token_budget(budget), pytest.raises(StructuredOutputError):
        await provider.astructured_output(MESSAGES, _Schema, max_attempts=3)

    assert budget.calls == 3


def test_the_wrapped_provider_keeps_its_identity() -> None:
    """A log line should name the provider that ran, not the wrapper."""
    provider = BudgetedProvider(_provider())

    assert provider.name == f"budgeted:{_provider().name}"
    assert provider.default_model == _provider().default_model
    assert provider.inner is not None
