"""Tests for honouring a service's ``Retry-After`` instruction.

Backoff that ignores the service is backoff that guesses. When a provider says
"come back in thirty seconds", waiting two seconds does not retry sooner; it earns
another rejection and spends a second of the retry budget to do it.

The interesting cases are the boundaries: the instruction must be a floor and not
a ceiling, it must be capped, and a malformed header must not become a delay of
zero or a crash.
"""

from __future__ import annotations

import httpx
import pytest

from app.core.constants import FailureKind
from app.core.exceptions import AppError, LLMRateLimitError
from app.services.execution import (
    MAX_RETRY_AFTER_SECONDS,
    RetryExhaustedError,
    RetryPolicy,
    retry_after_hint,
    retry_async,
)
from app.services.llm import OpenAICompatibleProvider

MESSAGES = [{"role": "user", "content": "hello"}]


class _Recorder:
    """Collects the delays a retry loop asked for."""

    def __init__(self) -> None:
        """Start with no recorded delays."""
        self.delays: list[float] = []

    async def sleep(self, seconds: float) -> None:
        """Record a delay instead of waiting it out."""
        self.delays.append(seconds)


def _provider(handler: httpx.MockTransport) -> OpenAICompatibleProvider:
    """Return a provider whose HTTP transport is scripted."""
    return OpenAICompatibleProvider(
        api_key="test-key",
        model="test-model",
        client=httpx.AsyncClient(transport=handler),
    )


# --------------------------------------------------------------------------- #
# Reading the instruction
# --------------------------------------------------------------------------- #


def test_no_instruction_yields_nothing() -> None:
    """An error without the attribute is not a hint of zero seconds."""
    assert retry_after_hint(RuntimeError("boom")) is None
    assert retry_after_hint(AppError("boom")) is None


def test_a_usable_instruction_is_returned() -> None:
    assert retry_after_hint(LLMRateLimitError("slow down", retry_after=12.5)) == 12.5


def test_a_nonsensical_instruction_is_ignored() -> None:
    """Zero, negative, and non-numeric all mean "no usable instruction"."""
    assert retry_after_hint(LLMRateLimitError("x", retry_after=0)) is None
    assert retry_after_hint(LLMRateLimitError("x", retry_after=-5)) is None
    assert retry_after_hint(LLMRateLimitError("x", retry_after=float("nan"))) is None
    assert retry_after_hint(LLMRateLimitError("x", retry_after=float("inf"))) is None


def test_an_excessive_instruction_is_capped() -> None:
    """An hour-long wait inside a request is a slower failure, not a recovery."""
    assert retry_after_hint(LLMRateLimitError("x", retry_after=3600)) == MAX_RETRY_AFTER_SECONDS


# --------------------------------------------------------------------------- #
# The parse
# --------------------------------------------------------------------------- #


async def test_a_throttled_response_carries_the_hint() -> None:
    """429 is where providers put the header, so it has to survive the parse."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(429, headers={"Retry-After": "30"}, json={})
    )
    provider = _provider(transport)

    with pytest.raises(LLMRateLimitError) as captured:
        await provider.ainvoke([])  # type: ignore[arg-type]

    assert captured.value.retry_after == 30.0
    assert captured.value.failure_kind is FailureKind.RATE_LIMIT


async def test_an_overloaded_response_is_also_a_throttle() -> None:
    """503 means the same thing as 429 here and is treated the same way."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(503, headers={"Retry-After": "5"}, json={})
    )
    provider = _provider(transport)

    with pytest.raises(LLMRateLimitError) as captured:
        await provider.ainvoke([])  # type: ignore[arg-type]

    assert captured.value.retry_after == 5.0


@pytest.mark.parametrize("header", ["soon", "", "Wed, 21 Oct 2026 07:28:00 GMT"])
async def test_a_malformed_header_is_ignored_rather_than_guessed(header: str) -> None:
    """A date would mean trusting someone else's clock; better to use the curve."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(429, headers={"Retry-After": header}, json={})
    )
    provider = _provider(transport)

    with pytest.raises(LLMRateLimitError) as captured:
        await provider.ainvoke([])  # type: ignore[arg-type]

    assert captured.value.retry_after is None


async def test_a_non_throttle_error_carries_no_hint() -> None:
    """A 400 will not succeed on retry, and its headers mean nothing here."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(400, headers={"Retry-After": "30"}, json={})
    )
    provider = _provider(transport)

    with pytest.raises(AppError) as captured:
        await provider.ainvoke([])  # type: ignore[arg-type]

    assert not isinstance(captured.value, LLMRateLimitError)
    assert captured.value.retry_after is None


# --------------------------------------------------------------------------- #
# The retry loop
# --------------------------------------------------------------------------- #


async def test_the_hint_lengthens_the_wait() -> None:
    """The instruction is a floor when it exceeds the backoff curve."""
    recorder = _Recorder()
    attempts = 0

    async def throttled() -> str:
        nonlocal attempts
        attempts += 1
        raise LLMRateLimitError("slow down", retry_after=45.0)

    with pytest.raises(RetryExhaustedError):
        await retry_async(throttled, policy=RetryPolicy(max_retries=1), sleep=recorder.sleep)

    assert recorder.delays == [45.0]
    assert attempts == 2


async def test_the_curve_wins_when_it_is_longer() -> None:
    """A small instruction must not shorten a deliberate backoff."""
    recorder = _Recorder()

    async def throttled() -> str:
        raise LLMRateLimitError("slow down", retry_after=0.01)

    with pytest.raises(RetryExhaustedError):
        await retry_async(throttled, policy=RetryPolicy(max_retries=1), sleep=recorder.sleep)

    assert recorder.delays[0] > 0.01


async def test_the_cap_applies_to_the_wait_as_well_as_the_parse() -> None:
    """A service cannot park a worker for an hour by asking politely."""
    recorder = _Recorder()

    async def throttled() -> str:
        raise LLMRateLimitError("slow down", retry_after=3600.0)

    with pytest.raises(RetryExhaustedError):
        await retry_async(throttled, policy=RetryPolicy(max_retries=1), sleep=recorder.sleep)

    assert recorder.delays == [MAX_RETRY_AFTER_SECONDS]


async def test_an_unhinted_failure_uses_the_curve() -> None:
    """The ordinary path is unchanged."""
    recorder = _Recorder()

    async def broken() -> str:
        raise TimeoutError("no response")

    with pytest.raises(RetryExhaustedError):
        await retry_async(broken, policy=RetryPolicy(max_retries=1), sleep=recorder.sleep)

    assert 0 < recorder.delays[0] <= MAX_RETRY_AFTER_SECONDS
