"""Phase 35 — failure injection.

The interesting question about an agent system is not whether it works when
everything is up. It is what it does when a tool hangs, a provider refuses, a
database disappears, or a memory store throws — and specifically whether it
degrades in a way that is *reported* or in a way that is *hidden*.

Every test here injects a fault at a real seam and asserts a specific behaviour.
None of them asserts that nothing goes wrong; they assert that the wrong thing is
contained and visible.
"""

from __future__ import annotations

import asyncio

import pytest

from app.core.constants import FailureKind, backoff_delay, is_retryable
from app.core.exceptions import (
    AuthenticationError,
    DatabaseError,
    InputValidationError,
    RateLimitExceededError,
    ToolError,
    classify_exception,
)
from app.services.execution import RetryExhaustedError, RetryPolicy, retry_async


async def no_sleep(seconds: float) -> None:
    """Swallow a backoff delay, so retry tests do not wait in real time."""
    del seconds


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        # The application's own errors carry their classification.
        (DatabaseError("down"), FailureKind.DATABASE_FAILURE),
        (RateLimitExceededError("slow down"), FailureKind.RATE_LIMIT),
        (ToolError("bad arguments"), FailureKind.TOOL_FAILURE),
        (InputValidationError("bad input"), FailureKind.VALIDATION),
        (AuthenticationError("nope"), FailureKind.AUTHENTICATION),
        # Foreign errors are inspected narrowly rather than guessed at.
        (TimeoutError(), FailureKind.TIMEOUT),
        (ConnectionResetError("reset"), FailureKind.TRANSIENT),
        (MemoryError(), FailureKind.PERMANENT),
        (FileNotFoundError("gone"), FailureKind.PERMANENT),
        (PermissionError("denied"), FailureKind.PERMANENT),
        (ValueError("bad input"), FailureKind.UNKNOWN),
        (RuntimeError("who knows"), FailureKind.UNKNOWN),
    ],
)
def test_each_error_class_maps_to_a_classification(
    error: BaseException, expected: FailureKind
) -> None:
    assert classify_exception(error) is expected


def test_a_missing_file_is_permanent_not_transient() -> None:
    """The tempting default is to retry anything that looks like I/O.

    A file that does not exist will not appear because it was asked for a second
    time, so retrying it only delays the failure and burns the budget.
    """
    assert classify_exception(FileNotFoundError("missing")) is FailureKind.PERMANENT


def test_only_recoverable_kinds_are_retryable() -> None:
    """The classification table is the whole retry policy."""
    retryable = {kind for kind in FailureKind if is_retryable(kind)}

    assert FailureKind.TRANSIENT in retryable
    assert FailureKind.RATE_LIMIT in retryable
    assert FailureKind.VALIDATION not in retryable
    assert FailureKind.AUTHENTICATION not in retryable
    assert FailureKind.PERMANENT not in retryable


def test_backoff_grows_and_is_capped() -> None:
    """Unbounded exponential backoff becomes an outage of its own."""
    first = backoff_delay(FailureKind.TRANSIENT, 1)
    second = backoff_delay(FailureKind.TRANSIENT, 2)
    far = backoff_delay(FailureKind.TRANSIENT, 40)

    assert second > first
    assert far <= 30.0


def test_rate_limits_back_off_harder_than_transient_errors() -> None:
    assert backoff_delay(FailureKind.RATE_LIMIT, 1) > backoff_delay(FailureKind.TRANSIENT, 1)


# --------------------------------------------------------------------------- #
# Retry behaviour
# --------------------------------------------------------------------------- #


async def test_a_transient_failure_is_retried_then_succeeds() -> None:
    attempts = 0

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionResetError("reset")
        return "ok"

    result = await retry_async(flaky, policy=RetryPolicy(max_retries=3), sleep=no_sleep)

    assert result == "ok"
    assert attempts == 3


async def test_a_permanent_failure_is_not_retried() -> None:
    """Retrying an authentication failure multiplies load and fixes nothing."""
    attempts = 0

    async def unauthorised() -> str:
        nonlocal attempts
        attempts += 1
        raise AuthenticationError("denied")

    with pytest.raises(RetryExhaustedError) as captured:
        await retry_async(unauthorised, policy=RetryPolicy(max_retries=5), sleep=no_sleep)

    assert attempts == 1
    assert captured.value.kind is FailureKind.AUTHENTICATION


async def test_retries_stop_at_the_ceiling() -> None:
    attempts = 0

    async def always_transient() -> str:
        nonlocal attempts
        attempts += 1
        raise ConnectionResetError("reset")

    with pytest.raises(RetryExhaustedError):
        await retry_async(always_transient, policy=RetryPolicy(max_retries=2), sleep=no_sleep)

    assert attempts == 3  # the first attempt plus two retries


async def test_never_retry_disables_retrying_entirely() -> None:
    """The switch an irreversible operation uses."""
    attempts = 0

    async def transient() -> str:
        nonlocal attempts
        attempts += 1
        raise ConnectionResetError("reset")

    with pytest.raises(RetryExhaustedError):
        await retry_async(transient, policy=RetryPolicy(never_retry=True), sleep=no_sleep)

    assert attempts == 1


async def test_a_negative_retry_budget_is_refused() -> None:
    """A nonsensical budget is a code defect, not a runtime condition."""
    with pytest.raises(ValueError, match="negative"):
        RetryPolicy(max_retries=-1)


async def test_every_retry_is_reported_before_it_happens() -> None:
    """Observability of a retry loop is what distinguishes slow from broken."""
    seen: list[tuple[int, FailureKind]] = []

    async def transient() -> str:
        raise ConnectionResetError("reset")

    with pytest.raises(RetryExhaustedError):
        await retry_async(
            transient,
            policy=RetryPolicy(max_retries=2),
            sleep=no_sleep,
            on_retry=lambda attempt, kind, exc: seen.append((attempt, kind)),
        )

    assert [attempt for attempt, _ in seen] == [1, 2]
    assert all(kind is FailureKind.TRANSIENT for _, kind in seen)


async def test_the_exhausted_error_keeps_its_cause() -> None:
    """A lost cause makes an incident impossible to diagnose."""

    async def broken() -> str:
        raise ConnectionResetError("the real reason")

    with pytest.raises(RetryExhaustedError) as captured:
        await retry_async(broken, policy=RetryPolicy(max_retries=0), sleep=no_sleep)

    assert isinstance(captured.value.cause, ConnectionResetError)
    assert "the real reason" in str(captured.value)


async def test_a_cancellation_is_not_swallowed_by_the_retry_loop() -> None:
    """Shutdown must not be retried into a hang."""

    async def cancelled() -> str:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await retry_async(cancelled, policy=RetryPolicy(max_retries=3), sleep=no_sleep)
