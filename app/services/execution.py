"""Retry and failure handling.

Retrying is a decision, not a reflex. Every failure is classified first, and
only retryable kinds are retried. A validation error or a permission denial is
never retried, because repeating it changes nothing and only burns budget.

Destructive actions are never retried automatically regardless of classification:
a write that may have succeeded before the connection dropped is not safe to
repeat blindly.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.core.constants import FailureKind, backoff_delay, is_retryable
from app.core.exceptions import classify_exception

__all__ = [
    "MAX_RETRY_AFTER_SECONDS",
    "RetryExhaustedError",
    "RetryPolicy",
    "retry_after_hint",
    "retry_async",
]

#: Signature of the optional progress callback: (attempt, kind, exception).
RetryCallback = Callable[[int, FailureKind, BaseException], None]

#: Ceiling on a ``Retry-After`` instruction from a remote service, in seconds.
#: The header is honoured because the service knows more than the backoff curve
#: does — but only up to a point. A provider asking for an hour is, in practice,
#: asking to fail slowly, and blocking a request that long is worse than
#: surfacing the failure.
MAX_RETRY_AFTER_SECONDS = 60.0


def retry_after_hint(exc: BaseException) -> float | None:
    """Return the service's requested delay, if it supplied a usable one.

    Args:
        exc: The failure that was just classified.

    Returns:
        A positive number of seconds, capped at :data:`MAX_RETRY_AFTER_SECONDS`,
        or ``None`` when the failure carried no instruction.
    """
    hint = getattr(exc, "retry_after", None)
    # ``isfinite`` rather than a range check: ``nan`` fails every comparison, so
    # ``nan <= 0`` is false and a bare sign test would let it through to
    # ``asyncio.sleep``, where it raises and turns a throttle into a crash.
    if (
        not isinstance(hint, int | float)
        or isinstance(hint, bool)
        or not math.isfinite(hint)
        or hint <= 0
    ):
        return None
    return min(float(hint), MAX_RETRY_AFTER_SECONDS)


class RetryExhaustedError(Exception):
    """Raised when an operation failed and its retry budget is spent."""

    def __init__(self, kind: FailureKind, attempts: int, cause: BaseException) -> None:
        super().__init__(f"operation failed after {attempts} attempt(s) with {kind.value}: {cause}")
        self.kind = kind
        self.attempts = attempts
        self.cause = cause


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How many retries are permitted, and for what."""

    max_retries: int = 3
    #: When true, no failure is retried. Used for irreversible operations.
    never_retry: bool = False

    def __post_init__(self) -> None:
        """Reject a nonsensical retry budget."""
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")

    def should_retry(self, kind: FailureKind, attempt: int) -> bool:
        """Return whether another attempt is permitted.

        Args:
            kind: The classified failure.
            attempt: 1-based number of the attempt that just failed.

        Returns:
            Whether to retry.
        """
        if self.never_retry:
            return False
        return attempt <= self.max_retries and is_retryable(kind)

    def delay_for(self, kind: FailureKind, attempt: int) -> float:
        """Return the backoff delay before the next attempt."""
        return backoff_delay(kind, attempt)


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy | None = None,
    on_retry: RetryCallback | None = None,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    classify: Callable[[BaseException], FailureKind] = classify_exception,
) -> T:
    """Run an async operation, retrying only failures worth retrying.

    Args:
        operation: A zero-argument callable returning an awaitable.
        policy: Retry budget. Defaults to three retries.
        on_retry: Called before each retry with the attempt, kind, and exception.
        sleep: Injectable sleep, so tests do not wait for real backoff.
        classify: Injectable classifier, defaulting to the shared one.

    Returns:
        The operation's result.

    Raises:
        RetryExhaustedError: If every permitted attempt fails.
    """
    active = policy or RetryPolicy()
    attempt = 0

    while True:
        attempt += 1
        try:
            return await operation()
        except Exception as exc:
            kind = classify(exc)
            if not active.should_retry(kind, attempt):
                raise RetryExhaustedError(kind, attempt, exc) from exc
            if on_retry is not None:
                on_retry(attempt, kind, exc)

            # The backoff curve is the floor, not the answer: when the service
            # said how long to wait, waiting less than that just earns another
            # rejection.
            delay = active.delay_for(kind, attempt)
            hint = retry_after_hint(exc)
            if hint is not None and hint > delay:
                delay = hint
            await sleep(delay)
