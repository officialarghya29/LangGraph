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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.core.constants import FailureKind, backoff_delay, is_retryable
from app.core.exceptions import classify_exception

__all__ = ["RetryExhaustedError", "RetryPolicy", "retry_async"]

#: Signature of the optional progress callback: (attempt, kind, exception).
RetryCallback = Callable[[int, FailureKind, BaseException], None]


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
            await sleep(active.delay_for(kind, attempt))
