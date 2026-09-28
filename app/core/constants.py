"""Cross-cutting constants.

Failure classification lives here rather than in the retry code, because both
the tool layer and the graph layer need to agree on what is retryable.
"""

from __future__ import annotations

from enum import StrEnum


class FailureKind(StrEnum):
    """How a failure should be treated.

    Only the members listed in :data:`RETRYABLE` may be retried. Everything
    else fails fast, so a permanent error can never be retried in a loop.
    """

    TRANSIENT = "transient"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    TOOL_FAILURE = "tool_failure"
    MODEL_FAILURE = "model_failure"
    DATABASE_FAILURE = "database_failure"
    PERMANENT = "permanent"
    UNKNOWN = "unknown"


RETRYABLE: frozenset[FailureKind] = frozenset(
    {
        FailureKind.TRANSIENT,
        FailureKind.RATE_LIMIT,
        FailureKind.TIMEOUT,
        FailureKind.TOOL_FAILURE,
        FailureKind.MODEL_FAILURE,
        FailureKind.DATABASE_FAILURE,
    }
)

#: Base delay in seconds for exponential backoff, per failure kind. Rate limits
#: are backed off harder because retrying them quickly usually makes it worse.
BACKOFF_BASE_SECONDS: dict[FailureKind, float] = {
    FailureKind.RATE_LIMIT: 2.0,
    FailureKind.TIMEOUT: 1.0,
    FailureKind.TRANSIENT: 0.5,
    FailureKind.TOOL_FAILURE: 0.5,
    FailureKind.MODEL_FAILURE: 1.0,
    FailureKind.DATABASE_FAILURE: 0.5,
}

DEFAULT_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 30.0


def is_retryable(kind: FailureKind) -> bool:
    """Return whether a failure kind is safe to retry."""
    return kind in RETRYABLE


def backoff_delay(kind: FailureKind, attempt: int) -> float:
    """Return the exponential backoff delay for a given attempt.

    Args:
        kind: The classified failure.
        attempt: 1-based retry attempt number.

    Returns:
        Delay in seconds, capped at :data:`MAX_BACKOFF_SECONDS`.
    """
    base = BACKOFF_BASE_SECONDS.get(kind, DEFAULT_BACKOFF_SECONDS)
    return float(min(base * (2 ** max(attempt - 1, 0)), MAX_BACKOFF_SECONDS))
