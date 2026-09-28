"""Application error hierarchy.

Every error the application raises deliberately derives from :class:`AppError`,
so the API layer can map errors to status codes without inspecting arbitrary
exception types. Nothing here carries a secret: ``__str__`` output is safe to
log and to return to a client.
"""

from __future__ import annotations

from app.core.constants import FailureKind

__all__ = [
    "AppError",
    "ApprovalRequiredError",
    "AuthenticationError",
    "CacheError",
    "ConfigurationError",
    "DatabaseError",
    "EmbeddingError",
    "ExecutionLimitError",
    "InputValidationError",
    "LLMError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "NotFoundError",
    "PermissionDeniedError",
    "ProviderError",
    "StructuredOutputError",
    "ToolError",
    "ToolPermissionError",
    "ToolTimeoutError",
    "classify_exception",
]


class AppError(Exception):
    """Base class for errors this application raises deliberately."""

    #: Failure kind used by the retry policy when this error escapes.
    failure_kind: FailureKind = FailureKind.UNKNOWN

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def __str__(self) -> str:
        """Render the error without any sensitive payload."""
        return f"{self.message} ({self.detail})" if self.detail else self.message


# --------------------------------------------------------------------------- #
# Configuration and input
# --------------------------------------------------------------------------- #


class ConfigurationError(AppError):
    """The application is misconfigured. Always fatal."""

    failure_kind = FailureKind.PERMANENT


class InputValidationError(AppError):
    """Caller-supplied input failed validation. Never retried as-is."""

    failure_kind = FailureKind.VALIDATION


class ExecutionLimitError(AppError):
    """A configured execution ceiling was reached."""

    failure_kind = FailureKind.PERMANENT


# --------------------------------------------------------------------------- #
# Authorisation and ownership
# --------------------------------------------------------------------------- #


class NotFoundError(AppError):
    """A requested resource does not exist, or is not visible to the caller."""

    failure_kind = FailureKind.VALIDATION


class PermissionDeniedError(AppError):
    """The caller is known but not permitted to perform the operation."""

    failure_kind = FailureKind.PERMANENT


class AuthenticationError(AppError):
    """Credentials are missing or invalid. Never retried."""

    failure_kind = FailureKind.AUTHENTICATION


class ApprovalRequiredError(AppError):
    """The action is gated behind human approval and cannot proceed."""

    failure_kind = FailureKind.PERMANENT


# --------------------------------------------------------------------------- #
# Providers (LLM, embeddings)
# --------------------------------------------------------------------------- #


class ProviderError(AppError):
    """Base class for upstream provider failures."""

    failure_kind = FailureKind.MODEL_FAILURE


class LLMError(ProviderError):
    """The language model provider failed."""


class LLMTimeoutError(LLMError):
    """The language model provider did not respond within the timeout."""

    failure_kind = FailureKind.TIMEOUT


class LLMRateLimitError(LLMError):
    """The language model provider throttled the request."""

    failure_kind = FailureKind.RATE_LIMIT


class StructuredOutputError(LLMError):
    """The model did not return output conforming to the required schema.

    Retryable, because a second attempt with the validation error fed back
    usually succeeds.
    """

    failure_kind = FailureKind.MODEL_FAILURE


class EmbeddingError(ProviderError):
    """The embedding provider failed."""


# --------------------------------------------------------------------------- #
# Infrastructure
# --------------------------------------------------------------------------- #


class DatabaseError(AppError):
    """The database is unavailable or rejected the operation."""

    failure_kind = FailureKind.DATABASE_FAILURE


class CacheError(AppError):
    """The cache or coordination store is unavailable.

    Separate from :class:`DatabaseError` because the two have different
    consequences: a cache outage degrades performance, a database outage loses
    correctness. Nothing durable is stored only in the cache.
    """

    failure_kind = FailureKind.TRANSIENT


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


class ToolError(AppError):
    """A tool failed during execution."""

    failure_kind = FailureKind.TOOL_FAILURE


class ToolTimeoutError(ToolError):
    """A tool exceeded its execution timeout."""

    failure_kind = FailureKind.TIMEOUT


class ToolPermissionError(ToolError):
    """The calling agent is not authorised to use the tool."""

    failure_kind = FailureKind.PERMANENT


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

_TRANSIENT_MARKERS = (
    "connection reset",
    "connection aborted",
    "connection refused",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "broken pipe",
)


def classify_exception(exc: BaseException) -> FailureKind:
    """Map an arbitrary exception to a failure kind.

    Errors raised by this application classify from their own class. Everything
    else is inspected narrowly: timeouts, then connection-level transients, then
    memory exhaustion. The default is :attr:`FailureKind.UNKNOWN`, which is
    retried at most once.

    Args:
        exc: The exception to classify.

    Returns:
        The failure kind that governs retry behaviour.
    """
    if isinstance(exc, AppError):
        return exc.failure_kind

    if isinstance(exc, TimeoutError):
        return FailureKind.TIMEOUT

    if isinstance(exc, MemoryError):
        return FailureKind.PERMANENT

    if isinstance(exc, ConnectionError):
        return FailureKind.TRANSIENT

    if isinstance(exc, (FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError)):
        # Retrying a missing file or a denied permission changes nothing.
        return FailureKind.PERMANENT

    if isinstance(exc, OSError):
        text = str(exc).lower()
        if any(marker in text for marker in _TRANSIENT_MARKERS):
            return FailureKind.TRANSIENT
        return FailureKind.UNKNOWN

    return FailureKind.UNKNOWN
