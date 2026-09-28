"""Tests for the error hierarchy and retry policy."""

from __future__ import annotations

import pytest
from app.core.constants import (
    MAX_BACKOFF_SECONDS,
    FailureKind,
    backoff_delay,
    is_retryable,
)
from app.core.exceptions import (
    AppError,
    ApprovalRequiredError,
    InputValidationError,
    LLMRateLimitError,
    LLMTimeoutError,
    PermissionDeniedError,
    StructuredOutputError,
    ToolPermissionError,
    classify_exception,
)


def test_str_includes_detail_when_present() -> None:
    assert str(AppError("boom", detail="ctx")) == "boom (ctx)"
    assert str(AppError("boom")) == "boom"


def test_app_errors_classify_from_their_own_type() -> None:
    assert classify_exception(LLMTimeoutError("slow")) is FailureKind.TIMEOUT
    assert classify_exception(LLMRateLimitError("throttled")) is FailureKind.RATE_LIMIT
    assert classify_exception(PermissionDeniedError("no")) is FailureKind.PERMANENT
    assert classify_exception(InputValidationError("bad")) is FailureKind.VALIDATION
    assert classify_exception(ApprovalRequiredError("gate")) is FailureKind.PERMANENT
    assert classify_exception(ToolPermissionError("no")) is FailureKind.PERMANENT


def test_timeout_classification() -> None:
    assert classify_exception(TimeoutError()) is FailureKind.TIMEOUT


def test_connection_errors_are_transient() -> None:
    assert classify_exception(ConnectionError("reset")) is FailureKind.TRANSIENT
    assert classify_exception(ConnectionResetError()) is FailureKind.TRANSIENT


def test_filesystem_errors_are_permanent() -> None:
    """Retrying a missing file or a denied permission changes nothing."""
    assert classify_exception(FileNotFoundError("nope")) is FailureKind.PERMANENT
    assert classify_exception(PermissionError("denied")) is FailureKind.PERMANENT
    assert classify_exception(IsADirectoryError("dir")) is FailureKind.PERMANENT


def test_memory_errors_are_permanent() -> None:
    assert classify_exception(MemoryError()) is FailureKind.PERMANENT


def test_unknown_error_is_unknown() -> None:
    assert classify_exception(ValueError("?")) is FailureKind.UNKNOWN


def test_structured_output_error_is_retryable() -> None:
    """A second attempt usually satisfies the schema."""
    assert is_retryable(classify_exception(StructuredOutputError("bad json"))) is True


# --------------------------------------------------------------------------- #
# Retry policy
# --------------------------------------------------------------------------- #


def test_only_documented_kinds_are_retryable() -> None:
    assert is_retryable(FailureKind.TRANSIENT) is True
    assert is_retryable(FailureKind.RATE_LIMIT) is True
    assert is_retryable(FailureKind.TIMEOUT) is True

    assert is_retryable(FailureKind.VALIDATION) is False
    assert is_retryable(FailureKind.AUTHENTICATION) is False
    assert is_retryable(FailureKind.PERMANENT) is False


def test_backoff_grows_exponentially() -> None:
    first = backoff_delay(FailureKind.TRANSIENT, 1)
    second = backoff_delay(FailureKind.TRANSIENT, 2)
    third = backoff_delay(FailureKind.TRANSIENT, 3)

    assert second == pytest.approx(first * 2)
    assert third == pytest.approx(first * 4)


def test_backoff_is_capped() -> None:
    assert backoff_delay(FailureKind.TRANSIENT, 50) == MAX_BACKOFF_SECONDS


def test_rate_limits_back_off_harder_than_transients() -> None:
    assert backoff_delay(FailureKind.RATE_LIMIT, 1) > backoff_delay(FailureKind.TRANSIENT, 1)


def test_backoff_handles_zero_attempt() -> None:
    assert backoff_delay(FailureKind.TRANSIENT, 0) > 0
