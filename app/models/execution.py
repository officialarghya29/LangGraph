"""Execution-related domain models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from app.core.constants import FailureKind

__all__ = ["ExecutionError", "ExecutionMetadata", "TaskStatus", "TokenUsageSummary"]


class TaskStatus(StrEnum):
    """Lifecycle state of a task."""

    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Return whether no further progress is possible."""
        return self in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}


class ExecutionError(BaseModel):
    """A structured, client-safe record of something that went wrong.

    Carries no traceback and no prompt content: it is safe to log and to return
    over the API.
    """

    node: str
    message: str
    failure_kind: FailureKind = FailureKind.UNKNOWN
    attempt: int = 1
    detail: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ExecutionMetadata(BaseModel):
    """Accounting for one graph run."""

    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    duration_ms: float = 0.0
    provider: str | None = None
    model: str | None = None
    llm_calls: int = 0
    tool_calls: int = 0
    iterations: int = 0
    retries: int = 0
    usage: TokenUsageSummary = Field(default_factory=lambda: TokenUsageSummary())

    def finish(self, *, duration_ms: float) -> ExecutionMetadata:
        """Mark the run as finished and return this instance."""
        self.finished_at = datetime.now(UTC)
        self.duration_ms = duration_ms
        return self


class TokenUsageSummary(BaseModel):
    """Aggregate token accounting across a run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, *, prompt: int, completion: int) -> None:
        """Accumulate one model call's usage."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
