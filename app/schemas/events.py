"""Structured execution events.

Events are the only channel through which internal progress reaches a client.
Every field here is deliberately safe to expose.

What must never appear in an event:

- model chain-of-thought or reasoning traces
- raw prompts or system prompts
- provider credentials or any secret
- raw tool arguments that may contain sensitive values

Anything with a free-text ``message`` must be a short, sanitised summary written
by the application, never text lifted verbatim from a model's internal reasoning.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

__all__ = ["SAFE_EVENT_TYPES", "EventType", "ExecutionEvent"]


class EventType(StrEnum):
    """Types of progress a client may observe."""

    TASK_STARTED = "task_started"
    TASK_ROUTING = "task_routing"
    TASK_PLANNING = "task_planning"
    TASK_RETRY = "task_retry"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"

    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"

    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"

    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_COMPLETED = "verification_completed"

    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_RECEIVED = "approval_received"

    MEMORY_RECALLED = "memory_recalled"
    MEMORY_WRITTEN = "memory_written"


#: Every event type is safe to stream. The constant exists so that any future
#: internal-only event type has to be named explicitly to be excluded.
SAFE_EVENT_TYPES: frozenset[EventType] = frozenset(EventType)


class ExecutionEvent(BaseModel):
    """One observable step in a task's execution."""

    event_type: EventType
    task_id: str
    sequence: int = Field(default=0, ge=0)
    conversation_id: str | None = None
    node: str | None = None
    agent: str | None = None
    tool: str | None = None
    status: str | None = None
    message: str | None = None
    duration_ms: float | None = None
    error_type: str | None = None
    retry_count: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_safe_to_expose(self) -> bool:
        """Return whether this event may be sent to a client."""
        return self.event_type in SAFE_EVENT_TYPES
