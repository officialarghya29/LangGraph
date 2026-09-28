"""ORM models.

The schema mirrors the domain: a user has conversations, a conversation has
tasks, and a task accumulates the steps, agent runs, tool calls, approvals, and
execution events that describe how it was carried out.

Two deliberate choices are worth stating, because both are the kind of thing
usually discovered late:

- **Enumerations are stored as text.** A native PostgreSQL ``ENUM`` type cannot
  have a value added or removed without a migration that rewrites the type, and
  reordering values in Python silently rewrites the meaning of existing rows.
  Text columns with a check constraint give the same integrity guarantee and stay
  migration-friendly. The Python-side enums remain the source of truth, and
  :mod:`app.database.repositories` converts between the two at the boundary.
- **Nothing here is a LangGraph checkpoint.** Checkpoint state is owned by the
  checkpointer's own tables, which it creates and manages. Duplicating that
  structure by hand would mean two writers competing over one lifecycle.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, JsonColumn, PrimaryKey, TimestampMixin

__all__ = [
    "AgentRun",
    "Approval",
    "Conversation",
    "ExecutionEvent",
    "MemoryRecord",
    "Task",
    "TaskStep",
    "ToolCall",
    "User",
]


# --------------------------------------------------------------------------- #
# Identity and conversation
# --------------------------------------------------------------------------- #


class User(Base, TimestampMixin):
    """A principal the system acts on behalf of."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(PrimaryKey, primary_key=True, default=uuid.uuid4)
    #: Stable identifier from the identity provider, or a caller-supplied id
    #: while authentication is a header. Unique so a principal cannot be
    #: duplicated by a second request.
    external_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    conversations: Mapped[list[Conversation]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )
    tasks: Mapped[list[Task]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )

    def __repr__(self) -> str:
        """Return a debug representation naming the principal, not their id."""
        return f"<User {self.external_id}>"


class Conversation(Base, TimestampMixin):
    """A thread of related requests."""

    __tablename__ = "conversations"
    __table_args__ = (
        # Conversations are listed per owner, newest first; one composite index
        # serves both the filter and the sort.
        Index("ix_conversations_user_id_created_at", "user_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(PrimaryKey, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)

    user: Mapped[User] = relationship(back_populates="conversations")
    tasks: Mapped[list[Task]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", passive_deletes=True
    )

    def __repr__(self) -> str:
        """Return a debug representation of this conversation."""
        return f"<Conversation {self.id}>"


# --------------------------------------------------------------------------- #
# Task and its execution trail
# --------------------------------------------------------------------------- #


class Task(Base, TimestampMixin):
    """One request and everything known about running it."""

    __tablename__ = "tasks"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','running','awaiting_approval','completed','failed','cancelled')",
            name="status_valid",
        ),
        CheckConstraint(
            "approval_status IN ('not_required','pending','approved','rejected')",
            name="approval_status_valid",
        ),
        # The dashboard lists a user's recent tasks, which is a filter on
        # owner plus a sort on creation time; one composite index serves both.
        Index("ix_tasks_user_id_created_at", "user_id", "created_at"),
        Index("ix_tasks_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(PrimaryKey, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=True, index=True
    )

    request: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    #: The route the router chose, recorded because it explains the shape of the
    #: run without storing any reasoning.
    route: Mapped[str | None] = mapped_column(String(32), nullable=True)
    complexity: Mapped[str | None] = mapped_column(String(16), nullable=True)

    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    approval_status: Mapped[str] = mapped_column(String(16), nullable=False, default="not_required")

    iteration_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tool_call_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    #: Per-agent cumulative accounting, so token usage survives the process that
    #: produced it.
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    user: Mapped[User] = relationship(back_populates="tasks")
    conversation: Mapped[Conversation | None] = relationship(back_populates="tasks")
    steps: Mapped[list[TaskStep]] = relationship(
        back_populates="task", cascade="all, delete-orphan", passive_deletes=True
    )
    agent_runs: Mapped[list[AgentRun]] = relationship(
        back_populates="task", cascade="all, delete-orphan", passive_deletes=True
    )
    tool_calls: Mapped[list[ToolCall]] = relationship(
        back_populates="task", cascade="all, delete-orphan", passive_deletes=True
    )
    approvals: Mapped[list[Approval]] = relationship(
        back_populates="task", cascade="all, delete-orphan", passive_deletes=True
    )
    events: Mapped[list[ExecutionEvent]] = relationship(
        back_populates="task", cascade="all, delete-orphan", passive_deletes=True
    )

    def __repr__(self) -> str:
        """Return a debug representation showing identity and lifecycle state."""
        return f"<Task {self.id} {self.status}>"


class TaskStep(Base, TimestampMixin):
    """One subtask from the plan, and how it turned out."""

    __tablename__ = "task_steps"
    __table_args__ = (
        # A subtask is attempted once per retry pass; the unique key is on the
        # plan-level identity so a retry updates rather than duplicates the row.
        UniqueConstraint("task_id", "subtask_id", name="task_id_subtask_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(PrimaryKey, primary_key=True, default=uuid.uuid4)
    # No separate index on ``task_id``: the unique constraint below already
    # indexes ``(task_id, subtask_id)``, and its leftmost column serves every
    # lookup by task.
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    #: The plan's own subtask id, kept as text because it is the planner's
    #: namespace, not this schema's.
    subtask_id: Mapped[str] = mapped_column(String(128), nullable=False)
    agent: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    output: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    duration_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    task: Mapped[Task] = relationship(back_populates="steps")

    def __repr__(self) -> str:
        """Return a debug representation of this step."""
        return f"<TaskStep {self.subtask_id} {self.status}>"


class AgentRun(Base, TimestampMixin):
    """One agent invocation, with its own cost and latency."""

    __tablename__ = "agent_runs"
    __table_args__ = (
        # The execution timeline reads one task's runs in order.
        Index("ix_agent_runs_task_id_created_at", "task_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(PrimaryKey, primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    agent: Mapped[str] = mapped_column(String(64), nullable=False)
    #: The graph node that triggered this run, for the execution timeline.
    node: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    task: Mapped[Task] = relationship(back_populates="agent_runs")
    tool_calls: Mapped[list[ToolCall]] = relationship(back_populates="agent_run")

    def __repr__(self) -> str:
        """Return a debug representation of this agent run."""
        return f"<AgentRun {self.agent} {self.status}>"


class ToolCall(Base, TimestampMixin):
    """One tool invocation, recorded whether it succeeded or not.

    Failures are stored deliberately. A tool call that was refused for lack of
    approval is exactly the record an audit needs, and it is the record most
    likely to be dropped if only successes are written.
    """

    __tablename__ = "tool_calls"
    __table_args__ = (
        CheckConstraint(
            "risk_level IN ('LOW','MEDIUM','HIGH','CRITICAL')", name="risk_level_valid"
        ),
        Index("ix_tool_calls_task_id_created_at", "task_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(PrimaryKey, primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    agent_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="SET NULL"), nullable=True
    )

    tool: Mapped[str] = mapped_column(String(64), nullable=False)
    agent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    arguments: Mapped[dict[str, object]] = mapped_column(JsonColumn, nullable=False, default=dict)
    ok: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False, default="LOW")
    approved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    approval_required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    failure_kind: Mapped[str | None] = mapped_column(String(32), nullable=True)
    duration_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    task: Mapped[Task] = relationship(back_populates="tool_calls")
    agent_run: Mapped[AgentRun | None] = relationship(back_populates="tool_calls")

    def __repr__(self) -> str:
        """Return a debug representation of this tool call."""
        return f"<ToolCall {self.tool} ok={self.ok}>"


class Approval(Base, TimestampMixin):
    """A request for a human to approve or reject a gated action."""

    __tablename__ = "approvals"
    __table_args__ = (
        CheckConstraint("decision IN ('pending','approve','reject')", name="decision_valid"),
        # One approval per task, which also indexes every lookup by task.
        UniqueConstraint("task_id", name="task_id"),
        # The pending queue filters on decision alone and has no task to lead
        # with, so it needs its own index.
        Index("ix_approvals_decision", "decision"),
    )

    id: Mapped[uuid.UUID] = mapped_column(PrimaryKey, primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    requested_action: Mapped[str] = mapped_column(Text, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False, default="HIGH")
    decision: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    decided_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    task: Mapped[Task] = relationship(back_populates="approvals")

    def __repr__(self) -> str:
        """Return a debug representation of this approval request."""
        return f"<Approval {self.task_id} {self.decision}>"


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #


class MemoryRecord(Base, TimestampMixin):
    """One durable memory, of one of four kinds.

    ``short_term`` and ``working`` rows are scoped to a conversation or a task
    and are expected to expire. ``long_term`` and ``execution`` rows are the ones
    worth keeping, and are subject to an importance threshold before they are
    written at all.
    """

    __tablename__ = "memory_records"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('short_term','working','long_term','execution')", name="kind_valid"
        ),
        # The read path is "this user's memories of this kind, newest first",
        # so one composite index covers the filter and the sort.
        Index("ix_memory_records_user_id_kind_created_at", "user_id", "kind", "created_at"),
        Index("ix_memory_records_conversation_id", "conversation_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(PrimaryKey, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=True
    )
    task_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), nullable=True
    )

    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: Stored as a float array rather than a pgvector column. Exact cosine
    #: similarity is computed in Python, which is correct at this scale and
    #: keeps the schema free of an extension the deployment may not have. An
    #: approximate index is the scaling path, not a correctness requirement.
    embedding: Mapped[list[float] | None] = mapped_column(JsonColumn, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)

    importance: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    source: Mapped[str | None] = mapped_column(String(128), nullable=True)
    meta: Mapped[dict[str, object]] = mapped_column(JsonColumn, nullable=False, default=dict)
    access_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    accessed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        """Return a debug representation of this memory record."""
        return f"<MemoryRecord {self.kind} {self.id}>"


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #


class ExecutionEvent(Base, TimestampMixin):
    """One client-safe event emitted during a run.

    Ordering is explicit rather than implied by timestamp: two events can share a
    millisecond, and the timeline on the dashboard must still be deterministic.
    """

    __tablename__ = "execution_events"
    __table_args__ = (
        # The unique constraint already indexes ``(task_id, seq)``, which is
        # precisely the stream's read pattern. A second index on the same
        # columns would only add write cost.
        UniqueConstraint("task_id", "seq", name="task_id_seq"),
    )

    id: Mapped[uuid.UUID] = mapped_column(PrimaryKey, primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, object]] = mapped_column(JsonColumn, nullable=False, default=dict)

    task: Mapped[Task] = relationship(back_populates="events")

    def __repr__(self) -> str:
        """Return a debug representation of this event."""
        return f"<ExecutionEvent {self.seq} {self.event_type}>"
