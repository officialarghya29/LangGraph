"""Repositories.

Every database read and write the application performs goes through one of
these classes. Agents and graph nodes never hold a session and never build a
statement: they depend on these narrow interfaces, which is what keeps the
persistence layer replaceable and the permission checks in one place.

Each repository takes a session and does no transaction management of its own.
The caller decides what constitutes one unit of work, via
:meth:`app.database.connection.Database.session`.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any, cast

from sqlalchemy import Result, delete, func, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DatabaseError
from app.database.base import utc_now
from app.database.models import (
    AgentRun,
    Approval,
    Conversation,
    ExecutionEvent,
    MemoryRecord,
    Task,
    TaskStep,
    ToolCall,
    User,
)

__all__ = [
    "AgentRunRepository",
    "ApprovalRepository",
    "ConversationRepository",
    "EventRepository",
    "MemoryRepository",
    "TaskRepository",
    "ToolCallRepository",
    "UserRepository",
    "affected_rows",
]


def affected_rows(result: Result[Any]) -> int:
    """Return how many rows a statement changed.

    ``AsyncSession.execute`` is typed as returning ``Result``, which does not
    expose ``rowcount``; a DML statement actually produces a ``CursorResult``,
    which does. Narrowing in one place keeps four call sites readable and keeps
    the cast honest — the only statements passed here are ``update``/``delete``.
    """
    return int(cast("CursorResult[Any]", result).rowcount or 0)


async def _flush_or_raise(session: AsyncSession, context: str) -> None:
    """Flush pending changes, converting driver failures into ``DatabaseError``.

    The message names the operation, never the values, so a constraint violation
    on a credential column cannot surface the credential in a log line.
    """
    try:
        await session.flush()
    except IntegrityError as exc:
        raise DatabaseError(f"{context} violated a database constraint") from exc
    except SQLAlchemyError as exc:
        raise DatabaseError(f"{context} failed: {type(exc).__name__}") from exc


# --------------------------------------------------------------------------- #
# Users and conversations
# --------------------------------------------------------------------------- #


class UserRepository:
    """Reads and writes for :class:`~app.database.models.User`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, user_id: uuid.UUID) -> User | None:
        """Return a user by primary key."""
        return await self._session.get(User, user_id)

    async def get_by_external_id(self, external_id: str) -> User | None:
        """Return a user by the identity provider's identifier."""
        result = await self._session.execute(select(User).where(User.external_id == external_id))
        return result.scalar_one_or_none()

    async def get_or_create(
        self,
        external_id: str,
        *,
        email: str | None = None,
        display_name: str | None = None,
    ) -> User:
        """Return the user for ``external_id``, creating it on first sight.

        Provisioning on first request is what lets an unauthenticated caller be
        treated as a real principal during development, without a second code
        path once authentication is switched on.
        """
        existing = await self.get_by_external_id(external_id)
        if existing is not None:
            return existing

        user = User(external_id=external_id, email=email, display_name=display_name)
        self._session.add(user)
        try:
            await session_flush(self._session, "user creation")
        except DatabaseError:
            # Two concurrent first requests for the same principal: the other one
            # won the unique constraint, so re-read rather than failing.
            await self._session.rollback()
            raced = await self.get_by_external_id(external_id)
            if raced is None:
                raise
            return raced
        return user

    async def deactivate(self, external_id: str) -> bool:
        """Mark a user inactive. Returns whether a row changed."""
        result = await self._session.execute(
            update(User).where(User.external_id == external_id).values(is_active=False)
        )
        return affected_rows(result) > 0


class ConversationRepository:
    """Reads and writes for :class:`~app.database.models.Conversation`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, user_id: uuid.UUID, *, title: str | None = None) -> Conversation:
        """Create a conversation for a user."""
        conversation = Conversation(user_id=user_id, title=title)
        self._session.add(conversation)
        await session_flush(self._session, "conversation creation")
        return conversation

    async def get_for_user(
        self, conversation_id: uuid.UUID, user_id: uuid.UUID
    ) -> Conversation | None:
        """Return a conversation only if ``user_id`` owns it.

        Returning ``None`` for somebody else's conversation is deliberate: a
        separate "forbidden" answer would confirm that it exists.
        """
        result = await self._session.execute(
            select(Conversation).where(
                Conversation.id == conversation_id, Conversation.user_id == user_id
            )
        )
        return result.scalar_one_or_none()

    async def list_for_user(self, user_id: uuid.UUID, *, limit: int = 50) -> Sequence[Conversation]:
        """Return a user's conversations, newest first."""
        result = await self._session.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #


class TaskRepository:
    """Reads and writes for :class:`~app.database.models.Task`."""

    #: Columns a caller may change through :meth:`update`. An allow-list rather
    #: than ``**kwargs`` splatted into ``values()``, which would let a typo add a
    #: column or an attacker set ``user_id`` and take ownership of a task.
    UPDATABLE: frozenset[str] = frozenset(
        {
            "status",
            "answer",
            "failure_reason",
            "route",
            "complexity",
            "approval_status",
            "iteration_count",
            "retry_count",
            "tool_call_count",
            "started_at",
            "finished_at",
            "prompt_tokens",
            "completion_tokens",
            "duration_ms",
        }
    )

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        user_id: uuid.UUID,
        request: str,
        conversation_id: uuid.UUID | None = None,
        task_id: uuid.UUID | None = None,
    ) -> Task:
        """Create a task.

        Args:
            user_id: Owner.
            request: The request text.
            conversation_id: Conversation to attach to, if any.
            task_id: Optional explicit id. Supplied when the id must match a
                LangGraph thread id created before the row is written.
        """
        task = Task(
            user_id=user_id,
            request=request,
            conversation_id=conversation_id,
            **({"id": task_id} if task_id is not None else {}),
        )
        self._session.add(task)
        await session_flush(self._session, "task creation")
        return task

    async def get(self, task_id: uuid.UUID) -> Task | None:
        """Return a task by primary key, ignoring ownership."""
        return await self._session.get(Task, task_id)

    async def get_for_user(self, task_id: uuid.UUID, user_id: uuid.UUID) -> Task | None:
        """Return a task only if ``user_id`` owns it."""
        result = await self._session.execute(
            select(Task).where(Task.id == task_id, Task.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def update(self, task_id: uuid.UUID, **changes: Any) -> Task | None:
        """Apply allow-listed changes to a task.

        Returns:
            The updated task, or ``None`` if it does not exist.

        Raises:
            ValueError: If ``changes`` names a column outside :attr:`UPDATABLE`.
        """
        unknown = set(changes) - self.UPDATABLE
        if unknown:
            raise ValueError(f"refusing to update non-updatable columns: {sorted(unknown)}")
        if not changes:
            return await self.get(task_id)

        await self._session.execute(
            update(Task).where(Task.id == task_id).values(**changes, updated_at=utc_now())
        )
        task = await self.get(task_id)
        if task is not None:
            # The bulk update above bypasses the identity map, so the in-memory
            # copy must be refreshed or it would keep serving stale values.
            await self._session.refresh(task)
        return task

    async def list_for_user(
        self,
        user_id: uuid.UUID,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[Task]:
        """Return a user's tasks, newest first."""
        statement = select(Task).where(Task.user_id == user_id)
        if status is not None:
            statement = statement.where(Task.status == status)
        result = await self._session.execute(
            statement.order_by(Task.created_at.desc()).limit(limit).offset(offset)
        )
        return result.scalars().all()

    async def list_by_status(self, status: str, *, limit: int = 100) -> Sequence[Task]:
        """Return every task in a given state, oldest first.

        Used by the dashboard's active-task view and by recovery, both of which
        want the longest-waiting task first rather than the newest.
        """
        result = await self._session.execute(
            select(Task).where(Task.status == status).order_by(Task.created_at).limit(limit)
        )
        return result.scalars().all()

    async def count_by_status(self) -> dict[str, int]:
        """Return how many tasks exist in each state."""
        result = await self._session.execute(
            select(Task.status, func.count()).group_by(Task.status)
        )
        return dict(result.all())

    async def active_count(self) -> int:
        """Return how many tasks are unfinished."""
        result = await self._session.execute(
            select(func.count()).select_from(Task).where(Task.status.in_(("pending", "running")))
        )
        return int(result.scalar_one())

    async def delete(self, task_id: uuid.UUID) -> bool:
        """Delete a task and everything cascading from it."""
        result = await self._session.execute(delete(Task).where(Task.id == task_id))
        return affected_rows(result) > 0


class _ChildRepository:
    """Shared behaviour for the repositories that hang off a task."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session


class TaskStepRepository(_ChildRepository):
    """Reads and writes for :class:`~app.database.models.TaskStep`."""

    async def upsert(
        self,
        *,
        task_id: uuid.UUID,
        subtask_id: str,
        agent: str,
        description: str,
        status: str = "pending",
        attempt: int = 1,
    ) -> TaskStep:
        """Create or update the row for one subtask.

        A retry revisits the same subtask, so this updates in place rather than
        appending a second row for the same plan identity.
        """
        existing = await self._session.execute(
            select(TaskStep).where(TaskStep.task_id == task_id, TaskStep.subtask_id == subtask_id)
        )
        step = existing.scalar_one_or_none()
        if step is None:
            step = TaskStep(
                task_id=task_id,
                subtask_id=subtask_id,
                agent=agent,
                description=description,
                status=status,
                attempt=attempt,
            )
            self._session.add(step)
        else:
            step.status = status
            step.agent = agent
            step.description = description
            step.attempt = attempt
        await session_flush(self._session, "task step upsert")
        return step

    async def finish(
        self,
        *,
        task_id: uuid.UUID,
        subtask_id: str,
        status: str,
        output: str | None = None,
        error: str | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Record the outcome of a subtask."""
        await self._session.execute(
            update(TaskStep)
            .where(TaskStep.task_id == task_id, TaskStep.subtask_id == subtask_id)
            .values(
                status=status,
                output=output,
                error=error,
                duration_ms=duration_ms,
                updated_at=utc_now(),
            )
        )

    async def list_for_task(self, task_id: uuid.UUID) -> Sequence[TaskStep]:
        """Return a task's steps in creation order."""
        result = await self._session.execute(
            select(TaskStep).where(TaskStep.task_id == task_id).order_by(TaskStep.created_at)
        )
        return result.scalars().all()


class AgentRunRepository(_ChildRepository):
    """Reads and writes for :class:`~app.database.models.AgentRun`."""

    async def start(
        self,
        *,
        task_id: uuid.UUID,
        agent: str,
        node: str | None = None,
        attempt: int = 1,
    ) -> AgentRun:
        """Open an agent run."""
        run = AgentRun(
            task_id=task_id,
            agent=agent,
            node=node,
            attempt=attempt,
            status="running",
            started_at=utc_now(),
        )
        self._session.add(run)
        await session_flush(self._session, "agent run creation")
        return run

    async def finish(
        self,
        run_id: uuid.UUID,
        *,
        status: str = "completed",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        duration_ms: float = 0.0,
        error: str | None = None,
    ) -> None:
        """Close an agent run and record its cost."""
        await self._session.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(
                status=status,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                duration_ms=duration_ms,
                error=error,
                finished_at=utc_now(),
                updated_at=utc_now(),
            )
        )

    async def list_for_task(self, task_id: uuid.UUID) -> Sequence[AgentRun]:
        """Return a task's agent runs in creation order, for the timeline."""
        result = await self._session.execute(
            select(AgentRun).where(AgentRun.task_id == task_id).order_by(AgentRun.created_at)
        )
        return result.scalars().all()


class ToolCallRepository(_ChildRepository):
    """Reads and writes for :class:`~app.database.models.ToolCall`."""

    async def record(
        self,
        *,
        task_id: uuid.UUID,
        tool: str,
        agent: str | None = None,
        agent_run_id: uuid.UUID | None = None,
        arguments: dict[str, object] | None = None,
        ok: bool = False,
        risk_level: str = "LOW",
        approved: bool = False,
        approval_required: bool = False,
        error: str | None = None,
        failure_kind: str | None = None,
        duration_ms: float = 0.0,
    ) -> ToolCall:
        """Append a tool call record.

        Failures and refusals are recorded alongside successes: an audit that
        only holds successful calls cannot answer why something did not happen.
        """
        call = ToolCall(
            task_id=task_id,
            agent_run_id=agent_run_id,
            tool=tool,
            agent=agent,
            arguments=arguments or {},
            ok=ok,
            risk_level=risk_level,
            approved=approved,
            approval_required=approval_required,
            error=error,
            failure_kind=failure_kind,
            duration_ms=duration_ms,
        )
        self._session.add(call)
        await session_flush(self._session, "tool call record")
        return call

    async def list_for_task(self, task_id: uuid.UUID) -> Sequence[ToolCall]:
        """Return a task's tool calls, oldest first."""
        result = await self._session.execute(
            select(ToolCall).where(ToolCall.task_id == task_id).order_by(ToolCall.created_at)
        )
        return result.scalars().all()


class ApprovalRepository(_ChildRepository):
    """Reads and writes for :class:`~app.database.models.Approval`."""

    async def request(
        self,
        *,
        task_id: uuid.UUID,
        requested_action: str,
        risk_level: str = "HIGH",
    ) -> Approval:
        """Open an approval request, or return the existing one.

        A task has at most one approval row, so a graph that is resumed twice
        cannot create a second request for the same decision.
        """
        existing = await self.get_for_task(task_id)
        if existing is not None:
            return existing

        approval = Approval(
            task_id=task_id,
            requested_action=requested_action,
            risk_level=risk_level,
            decision="pending",
            requested_at=utc_now(),
        )
        self._session.add(approval)
        await session_flush(self._session, "approval request")
        return approval

    async def get_for_task(self, task_id: uuid.UUID) -> Approval | None:
        """Return a task's approval request, if one exists."""
        result = await self._session.execute(select(Approval).where(Approval.task_id == task_id))
        return result.scalar_one_or_none()

    async def decide(
        self,
        task_id: uuid.UUID,
        *,
        decision: str,
        decided_by: str,
        note: str | None = None,
    ) -> Approval | None:
        """Record a human decision.

        Returns:
            The updated approval, or ``None`` if there was no pending request.

        Raises:
            ValueError: If ``decision`` is not ``approve`` or ``reject``, so a
                caller cannot resolve an approval with a pending value.
        """
        if decision not in {"approve", "reject"}:
            raise ValueError(f"decision must be 'approve' or 'reject', got {decision!r}")

        approval = await self.get_for_task(task_id)
        if approval is None or approval.decision != "pending":
            return None

        approval.decision = decision
        approval.decided_by = decided_by
        approval.decided_at = utc_now()
        approval.note = note
        await session_flush(self._session, "approval decision")
        return approval

    async def list_pending(self, *, limit: int = 100) -> Sequence[Approval]:
        """Return every unresolved approval request, oldest first."""
        result = await self._session.execute(
            select(Approval)
            .where(Approval.decision == "pending")
            .order_by(Approval.requested_at)
            .limit(limit)
        )
        return result.scalars().all()


class EventRepository(_ChildRepository):
    """Reads and writes for :class:`~app.database.models.ExecutionEvent`."""

    async def append(
        self,
        *,
        task_id: uuid.UUID,
        event_type: str,
        payload: dict[str, object] | None = None,
    ) -> ExecutionEvent:
        """Append an event, assigning the next sequence number.

        Concurrency is handled by locking the *task* row, not the events. Two
        properties make this correct:

        - PostgreSQL forbids ``FOR UPDATE`` alongside an aggregate, so the
          obvious ``SELECT max(seq) ... FOR UPDATE`` is a runtime error rather
          than the lock it looks like.
        - Locking the parent row serialises every writer for this task, and the
          lock is held until the transaction ends, which is exactly the window
          in which a colliding ``(task_id, seq)`` could otherwise be written.
        """
        await self._session.execute(select(Task.id).where(Task.id == task_id).with_for_update())

        result = await self._session.execute(
            select(func.coalesce(func.max(ExecutionEvent.seq), 0)).where(
                ExecutionEvent.task_id == task_id
            )
        )
        next_seq = int(result.scalar_one()) + 1

        event = ExecutionEvent(
            task_id=task_id, seq=next_seq, event_type=event_type, payload=payload or {}
        )
        self._session.add(event)
        await session_flush(self._session, "event append")
        return event

    async def list_for_task(
        self, task_id: uuid.UUID, *, after_seq: int = 0, limit: int = 500
    ) -> Sequence[ExecutionEvent]:
        """Return a task's events after ``after_seq``, in order.

        ``after_seq`` is what makes a reconnect to the event stream resumable
        without replaying everything the client already saw.
        """
        result = await self._session.execute(
            select(ExecutionEvent)
            .where(ExecutionEvent.task_id == task_id, ExecutionEvent.seq > after_seq)
            .order_by(ExecutionEvent.seq)
            .limit(limit)
        )
        return result.scalars().all()


class MemoryRepository(_ChildRepository):
    """Reads and writes for :class:`~app.database.models.MemoryRecord`."""

    async def create(
        self,
        *,
        user_id: uuid.UUID,
        kind: str,
        content: str,
        conversation_id: uuid.UUID | None = None,
        task_id: uuid.UUID | None = None,
        summary: str | None = None,
        embedding: list[float] | None = None,
        embedding_model: str | None = None,
        importance: float = 0.5,
        source: str | None = None,
        meta: dict[str, object] | None = None,
    ) -> MemoryRecord:
        """Persist a memory record."""
        record = MemoryRecord(
            user_id=user_id,
            conversation_id=conversation_id,
            task_id=task_id,
            kind=kind,
            content=content,
            summary=summary,
            embedding=embedding,
            embedding_model=embedding_model,
            importance=max(0.0, min(1.0, importance)),
            source=source,
            meta=meta or {},
        )
        self._session.add(record)
        await session_flush(self._session, "memory creation")
        return record

    async def list_for_user(
        self,
        user_id: uuid.UUID,
        *,
        kind: str | None = None,
        conversation_id: uuid.UUID | None = None,
        task_id: uuid.UUID | None = None,
        limit: int = 200,
    ) -> Sequence[MemoryRecord]:
        """Return a user's memories, newest first, with optional narrowings."""
        statement = select(MemoryRecord).where(MemoryRecord.user_id == user_id)
        if kind is not None:
            statement = statement.where(MemoryRecord.kind == kind)
        if conversation_id is not None:
            statement = statement.where(MemoryRecord.conversation_id == conversation_id)
        if task_id is not None:
            statement = statement.where(MemoryRecord.task_id == task_id)
        result = await self._session.execute(
            statement.order_by(MemoryRecord.created_at.desc()).limit(limit)
        )
        return result.scalars().all()

    async def mark_accessed(self, record_ids: Sequence[uuid.UUID]) -> None:
        """Record that memories were retrieved, for recency ranking."""
        if not record_ids:
            return
        await self._session.execute(
            update(MemoryRecord)
            .where(MemoryRecord.id.in_(record_ids))
            .values(access_count=MemoryRecord.access_count + 1, accessed_at=utc_now())
        )

    async def delete(self, record_id: uuid.UUID, *, user_id: uuid.UUID) -> bool:
        """Delete one of a user's memories. Returns whether a row was removed."""
        result = await self._session.execute(
            delete(MemoryRecord).where(
                MemoryRecord.id == record_id, MemoryRecord.user_id == user_id
            )
        )
        return affected_rows(result) > 0

    async def prune(
        self,
        *,
        kind: str,
        older_than: datetime,
        limit: int = 1000,
    ) -> int:
        """Delete old records of one kind. Returns how many were removed.

        Used to stop short-term and working memory growing without bound. Only
        the volatile kinds are pruned; long-term memory is curated, not expired.
        """
        if kind not in {"short_term", "working"}:
            raise ValueError(f"refusing to prune durable memory kind {kind!r}")
        result = await self._session.execute(
            delete(MemoryRecord)
            .where(MemoryRecord.kind == kind, MemoryRecord.created_at < older_than)
            .execution_options(synchronize_session=False)
        )
        return affected_rows(result)

    async def count_for_user(self, user_id: uuid.UUID) -> dict[str, int]:
        """Return how many memories a user has of each kind."""
        result = await self._session.execute(
            select(MemoryRecord.kind, func.count())
            .where(MemoryRecord.user_id == user_id)
            .group_by(MemoryRecord.kind)
        )
        return dict(result.all())


#: Local alias so :class:`UserRepository` can flush without importing the helper
#: name, which would collide with the module-level ``_flush_or_raise`` signature.
async def session_flush(session: AsyncSession, context: str) -> None:
    """Flush a session, converting driver failures into ``DatabaseError``."""
    await _flush_or_raise(session, context)
