"""PostgreSQL-backed task storage.

The API is coded against :class:`~app.services.tasks.TaskStore`, so a durable
implementation drops in without a route changing. Two things make this more than
a translation layer:

- **Every read is owner-scoped.** ``get`` exists only for internal callers that
  already resolved ownership; everything a request touches goes through
  ``get_for_user``. A task visible to the wrong caller is the failure this
  layer exists to prevent.
- **The durable side trail is written as the run proceeds.** Steps, agent runs,
  tool calls, approvals, and events land in their own tables rather than being
  reconstructed at the end. A run that crashes is exactly the run whose history
  matters, and reconstruction after the fact cannot recover it.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from app.core.config import Settings
from app.database.connection import Database
from app.database.repositories import (
    AgentRunRepository,
    ApprovalRepository,
    EventRepository,
    TaskRepository,
    TaskStepRepository,
    ToolCallRepository,
    UserRepository,
)
from app.graph.checkpoints import thread_config
from app.graph.nodes import event_sink_for
from app.graph.state import initial_state
from app.models.approval import ApprovalStatus
from app.models.execution import TaskStatus
from app.observability.metrics import MetricRegistry
from app.observability.tracing import Tracer
from app.services.budget import budget_for, token_budget

__all__ = ["PostgresTaskStore", "TaskRecord", "TaskStore", "execute_task_with_store"]

logger = logging.getLogger(__name__)


class TaskRecord(BaseModel):
    """The client-visible state of one task.

    Carries no prompt, no reasoning, and no tool arguments — only what a caller
    is entitled to see about their own task. The durable task id is a string
    here because it crosses the API boundary, where a UUID would serialise to
    the same text anyway.
    """

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "anonymous"
    request: str
    status: TaskStatus = TaskStatus.PENDING
    answer: str | None = None
    failure_reason: str | None = None
    route: str | None = None
    approval_status: ApprovalStatus = ApprovalStatus.NOT_REQUIRED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_terminal(self) -> bool:
        """Return whether the task has finished."""
        return self.status.is_terminal


#: Columns a caller may change. Mirrors
#: :attr:`app.database.repositories.TaskRepository.UPDATABLE`.
_UPDATABLE = frozenset(
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


def _coerce(column: str, value: object) -> object:
    """Convert an API-facing value to what the column stores.

    The API speaks in ``TaskStatus`` and ``ApprovalStatus`` enums; the database
    stores text. Converting at one boundary keeps every repository free of
    enum awareness and makes an unexpected type a visible error rather than a
    value stored as its ``repr``.
    """
    if column in {"status", "approval_status"}:
        return value.value if hasattr(value, "value") else value
    return value


class TaskStore:
    """The storage surface the API is coded against.

    Declared as a real base class rather than a ``Protocol`` so an incomplete
    implementation fails at import with a clear error instead of failing later at
    whichever method a request happens to call first.

    The side-trail methods default to doing nothing. A store that does not keep a
    durable trail says so by inheriting that default, rather than by the runner
    branching on which store it was given — which would put the difference in
    every call site instead of in one place.
    """

    async def create(self, record: TaskRecord) -> TaskRecord:
        """Persist a new record."""
        raise NotImplementedError

    async def get(self, task_id: str) -> TaskRecord | None:
        """Return a record by id, or ``None``."""
        raise NotImplementedError

    async def get_for_user(self, task_id: str, user_id: str) -> TaskRecord | None:
        """Return a record only if ``user_id`` owns it."""
        raise NotImplementedError

    async def update(self, task_id: str, **changes: object) -> TaskRecord | None:
        """Apply changes to a record, or return ``None`` if it is absent."""
        raise NotImplementedError

    async def list(self, *, user_id: str | None = None, limit: int = 50) -> Sequence[TaskRecord]:
        """Return recent records, newest first."""
        raise NotImplementedError

    # --- Durable side trail --------------------------------------------- #

    async def append_event(
        self, task_id: str, event_type: str, payload: dict[str, object] | None = None
    ) -> None:
        """Append an execution event. No-op when the store is not durable."""

    async def list_events(self, task_id: str, *, after_seq: int = 0) -> Sequence[dict[str, object]]:
        """Return persisted events after a sequence number."""
        return []

    async def request_approval(
        self, task_id: str, *, requested_action: str, risk_level: str = "HIGH"
    ) -> None:
        """Persist an approval request. No-op when the store is not durable."""

    async def decide_approval(
        self, task_id: str, *, decision: str, decided_by: str, note: str | None = None
    ) -> bool:
        """Record a human decision. Returns whether one was outstanding."""
        return False

    async def list_pending_approvals(self) -> Sequence[dict[str, object]]:
        """Return every unresolved approval, oldest first."""
        return []

    async def counts(self) -> dict[str, int]:
        """Return task counts by status."""
        return {}


class PostgresTaskStore(TaskStore):
    """A durable task store on PostgreSQL.

    Implements the same surface as :class:`~app.services.tasks.InMemoryTaskStore`,
    including the invariant that a task belonging to another caller is
    indistinguishable from one that does not exist.
    """

    def __init__(self, database: Database) -> None:
        self._database = database

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_record(task: Any) -> TaskRecord:
        """Convert an ORM row into the API-facing record."""
        return TaskRecord(
            task_id=str(task.id),
            user_id=str(task.user_id),
            request=task.request,
            status=TaskStatus(task.status),
            answer=task.answer,
            failure_reason=task.failure_reason,
            route=task.route,
            approval_status=ApprovalStatus(task.approval_status),
            created_at=task.created_at,
            updated_at=task.updated_at,
        )

    @staticmethod
    async def _resolve_user(session: Any, external_id: str) -> uuid.UUID:
        """Return the database id for a principal, provisioning on first sight."""
        user = await UserRepository(session).get_or_create(external_id)
        return uuid.UUID(str(user.id))

    # ------------------------------------------------------------------ #
    # TaskStore surface
    # ------------------------------------------------------------------ #

    async def create(self, record: TaskRecord) -> TaskRecord:
        """Persist a new task.

        The caller-supplied id is honoured, so the record, the LangGraph thread,
        and any log line all refer to the run by one identifier.
        """
        task_id = uuid.UUID(record.task_id)
        async with self._database.session() as session:
            user_id = await self._resolve_user(session, record.user_id)
            task = await TaskRepository(session).create(
                user_id=user_id,
                request=record.request,
                task_id=task_id,
            )
            await EventRepository(session).append(
                task_id=task.id, event_type="task_created", payload={"route": None}
            )
            return self._to_record(task)

    async def get(self, task_id: str) -> TaskRecord | None:
        """Return a task by id, ignoring ownership.

        For internal callers that have already established ownership. A request
        handler must use :meth:`get_for_user`.
        """
        try:
            key = uuid.UUID(task_id)
        except ValueError:
            return None

        async with self._database.session() as session:
            task = await TaskRepository(session).get(key)
            return None if task is None else self._to_record(task)

    async def get_for_user(self, task_id: str, user_id: str) -> TaskRecord | None:
        """Return a task only if ``user_id`` owns it.

        A malformed id returns ``None`` rather than raising: the caller asked for
        a task, and the answer is that they do not have one. Distinguishing
        "malformed" from "not yours" would leak the existence of other tasks.
        """
        try:
            key = uuid.UUID(task_id)
        except ValueError:
            return None

        async with self._database.session() as session:
            owner = await UserRepository(session).get_by_external_id(user_id)
            if owner is None:
                return None
            task = await TaskRepository(session).get_for_user(key, uuid.UUID(str(owner.id)))
            return None if task is None else self._to_record(task)

    async def update(self, task_id: str, **changes: object) -> TaskRecord | None:
        """Apply changes to a task.

        Raises:
            ValueError: If ``changes`` names a column that is not updatable, or
                if the id is not a UUID.
        """
        unknown = set(changes) - _UPDATABLE
        if unknown:
            raise ValueError(f"refusing to update non-updatable columns: {sorted(unknown)}")

        key = uuid.UUID(task_id)
        coerced = {name: _coerce(name, value) for name, value in changes.items()}

        async with self._database.session() as session:
            task = await TaskRepository(session).update(key, **coerced)
            return None if task is None else self._to_record(task)

    async def list(self, *, user_id: str | None = None, limit: int = 50) -> list[TaskRecord]:
        """Return recent tasks, newest first, optionally scoped to one owner."""
        async with self._database.session() as session:
            if user_id is None:
                # Internal use only — the dashboard's aggregate view. It is never
                # reachable from a task-scoped request handler.
                tasks = await TaskRepository(session).list_by_status("pending", limit=limit)
                tasks = list(tasks) + list(
                    await TaskRepository(session).list_by_status("running", limit=limit)
                )
                return [self._to_record(task) for task in tasks[:limit]]

            owner = await UserRepository(session).get_by_external_id(user_id)
            if owner is None:
                return []
            rows = await TaskRepository(session).list_for_user(
                uuid.UUID(str(owner.id)), limit=limit
            )
            return [self._to_record(task) for task in rows]

    # ------------------------------------------------------------------ #
    # Durable side trail
    # ------------------------------------------------------------------ #

    async def record_step(
        self,
        task_id: str,
        *,
        subtask_id: str,
        agent: str,
        description: str,
        status: str = "running",
        attempt: int = 1,
    ) -> None:
        """Open or update a step in the durable trail."""
        async with self._database.session() as session:
            await TaskStepRepository(session).upsert(
                task_id=uuid.UUID(task_id),
                subtask_id=subtask_id,
                agent=agent,
                description=description,
                status=status,
                attempt=attempt,
            )

    async def finish_step(
        self,
        task_id: str,
        *,
        subtask_id: str,
        status: str,
        output: str | None = None,
        error: str | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Close a step in the durable trail."""
        async with self._database.session() as session:
            await TaskStepRepository(session).finish(
                task_id=uuid.UUID(task_id),
                subtask_id=subtask_id,
                status=status,
                output=output,
                error=error,
                duration_ms=duration_ms,
            )

    async def record_tool_call(
        self,
        task_id: str,
        *,
        tool: str,
        agent: str | None = None,
        arguments: dict[str, object] | None = None,
        ok: bool = False,
        risk_level: str = "LOW",
        approved: bool = False,
        approval_required: bool = False,
        error: str | None = None,
        failure_kind: str | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Append a tool call to the audit trail, success or failure."""
        async with self._database.session() as session:
            await ToolCallRepository(session).record(
                task_id=uuid.UUID(task_id),
                tool=tool,
                agent=agent,
                arguments=arguments,
                ok=ok,
                risk_level=risk_level,
                approved=approved,
                approval_required=approval_required,
                error=error,
                failure_kind=failure_kind,
                duration_ms=duration_ms,
            )

    async def record_agent_run(
        self,
        task_id: str,
        *,
        agent: str,
        node: str | None = None,
        status: str = "completed",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        duration_ms: float = 0.0,
        error: str | None = None,
    ) -> None:
        """Record a completed agent run with its cost."""
        async with self._database.session() as session:
            repo = AgentRunRepository(session)
            run = await repo.start(task_id=uuid.UUID(task_id), agent=agent, node=node)
            await repo.finish(
                run.id,
                status=status,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_ms=duration_ms,
                error=error,
            )

    async def append_event(
        self, task_id: str, event_type: str, payload: dict[str, object] | None = None
    ) -> None:
        """Append an execution event, assigning the next sequence number."""
        async with self._database.session() as session:
            await EventRepository(session).append(
                task_id=uuid.UUID(task_id), event_type=event_type, payload=payload
            )

    async def request_approval(
        self, task_id: str, *, requested_action: str, risk_level: str = "HIGH"
    ) -> None:
        """Persist an approval request so it survives a restart."""
        async with self._database.session() as session:
            await ApprovalRepository(session).request(
                task_id=uuid.UUID(task_id),
                requested_action=requested_action,
                risk_level=risk_level,
            )

    async def decide_approval(
        self, task_id: str, *, decision: str, decided_by: str, note: str | None = None
    ) -> bool:
        """Record a human decision. Returns whether one was outstanding."""
        async with self._database.session() as session:
            decided = await ApprovalRepository(session).decide(
                task_id=uuid.UUID(task_id),
                decision=decision,
                decided_by=decided_by,
                note=note,
            )
            return decided is not None

    # The return annotations below say ``Sequence`` rather than ``list``, because
    # the ``list`` method above shadows the builtin inside this class body and a
    # later ``list[...]`` would resolve to the method.
    async def list_events(self, task_id: str, *, after_seq: int = 0) -> Sequence[dict[str, object]]:
        """Return a task's persisted events after a sequence number."""
        async with self._database.session() as session:
            events = await EventRepository(session).list_for_task(
                uuid.UUID(task_id), after_seq=after_seq
            )
            return [
                {
                    "seq": event.seq,
                    "type": event.event_type,
                    "payload": event.payload,
                    "created_at": event.created_at.isoformat(),
                }
                for event in events
            ]

    async def list_pending_approvals(self) -> Sequence[dict[str, object]]:
        """Return every unresolved approval, oldest first, for the dashboard."""
        async with self._database.session() as session:
            pending = await ApprovalRepository(session).list_pending()
            return [
                {
                    "task_id": str(approval.task_id),
                    "requested_action": approval.requested_action,
                    "risk_level": approval.risk_level,
                    "requested_at": approval.requested_at.isoformat(),
                }
                for approval in pending
            ]

    async def counts(self) -> dict[str, int]:
        """Return task counts by status, for the dashboard summary."""
        async with self._database.session() as session:
            return await TaskRepository(session).count_by_status()


async def execute_task_with_store(
    task_id: str,
    *,
    graph: Any,
    store: TaskStore,
    settings: Settings,
    tracer: Tracer | None = None,
    metrics: MetricRegistry | None = None,
) -> None:
    """Run one task through the graph and record the outcome durably.

    Intended to run after the HTTP response has been sent. Never raises: a
    background failure is recorded on the task, because there is no longer a
    request to fail.

    Args:
        task_id: Task to run. Must already exist in the store.
        graph: The compiled orchestration graph.
        store: Where to record progress.
        settings: Application settings, for the execution ceilings.
        tracer: Optional tracer. The span covers the whole run, so its duration
            is the number that matters when a task feels slow.
        metrics: Optional registry for task outcomes.
    """

    def record_outcome(outcome: str) -> None:
        """Count a terminal outcome. Only three exist, so cardinality is fixed."""
        if metrics is not None:
            metrics.increment("tasks_total", outcome=outcome)

    span_context = tracer.span("task.execute", task_id=task_id) if tracer else nullcontext()
    with span_context:
        await _execute_task(
            task_id, graph=graph, store=store, settings=settings, record_outcome=record_outcome
        )


async def _execute_task(
    task_id: str,
    *,
    graph: Any,
    store: TaskStore,
    settings: Settings,
    record_outcome: Callable[[str], None],
) -> None:
    """Carry out one run and record its outcome.

    Split from :func:`execute_task_with_store` so the span and the work have the
    same lifetime without the whole body being indented under a ``with``.
    """
    record = await store.get(task_id)
    if record is None:  # pragma: no cover - defensive
        logger.warning("task.missing", extra={"task_id": task_id})
        return

    await store.update(task_id, status=TaskStatus.RUNNING, started_at=datetime.now(UTC))
    await store.append_event(task_id, "task_started", {"request_length": len(record.request)})

    try:
        state = initial_state(
            record.request,
            user_id=record.user_id,
            task_id=task_id,
            iteration_limit=settings.max_agent_iterations,
            retry_limit=settings.max_retries,
        )

        async def publish(event_type: str, payload: dict[str, object]) -> None:
            await store.append_event(task_id, event_type, payload)

        # Both contexts are bound to this run, not to the shared compiled graph,
        # so two concurrent tasks never write into each other's event history or
        # pool their token counts. The budget writes into this state's usage
        # object, which is the same one the terminal update reads below, so the
        # task record ends up with the real numbers rather than zeros.
        budget = budget_for(state, limit=settings.max_token_budget)
        with event_sink_for(publish), token_budget(budget):
            result = await graph.ainvoke(state, thread_config(task_id))
    except Exception as exc:
        # The class name only. An exception string can carry a prompt fragment,
        # a URL, or a credential.
        logger.exception("task.failed", extra={"task_id": task_id})
        reason = f"the run failed with {type(exc).__name__}"
        await store.update(
            task_id,
            status=TaskStatus.FAILED,
            failure_reason=reason,
            finished_at=datetime.now(UTC),
        )
        await store.append_event(task_id, "task_failed", {"reason": reason})
        record_outcome("failed")
        return

    approval_status = result.get("approval_status")
    route = result.get("route")
    route_name = getattr(getattr(route, "route", None), "value", None)

    if approval_status is ApprovalStatus.PENDING:
        # A run that stopped at an interrupt is awaiting a human, not complete.
        await store.update(
            task_id,
            status=TaskStatus.AWAITING_APPROVAL,
            approval_status=ApprovalStatus.PENDING,
            route=route_name,
        )
        await store.request_approval(
            task_id,
            requested_action=record.request,
            risk_level="HIGH",
        )
        await store.append_event(task_id, "approval_required", {"route": route_name})
        record_outcome("awaiting_approval")
        return

    metadata = result.get("execution_metadata")
    await store.update(
        task_id,
        status=TaskStatus.COMPLETED,
        answer=result.get("final_answer"),
        route=route_name,
        approval_status=approval_status or ApprovalStatus.NOT_REQUIRED,
        finished_at=datetime.now(UTC),
        duration_ms=float(getattr(metadata, "duration_ms", 0.0) or 0.0),
        prompt_tokens=int(getattr(getattr(metadata, "usage", None), "prompt_tokens", 0) or 0),
        completion_tokens=int(
            getattr(getattr(metadata, "usage", None), "completion_tokens", 0) or 0
        ),
    )
    await store.append_event(task_id, "task_completed", {"route": route_name})
    record_outcome("completed")
