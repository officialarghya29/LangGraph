"""Task records and storage.

A task is the durable handle a caller holds while a graph run proceeds. The
store keeps only what a client is allowed to see: status, the final answer, and
a coarse failure reason. No prompts, no reasoning, no tool arguments.

The in-memory store is real and fully functional within one process. A
PostgreSQL-backed store replaces it in Phase 22, at which point this interface
is what the API is already coded against.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.core.config import Settings
from app.graph.checkpoints import thread_config
from app.graph.state import initial_state
from app.models.approval import ApprovalStatus
from app.models.execution import TaskStatus

__all__ = ["InMemoryTaskStore", "TaskRecord", "TaskStore", "execute_task"]

logger = logging.getLogger(__name__)


class TaskRecord(BaseModel):
    """The client-visible state of one task."""

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


class TaskStore(Protocol):
    """Storage for task records."""

    async def create(self, record: TaskRecord) -> TaskRecord:
        """Persist a new record."""
        ...

    async def get(self, task_id: str) -> TaskRecord | None:
        """Return a record by id, or ``None``."""
        ...

    async def update(self, task_id: str, **changes: object) -> TaskRecord | None:
        """Apply changes to a record, or return ``None`` if it is absent."""
        ...

    async def list(self, *, user_id: str | None = None, limit: int = 50) -> list[TaskRecord]:
        """Return recent records, newest first."""
        ...


class InMemoryTaskStore:
    """A task store held in process memory."""

    def __init__(self) -> None:
        self._records: dict[str, TaskRecord] = {}

    async def create(self, record: TaskRecord) -> TaskRecord:
        """Store a new record."""
        self._records[record.task_id] = record
        return record

    async def get(self, task_id: str) -> TaskRecord | None:
        """Return a record by id."""
        return self._records.get(task_id)

    async def get_for_user(self, task_id: str, user_id: str) -> TaskRecord | None:
        """Return a record only if ``user_id`` owns it.

        Returning ``None`` rather than raising lets the API answer 404. That is
        deliberate: a 403 would confirm that somebody else's task exists.
        """
        record = self._records.get(task_id)
        if record is None or record.user_id != user_id:
            return None
        return record

    async def update(self, task_id: str, **changes: object) -> TaskRecord | None:
        """Apply changes to a record."""
        record = self._records.get(task_id)
        if record is None:
            return None

        updated = record.model_copy(update={**changes, "updated_at": datetime.now(UTC)})
        self._records[task_id] = updated
        return updated

    async def list(self, *, user_id: str | None = None, limit: int = 50) -> list[TaskRecord]:
        """Return recent records, newest first, optionally scoped to one user."""
        records = list(self._records.values())
        if user_id is not None:
            records = [record for record in records if record.user_id == user_id]
        records.sort(key=lambda record: record.created_at, reverse=True)
        return records[:limit]


def _route_name(decision: Any) -> str | None:
    """Return a routing decision's route name, if the run got as far as routing.

    The decision comes back from graph state as an untyped value, so this reads
    it defensively rather than assuming routing succeeded.
    """
    route = getattr(decision, "route", None)
    if route is None:
        return None
    return str(getattr(route, "value", route))


async def execute_task(
    task_id: str,
    *,
    graph: Any,
    store: TaskStore,
    settings: Settings,
) -> None:
    """Run one task through the graph and record the outcome.

    Intended to run after the HTTP response has been sent. It never raises: a
    background failure is recorded on the task, because there is no longer a
    request to fail.

    Args:
        task_id: The task to execute. Must already exist in the store.
        graph: The compiled orchestration graph.
        store: Where to record progress.
        settings: Application settings, for the execution ceilings.
    """
    record = await store.get(task_id)
    if record is None:  # pragma: no cover - defensive
        logger.warning("task.missing", extra={"task_id": task_id})
        return

    await store.update(task_id, status=TaskStatus.RUNNING)

    try:
        state = initial_state(
            record.request,
            user_id=record.user_id,
            task_id=task_id,
            iteration_limit=settings.max_agent_iterations,
            retry_limit=settings.max_retries,
        )
        result = await graph.ainvoke(state, thread_config(task_id))
    except Exception as exc:
        # The class name only. An exception string can carry a prompt fragment,
        # a URL, or a credential.
        logger.exception("task.failed", extra={"task_id": task_id})
        await store.update(
            task_id,
            status=TaskStatus.FAILED,
            failure_reason=f"the run failed with {type(exc).__name__}",
        )
        return

    approval_status = result.get("approval_status")

    # A run that stopped at an interrupt is awaiting a human, not complete.
    if approval_status is ApprovalStatus.PENDING:
        await store.update(
            task_id,
            status=TaskStatus.AWAITING_APPROVAL,
            approval_status=ApprovalStatus.PENDING,
            route=_route_name(result.get("route")),
        )
        return

    await store.update(
        task_id,
        status=TaskStatus.COMPLETED,
        answer=result.get("final_answer"),
        route=_route_name(result.get("route")),
        approval_status=approval_status or ApprovalStatus.NOT_REQUIRED,
    )
