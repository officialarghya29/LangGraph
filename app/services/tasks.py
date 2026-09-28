"""In-memory task storage.

The real store is :class:`~app.services.task_store.PostgresTaskStore`. This one
exists for tests that exercise the API without a database, and it is a complete
implementation of the same surface rather than a stub — the ownership rule is
enforced here exactly as it is there, so a route tested against this store is
tested against the real contract.

It is deliberately not a selectable deployment option: tasks, approvals, and
memories are durable state, and a process-scoped store would lose all of them on
a restart while still appearing to work.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from app.services.task_store import TaskRecord, TaskStore

__all__ = ["InMemoryTaskStore", "TaskRecord", "TaskStore"]


class InMemoryTaskStore(TaskStore):
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

    async def list(self, *, user_id: str | None = None, limit: int = 50) -> Sequence[TaskRecord]:
        """Return recent records, newest first, optionally scoped to one user."""
        records = list(self._records.values())
        if user_id is not None:
            records = [record for record in records if record.user_id == user_id]
        records.sort(key=lambda record: record.created_at, reverse=True)
        return records[:limit]
