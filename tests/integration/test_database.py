"""Integration tests for the persistence layer, against real PostgreSQL.

These exercise the parts of the schema that only a real server can prove:
foreign key enforcement, cascades, unique and check constraints, JSONB columns,
and the row locking used to allocate event sequence numbers.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DatabaseError
from app.database.connection import Database
from app.database.models import Approval, Task, ToolCall, User
from app.database.repositories import (
    AgentRunRepository,
    ApprovalRepository,
    ConversationRepository,
    EventRepository,
    MemoryRepository,
    TaskRepository,
    TaskStepRepository,
    ToolCallRepository,
    UserRepository,
)

#: An error either the driver or the repository raises for a bad write. Which one
#: depends on whether the constraint is reported at flush or mapped first.
BadWrite = (IntegrityError, DatabaseError)


# --------------------------------------------------------------------------- #
# Connectivity and schema
# --------------------------------------------------------------------------- #


async def test_the_database_answers_a_ping(database: Database) -> None:
    ok, detail = await database.ping()

    assert ok is True
    assert detail == "ok"


async def test_every_migration_is_applied(session: AsyncSession) -> None:
    """The test database must be at the head revision, not partially built."""
    result = await session.execute(text("SELECT version_num FROM alembic_version"))

    assert result.scalar_one()


async def test_the_expected_tables_exist(session: AsyncSession) -> None:
    result = await session.execute(
        text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
    )
    tables = {row[0] for row in result.all()}

    assert {
        "users",
        "conversations",
        "tasks",
        "task_steps",
        "agent_runs",
        "tool_calls",
        "approvals",
        "memory_records",
        "execution_events",
    } <= tables


async def test_json_columns_are_stored_as_jsonb(session: AsyncSession) -> None:
    """JSONB is what makes a payload queryable rather than an opaque string."""
    result = await session.execute(
        text(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'execution_events' AND column_name = 'payload'"
        )
    )

    assert result.scalar_one() == "jsonb"


async def test_uuid_primary_keys_are_native_uuids(session: AsyncSession) -> None:
    """A char column would double the storage and defeat range comparisons."""
    result = await session.execute(
        text(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'tasks' AND column_name = 'id'"
        )
    )

    assert result.scalar_one() == "uuid"


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #


async def test_a_user_is_provisioned_on_first_sight(session: AsyncSession) -> None:
    user = await UserRepository(session).get_or_create("alice")

    assert user.id is not None
    assert user.external_id == "alice"
    assert user.is_active is True


async def test_provisioning_the_same_principal_is_idempotent(session: AsyncSession) -> None:
    repo = UserRepository(session)

    first = await repo.get_or_create("alice")
    second = await repo.get_or_create("alice")

    assert first.id == second.id


async def test_external_identifiers_are_unique(session: AsyncSession) -> None:
    """Two principals cannot share one identity-provider subject."""
    session.add(User(external_id="duplicate"))
    await session.flush()

    session.add(User(external_id="duplicate"))
    with pytest.raises(IntegrityError):
        await session.flush()


async def test_a_user_can_be_deactivated(session: AsyncSession) -> None:
    repo = UserRepository(session)
    await repo.get_or_create("alice")

    assert await repo.deactivate("alice") is True

    fetched = await repo.get_by_external_id("alice")
    assert fetched is not None
    assert fetched.is_active is False


async def test_deactivating_someone_who_does_not_exist_changes_nothing(
    session: AsyncSession,
) -> None:
    assert await UserRepository(session).deactivate("nobody") is False


# --------------------------------------------------------------------------- #
# Conversations and ownership
# --------------------------------------------------------------------------- #


async def test_a_conversation_belongs_to_its_owner(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    repo = ConversationRepository(session)

    conversation = await repo.create(alice.id, title="first")

    assert await repo.get_for_user(conversation.id, alice.id) is not None


async def test_another_user_cannot_read_a_conversation(session: AsyncSession) -> None:
    users = UserRepository(session)
    alice = await users.get_or_create("alice")
    mallory = await users.get_or_create("mallory")
    repo = ConversationRepository(session)
    conversation = await repo.create(alice.id, title="private")

    assert await repo.get_for_user(conversation.id, mallory.id) is None


async def test_conversations_are_listed_newest_first(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    repo = ConversationRepository(session)
    await repo.create(alice.id, title="older")
    await repo.create(alice.id, title="newer")

    titles = [conversation.title for conversation in await repo.list_for_user(alice.id)]

    assert titles == ["newer", "older"]


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #


async def test_a_task_is_created_owned_and_pending(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")

    task = await TaskRepository(session).create(user_id=alice.id, request="do it")

    assert task.status == "pending"
    assert task.approval_status == "not_required"
    assert task.user_id == alice.id


async def test_a_task_can_be_created_under_a_caller_supplied_id(
    session: AsyncSession,
) -> None:
    """The id must be settable so it can double as the LangGraph thread id."""
    alice = await UserRepository(session).get_or_create("alice")
    chosen = uuid.uuid4()

    task = await TaskRepository(session).create(user_id=alice.id, request="do it", task_id=chosen)

    assert task.id == chosen


async def test_owner_scoped_lookup_hides_another_users_task(session: AsyncSession) -> None:
    users = UserRepository(session)
    alice = await users.get_or_create("alice")
    mallory = await users.get_or_create("mallory")
    repo = TaskRepository(session)
    task = await repo.create(user_id=alice.id, request="private")

    assert await repo.get_for_user(task.id, alice.id) is not None
    assert await repo.get_for_user(task.id, mallory.id) is None


async def test_updating_a_task_reflects_immediately(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    repo = TaskRepository(session)
    task = await repo.create(user_id=alice.id, request="do it")

    updated = await repo.update(task.id, status="completed", answer="done")

    assert updated is not None
    assert updated.status == "completed"
    assert updated.answer == "done"


async def test_an_invalid_status_is_rejected_by_the_database(session: AsyncSession) -> None:
    """The check constraint is the last line of defence behind the Python enum."""
    alice = await UserRepository(session).get_or_create("alice")
    task = await TaskRepository(session).create(user_id=alice.id, request="do it")

    with pytest.raises(BadWrite):
        await session.execute(
            text("UPDATE tasks SET status = 'nonsense' WHERE id = :tid"), {"tid": task.id}
        )
        await session.flush()


async def test_updating_a_non_updatable_column_is_refused(session: AsyncSession) -> None:
    """``user_id`` is not updatable: it would hand the task to somebody else."""
    alice = await UserRepository(session).get_or_create("alice")
    repo = TaskRepository(session)
    task = await repo.create(user_id=alice.id, request="do it")

    with pytest.raises(ValueError, match="non-updatable"):
        await repo.update(task.id, user_id=uuid.uuid4())


async def test_updating_a_missing_task_returns_none(session: AsyncSession) -> None:
    assert await TaskRepository(session).update(uuid.uuid4(), status="completed") is None


async def test_task_counts_group_by_status(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    repo = TaskRepository(session)
    await repo.create(user_id=alice.id, request="a")
    second = await repo.create(user_id=alice.id, request="b")
    await repo.update(second.id, status="completed")

    counts = await repo.count_by_status()

    assert counts.get("pending") == 1
    assert counts.get("completed") == 1


async def test_active_count_excludes_finished_tasks(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    repo = TaskRepository(session)
    await repo.create(user_id=alice.id, request="a")
    finished = await repo.create(user_id=alice.id, request="b")
    await repo.update(finished.id, status="completed")

    assert await repo.active_count() == 1


async def test_a_task_requires_an_existing_owner(session: AsyncSession) -> None:
    with pytest.raises(BadWrite):
        await TaskRepository(session).create(user_id=uuid.uuid4(), request="orphan")


async def test_tasks_can_be_listed_by_status(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    repo = TaskRepository(session)
    await repo.create(user_id=alice.id, request="a")
    finished = await repo.create(user_id=alice.id, request="b")
    await repo.update(finished.id, status="completed")

    completed = await repo.list_by_status("completed")

    assert [task.id for task in completed] == [finished.id]


# --------------------------------------------------------------------------- #
# Cascades
# --------------------------------------------------------------------------- #


async def test_deleting_a_user_removes_everything_they_own(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    conversation = await ConversationRepository(session).create(alice.id, title="c")
    task = await TaskRepository(session).create(
        user_id=alice.id, request="do it", conversation_id=conversation.id
    )
    await EventRepository(session).append(task_id=task.id, event_type="task_started")
    await session.commit()

    await session.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": alice.id})
    await session.commit()

    for table in ("conversations", "tasks", "execution_events"):
        remaining = await session.execute(text(f"SELECT count(*) FROM {table}"))
        assert remaining.scalar_one() == 0, f"{table} rows survived the cascade"


async def test_deleting_a_task_removes_its_trail(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    task = await TaskRepository(session).create(user_id=alice.id, request="do it")
    await ToolCallRepository(session).record(task_id=task.id, tool="read_file", ok=True)
    await ApprovalRepository(session).request(task_id=task.id, requested_action="drop")
    await session.commit()

    assert await TaskRepository(session).delete(task.id) is True
    await session.commit()

    for model in (Task, ToolCall, Approval):
        count = await session.execute(select(func.count()).select_from(model))
        assert count.scalar_one() == 0, f"{model.__tablename__} rows survived the cascade"


# --------------------------------------------------------------------------- #
# Steps, runs, and tool calls
# --------------------------------------------------------------------------- #


async def test_a_step_is_updated_in_place_across_retries(session: AsyncSession) -> None:
    """A retry revisits a subtask; it must not append a second row."""
    alice = await UserRepository(session).get_or_create("alice")
    task = await TaskRepository(session).create(user_id=alice.id, request="do it")
    repo = TaskStepRepository(session)

    await repo.upsert(
        task_id=task.id, subtask_id="a", agent="researcher", description="first", attempt=1
    )
    await repo.finish(task_id=task.id, subtask_id="a", status="failed", error="boom")
    await repo.upsert(
        task_id=task.id, subtask_id="a", agent="researcher", description="first", attempt=2
    )

    steps = await repo.list_for_task(task.id)

    assert len(steps) == 1
    assert steps[0].attempt == 2


async def test_the_child_reads_are_bounded(session: AsyncSession) -> None:
    """Every list in this module has a ceiling, including the per-task ones.

    These were the last unbounded reads in the file. The true row count is capped
    by the graph's iteration and tool-call ceilings, so this changes nothing
    today — and stops being true the day one of those ceilings is raised, which
    is the kind of change nobody remembers to audit.
    """
    alice = await UserRepository(session).get_or_create("alice")
    task = await TaskRepository(session).create(user_id=alice.id, request="do it")
    steps = TaskStepRepository(session)
    runs = AgentRunRepository(session)

    for index in range(3):
        await steps.upsert(
            task_id=task.id, subtask_id=f"s{index}", agent="researcher", description="work"
        )
        await runs.start(task_id=task.id, agent="researcher", node="agent_execution")

    assert len(await steps.list_for_task(task.id)) == 3
    assert len(await steps.list_for_task(task.id, limit=2)) == 2
    assert len(await steps.list_for_task(task.id, limit=2, offset=2)) == 1
    assert len(await runs.list_for_task(task.id, limit=1)) == 1
    # A limit of zero is a legal statement, not an error: it reads nothing.
    assert await steps.list_for_task(task.id, limit=0) == []


async def test_tool_call_reads_are_bounded_too(session: AsyncSession) -> None:
    """The tool-call timeline is the one that grows fastest per task."""
    alice = await UserRepository(session).get_or_create("alice")
    task = await TaskRepository(session).create(user_id=alice.id, request="do it")
    repo = ToolCallRepository(session)

    for index in range(3):
        await repo.record(task_id=task.id, tool=f"tool_{index}", arguments={})

    assert len(await repo.list_for_task(task.id)) == 3
    assert len(await repo.list_for_task(task.id, limit=1)) == 1


async def test_a_step_records_its_outcome(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    task = await TaskRepository(session).create(user_id=alice.id, request="do it")
    repo = TaskStepRepository(session)
    await repo.upsert(task_id=task.id, subtask_id="a", agent="researcher", description="find it")

    await repo.finish(
        task_id=task.id, subtask_id="a", status="completed", output="found", duration_ms=12.5
    )

    step = (await repo.list_for_task(task.id))[0]
    assert step.status == "completed"
    assert step.output == "found"
    assert step.duration_ms == 12.5


async def test_an_agent_run_records_its_cost(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    task = await TaskRepository(session).create(user_id=alice.id, request="do it")
    repo = AgentRunRepository(session)

    run = await repo.start(task_id=task.id, agent="researcher", node="agent_execution")
    await repo.finish(run.id, prompt_tokens=120, completion_tokens=40, duration_ms=8.5)

    runs = await repo.list_for_task(task.id)

    assert runs[0].total_tokens == 160
    assert runs[0].status == "completed"
    assert runs[0].finished_at is not None


async def test_an_agent_run_can_record_a_failure(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    task = await TaskRepository(session).create(user_id=alice.id, request="do it")
    repo = AgentRunRepository(session)

    run = await repo.start(task_id=task.id, agent="research")
    await repo.finish(run.id, status="failed", error="model unavailable")

    runs = await repo.list_for_task(task.id)
    assert runs[0].status == "failed"
    assert runs[0].error == "model unavailable"


async def test_a_refused_tool_call_is_recorded(session: AsyncSession) -> None:
    """An audit that keeps only successes cannot explain what did not happen."""
    alice = await UserRepository(session).get_or_create("alice")
    task = await TaskRepository(session).create(user_id=alice.id, request="do it")

    await ToolCallRepository(session).record(
        task_id=task.id,
        tool="python_executor",
        ok=False,
        risk_level="HIGH",
        approval_required=True,
        error="HIGH risk action requires human approval",
        failure_kind="permanent",
    )

    calls = await ToolCallRepository(session).list_for_task(task.id)

    assert len(calls) == 1
    assert calls[0].ok is False
    assert calls[0].approval_required is True


async def test_tool_call_arguments_round_trip_as_json(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    task = await TaskRepository(session).create(user_id=alice.id, request="do it")

    await ToolCallRepository(session).record(
        task_id=task.id,
        tool="read_file",
        arguments={"path": "notes.txt", "nested": {"limit": 5}},
        ok=True,
    )

    calls = await ToolCallRepository(session).list_for_task(task.id)

    assert calls[0].arguments == {"path": "notes.txt", "nested": {"limit": 5}}


async def test_an_invalid_risk_level_is_rejected(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    task = await TaskRepository(session).create(user_id=alice.id, request="do it")
    await ToolCallRepository(session).record(task_id=task.id, tool="read_file")

    with pytest.raises(BadWrite):
        await session.execute(
            text("UPDATE tool_calls SET risk_level = 'SILLY' WHERE task_id = :tid"),
            {"tid": task.id},
        )
        await session.flush()


async def test_a_tool_call_can_be_linked_to_its_agent_run(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    task = await TaskRepository(session).create(user_id=alice.id, request="do it")
    run = await AgentRunRepository(session).start(task_id=task.id, agent="researcher")

    call = await ToolCallRepository(session).record(
        task_id=task.id, agent_run_id=run.id, tool="web_search", ok=True
    )

    assert call.agent_run_id == run.id


# --------------------------------------------------------------------------- #
# Approvals
# --------------------------------------------------------------------------- #


async def test_only_one_approval_exists_per_task(session: AsyncSession) -> None:
    """A resumed graph must not be able to create a second request."""
    alice = await UserRepository(session).get_or_create("alice")
    task = await TaskRepository(session).create(user_id=alice.id, request="drop it")
    repo = ApprovalRepository(session)

    first = await repo.request(task_id=task.id, requested_action="drop the table")
    second = await repo.request(task_id=task.id, requested_action="something else")

    assert first.id == second.id
    assert second.requested_action == "drop the table"


async def test_deciding_an_approval_records_who_and_when(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    task = await TaskRepository(session).create(user_id=alice.id, request="drop it")
    repo = ApprovalRepository(session)
    await repo.request(task_id=task.id, requested_action="drop the table")

    decided = await repo.decide(task.id, decision="approve", decided_by="alice", note="reviewed")

    assert decided is not None
    assert decided.decision == "approve"
    assert decided.decided_by == "alice"
    assert decided.decided_at is not None


async def test_an_approval_cannot_be_decided_twice(session: AsyncSession) -> None:
    """A second decision must not overwrite the first, whatever a client sends."""
    alice = await UserRepository(session).get_or_create("alice")
    task = await TaskRepository(session).create(user_id=alice.id, request="drop it")
    repo = ApprovalRepository(session)
    await repo.request(task_id=task.id, requested_action="drop the table")
    await repo.decide(task.id, decision="reject", decided_by="alice")

    assert await repo.decide(task.id, decision="approve", decided_by="mallory") is None

    unchanged = await repo.get_for_task(task.id)
    assert unchanged is not None
    assert unchanged.decision == "reject"


async def test_a_pending_decision_is_refused(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    task = await TaskRepository(session).create(user_id=alice.id, request="drop it")
    repo = ApprovalRepository(session)
    await repo.request(task_id=task.id, requested_action="drop the table")

    with pytest.raises(ValueError, match="approve"):
        await repo.decide(task.id, decision="pending", decided_by="alice")


async def test_deciding_a_task_with_no_request_returns_none(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    task = await TaskRepository(session).create(user_id=alice.id, request="harmless")

    assert (
        await ApprovalRepository(session).decide(task.id, decision="approve", decided_by="alice")
        is None
    )


async def test_the_pending_queue_is_listed(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    tasks = TaskRepository(session)
    approvals = ApprovalRepository(session)
    waiting = await tasks.create(user_id=alice.id, request="one")
    decided = await tasks.create(user_id=alice.id, request="two")
    await approvals.request(task_id=waiting.id, requested_action="a")
    await approvals.request(task_id=decided.id, requested_action="b")
    await approvals.decide(decided.id, decision="approve", decided_by="alice")

    pending = await approvals.list_pending()

    assert [approval.task_id for approval in pending] == [waiting.id]


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #


async def test_event_sequence_numbers_are_monotonic(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    task = await TaskRepository(session).create(user_id=alice.id, request="do it")
    repo = EventRepository(session)

    for kind in ("task_started", "planning", "task_completed"):
        await repo.append(task_id=task.id, event_type=kind)

    events = await repo.list_for_task(task.id)

    assert [event.seq for event in events] == [1, 2, 3]
    assert [event.event_type for event in events] == [
        "task_started",
        "planning",
        "task_completed",
    ]


async def test_concurrent_appends_do_not_collide(database: Database) -> None:
    """The sequence is allocated under a row lock, so concurrent writers serialise.

    This is the behaviour a naive ``MAX(seq) + 1`` gets wrong: two writers read
    the same maximum and one of them loses the unique constraint.
    """
    async with database.session() as setup:
        alice = await UserRepository(setup).get_or_create("alice")
        task = await TaskRepository(setup).create(user_id=alice.id, request="do it")
        task_id = task.id

    async def append(kind: str) -> None:
        async with database.session() as worker:
            await EventRepository(worker).append(task_id=task_id, event_type=kind)

    await asyncio.gather(*(append(f"event_{index}") for index in range(8)))

    async with database.session() as check:
        events = await EventRepository(check).list_for_task(task_id)

    assert len(events) == 8
    assert sorted(event.seq for event in events) == list(range(1, 9))


async def test_events_can_be_read_after_a_sequence_number(session: AsyncSession) -> None:
    """Resuming a stream must not replay what the client already saw."""
    alice = await UserRepository(session).get_or_create("alice")
    task = await TaskRepository(session).create(user_id=alice.id, request="do it")
    repo = EventRepository(session)
    for index in range(5):
        await repo.append(task_id=task.id, event_type=f"e{index}")

    tail = await repo.list_for_task(task.id, after_seq=2)

    assert [event.seq for event in tail] == [3, 4, 5]


async def test_an_event_payload_round_trips(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    task = await TaskRepository(session).create(user_id=alice.id, request="do it")

    await EventRepository(session).append(
        task_id=task.id,
        event_type="agent_started",
        payload={"agent": "researcher", "subtask": "a", "attempt": 2},
    )

    events = await EventRepository(session).list_for_task(task.id)

    assert events[0].payload["agent"] == "researcher"
    assert events[0].payload["attempt"] == 2


async def test_events_are_scoped_to_their_task(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    tasks = TaskRepository(session)
    events = EventRepository(session)
    first = await tasks.create(user_id=alice.id, request="one")
    second = await tasks.create(user_id=alice.id, request="two")
    await events.append(task_id=first.id, event_type="a")
    await events.append(task_id=second.id, event_type="b")

    assert [event.event_type for event in await events.list_for_task(first.id)] == ["a"]


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #


async def test_a_memory_record_is_owned_and_typed(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")

    record = await MemoryRepository(session).create(
        user_id=alice.id, kind="long_term", content="prefers concise answers"
    )

    assert record.kind == "long_term"
    assert record.importance == 0.5


async def test_importance_is_clamped_to_a_unit_interval(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    repo = MemoryRepository(session)

    high = await repo.create(user_id=alice.id, kind="long_term", content="x", importance=5.0)
    low = await repo.create(user_id=alice.id, kind="long_term", content="y", importance=-3.0)

    assert high.importance == 1.0
    assert low.importance == 0.0


async def test_an_unknown_memory_kind_is_rejected(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")

    with pytest.raises(BadWrite):
        await MemoryRepository(session).create(user_id=alice.id, kind="nonsense", content="x")


async def test_memory_is_scoped_to_its_owner(session: AsyncSession) -> None:
    users = UserRepository(session)
    alice = await users.get_or_create("alice")
    mallory = await users.get_or_create("mallory")
    repo = MemoryRepository(session)
    await repo.create(user_id=alice.id, kind="long_term", content="alice only")

    assert len(await repo.list_for_user(alice.id)) == 1
    assert await repo.list_for_user(mallory.id) == []


async def test_memory_can_be_filtered_by_kind(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    repo = MemoryRepository(session)
    await repo.create(user_id=alice.id, kind="long_term", content="durable")
    await repo.create(user_id=alice.id, kind="working", content="transient")

    durable = await repo.list_for_user(alice.id, kind="long_term")

    assert [record.content for record in durable] == ["durable"]


async def test_embeddings_round_trip_as_float_arrays(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    repo = MemoryRepository(session)

    await repo.create(
        user_id=alice.id,
        kind="long_term",
        content="x",
        embedding=[0.1, -0.25, 0.75],
        embedding_model="local-hashing-256",
    )

    record = (await repo.list_for_user(alice.id))[0]

    assert record.embedding == [0.1, -0.25, 0.75]
    assert record.embedding_model == "local-hashing-256"


async def test_retrieval_updates_the_access_counter(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    repo = MemoryRepository(session)
    record = await repo.create(user_id=alice.id, kind="long_term", content="x")

    await repo.mark_accessed([record.id])
    await repo.mark_accessed([record.id])
    await session.refresh(record)

    assert record.access_count == 2
    assert record.accessed_at is not None


async def test_marking_nothing_accessed_is_a_no_op(session: AsyncSession) -> None:
    await MemoryRepository(session).mark_accessed([])


async def test_only_volatile_memory_may_be_pruned(session: AsyncSession) -> None:
    """Long-term memory is curated, not expired, so pruning it is refused."""
    repo = MemoryRepository(session)
    alice = await UserRepository(session).get_or_create("alice")

    with pytest.raises(ValueError, match="durable"):
        await repo.prune(kind="long_term", older_than=datetime.now(UTC))

    await repo.create(user_id=alice.id, kind="short_term", content="stale")
    removed = await repo.prune(kind="short_term", older_than=datetime.now(UTC))

    assert removed == 1


async def test_pruning_leaves_recent_memory_alone(session: AsyncSession) -> None:
    repo = MemoryRepository(session)
    alice = await UserRepository(session).get_or_create("alice")
    await repo.create(user_id=alice.id, kind="working", content="fresh")

    removed = await repo.prune(kind="working", older_than=datetime(2000, 1, 1, tzinfo=UTC))

    assert removed == 0
    assert len(await repo.list_for_user(alice.id, kind="working")) == 1


async def test_memory_counts_group_by_kind(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    repo = MemoryRepository(session)
    await repo.create(user_id=alice.id, kind="long_term", content="a")
    await repo.create(user_id=alice.id, kind="long_term", content="b")
    await repo.create(user_id=alice.id, kind="working", content="c")

    assert await repo.count_for_user(alice.id) == {"long_term": 2, "working": 1}


async def test_deleting_a_memory_requires_ownership(session: AsyncSession) -> None:
    users = UserRepository(session)
    alice = await users.get_or_create("alice")
    mallory = await users.get_or_create("mallory")
    repo = MemoryRepository(session)
    record = await repo.create(user_id=alice.id, kind="long_term", content="x")

    assert await repo.delete(record.id, user_id=mallory.id) is False
    assert await repo.delete(record.id, user_id=alice.id) is True


async def test_a_memory_can_be_linked_to_its_task(session: AsyncSession) -> None:
    alice = await UserRepository(session).get_or_create("alice")
    task = await TaskRepository(session).create(user_id=alice.id, request="do it")
    repo = MemoryRepository(session)

    record = await repo.create(
        user_id=alice.id, kind="execution", content="ran it", task_id=task.id
    )

    assert record.task_id == task.id
    linked = await repo.list_for_user(alice.id, task_id=task.id)
    assert [item.id for item in linked] == [record.id]


async def test_a_memory_cannot_reference_a_missing_task(session: AsyncSession) -> None:
    """The foreign key is real, so an orphaned link cannot be written."""
    alice = await UserRepository(session).get_or_create("alice")

    with pytest.raises(BadWrite):
        await MemoryRepository(session).create(
            user_id=alice.id, kind="execution", content="x", task_id=uuid.uuid4()
        )
