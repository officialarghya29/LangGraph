"""Tests for the run sink that records progress in both trails.

The graph publishes an event for every step it takes. Those events went to the
append-only log and nowhere else: ``record_step``, ``finish_step``, and
``record_tool_call`` were implemented, tested on their own, and called by no
production code, so the specialised tables an operator queries were empty. These
tests cover the mapping between the two, including the shapes that must be
ignored rather than crash a run.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.services.task_store import task_event_sink
from app.services.tasks import InMemoryTaskStore


class RecordingStore:
    """A store that keeps what the sink asked it to write."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []
        self.steps: list[dict[str, Any]] = []
        self.finished: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.agent_runs: list[dict[str, Any]] = []

    async def append_event(
        self, task_id: str, event_type: str, payload: dict[str, object] | None = None
    ) -> None:
        """Record an event."""
        del task_id
        self.events.append((event_type, payload or {}))

    async def record_step(self, task_id: str, **fields: Any) -> None:
        """Record an opened step."""
        del task_id
        self.steps.append(fields)

    async def finish_step(self, task_id: str, **fields: Any) -> None:
        """Record a closed step."""
        del task_id
        self.finished.append(fields)

    async def record_tool_call(self, task_id: str, **fields: Any) -> None:
        """Record a tool call."""
        del task_id
        self.tool_calls.append(fields)

    async def record_agent_run(self, task_id: str, **fields: Any) -> None:
        """Record one agent invocation with its cost."""
        del task_id
        self.agent_runs.append(fields)


class ExplodingStore(RecordingStore):
    """A store whose mirror write fails, to pin down the ordering."""

    async def record_tool_call(self, task_id: str, **fields: Any) -> None:
        """Fail, as a database or a constraint violation would."""
        raise RuntimeError("the mirror write failed")


def sink_for(store: Any) -> Any:
    """Return the run sink for a store, as the task runner builds it."""
    return task_event_sink("11111111-1111-1111-1111-111111111111", store)


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #


async def test_an_agent_start_opens_a_step() -> None:
    store = RecordingStore()

    await sink_for(store)(
        "agent_started",
        {"agent": "researcher", "subtask": "a", "description": "research postgres"},
    )

    assert store.steps == [
        {"subtask_id": "a", "agent": "researcher", "description": "research postgres"}
    ]


async def test_an_agent_completion_closes_its_step() -> None:
    store = RecordingStore()

    await sink_for(store)(
        "agent_completed",
        {
            "agent": "researcher",
            "subtask": "a",
            "status": "completed",
            "duration_ms": 12.5,
            "summary": "postgres is relational",
        },
    )

    assert store.finished == [
        {
            "subtask_id": "a",
            "status": "completed",
            "output": "postgres is relational",
            "error": None,
            "duration_ms": 12.5,
        }
    ]


async def test_an_agent_completion_records_a_run_with_its_own_cost() -> None:
    """The per-agent figures are the invocation's, not a figure shared with peers."""
    store = RecordingStore()

    await sink_for(store)(
        "agent_completed",
        {
            "agent": "researcher",
            "subtask": "a",
            "status": "completed",
            "duration_ms": 12.5,
            "prompt_tokens": 120,
            "completion_tokens": 40,
        },
    )

    assert store.agent_runs == [
        {
            "agent": "researcher",
            "status": "completed",
            "prompt_tokens": 120,
            "completion_tokens": 40,
            "duration_ms": 12.5,
            "error": None,
        }
    ]


async def test_an_agent_run_without_a_subtask_still_gets_a_row() -> None:
    """A planner and a critic are agent invocations, so they belong in the table."""
    store = RecordingStore()

    await sink_for(store)(
        "agent_completed",
        {"agent": "planner", "status": "completed", "prompt_tokens": 10},
    )

    assert store.finished == [], "there is no step to close"
    assert [run["agent"] for run in store.agent_runs] == ["planner"]


async def test_a_failed_agent_run_is_recorded_as_failed() -> None:
    store = RecordingStore()

    await sink_for(store)(
        "agent_completed",
        {
            "agent": "synthesizer",
            "subtask": "a",
            "status": "failed",
            "reason": "model_failure",
            "prompt_tokens": 5,
        },
    )

    assert store.agent_runs[0]["status"] == "failed"
    assert store.agent_runs[0]["error"] == "model_failure"
    assert store.agent_runs[0]["prompt_tokens"] == 5


async def test_missing_token_counts_are_reported_as_zero_not_crashed() -> None:
    """A payload from an older graph must not fail the run it is describing."""
    store = RecordingStore()

    await sink_for(store)("agent_completed", {"agent": "critic"})

    assert store.agent_runs[0]["prompt_tokens"] == 0
    assert store.agent_runs[0]["completion_tokens"] == 0


async def test_a_failed_step_records_the_failure_kind_as_its_error() -> None:
    """The trail records what went wrong without copying an exception string."""
    store = RecordingStore()

    await sink_for(store)(
        "agent_completed",
        {"agent": "researcher", "subtask": "a", "status": "failed", "reason": "timeout"},
    )

    assert store.finished[0]["status"] == "failed"
    assert store.finished[0]["error"] == "timeout"
    assert store.finished[0]["output"] is None


# --------------------------------------------------------------------------- #
# Tool calls
# --------------------------------------------------------------------------- #


async def test_a_completed_tool_call_becomes_an_audit_row() -> None:
    store = RecordingStore()

    await sink_for(store)(
        "tool_completed",
        {
            "tool": "write_file",
            "agent": "executor",
            "ok": True,
            "risk_level": "MEDIUM",
            "approved": True,
            "approval_required": False,
            "failure_kind": "unknown",
            "duration_ms": 3.5,
        },
    )

    assert store.tool_calls == [
        {
            "tool": "write_file",
            "agent": "executor",
            "ok": True,
            "risk_level": "MEDIUM",
            "approved": True,
            "approval_required": False,
            "failure_kind": "unknown",
            "duration_ms": 3.5,
        }
    ]


async def test_a_failed_tool_call_is_recorded_too() -> None:
    """The refused call is the record an audit most needs, so it must not be dropped."""
    store = RecordingStore()

    await sink_for(store)(
        "tool_completed",
        {"tool": "write_file", "ok": False, "risk_level": "CRITICAL"},
    )

    assert store.tool_calls[0]["ok"] is False
    assert store.tool_calls[0]["risk_level"] == "CRITICAL"


async def test_a_tool_event_without_a_name_mirrors_nothing() -> None:
    """A malformed payload must not reach a column with a name constraint."""
    store = RecordingStore()

    await sink_for(store)("tool_completed", {"ok": True})

    assert store.tool_calls == []


# --------------------------------------------------------------------------- #
# Both trails
# --------------------------------------------------------------------------- #


async def test_every_event_reaches_the_log() -> None:
    """An event with no specialised row still belongs in the stream."""
    store = RecordingStore()
    sink = sink_for(store)

    await sink("task_routing", {"route": "direct"})
    await sink("agent_completed", {"agent": "researcher", "subtask": "a"})
    await sink("memory_written", {"tier": "short_term"})

    assert [name for name, _ in store.events] == [
        "task_routing",
        "agent_completed",
        "memory_written",
    ]
    assert store.finished, "the event with a row still wrote one"


async def test_an_unexpected_event_type_mirrors_nothing() -> None:
    store = RecordingStore()

    await sink_for(store)("something_new", {"anything": 1})

    assert store.steps == []
    assert store.finished == []
    assert store.tool_calls == []
    assert store.agent_runs == []
    assert store.events


async def test_the_event_is_appended_before_the_mirror_is_attempted() -> None:
    """A failed mirror write must not lose the record of what happened."""
    store = ExplodingStore()

    with pytest.raises(RuntimeError):
        await sink_for(store)("tool_completed", {"tool": "write_file"})

    assert [name for name, _ in store.events] == ["tool_completed"]


async def test_a_store_without_a_durable_trail_is_harmless() -> None:
    """The in-memory store inherits the no-op writers, so its sink must be quiet."""
    store = InMemoryTaskStore()

    await sink_for(store)("tool_completed", {"tool": "write_file", "ok": True})
    await sink_for(store)("agent_started", {"agent": "researcher", "subtask": "a"})
