"""Tests for the structured execution event model."""

from __future__ import annotations

import pytest

from app.schemas.events import SAFE_EVENT_TYPES, EventType, ExecutionEvent


def make_event(event_type: EventType, **overrides: object) -> ExecutionEvent:
    return ExecutionEvent(event_type=event_type, task_id="t-1", **overrides)


def test_every_documented_event_type_exists() -> None:
    """The streamable vocabulary is fixed by the design."""
    expected = {
        "task_started",
        "task_routing",
        "task_planning",
        "task_retry",
        "task_completed",
        "task_failed",
        "agent_started",
        "agent_completed",
        "tool_started",
        "tool_completed",
        "verification_started",
        "verification_completed",
        "approval_required",
        "approval_received",
        "memory_recalled",
        "memory_written",
    }

    assert {member.value for member in EventType} == expected


def test_every_event_type_is_safe_to_expose() -> None:
    assert frozenset(EventType) == SAFE_EVENT_TYPES


def test_event_reports_itself_as_safe() -> None:
    assert make_event(EventType.TASK_STARTED).is_safe_to_expose is True


def test_event_requires_a_task_id() -> None:
    with pytest.raises(Exception, match="task_id"):
        ExecutionEvent(event_type=EventType.TASK_STARTED)  # type: ignore[call-arg]


def test_event_rejects_a_negative_sequence() -> None:
    with pytest.raises(Exception, match="sequence"):
        make_event(EventType.TASK_STARTED, sequence=-1)


def test_event_carries_only_safe_operational_fields() -> None:
    """A guard against someone adding a reasoning field later."""
    forbidden = {"chain_of_thought", "reasoning", "prompt", "system_prompt", "thoughts"}

    assert forbidden.isdisjoint(ExecutionEvent.model_fields)


def test_event_carries_timing_and_error_metadata() -> None:
    event = make_event(
        EventType.TOOL_COMPLETED,
        tool="web_search",
        duration_ms=12.5,
        error_type="timeout",
        retry_count=2,
    )

    assert event.tool == "web_search"
    assert event.duration_ms == 12.5
    assert event.error_type == "timeout"
    assert event.retry_count == 2


def test_event_serialises_for_streaming() -> None:
    payload = make_event(EventType.AGENT_STARTED, agent="researcher").model_dump(mode="json")

    assert payload["event_type"] == "agent_started"
    assert payload["agent"] == "researcher"
    assert isinstance(payload["created_at"], str)


def test_events_are_immutable_enough_to_serialise_deterministically() -> None:
    first = make_event(EventType.TASK_STARTED).model_dump(mode="json")
    second = make_event(EventType.TASK_STARTED).model_dump(mode="json")

    assert first.keys() == second.keys()
