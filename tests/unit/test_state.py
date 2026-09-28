"""Tests for the typed graph state."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from app.graph.state import AgentState, initial_state
from app.models.approval import ApprovalStatus
from app.services.llm import Role


def test_initial_state_populates_every_key() -> None:
    """Nodes read keys without guarding, so none may be missing."""
    state = initial_state("do the thing")

    for key in AgentState.__annotations__:
        assert key in state, f"initial_state is missing {key!r}"


def test_initial_state_records_the_request_and_a_user_message() -> None:
    state = initial_state("explain quantum tunnelling", user_id="u-1")

    assert state["user_request"] == "explain quantum tunnelling"
    assert len(state["messages"]) == 1
    assert state["messages"][0].role is Role.USER
    assert state["messages"][0].content == "explain quantum tunnelling"
    assert state["user_id"] == "u-1"


def test_initial_state_strips_surrounding_whitespace() -> None:
    assert initial_state("   padded request   ")["user_request"] == "padded request"


def test_initial_state_rejects_an_empty_request() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        initial_state("")


def test_initial_state_rejects_a_whitespace_only_request() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        initial_state("   \n\t  ")


def test_initial_state_generates_distinct_identifiers() -> None:
    first = initial_state("a")
    second = initial_state("b")

    assert first["task_id"] != second["task_id"]
    assert first["conversation_id"] != second["conversation_id"]


def test_initial_state_honours_explicit_identifiers() -> None:
    state = initial_state("a", task_id="t-1", conversation_id="c-1")

    assert state["task_id"] == "t-1"
    assert state["conversation_id"] == "c-1"


def test_initial_state_starts_with_no_approval_required() -> None:
    state = initial_state("a")

    assert state["requires_human_approval"] is False
    assert state["approval_status"] is ApprovalStatus.NOT_REQUIRED


def test_initial_state_starts_with_zeroed_counters() -> None:
    state = initial_state("a")

    assert state["retry_count"] == 0
    assert state["iteration_count"] == 0
    assert state["tool_call_count"] == 0


def test_initial_state_starts_with_empty_collections() -> None:
    state = initial_state("a")

    assert state["agent_outputs"] == []
    assert state["tool_results"] == []
    assert state["errors"] == []
    assert state["subtasks"] == []
    assert state["retrieved_context"] == []


def test_state_is_fully_json_serializable() -> None:
    """The checkpoint serializer round-trips the state, so nothing opaque may hide in it."""

    def encode(value: object) -> object:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        raise TypeError(f"state contains a non-serializable {type(value).__name__}")

    state = initial_state("write a report", user_id="u-9")

    encoded = json.dumps(state, default=encode)

    assert isinstance(encoded, str)
    assert json.loads(encoded)["user_request"] == "write a report"


def test_append_reducers_concatenate_rather_than_overwrite() -> None:
    """Keys written by parallel branches must merge, not race."""
    from typing import get_type_hints

    append_reduced = {
        "messages",
        "agent_outputs",
        "tool_results",
        "errors",
        "retrieved_context",
        "completed_subtasks",
        "events",
    }

    for key in append_reduced:
        hint = get_type_hints(AgentState, include_extras=True)[key]
        metadata = getattr(hint, "__metadata__", ())
        assert metadata, f"{key!r} is written by multiple nodes but has no reducer"
