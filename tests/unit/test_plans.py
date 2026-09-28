"""Tests for routing and planning schemas."""

from __future__ import annotations

import pytest
from app.schemas.plans import Complexity, Plan, Route, RouteDecision, Subtask
from pydantic import ValidationError


def subtask(
    id_: str,
    *,
    agent: str = "researcher",
    dependencies: list[str] | None = None,
) -> Subtask:
    return Subtask(
        id=id_,
        description=f"do {id_}",
        agent=agent,
        dependencies=dependencies or [],
    )


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("route", "needs_planning"),
    [
        (Route.DIRECT, False),
        (Route.HUMAN_APPROVAL, False),
        (Route.RESEARCH, True),
        (Route.CODING, True),
        (Route.DATA_ANALYSIS, True),
        (Route.DOCUMENT, True),
        (Route.MULTI_AGENT, True),
    ],
)
def test_needs_planning_matches_the_route(route: Route, needs_planning: bool) -> None:
    assert route.needs_planning is needs_planning


def test_route_decision_forces_the_planning_flag_from_the_route() -> None:
    """A model cannot claim a complex route needs no planning."""
    decision = RouteDecision(route=Route.RESEARCH, requires_planning=False)

    assert decision.requires_planning is True


def test_route_decision_clears_the_flag_for_direct_routes() -> None:
    decision = RouteDecision(route=Route.DIRECT, requires_planning=True)

    assert decision.requires_planning is False


def test_route_decision_defaults_to_simple() -> None:
    assert RouteDecision(route=Route.DIRECT).complexity is Complexity.SIMPLE


def test_route_decision_rejects_an_unknown_route() -> None:
    with pytest.raises(ValidationError):
        RouteDecision(route="teleport")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Plan validation
# --------------------------------------------------------------------------- #


def test_plan_requires_at_least_one_subtask() -> None:
    with pytest.raises(ValidationError):
        Plan(objective="something", subtasks=[])


def test_plan_requires_a_non_empty_objective() -> None:
    with pytest.raises(ValidationError):
        Plan(objective="", subtasks=[subtask("a")])


def test_plan_accepts_a_well_formed_plan() -> None:
    plan = Plan(
        objective="compare two databases",
        subtasks=[subtask("gather"), subtask("analyse", agent="analyst", dependencies=["gather"])],
    )

    assert plan.subtask_ids == ["gather", "analyse"]


def test_plan_rejects_duplicate_subtask_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate subtask ids"):
        Plan(objective="x", subtasks=[subtask("dup"), subtask("dup")])


def test_plan_rejects_unknown_dependencies() -> None:
    with pytest.raises(ValidationError, match="unknown"):
        Plan(objective="x", subtasks=[subtask("a", dependencies=["ghost"])])


def test_plan_rejects_a_dependency_cycle() -> None:
    with pytest.raises(ValidationError, match="cycle"):
        Plan(
            objective="x",
            subtasks=[
                subtask("a", dependencies=["b"]),
                subtask("b", dependencies=["a"]),
            ],
        )


def test_plan_rejects_a_self_dependency() -> None:
    with pytest.raises(ValidationError, match="cycle"):
        Plan(objective="x", subtasks=[subtask("a", dependencies=["a"])])


def test_plan_rejects_a_longer_cycle() -> None:
    with pytest.raises(ValidationError, match="cycle"):
        Plan(
            objective="x",
            subtasks=[
                subtask("a", dependencies=["c"]),
                subtask("b", dependencies=["a"]),
                subtask("c", dependencies=["b"]),
            ],
        )


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #


def test_ready_subtasks_returns_root_tasks_first() -> None:
    plan = Plan(
        objective="x",
        subtasks=[
            subtask("first"),
            subtask("second"),
            subtask("third", dependencies=["first", "second"]),
        ],
    )

    ready = plan.ready_subtasks(completed=set())

    assert [task.id for task in ready] == ["first", "second"]


def test_ready_subtasks_unlocks_dependents_once_satisfied() -> None:
    plan = Plan(
        objective="x",
        subtasks=[subtask("first"), subtask("second", dependencies=["first"])],
    )

    ready = plan.ready_subtasks(completed={"first"})

    assert [task.id for task in ready] == ["second"]


def test_ready_subtasks_excludes_completed_work() -> None:
    plan = Plan(objective="x", subtasks=[subtask("first"), subtask("second")])

    ready = plan.ready_subtasks(completed={"first"})

    assert [task.id for task in ready] == ["second"]


def test_ready_subtasks_is_empty_when_everything_is_done() -> None:
    plan = Plan(objective="x", subtasks=[subtask("first")])

    assert plan.ready_subtasks(completed={"first"}) == []


def test_independent_subtasks_are_all_ready_at_once() -> None:
    """This is what makes parallel dispatch possible."""
    plan = Plan(objective="x", subtasks=[subtask("a"), subtask("b"), subtask("c")])

    assert len(plan.ready_subtasks(completed=set())) == 3


def test_diamond_plan_serialises_correctly() -> None:
    plan = Plan(
        objective="diamond",
        subtasks=[
            subtask("root"),
            subtask("left", dependencies=["root"]),
            subtask("right", dependencies=["root"]),
            subtask("join", dependencies=["left", "right"]),
        ],
    )

    assert [t.id for t in plan.ready_subtasks(completed={"root"})] == ["left", "right"]
    assert [t.id for t in plan.ready_subtasks(completed={"root", "left"})] == ["right"]
    assert [t.id for t in plan.ready_subtasks(completed={"root", "left", "right"})] == ["join"]
