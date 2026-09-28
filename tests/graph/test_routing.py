"""Tests for the graph's conditional routing.

These are pure functions of state, so they are tested in isolation rather than
through a compiled graph. Branches that can loop are the ones most worth
pinning down.
"""

from __future__ import annotations

from app.graph.builder import (
    route_after_aggregate,
    route_after_approval,
    route_after_critic,
    route_after_planning,
    route_after_risk,
    route_after_routing,
    route_after_validation,
)
from app.models.agent import VerificationResult
from app.models.approval import ApprovalStatus
from app.schemas.plans import Plan, Route, RouteDecision, Subtask


def plan_with(*ids: str) -> Plan:
    return Plan(
        objective="objective",
        subtasks=[Subtask(id=i, description=f"do {i}", agent="researcher") for i in ids],
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_validation_fails_when_errors_are_present() -> None:
    from app.core.constants import FailureKind
    from app.models.execution import ExecutionError

    state = {"errors": [ExecutionError(node="n", message="m", failure_kind=FailureKind.VALIDATION)]}

    assert route_after_validation(state) == "fail"  # type: ignore[arg-type]


def test_validation_continues_when_clean() -> None:
    assert route_after_validation({"errors": []}) == "continue"  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def test_direct_route_short_circuits_planning() -> None:
    state = {"route": RouteDecision(route=Route.DIRECT)}

    assert route_after_routing(state) == "direct"  # type: ignore[arg-type]


def test_a_missing_route_falls_back_to_direct() -> None:
    assert route_after_routing({"route": None}) == "direct"  # type: ignore[arg-type]


def test_approval_route_goes_straight_to_the_risk_check() -> None:
    state = {"route": RouteDecision(route=Route.HUMAN_APPROVAL)}

    assert route_after_routing(state) == "approval"  # type: ignore[arg-type]


def test_complex_routes_are_planned() -> None:
    for route in (Route.RESEARCH, Route.CODING, Route.DATA_ANALYSIS, Route.MULTI_AGENT):
        state = {"route": RouteDecision(route=route)}
        assert route_after_routing(state) == "plan"  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


def test_planning_continues_with_a_plan() -> None:
    assert route_after_planning({"plan": plan_with("a")}) == "execute"  # type: ignore[arg-type]


def test_planning_fails_without_a_plan() -> None:
    assert route_after_planning({"plan": None}) == "fail"  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Dispatch loop
# --------------------------------------------------------------------------- #


def test_dispatch_loops_while_subtasks_remain() -> None:
    state = {
        "plan": plan_with("a", "b"),
        "completed_subtasks": ["a"],
        "iteration_count": 1,
        "iteration_limit": 10,
    }

    assert route_after_aggregate(state) == "continue"  # type: ignore[arg-type]


def test_dispatch_moves_on_when_everything_is_complete() -> None:
    state = {
        "plan": plan_with("a", "b"),
        "completed_subtasks": ["a", "b"],
        "iteration_count": 2,
        "iteration_limit": 10,
    }

    assert route_after_aggregate(state) == "critic"  # type: ignore[arg-type]


def test_dispatch_stops_at_the_iteration_ceiling() -> None:
    """The loop must terminate even if subtasks never report completion."""
    state = {
        "plan": plan_with("a", "b"),
        "completed_subtasks": [],
        "iteration_count": 10,
        "iteration_limit": 10,
    }

    assert route_after_aggregate(state) == "critic"  # type: ignore[arg-type]


def test_dispatch_stops_well_before_the_ceiling() -> None:
    state = {
        "plan": plan_with("a", "b"),
        "completed_subtasks": [],
        "iteration_count": 9,
        "iteration_limit": 10,
    }

    assert route_after_aggregate(state) == "continue"  # type: ignore[arg-type]


def test_dispatch_handles_a_missing_plan() -> None:
    assert route_after_aggregate({"plan": None}) == "critic"  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Retry loop
# --------------------------------------------------------------------------- #


def test_a_passing_verdict_proceeds_to_synthesis() -> None:
    state = {
        "verification_result": VerificationResult(passed=True),
        "retry_count": 0,
        "retry_limit": 3,
    }

    assert route_after_critic(state) == "synthesize"  # type: ignore[arg-type]


def test_a_failing_verdict_retries_while_budget_remains() -> None:
    state = {
        "verification_result": VerificationResult(passed=False, issues=["thin"]),
        "retry_count": 0,
        "retry_limit": 3,
    }

    assert route_after_critic(state) == "retry"  # type: ignore[arg-type]


def test_a_failing_verdict_stops_retrying_at_the_ceiling() -> None:
    """The retry loop must terminate, and still deliver an answer."""
    state = {
        "verification_result": VerificationResult(passed=False, issues=["thin"]),
        "retry_count": 3,
        "retry_limit": 3,
    }

    assert route_after_critic(state) == "synthesize"  # type: ignore[arg-type]


def test_a_missing_verdict_proceeds_to_synthesis() -> None:
    assert route_after_critic({"verification_result": None}) == "synthesize"  # type: ignore[arg-type]


def test_a_zero_retry_budget_never_retries() -> None:
    state = {
        "verification_result": VerificationResult(passed=False),
        "retry_count": 0,
        "retry_limit": 0,
    }

    assert route_after_critic(state) == "synthesize"  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Approval
# --------------------------------------------------------------------------- #


def test_risk_check_gates_work_that_needs_approval() -> None:
    assert route_after_risk({"requires_human_approval": True}) == "approval"  # type: ignore[arg-type]


def test_risk_check_lets_safe_work_through() -> None:
    assert route_after_risk({"requires_human_approval": False}) == "finalize"  # type: ignore[arg-type]


def test_approval_executes_when_granted() -> None:
    state = {"approval_status": ApprovalStatus.APPROVED}

    assert route_after_approval(state) == "execute"  # type: ignore[arg-type]


def test_rejection_cancels_rather_than_executes() -> None:
    """A rejected action must never fall through to execution."""
    for status in (ApprovalStatus.REJECTED, ApprovalStatus.PENDING, ApprovalStatus.NOT_REQUIRED):
        state = {"approval_status": status}
        assert route_after_approval(state) == "cancel"  # type: ignore[arg-type]
