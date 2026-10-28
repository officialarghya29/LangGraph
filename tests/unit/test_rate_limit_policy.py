"""Tests for which rate-limit policy a request is matched against.

The table in :mod:`app.api.middleware` describes a budget per operation, and one
of those budgets was never applied: the creation prefix is a prefix of every
approval path, so a first-match lookup returned the creation multiplier for
decisions as well. Nothing asserted the multiplication, so nothing noticed.
"""

from __future__ import annotations

import pytest

from app.api.middleware import RATE_LIMIT_POLICIES, _match_policy, _policy_for


def test_creating_a_task_gets_the_ordinary_budget() -> None:
    assert _policy_for("POST", "/api/v1/tasks") == 1


@pytest.mark.parametrize("action", ["approve", "reject", "cancel"])
def test_a_decision_gets_the_larger_budget(action: str) -> None:
    """These paths are prefixes of the creation path and were shadowed by it."""
    assert _policy_for("POST", f"/api/v1/tasks/{action}-me/{action}") == 2


def test_streaming_gets_the_reconnect_budget() -> None:
    """Reconnects are expected traffic, so they are allowed more of them."""
    assert _policy_for("GET", "/api/v1/events/abc/history") == 4


def test_an_unlisted_shape_is_unlimited() -> None:
    assert _policy_for("GET", "/api/v1/tasks") is None
    assert _policy_for("DELETE", "/api/v1/tasks/abc") is None


def test_a_method_only_matches_its_own_policy() -> None:
    """Creating by GET would otherwise inherit the creation budget's limit."""
    assert _policy_for("GET", "/api/v1/tasks") is None
    assert _policy_for("POST", "/api/v1/events/abc") is None


def test_creating_and_deciding_count_against_different_budgets() -> None:
    """One counter per policy, or the two limits fight over the same count.

    The bucket used to be a path segment, which is ``tasks`` for both, so a
    client's approvals spent its creation budget and the doubled allowance for
    decisions was unenforceable whatever multiplier was configured.
    """
    created = _match_policy("POST", "/api/v1/tasks")
    approved = _match_policy("POST", "/api/v1/tasks/abc/approve")
    streamed = _match_policy("GET", "/api/v1/events/abc")

    assert created is not None and approved is not None and streamed is not None
    scopes = {created[0], approved[0], streamed[0]}
    assert len(scopes) == 3, scopes


def test_every_row_resolves_to_itself() -> None:
    """Two rows can disagree only by both claiming the same prefix.

    A row whose prefix is exactly the request path is matched by that row: no
    longer prefix can be a prefix of itself. So the assertion below fails only
    when the table contradicts itself, which is the shape the shadowing bug had
    before it was found by hand.
    """
    for method, prefix, multiplier in RATE_LIMIT_POLICIES:
        if method == "*":
            continue
        assert _policy_for(method, prefix) == multiplier, (method, prefix)
