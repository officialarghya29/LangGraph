"""Tests for the streaming poll schedule.

The interval is the whole trade in one number: how quickly a client hears about
the next event against how much database work an idle connection costs. It is
asserted directly because the failure it prevents is quantitative — a stream that
polls at a fixed rate looks completely correct from the outside and quietly
multiplies its own cost by the number of viewers.
"""

from __future__ import annotations

import pytest

from app.api.routes.events import (
    MAX_POLL_INTERVAL_SECONDS,
    POLL_INTERVAL_SECONDS,
    _next_interval,
)

#: The steady state an idle stream settles into, and the reason the change was
#: worth making: 0.4 s is 2.5 polls a second per viewer, 2.0 s is 0.5.
IDLE_POLLS_PER_SECOND = 1 / MAX_POLL_INTERVAL_SECONDS
ACTIVE_POLLS_PER_SECOND = 1 / POLL_INTERVAL_SECONDS


def test_an_idle_stream_backs_off_geometrically() -> None:
    """Each empty poll doubles the wait, which reaches the cap in two steps."""
    first = _next_interval(POLL_INTERVAL_SECONDS, produced=False)
    second = _next_interval(first, produced=False)

    assert first == POLL_INTERVAL_SECONDS * 2
    assert second == POLL_INTERVAL_SECONDS * 4


def test_the_interval_never_exceeds_the_cap() -> None:
    """The cap is the client's worst-case delay; exceeding it changes the product."""
    interval = POLL_INTERVAL_SECONDS
    for _ in range(20):
        interval = _next_interval(interval, produced=False)

    assert interval == MAX_POLL_INTERVAL_SECONDS


def test_work_resets_the_interval_to_the_floor() -> None:
    """A task that starts moving again is watched at the floor, not the cap.

    Without the reset the stream would be slowest exactly when the run is
    busiest, which is the opposite of what a viewer wants.
    """
    assert _next_interval(MAX_POLL_INTERVAL_SECONDS, produced=True) == POLL_INTERVAL_SECONDS


def test_the_idle_rate_is_materially_cheaper_than_the_active_one() -> None:
    """The quantitative claim, stated where a change to it will fail a test."""
    assert pytest.approx(5.0) == ACTIVE_POLLS_PER_SECOND / IDLE_POLLS_PER_SECOND


def test_the_cap_is_short_enough_to_be_invisible() -> None:
    """Two seconds is the budget for "the task finished and nobody said"."""
    assert MAX_POLL_INTERVAL_SECONDS <= 2.0
    assert MAX_POLL_INTERVAL_SECONDS > POLL_INTERVAL_SECONDS


def test_a_fractional_interval_still_doubles_cleanly() -> None:
    """The schedule starts from a float, and must not accumulate drift."""
    interval = POLL_INTERVAL_SECONDS
    seen = [interval]
    for _ in range(10):
        interval = _next_interval(interval, produced=False)
        seen.append(interval)

    assert seen == sorted(seen), "the interval must be monotonic while idle"
    assert all(value <= MAX_POLL_INTERVAL_SECONDS for value in seen)
