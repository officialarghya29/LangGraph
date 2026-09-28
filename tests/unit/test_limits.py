"""Tests for the per-run execution ceilings.

``MAX_TOOL_CALLS`` and ``MAX_EXECUTION_TIME`` were configured, bounded, and
documented, and read by nothing. These tests exist so that the setting and the
enforcement cannot drift apart again: each ceiling is asserted to refuse the work
it is supposed to refuse, and to leave the work it permits alone.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.constants import FailureKind
from app.core.exceptions import ExecutionLimitError
from app.services.limits import RunLimits, current_limits, limits_for, run_limits

SETTINGS = Settings(_env_file=None)


class FakeClock:
    """A monotonic clock a test can advance.

    The alternative — really waiting out a deadline — makes the suite slower the
    more carefully the ceiling is tested, which is exactly backwards.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Move time forward."""
        self.now += seconds


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_a_ceiling_of_zero_tool_calls_is_rejected() -> None:
    """A run allowed no tools is indistinguishable from a broken one."""
    with pytest.raises(ValueError, match="at least one tool call"):
        RunLimits(max_tool_calls=0, max_seconds=30)


def test_a_non_positive_time_ceiling_is_rejected() -> None:
    """A deadline that has already passed would refuse the first tool call."""
    with pytest.raises(ValueError, match="must be positive"):
        RunLimits(max_tool_calls=5, max_seconds=0)


def test_limits_read_their_ceilings_from_settings() -> None:
    limits = limits_for(SETTINGS)

    assert limits.max_tool_calls == SETTINGS.max_tool_calls
    assert limits.max_seconds == SETTINGS.max_execution_time


def test_changing_a_setting_changes_the_ceiling() -> None:
    """The wiring is the setting, not a constant that happens to match it."""
    limits = limits_for(Settings(_env_file=None, max_tool_calls=3, max_execution_time=7))

    assert limits.max_tool_calls == 3
    assert limits.max_seconds == 7


# --------------------------------------------------------------------------- #
# The tool-call ceiling
# --------------------------------------------------------------------------- #


def test_calls_up_to_the_ceiling_are_allowed() -> None:
    limits = RunLimits(max_tool_calls=3, max_seconds=60)

    for _ in range(3):
        limits.note_tool_call()

    assert limits.tool_calls == 3


def test_the_call_past_the_ceiling_is_refused() -> None:
    """The ceiling is a maximum, not an off-by-one invitation."""
    limits = RunLimits(max_tool_calls=2, max_seconds=60)
    limits.note_tool_call()
    limits.note_tool_call()

    with pytest.raises(ExecutionLimitError, match="tool-call ceiling") as captured:
        limits.note_tool_call()

    assert "2 of 2" in (captured.value.detail or "")


def test_a_refused_call_is_not_counted() -> None:
    """The reported number is work performed, not work attempted."""
    limits = RunLimits(max_tool_calls=1, max_seconds=60)
    limits.note_tool_call()

    for _ in range(3):
        with pytest.raises(ExecutionLimitError):
            limits.note_tool_call()

    assert limits.tool_calls == 1


def test_a_refused_call_is_permanent() -> None:
    """Retrying the same request hits the same ceiling, so it is not retryable."""
    assert ExecutionLimitError.failure_kind is FailureKind.PERMANENT


# --------------------------------------------------------------------------- #
# The wall-clock ceiling
# --------------------------------------------------------------------------- #


def test_time_within_the_ceiling_is_not_refused() -> None:
    clock = FakeClock()
    limits = RunLimits(max_tool_calls=5, max_seconds=30, clock=clock)

    clock.advance(29.9)

    assert limits.expired is False
    limits.check_deadline()


def test_time_past_the_ceiling_is_refused() -> None:
    clock = FakeClock()
    limits = RunLimits(max_tool_calls=5, max_seconds=30, clock=clock)

    clock.advance(30.0)

    assert limits.expired is True
    with pytest.raises(ExecutionLimitError, match="execution-time ceiling") as captured:
        limits.check_deadline()

    assert "30.0 of 30 seconds" in (captured.value.detail or "")


def test_remaining_time_never_goes_negative() -> None:
    """A negative remainder would be reported to a caller as reclaimed time."""
    clock = FakeClock()
    limits = RunLimits(max_tool_calls=5, max_seconds=10, clock=clock)

    clock.advance(1_000.0)

    assert limits.remaining_seconds == 0.0


def test_the_clock_starts_when_the_limits_are_built() -> None:
    """Elapsed time is measured from the run, not from the process."""
    clock = FakeClock()
    limits = RunLimits(max_tool_calls=5, max_seconds=10, clock=clock)

    clock.advance(4)

    assert limits.elapsed_seconds == 4.0


# --------------------------------------------------------------------------- #
# Scoping
# --------------------------------------------------------------------------- #


def test_no_limits_are_in_scope_by_default() -> None:
    """Outside a run there is nothing to charge, so there is no ceiling."""
    assert current_limits() is None


def test_limits_are_visible_inside_their_scope() -> None:
    limits = RunLimits(max_tool_calls=5, max_seconds=30)

    with run_limits(limits) as bound:
        assert current_limits() is limits
        assert bound is limits

    assert current_limits() is None


def test_scopes_nest_and_restore() -> None:
    """Two runs on one task must not charge each other's calls."""
    outer = RunLimits(max_tool_calls=5, max_seconds=30)
    inner = RunLimits(max_tool_calls=1, max_seconds=30)

    with run_limits(outer):
        with run_limits(inner):
            assert current_limits() is inner
        assert current_limits() is outer

    assert current_limits() is None


def test_limits_are_restored_even_when_the_body_raises() -> None:
    """A leaked ceiling would silently bound the next run."""
    limits = RunLimits(max_tool_calls=1, max_seconds=30)

    with pytest.raises(RuntimeError), run_limits(limits):
        raise RuntimeError("boom")

    assert current_limits() is None


def test_the_summary_describes_both_ceilings() -> None:
    clock = FakeClock()
    limits = RunLimits(max_tool_calls=4, max_seconds=60, clock=clock)
    limits.note_tool_call()
    clock.advance(12.5)

    summary = limits.summary()

    assert "1/4 tool calls" in summary
    assert "12.5/60 seconds" in summary
