"""Per-run execution ceilings: tool calls and wall-clock time.

``MAX_TOOL_CALLS`` and ``MAX_EXECUTION_TIME`` were declared, validated, given
bounds in the settings layer, documented in the README as hard ceilings, and read
by nothing. The same was true of the ``tool_call_count`` column and of the
``TOOL_STARTED``/``TOOL_COMPLETED`` event types: the vocabulary for bounded,
auditable tool execution existed in full and was never connected to the tool
pipeline. A run could therefore call tools and consume wall-clock time without
limit, while three separate documents said otherwise. This module is the
enforcement those settings were always meant to have.

The ceilings are two questions with one answer:

- *How many tools may this run call?* Counted at the tool choke point, so no
  caller can bypass the count by going around the graph.
- *How long may this run take?* Measured from the start of the run, checked
  before each tool call and by the graph's own loop guard, and enforced as a hard
  cancellation by the callers that invoke the graph.

Time is deliberately checked in more than one place. The loop guard lets a run
exit with the work it has rather than being cancelled mid-flight, which is the
better outcome; the hard cancellation is the backstop for a run that is inside a
single long call when the ceiling passes.

Enforcement is a :class:`~contextvars.ContextVar` for the same reason the token
budget is: the compiled graph is built once and invoked concurrently, so
counters stored on it would pool two runs into one number. Each run binds its
own, and a caller that binds none — a script driving an agent directly — is
unbounded, which is the honest default outside a graph run.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from app.core.config import Settings
from app.core.exceptions import ExecutionLimitError

__all__ = ["RunLimits", "current_limits", "limits_for", "run_limits"]


class RunLimits:
    """The ceilings one run must stay inside, and what it has used so far."""

    def __init__(
        self,
        *,
        max_tool_calls: int,
        max_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a run's limits, starting the clock.

        Args:
            max_tool_calls: The most tools this run may call. At least one.
            max_seconds: The longest this run may take, in seconds. Positive.
            clock: Monotonic time source, injectable so a test can advance the
                deadline without waiting for it.

        Raises:
            ValueError: If either ceiling is not usable. A zero tool-call ceiling
                would refuse the first call and a negative deadline would refuse
                everything, neither of which is distinguishable from a broken
                system, so both are rejected at construction.
        """
        if max_tool_calls < 1:
            raise ValueError("a run must be allowed at least one tool call")
        if max_seconds <= 0:
            raise ValueError("a run's execution ceiling must be positive")
        self.max_tool_calls = max_tool_calls
        self.max_seconds = max_seconds
        self._clock = clock
        self._started = clock()
        #: Tool calls actually made. A refused call is not counted, so this is
        #: the number worth reporting rather than the number attempted.
        self.tool_calls = 0

    @property
    def elapsed_seconds(self) -> float:
        """Return how long the run has been going."""
        return self._clock() - self._started

    @property
    def remaining_seconds(self) -> float:
        """Return how much of the wall-clock ceiling is left, never below zero."""
        return max(0.0, self.max_seconds - self.elapsed_seconds)

    @property
    def expired(self) -> bool:
        """Return whether the wall-clock ceiling has been reached."""
        return self.remaining_seconds <= 0.0

    def check_deadline(self) -> None:
        """Refuse further work if the run has outlived its ceiling.

        Raises:
            ExecutionLimitError: If the ceiling has passed.
        """
        if self.expired:
            raise ExecutionLimitError(
                "the run exceeded its execution-time ceiling",
                detail=(
                    f"{self.elapsed_seconds:.1f} of {self.max_seconds:.0f} seconds, "
                    f"after {self.tool_calls} tool call(s)"
                ),
            )

    def note_tool_call(self) -> None:
        """Reserve one tool call against the ceiling, or refuse before making it.

        Checked before the call rather than after, like the token budget: a call
        that has already run cannot be un-run, and discarding its result would
        waste work that was really performed. What is prevented is starting
        another one.

        Raises:
            ExecutionLimitError: If the allowance is already spent.
        """
        if self.tool_calls >= self.max_tool_calls:
            raise ExecutionLimitError(
                "the run's tool-call ceiling is reached",
                detail=f"{self.tool_calls} of {self.max_tool_calls} calls already made",
            )
        self.tool_calls += 1

    def summary(self) -> str:
        """Return a one-line description of the limits' state."""
        return (
            f"{self.tool_calls}/{self.max_tool_calls} tool calls, "
            f"{self.elapsed_seconds:.1f}/{self.max_seconds:.0f} seconds elapsed"
        )


#: The limits in scope for the current run. ``None`` means unbounded, which is
#: the behaviour outside a graph run.
_CURRENT: ContextVar[RunLimits | None] = ContextVar("run_limits", default=None)


def limits_for(settings: Settings) -> RunLimits:
    """Return limits sized from configuration.

    Both entry points into the graph — the asynchronous task path and the
    synchronous chat path — call this, and so does the approval resume, so a
    ceiling configured for one cannot go unenforced in another.

    Args:
        settings: Application settings carrying the configured ceilings.

    Returns:
        A limit set whose clock starts now.
    """
    return RunLimits(
        max_tool_calls=settings.max_tool_calls,
        max_seconds=float(settings.max_execution_time),
    )


def current_limits() -> RunLimits | None:
    """Return the limits in scope for this context, or ``None``.

    Returns:
        The active limits, if a run bound any.
    """
    return _CURRENT.get()


@contextmanager
def run_limits(limits: RunLimits) -> Iterator[RunLimits]:
    """Scope ``limits`` to the current run.

    Args:
        limits: The ceilings every tool call in this context is charged to.

    Yields:
        The limits, so the caller can report what the run consumed.
    """
    token = _CURRENT.set(limits)
    try:
        yield limits
    finally:
        _CURRENT.reset(token)
