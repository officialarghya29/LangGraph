"""Checkpoint persistence.

Checkpointing is what makes a run resumable: it allows execution to be
interrupted, inspected, resumed, and recovered after a crash, and it is the same
mechanism that human approval uses to suspend a graph mid-run.

Every run is keyed by a stable thread identifier derived from the task id, so a
task can always be resumed by the same id it started with.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from app.core.config import Settings

__all__ = ["build_checkpointer", "thread_config"]


def build_checkpointer(settings: Settings) -> BaseCheckpointSaver[Any]:
    """Build the checkpointer for the configured environment.

    Current state: the in-memory saver. It provides real checkpoint, interrupt,
    and resume semantics within one process, which is what the graph tests
    exercise.

    Durable cross-process persistence needs the PostgreSQL saver
    (``langgraph-checkpoint-postgres``). That is deliberately not wired up yet:
    PostgreSQL is not available in this environment, and shipping an unverified
    durable backend would be worse than shipping a verified volatile one. Until
    then, a process restart loses in-flight runs.

    Args:
        settings: Application settings. Reserved for selecting the backend.

    Returns:
        A checkpoint saver.
    """
    del settings  # Backend selection arrives with the PostgreSQL saver.
    return InMemorySaver()


def thread_config(task_id: str, *, checkpoint_id: str | None = None) -> dict[str, Any]:
    """Build the LangGraph config that identifies a task's checkpoint thread.

    Args:
        task_id: Stable task identifier.
        checkpoint_id: Optional specific checkpoint to resume from.

    Returns:
        A config mapping suitable for ``graph.ainvoke``.

    Raises:
        ValueError: If ``task_id`` is empty.
    """
    if not task_id:
        raise ValueError("task_id must not be empty")
    configurable: dict[str, str] = {"thread_id": task_id}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}
