"""Checkpoint persistence.

Checkpointing is what makes a run resumable: it allows execution to be
interrupted, inspected, resumed, and recovered after a crash, and it is the same
mechanism human approval uses to suspend a graph mid-run.

Every run is keyed by a stable thread identifier derived from the task id, so a
task can always be resumed under the id it started with.

Two backends are supported, and the difference matters:

- ``postgres`` — durable. A run survives a process restart, a deploy, and a
  crash, and several workers can serve the same task. This is the default.
- ``memory`` — volatile, and correct only within one process. It exists so the
  graph can be exercised in a unit test without a database. Selecting it in a
  deployment silently removes the durability guarantee, so the choice is
  explicit and is reported by ``/ready``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from psycopg import AsyncConnection

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.database.connection import plain_dsn

__all__ = [
    "CheckpointHandle",
    "build_checkpointer",
    "open_checkpointer",
    "thread_config",
]

logger = logging.getLogger(__name__)


def thread_config(task_id: str, *, checkpoint_id: str | None = None) -> dict[str, Any]:
    """Build the LangGraph config that identifies a task's checkpoint thread.

    Args:
        task_id: Stable task identifier.
        checkpoint_id: Optional specific checkpoint to resume from.

    Returns:
        A config mapping suitable for ``graph.ainvoke``.

    Raises:
        ValueError: If ``task_id`` is empty. An empty thread id would make every
            task share one checkpoint thread, which fails in the worst possible
            way: runs would resume each other's state.
    """
    if not task_id:
        raise ValueError("task_id must not be empty")
    configurable: dict[str, str] = {"thread_id": task_id}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def build_checkpointer(settings: Settings) -> BaseCheckpointSaver[Any]:
    """Return a volatile in-memory checkpointer.

    Deliberately not the default path in a running application: this exists so
    the graph can be compiled and exercised in a test without a database. Use
    :func:`open_checkpointer` to obtain the configured backend.

    Args:
        settings: Application settings. Unused; present for symmetry with
            :func:`open_checkpointer` so call sites read the same.

    Returns:
        An in-memory checkpoint saver.
    """
    del settings
    return InMemorySaver()


@dataclass(slots=True)
class CheckpointHandle:
    """A checkpointer plus whatever it needs to be shut down.

    The PostgreSQL saver owns a connection pool. Holding it behind a handle
    means the lifespan closes exactly what it opened, rather than leaking a pool
    on every application start.
    """

    #: The saver to compile the graph with.
    saver: BaseCheckpointSaver[Any]
    #: Backend actually in use: ``postgres`` or ``memory``.
    backend: str
    #: Set when the durable backend was requested and could not be opened.
    error: str | None = None
    #: Closes the backend. Safe to call more than once.
    _close: Any = None

    @property
    def durable(self) -> bool:
        """Return whether runs survive a restart on this backend."""
        return self.backend == "postgres" and self.error is None

    async def close(self) -> None:
        """Release backend resources."""
        if self._close is not None:
            await self._close()
            self._close = None


async def open_checkpointer(settings: Settings) -> CheckpointHandle:
    """Open the configured checkpoint backend.

    Args:
        settings: Application settings, including ``checkpoint_backend``.

    Returns:
        A handle holding the saver and its lifecycle.

    Raises:
        ConfigurationError: If the settings name an unknown backend, or if the
            durable backend was requested but could not be opened. Refusing to
            start is the correct outcome: falling back to memory would leave the
            application running with the durability guarantee silently removed,
            and every later "resumed successfully" claim would be untrustworthy.
    """
    backend = settings.checkpoint_backend
    if backend == "memory":
        logger.warning(
            "checkpoint.volatile_backend",
            extra={"detail": "runs will not survive a restart"},
        )
        return CheckpointHandle(saver=InMemorySaver(), backend="memory")

    if backend != "postgres":
        raise ConfigurationError(
            "unknown checkpoint backend",
            detail=f"{backend!r} is not one of 'postgres', 'memory'",
        )

    # Imported here rather than at module scope: psycopg needs libpq present, and
    # a deployment that only ever uses the memory backend should not need it.
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    dsn = plain_dsn(settings.database_url)
    # The saver requires connections that yield dict rows, and the pool's type
    # parameter follows from the row factory. Naming that type explicitly is
    # what lets a type checker see the pool as the saver's expected input
    # rather than as a bare connection pool.
    pool = cast(
        "AsyncConnectionPool[AsyncConnection[dict[str, Any]]]",
        AsyncConnectionPool(
            conninfo=dsn,
            min_size=1,
            max_size=settings.checkpoint_pool_size,
            # Open explicitly, so a connection failure is raised here and reported
            # rather than surfacing later inside an unrelated request.
            open=False,
            kwargs={
                "autocommit": True,
                "row_factory": dict_row,
                "prepare_threshold": 0,
            },
        ),
    )

    try:
        await pool.open(wait=True, timeout=settings.checkpoint_open_timeout_seconds)
    except Exception as exc:
        await pool.close()
        raise ConfigurationError(
            "the checkpoint database is unreachable",
            detail=type(exc).__name__,
        ) from exc

    saver = AsyncPostgresSaver(pool)
    try:
        # Creates the checkpoint tables if they are absent. Idempotent, so it is
        # safe on every start and does not need a migration of its own: LangGraph
        # owns this schema and must be free to change it with its own version.
        await saver.setup()
    except Exception as exc:
        await pool.close()
        raise ConfigurationError(
            "the checkpoint schema could not be prepared",
            detail=type(exc).__name__,
        ) from exc

    async def close() -> None:
        await pool.close()

    logger.info("checkpoint.durable_backend", extra={"backend": "postgres"})
    return CheckpointHandle(saver=saver, backend="postgres", _close=close)
