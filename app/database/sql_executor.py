"""A bounded SQL runner for the ``database`` tool.

Two decisions shape this module, and both are about blast radius.

**It gets its own engine, aimed at ``DATABASE_TOOL_URL``.** It is deliberately
not built on the application's :class:`~app.database.connection.Database`, whose
engine holds connections to the tables that store users, tasks, and memory. An
agent that can write ``SELECT * FROM users`` is a data-leak primitive no matter
how the classifier is written, so the tool is pointed somewhere else or not
registered at all.

**Read-only is enforced by the server, not by this code.** The connection sets
``default_transaction_read_only`` at startup, so a statement that somehow
classified as a read but writes is refused by PostgreSQL. A Python-side check is
policy; a server-side one is a property of the connection. The classifier in
:mod:`app.tools.database` still runs first, because a clear error beats a
permission failure, but correctness does not rest on it.

Statement time and row count are bounded per call, so one query cannot hold a
connection open or materialise a table into memory.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.database.connection import build_engine

__all__ = ["SqlQueryExecutor"]

logger = logging.getLogger(__name__)


class SqlQueryExecutor:
    """Runs a classified SQL statement against a dedicated engine.

    Structurally satisfies :class:`app.tools.database.QueryExecutor`; the import
    is deliberately not taken so the tool layer stays free of database code.
    """

    def __init__(
        self,
        url: str,
        *,
        read_only: bool = True,
        pool_size: int = 2,
        echo: bool = False,
    ) -> None:
        """Build an executor and its engine.

        Args:
            url: SQLAlchemy URL for the database the tool may reach.
            read_only: Pin every connection to a read-only transaction. Leave on
                unless an operator has deliberately enabled writes.
            pool_size: Connections to keep. Small on purpose: the tool is called
                by one agent at a time, and a large pool against a reporting
                replica is a footprint nobody asked for.
            echo: Log every statement. Off by default; a statement can carry a
                literal from a prompt.
        """
        self._read_only = read_only
        self._engine: AsyncEngine = build_engine(
            url,
            pool_size=pool_size,
            max_overflow=0,
            echo=echo,
            connect_args=self._connect_args(read_only),
        )

    @staticmethod
    def _connect_args(read_only: bool) -> Mapping[str, Any]:
        """Return driver arguments that apply the posture at connect time.

        ``server_settings`` is asyncpg's startup-packet hook: these values are
        negotiated when the connection is established, before any statement from
        a caller can run. A ``postgresql+asyncpg`` URL is the only form the
        engine builder accepts, so no other driver needs handling here.
        """
        if not read_only:
            return {}
        return {"server_settings": {"default_transaction_read_only": "on"}}

    async def run(
        self,
        sql: str,
        *,
        parameters: Mapping[str, object],
        max_rows: int,
        timeout_ms: int,
    ) -> tuple[list[str], list[list[object]]]:
        """Execute ``sql`` and return its columns and rows.

        Fetching is capped one row past ``max_rows`` so the caller can tell a
        complete result from a truncated one; the extra row is never returned.

        Args:
            sql: The statement, with any placeholders in the driver's own syntax.
            parameters: Values bound by the driver. Never interpolated.
            max_rows: Hard ceiling on returned rows.
            timeout_ms: Server-side statement timeout for this transaction.

        Returns:
            ``(columns, rows)``. A statement that returns no result set yields an
            empty column list.
        """
        statement = text(sql)
        async with self._engine.connect() as connection, connection.begin():
            # ``set_config(..., true)`` is transaction-local, so the ceiling
            # cannot leak into the next caller's connection via the pool.
            await connection.execute(
                text("SELECT set_config('statement_timeout', :timeout, true)"),
                {"timeout": str(timeout_ms)},
            )
            result = await connection.execute(statement, dict(parameters))

            if not result.returns_rows:
                return [], []

            columns = list(result.keys())
            fetched = result.fetchmany(max_rows + 1)
            return columns, [list(row) for row in fetched]

    async def close(self) -> None:
        """Release the engine's pooled connections."""
        await self._engine.dispose()
