"""Database tool.

Default posture is read-only. A model does not get to construct administrative
SQL, and it does not get to delete anything without a human saying so.

Statement classification is done by parsing the SQL here rather than by trusting
the caller or the `risk_level` declared on the tool class. A single database tool
covers three statement classes, so the risk cannot be a class-level constant: a
``SELECT`` must not require approval while a ``DROP TABLE`` must. The gate is
therefore evaluated per statement, inside :meth:`run`.

Execution is delegated to an injected runner so this module holds no connection
management. The runner is supplied by the composition root once Phase 3 lands.
"""

from __future__ import annotations

import logging
import re
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, Field

from app.core.exceptions import ToolError
from app.models.tool import AccessMode
from app.tools.base import Tool, ToolContext

__all__ = [
    "DatabaseTool",
    "QueryExecutor",
    "QueryInput",
    "QueryResult",
    "SqlClass",
    "classify_sql",
]

logger = logging.getLogger(__name__)


class SqlClass(StrEnum):
    """What a SQL statement does."""

    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


_READ_LEADERS = frozenset(
    {"select", "with", "explain", "show", "table", "values", "describe", "desc"}
)
_WRITE_LEADERS = frozenset({"insert", "update", "merge", "upsert", "copy", "set", "reset"})

#: Scanned for anywhere in the statement, not just at the front: a destructive
#: keyword can hide behind a leading ``WITH`` or ``EXPLAIN``.
#:
#: ``into`` is deliberately absent. It appears in every ``INSERT INTO`` and
#: ``MERGE INTO``, so listing it here would classify ordinary writes as
#: destructive. It is handled in the read branch instead, where
#: ``SELECT ... INTO`` is what it actually signals.
_DESTRUCTIVE_KEYWORDS = frozenset(
    {
        "delete",
        "drop",
        "truncate",
        "alter",
        "create",
        "grant",
        "revoke",
        "vacuum",
        "reindex",
        "refresh",
        "cluster",
        "comment",
        "call",
        "do",
    }
)

_WRITE_KEYWORDS = frozenset({"insert", "update", "merge", "delete", "set", "copy"})

_WORD = re.compile(r"\b([a-z_]+)\b")


def _strip_noise(sql: str) -> str:
    """Remove comments and string literals from SQL.

    Literals and comments are removed before any keyword scan, so a keyword
    appearing inside a string — or a destructive keyword hidden in a comment —
    cannot influence classification in either direction.
    """
    out: list[str] = []
    index = 0
    length = len(sql)

    while index < length:
        char = sql[index]

        if char == "-" and index + 1 < length and sql[index + 1] == "-":
            while index < length and sql[index] != "\n":
                index += 1
        elif char == "/" and index + 1 < length and sql[index + 1] == "*":
            index += 2
            while index + 1 < length and not (sql[index] == "*" and sql[index + 1] == "/"):
                index += 1
            index += 2
        elif char in {"'", '"'}:
            quote = char
            index += 1
            while index < length:
                if sql[index] == quote:
                    if index + 1 < length and sql[index + 1] == quote:
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            out.append(" ")
        elif char == "$" and sql[index : index + 2] == "$$":
            index += 2
            while index + 1 < length and sql[index : index + 2] != "$$":
                index += 1
            index += 2
            out.append(" ")
        else:
            out.append(char)
            index += 1

    return "".join(out)


def classify_sql(sql: str) -> SqlClass:
    """Classify a SQL statement.

    Args:
        sql: The statement to classify.

    Returns:
        Whether the statement reads, writes, or is destructive.

    Raises:
        ToolError: If the statement is empty, is a batch of several statements,
            or uses a leading keyword that is not recognised. Unrecognised SQL
            is refused rather than guessed at.
    """
    cleaned = _strip_noise(sql).strip()
    if not cleaned:
        raise ToolError("the query is empty")

    statements = [part.strip() for part in cleaned.split(";") if part.strip()]
    if not statements:
        raise ToolError("the query is empty")
    if len(statements) > 1:
        raise ToolError(
            "only one statement may be executed at a time", detail=f"{len(statements)} found"
        )

    statement = statements[0]
    words = {match.group(1) for match in _WORD.finditer(statement.lower())}

    # Destructive keywords anywhere in the statement take precedence, which is
    # what catches `WITH x AS (...) DELETE FROM ...`.
    if words & _DESTRUCTIVE_KEYWORDS:
        return SqlClass.DESTRUCTIVE

    leader = statement.split(None, 1)[0].lower()

    if leader in _READ_LEADERS:
        if "into" in words:
            # `SELECT ... INTO` writes a table or a file, despite the SELECT.
            return SqlClass.DESTRUCTIVE
        if words & _WRITE_KEYWORDS:
            return SqlClass.WRITE
        return SqlClass.READ

    if leader in _WRITE_LEADERS:
        return SqlClass.WRITE

    raise ToolError("the statement could not be classified and was refused", detail=leader)


class QueryExecutor(Protocol):
    """Runs a SQL statement and returns its columns and rows.

    Injected rather than imported so this module holds no connection logic. The
    SQLAlchemy implementation arrives with the database layer in Phase 3.
    """

    async def run(
        self, sql: str, *, max_rows: int, timeout_ms: int
    ) -> tuple[list[str], list[list[object]]]:
        """Execute ``sql`` and return ``(columns, rows)``."""
        ...


class QueryInput(BaseModel):
    """A SQL statement to run."""

    sql: str = Field(min_length=1, max_length=20_000)
    parameters: dict[str, object] = Field(default_factory=dict)


class QueryResult(BaseModel):
    """The result of a query."""

    sql_class: SqlClass
    columns: list[str] = Field(default_factory=list)
    rows: list[list[object]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False


class DatabaseTool(Tool[QueryInput, QueryResult]):
    """Runs classified SQL against an injected executor.

    Declared ``WRITE`` so its base risk is ``MEDIUM``: reads must not be gated,
    and destructive statements are gated inside :meth:`run` where the statement
    class is actually known.
    """

    name = "database"
    description = "Run read-only SQL by default; writes and destructive statements are gated"
    access_mode = AccessMode.WRITE
    timeout_seconds = 60.0
    input_model = QueryInput
    output_model = QueryResult

    def __init__(self, executor: QueryExecutor) -> None:
        self._executor = executor

    async def run(self, payload: QueryInput, context: ToolContext) -> QueryResult:
        """Classify, authorise, then execute the statement.

        Args:
            payload: The statement and its bind parameters.
            context: Caller context, including whether a human approved.

        Returns:
            The query result.

        Raises:
            ToolError: If the statement is refused by policy or execution fails.
        """
        statement_class = classify_sql(payload.sql)
        settings = context.settings

        if statement_class is SqlClass.DESTRUCTIVE:
            if not settings.database_allow_destructive:
                raise ToolError(
                    "destructive statements are disabled",
                    detail="set DATABASE_ALLOW_DESTRUCTIVE to enable them",
                )
            if not context.approved:
                raise ToolError(
                    "destructive statements require human approval",
                    detail="this statement would not be safe to repeat automatically",
                )
        elif statement_class is SqlClass.WRITE and not settings.database_allow_writes:
            raise ToolError(
                "write statements are disabled",
                detail="set DATABASE_ALLOW_WRITES to enable them",
            )

        logger.info(
            "database.query",
            extra={
                "sql_class": statement_class.value,
                "task_id": context.task_id,
                "approved": context.approved,
            },
        )

        try:
            columns, rows = await self._executor.run(
                payload.sql,
                max_rows=settings.database_max_rows,
                timeout_ms=settings.database_statement_timeout_ms,
            )
        except ToolError:
            raise
        except Exception as exc:  # the runner is external and may raise anything
            raise ToolError("the query failed", detail=type(exc).__name__) from exc

        limit = settings.database_max_rows
        truncated = len(rows) > limit
        rows = rows[:limit]

        return QueryResult(
            sql_class=statement_class,
            columns=list(columns),
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
        )
