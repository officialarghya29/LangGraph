"""Tests for the database tool's SQL classifier and policy gates."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.exceptions import ToolError
from app.tools.base import ToolContext, ToolRequest
from app.tools.database import DatabaseTool, QueryExecutor, SqlClass, classify_sql


class RecordingExecutor(QueryExecutor):
    """An executor that records calls and returns canned rows."""

    def __init__(self, rows: list[list[object]] | None = None) -> None:
        self.calls: list[str] = []
        self._rows = rows if rows is not None else [[1]]

    async def run(
        self, sql: str, *, max_rows: int, timeout_ms: int
    ) -> tuple[list[str], list[list[object]]]:
        self.calls.append(sql)
        return ["value"], self._rows


class ExplodingExecutor(QueryExecutor):
    """An executor that always fails."""

    async def run(
        self, sql: str, *, max_rows: int, timeout_ms: int
    ) -> tuple[list[str], list[list[object]]]:
        raise RuntimeError("connection lost")


def context_for(*, approved: bool = False, **settings_overrides: object) -> ToolContext:
    """Build a context.

    ``approved`` is a :class:`ToolContext` field, not a setting. It is pulled
    out explicitly because ``Settings`` ignores unknown keys, so passing it
    through would silently drop it and the approval tests would pass for the
    wrong reason.
    """
    settings = Settings(_env_file=None, **settings_overrides)  # type: ignore[arg-type]
    return ToolContext(settings=settings, approved=approved)


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "select * from users",
        "SELECT count(*) FROM orders WHERE total > 10",
        "TABLE users",
        "SHOW TABLES",
        "DESCRIBE users",
        "EXPLAIN SELECT 1",
        "WITH recent AS (SELECT * FROM orders) SELECT * FROM recent",
    ],
)
def test_read_statements_classify_as_read(sql: str) -> None:
    assert classify_sql(sql) is SqlClass.READ


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO users (name) VALUES ('x')",
        "UPDATE users SET name = 'x'",
        "MERGE INTO users USING staging ON users.id = staging.id",
        "SET search_path TO public",
    ],
)
def test_write_statements_classify_as_write(sql: str) -> None:
    assert classify_sql(sql) is SqlClass.WRITE


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM users",
        "DROP TABLE users",
        "TRUNCATE users",
        "ALTER TABLE users ADD COLUMN x int",
        "CREATE TABLE t (id int)",
        "GRANT ALL ON users TO public",
        "REVOKE ALL ON users FROM public",
        "VACUUM FULL",
    ],
)
def test_destructive_statements_classify_as_destructive(sql: str) -> None:
    assert classify_sql(sql) is SqlClass.DESTRUCTIVE


def test_a_destructive_statement_hidden_behind_a_cte_is_caught() -> None:
    """The leading keyword alone would miss this."""
    sql = (
        "WITH doomed AS (SELECT id FROM users) "
        "DELETE FROM users WHERE id IN (SELECT id FROM doomed)"
    )

    assert classify_sql(sql) is SqlClass.DESTRUCTIVE


def test_a_destructive_statement_hidden_behind_explain_is_caught() -> None:
    assert classify_sql("EXPLAIN DROP TABLE users") is SqlClass.DESTRUCTIVE


def test_keywords_inside_string_literals_are_ignored() -> None:
    """A literal must not be able to change classification."""
    assert classify_sql("SELECT * FROM logs WHERE message = 'drop table users'") is SqlClass.READ


def test_keywords_inside_comments_are_ignored() -> None:
    assert classify_sql("-- DROP TABLE users\nSELECT 1") is SqlClass.READ
    assert classify_sql("/* DELETE FROM users */ SELECT 1") is SqlClass.READ


def test_a_select_that_writes_is_classified_as_a_write() -> None:
    assert classify_sql("SELECT * FROM users INTO OUTFILE '/tmp/x'") is SqlClass.DESTRUCTIVE


def test_quoted_identifiers_do_not_confuse_classification() -> None:
    assert classify_sql('SELECT "delete" FROM users') is SqlClass.READ


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


def test_an_empty_statement_is_refused() -> None:
    with pytest.raises(ToolError, match="empty"):
        classify_sql("   ")


def test_multiple_statements_are_refused() -> None:
    """Stacking statements is how a read guard gets bypassed."""
    with pytest.raises(ToolError, match="only one statement"):
        classify_sql("SELECT 1; DROP TABLE users")


def test_a_trailing_semicolon_is_fine() -> None:
    assert classify_sql("SELECT 1;") is SqlClass.READ


def test_unrecognised_sql_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ToolError, match="could not be classified"):
        classify_sql("FROBNICATE the database")


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #


async def test_reads_run_without_any_opt_in() -> None:
    executor = RecordingExecutor()
    tool = DatabaseTool(executor)

    result = await tool.execute(
        ToolRequest(tool="database", arguments={"sql": "SELECT 1"}), context_for()
    )

    assert result.ok is True
    assert executor.calls == ["SELECT 1"]


async def test_writes_are_refused_by_default() -> None:
    executor = RecordingExecutor()
    tool = DatabaseTool(executor)

    result = await tool.execute(
        ToolRequest(tool="database", arguments={"sql": "UPDATE users SET a = 1"}), context_for()
    )

    assert result.ok is False
    assert "write statements are disabled" in (result.error or "")
    assert executor.calls == [], "the statement must not reach the database"


async def test_writes_run_when_enabled() -> None:
    executor = RecordingExecutor()
    tool = DatabaseTool(executor)

    result = await tool.execute(
        ToolRequest(tool="database", arguments={"sql": "UPDATE users SET a = 1"}),
        context_for(database_allow_writes=True),
    )

    assert result.ok is True


async def test_destructive_statements_are_refused_by_default() -> None:
    executor = RecordingExecutor()
    tool = DatabaseTool(executor)

    result = await tool.execute(
        ToolRequest(tool="database", arguments={"sql": "DROP TABLE users"}), context_for()
    )

    assert result.ok is False
    assert executor.calls == []


async def test_destructive_statements_need_both_a_flag_and_approval() -> None:
    """Enabling the flag is not the same as a human saying yes."""
    executor = RecordingExecutor()
    tool = DatabaseTool(executor)

    result = await tool.execute(
        ToolRequest(tool="database", arguments={"sql": "DROP TABLE users"}),
        context_for(database_allow_destructive=True),
    )

    assert result.ok is False
    assert "human approval" in (result.error or "")
    assert executor.calls == []


async def test_destructive_statements_run_when_flagged_and_approved() -> None:
    executor = RecordingExecutor()
    tool = DatabaseTool(executor)

    result = await tool.execute(
        ToolRequest(tool="database", arguments={"sql": "DROP TABLE users"}),
        context_for(database_allow_destructive=True, approved=True),
    )

    assert result.ok is True


async def test_an_executor_failure_is_reported_structurally() -> None:
    tool = DatabaseTool(ExplodingExecutor())

    result = await tool.execute(
        ToolRequest(tool="database", arguments={"sql": "SELECT 1"}), context_for()
    )

    assert result.ok is False
    assert "the query failed" in (result.error or "")


async def test_results_are_truncated_to_the_row_ceiling() -> None:
    executor = RecordingExecutor(rows=[[i] for i in range(1000)])
    tool = DatabaseTool(executor)

    result = await tool.execute(
        ToolRequest(tool="database", arguments={"sql": "SELECT 1"}),
        context_for(database_max_rows=10),
    )

    assert result.ok is True
    assert result.output["row_count"] == 10
    assert result.output["truncated"] is True
