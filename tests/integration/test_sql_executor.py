"""Integration tests for the SQL runner backing the database tool.

What only a real server can prove is here: that a read-only connection is
actually read-only, that the statement timeout is enforced by PostgreSQL rather
than by a Python timer, and that bind parameters survive the round trip. A fake
executor can show none of that, because none of it is behaviour of this code —
it is behaviour of the connection this code configures.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.exc import SQLAlchemyError

from app.core.exceptions import DatabaseError
from app.database.sql_executor import SqlQueryExecutor


@pytest_asyncio.fixture
async def executor(test_database_url: str) -> AsyncIterator[SqlQueryExecutor]:
    """Yield a read-only executor pointed at the throwaway test database."""
    instance = SqlQueryExecutor(test_database_url)
    try:
        yield instance
    finally:
        await instance.close()


# --------------------------------------------------------------------------- #
# Reads and parameter binding
# --------------------------------------------------------------------------- #


async def test_a_select_returns_columns_and_rows(executor: SqlQueryExecutor) -> None:
    columns, rows = await executor.run(
        "SELECT 1 AS one, 'two' AS two", parameters={}, max_rows=10, timeout_ms=5000
    )

    assert columns == ["one", "two"]
    assert rows == [[1, "two"]]


async def test_bind_parameters_are_sent_as_values(executor: SqlQueryExecutor) -> None:
    """A parameter is bound, not interpolated.

    The value contains a quote and a statement terminator, which would break out
    of an interpolated literal. Receiving it back unchanged is evidence the
    driver got it as data.
    """
    _, rows = await executor.run(
        "SELECT :value AS echoed",
        parameters={"value": "'; DROP TABLE no_such_table; --"},
        max_rows=1,
        timeout_ms=5000,
    )

    assert rows == [["'; DROP TABLE no_such_table; --"]]


async def test_the_row_cap_fetches_one_extra_row_for_truncation(
    executor: SqlQueryExecutor,
) -> None:
    """The executor over-fetches by one so the caller can detect truncation."""
    _, rows = await executor.run(
        "SELECT generate_series(1, 50) AS n", parameters={}, max_rows=10, timeout_ms=5000
    )

    assert len(rows) == 11
    assert rows[-1] == [11]


# --------------------------------------------------------------------------- #
# The posture
# --------------------------------------------------------------------------- #


async def test_a_read_only_executor_refuses_a_write_at_the_server(
    executor: SqlQueryExecutor,
) -> None:
    """The database refuses the write, not a rule in Python.

    This is the property that matters. Classification lives in application code
    and could be wrong, so read-only is pinned to the connection itself, where a
    mistake in the classifier cannot defeat it.
    """
    with pytest.raises(SQLAlchemyError) as captured:
        await executor.run(
            "CREATE TABLE should_not_exist (id int)",
            parameters={},
            max_rows=1,
            timeout_ms=5000,
        )

    assert "read-only" in str(captured.value).lower()


async def test_the_statement_timeout_is_enforced_by_the_server(
    executor: SqlQueryExecutor,
) -> None:
    with pytest.raises(SQLAlchemyError) as captured:
        await executor.run("SELECT pg_sleep(2)", parameters={}, max_rows=1, timeout_ms=250)

    message = str(captured.value).lower()
    assert "timeout" in message or "canceling" in message


async def test_the_timeout_does_not_leak_to_the_next_statement(
    executor: SqlQueryExecutor,
) -> None:
    """``set_config(..., true)`` is transaction-local, so the pool stays clean.

    Without the ``true`` this would set the ceiling on the physical connection
    and the next caller to check it out would inherit it.
    """
    with pytest.raises(SQLAlchemyError):
        await executor.run("SELECT pg_sleep(2)", parameters={}, max_rows=1, timeout_ms=250)

    _, rows = await executor.run("SELECT 1", parameters={}, max_rows=1, timeout_ms=5000)

    assert rows == [[1]]


async def test_an_executor_that_allows_writes_can_write(test_database_url: str) -> None:
    """The opt-in changes the connection, rather than only relabelling it."""
    writable = SqlQueryExecutor(test_database_url, read_only=False)
    try:
        await writable.run(
            "CREATE TEMP TABLE scratch (id int)", parameters={}, max_rows=1, timeout_ms=5000
        )
        columns, rows = await writable.run(
            "INSERT INTO scratch (id) VALUES (1) RETURNING id",
            parameters={},
            max_rows=1,
            timeout_ms=5000,
        )
    finally:
        await writable.close()

    assert columns == ["id"]
    assert rows == [[1]]


async def test_an_unreachable_target_fails_rather_than_hanging() -> None:
    """A bad target must surface as an error, not as a stuck request."""
    dead = SqlQueryExecutor("postgresql+asyncpg://127.0.0.1:1/nothing")
    try:
        # A refused connection surfaces as the socket error itself: the driver
        # never got far enough for SQLAlchemy to wrap it.
        with pytest.raises((SQLAlchemyError, OSError)):
            await dead.run("SELECT 1", parameters={}, max_rows=1, timeout_ms=500)
    finally:
        await dead.close()


def test_a_synchronous_driver_is_rejected_at_construction() -> None:
    """The shared engine builder's guard applies to this executor too.

    No server is involved: the URL is rejected before a connection is attempted.
    """
    with pytest.raises(DatabaseError) as captured:
        SqlQueryExecutor("postgresql://localhost/langgraph")

    assert "async driver" in str(captured.value)
