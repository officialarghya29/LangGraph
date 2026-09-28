"""Shared test fixtures.

Two things live here because both the API tests and the integration tests need
them, and duplicating them would let the two drift apart:

- **A throwaway database.** Tests must never touch the development database.
  A separate one is created, migrated with the real migrations, and dropped with
  the process. Everything that runs against PostgreSQL — including the API
  suite — is pointed at it.
- **A synchronous SQL helper.** Assertions about durability are more convincing
  when they read the row over a *different* connection than the application
  used, and doing that synchronously avoids creating an event loop alongside the
  one the test client is already running.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from app.core.config import Settings, reset_settings_cache
from app.database.connection import plain_dsn

ROOT = Path(__file__).resolve().parents[1]

#: Name of the throwaway database. Prefixed so it is obvious in a table listing.
TEST_DATABASE = "langgraph_test"

#: Tables emptied between tests, ordered child-first so foreign keys do not
#: block the truncation. Declared once because three suites truncate the same
#: schema, and a table added to the models but missing here would leak rows from
#: one test into the next.
TABLES: tuple[str, ...] = (
    "execution_events",
    "tool_calls",
    "approvals",
    "agent_runs",
    "task_steps",
    "memory_records",
    "tasks",
    "conversations",
    "users",
)

#: The ``TRUNCATE`` statement that empties every table and restarts identities.
TRUNCATE_ALL = f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE"

#: Set to 1 to turn "services unavailable" from a skip into a failure.
REQUIRE_SERVICES = os.environ.get("REQUIRE_SERVICES", "") == "1"


def swap_database(url: str, database: str) -> str:
    """Return ``url`` pointing at a different database on the same server.

    Parsed by hand rather than with ``urllib.parse`` so the SQLAlchemy driver
    suffix survives intact.
    """
    prefix, _, remainder = url.partition("://")
    credentials, _, host_part = remainder.rpartition("@")
    host, _, _ = host_part.partition("/")
    return f"{prefix}://{credentials}@{host}/{database}"


def services_required_or_skip(reason: str, exc: BaseException) -> None:
    """Skip a test when a service is missing, unless ``REQUIRE_SERVICES`` is set.

    The flag exists so a verification run cannot report success because the
    infrastructure was absent: with it set, a missing service is a failure.
    """
    if REQUIRE_SERVICES:
        raise exc
    pytest.skip(f"{reason} ({exc})")


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """Create and migrate the throwaway database, and point settings at it.

    Returns:
        The SQLAlchemy URL of the test database.
    """
    import asyncio

    import asyncpg

    maintenance = plain_dsn(swap_database(Settings().database_url, "postgres"))
    target = swap_database(Settings().database_url, TEST_DATABASE)

    async def recreate() -> None:
        connection = await asyncpg.connect(maintenance)
        try:
            # Terminate stragglers first: a connection left open by a previous
            # crashed run makes DROP DATABASE fail.
            await connection.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = $1 AND pid <> pg_backend_pid()",
                TEST_DATABASE,
            )
            await connection.execute(f'DROP DATABASE IF EXISTS "{TEST_DATABASE}"')
            await connection.execute(f'CREATE DATABASE "{TEST_DATABASE}"')
        finally:
            await connection.close()

    try:
        asyncio.run(recreate())
    except Exception as exc:
        services_required_or_skip("PostgreSQL is not reachable", exc)
        raise  # unreachable: the helper raised

    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", target)
    command.upgrade(config, "head")

    # Point the whole session at the test database. Set here rather than in each
    # suite so no test can accidentally build an application against the
    # development database.
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = target
    reset_settings_cache()
    try:
        yield target
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous
        reset_settings_cache()


@pytest.fixture(scope="session", autouse=True)
def _use_test_database(test_database_url: str) -> None:
    """Ensure every suite runs against the throwaway database."""
    del test_database_url


@pytest.fixture
def sql(test_database_url: str) -> Iterator[Callable[[str, tuple[object, ...]], list[tuple]]]:
    """Return a function that runs a statement on its own connection.

    Autocommitting and outside the application's engine, so a row it reads has
    genuinely been committed rather than merely flushed in another session.

    Yields:
        ``run(statement, params) -> rows``.
    """
    import psycopg

    del test_database_url
    with psycopg.connect(plain_dsn(Settings().database_url), autocommit=True) as connection:

        def run(statement: str, params: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
            with connection.cursor() as cursor:
                cursor.execute(statement, params)
                if cursor.description is None:
                    return []
                return [tuple(row) for row in cursor.fetchall()]  # type: ignore[misc]

        yield run
