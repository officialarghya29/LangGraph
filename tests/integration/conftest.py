"""Fixtures for tests that require a real PostgreSQL and Redis.

These tests are not mocked, and that is the point. A mocked session proves the
test double works, not that the SQL, the constraints, the cascades, or the
migration do. Everything here runs against a real database using the same
migrations the deployment does.

The database is a separate one on the same server, created and dropped per test
session, so a run cannot damage development data. Isolation between tests is by
truncation rather than a rolled-back transaction, because some behaviour under
test — a cascade, a unique violation — only becomes visible once a statement
has actually committed.

If no server is reachable these tests skip, unless ``REQUIRE_SERVICES=1`` is
set, in which case they fail. That flag exists so a verification run cannot
accidentally report success because the infrastructure was missing.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.database.connection import Database

ROOT = Path(__file__).resolve().parents[2]

#: Name of the throwaway database. Prefixed so it is obvious in a table listing.
TEST_DATABASE = "langgraph_test"

#: Set to 1 to turn "services unavailable" from a skip into a failure.
REQUIRE_SERVICES = os.environ.get("REQUIRE_SERVICES", "") == "1"


def _base_settings() -> Settings:
    """Return settings from the environment, ignoring an unreadable .env."""
    return Settings()


def _swap_database(url: str, database: str) -> str:
    """Return ``url`` pointing at a different database on the same server.

    Parsed by hand rather than with ``urllib.parse`` because a SQLAlchemy URL
    carries a driver suffix (``postgresql+asyncpg``) that must survive intact.
    """
    prefix, _, remainder = url.partition("://")
    credentials, _, host_part = remainder.rpartition("@")
    host, _, _ = host_part.partition("/")
    return f"{prefix}://{credentials}@{host}/{database}"


def _plain_dsn(url: str) -> str:
    """Return ``url`` without its SQLAlchemy driver suffix.

    SQLAlchemy writes the driver as ``postgresql+asyncpg``; the raw driver does
    not recognise that form and refuses the DSN. Stripping it here keeps one
    configured URL serving both.
    """
    scheme, _, rest = url.partition("://")
    return f"{scheme.split('+', 1)[0]}://{rest}"


def _maintenance_url(url: str) -> str:
    """Return a URL for the server's default database, for CREATE/DROP DATABASE."""
    return _plain_dsn(_swap_database(url, "postgres"))


@pytest.fixture(scope="session")
def database_url() -> str:
    """Return the URL of the test database, creating the database itself.

    Raises:
        pytest.skip: When PostgreSQL is unreachable, unless ``REQUIRE_SERVICES``
            is set, in which case the original error propagates as a failure.
    """
    import asyncio

    import asyncpg

    target = _swap_database(_base_settings().database_url, TEST_DATABASE)
    maintenance = _maintenance_url(_base_settings().database_url)

    async def recreate() -> None:
        connection = await asyncpg.connect(maintenance)
        try:
            # Terminate stragglers first: a connection left open by a previous
            # crashed run would make DROP DATABASE fail.
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
        if REQUIRE_SERVICES:
            raise
        pytest.skip(f"PostgreSQL is not reachable at {_base_settings().database_url} ({exc})")

    return target


@pytest.fixture(scope="session")
def migrated_database(database_url: str) -> str:
    """Apply every migration to the test database before any test runs."""
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")
    return database_url


@pytest.fixture(scope="session")
def database(migrated_database: str) -> Iterator[Database]:
    """Return a :class:`Database` bound to the migrated test database."""
    import asyncio

    instance = Database(migrated_database)
    yield instance
    asyncio.run(instance.dispose())


#: Tables truncated between tests, ordered child-first so foreign keys do not
#: block the truncation.
_TABLES = (
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


@pytest_asyncio.fixture
async def session(database: Database) -> AsyncIterator[AsyncSession]:
    """Yield a session against a database emptied of the previous test's rows.

    The session is rolled back rather than committed on teardown. A test that
    deliberately provokes an ``IntegrityError`` leaves the transaction aborted,
    and committing it would fail during teardown and be reported as an error in
    an otherwise passing test. A test that needs its writes to outlive the
    session — because another connection must see them — commits explicitly.
    """
    async with database.session() as purge:
        await purge.execute(text(f"TRUNCATE {', '.join(_TABLES)} RESTART IDENTITY CASCADE"))

    active = database.session_factory()
    try:
        yield active
    finally:
        await active.rollback()
        await active.close()


@pytest.fixture
def unique_name() -> str:
    """Return a collision-proof identifier, so parallel runs cannot clash."""
    return uuid.uuid4().hex


# --------------------------------------------------------------------------- #
# Redis
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def redis_url() -> str:
    """Return the Redis URL, skipping if the server is unreachable."""
    import asyncio

    from redis.asyncio import Redis

    url = _base_settings().redis_url

    async def check() -> None:
        client = Redis.from_url(url)
        try:
            await client.ping()
        finally:
            await client.aclose()

    try:
        asyncio.run(check())
    except Exception as exc:
        if REQUIRE_SERVICES:
            raise
        pytest.skip(f"Redis is not reachable at {url} ({exc})")

    return url
