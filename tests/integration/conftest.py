"""Fixtures for tests that require a real PostgreSQL and Redis.

These tests are not mocked, and that is the point. A mocked session proves the
test double works, not that the SQL, the constraints, the cascades, or the
migration do. Everything here runs against a real database using the same
migrations the deployment does.

The database itself — created, migrated, dropped, and pointed at by the
environment — comes from :mod:`tests.conftest`, so every suite in this project
shares one throwaway database rather than each creating its own. Only the
connection handle and the per-test truncation live here, because only these
tests need a live :class:`~app.database.connection.Database`.

Isolation between tests is by truncation rather than a rolled-back transaction,
because some behaviour under test — a cascade, a unique violation — only becomes
visible once a statement has actually committed.

If no server is reachable these tests skip, unless ``REQUIRE_SERVICES=1`` is
set, in which case they fail. That flag exists so a verification run cannot
accidentally report success because the infrastructure was missing.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Iterator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.database.connection import Database
from tests.conftest import TRUNCATE_ALL, services_required_or_skip


@pytest.fixture(scope="session")
def database(test_database_url: str) -> Iterator[Database]:
    """Return a :class:`Database` bound to the migrated test database."""
    instance = Database(test_database_url)
    try:
        yield instance
    finally:
        asyncio.run(instance.dispose())


@pytest.fixture(autouse=True)
async def _empty_tables(database: Database) -> None:
    """Empty every table before each test in this package.

    Autouse and dependent on ``database`` rather than on ``session`` because not
    every test here wants a session: the memory tests build their own stores and
    open their own connections. Truncating per test — rather than only for tests
    that happen to request a session — is what keeps one test's rows from being
    read back as another's, which is an isolation bug that shows up as an
    assertion about counts or ordering rather than as an obvious crash.

    Truncation is chosen over a rolled-back transaction because some behaviour
    under test — a cascade, a unique violation — only becomes visible once a
    statement has actually committed.
    """
    async with database.session() as purge:
        await purge.execute(text(TRUNCATE_ALL))


@pytest_asyncio.fixture
async def session(database: Database, _empty_tables: None) -> AsyncIterator[AsyncSession]:
    """Yield a session against a database emptied of the previous test's rows.

    The session is rolled back rather than committed on teardown. A test that
    deliberately provokes an ``IntegrityError`` leaves the transaction aborted,
    and committing it would fail during teardown and be reported as an error in
    an otherwise passing test. A test that needs its writes to outlive the
    session — because another connection must see them — commits explicitly.
    """
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
    from redis.asyncio import Redis

    url = Settings().redis_url

    async def check() -> None:
        client = Redis.from_url(url)
        try:
            await client.ping()
        finally:
            await client.aclose()

    try:
        asyncio.run(check())
    except Exception as exc:
        services_required_or_skip(f"Redis is not reachable at {url}", exc)
        raise  # unreachable: the helper raised

    return url
