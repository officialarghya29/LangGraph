"""Migration verification: the migrations and the models must describe one schema.

Alembic and the ORM are two descriptions of the same database, and nothing in
Python keeps them in step. Write a column in the model and forget the migration,
and the application works perfectly on the developer's machine — where the table
was created by ``create_all`` at some point — and fails in every environment built
from migrations. The failure surfaces as a missing column in production, which is
the worst place to learn about it.

These tests close that gap by inspecting the schema that the migrations actually
built and comparing it against the metadata the application actually queries.
They run against real PostgreSQL, and each check reports the specific difference
rather than a bare assertion, because "six columns disagree" is not actionable
and "``tasks.owner_id`` exists in the model and not in the database" is.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, text

from app.core.config import get_settings
from app.database.base import Base
from app.database.connection import Database

#: Alembic's own bookkeeping table, which is deliberately absent from the models.
_ALEMBIC_TABLE = "alembic_version"

#: Tables the checkpointer library owns. It creates them itself, idempotently, at
#: start-up, because it has to be free to change its own schema with its own
#: version rather than waiting for a migration in this repository. They are
#: expected to be present and, along with ``alembic_version``, they are the only
#: tables allowed to exist without being declared by a model.
_CHECKPOINTER_TABLES = frozenset(
    {"checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations"}
)

#: Tables nobody in this repository declares, but that a correct run leaves behind.
_SUPPORTED_EXTRAS = _CHECKPOINTER_TABLES | {_ALEMBIC_TABLE}

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def settings_database_url() -> str:
    """Return the database URL the session is pointed at.

    Read from the settings rather than from the environment so the test migrates
    the throwaway database the rest of the suite uses, not whichever database
    ``DATABASE_URL`` named before the session fixture replaced it.
    """
    return get_settings().database_url


@pytest.fixture
def alembic_config() -> Config:
    """Return the Alembic configuration used by this repository."""
    return Config(str(REPO_ROOT / "alembic.ini"))


def _inspect(connection: Any) -> dict[str, Any]:
    """Return a snapshot of the live schema.

    Synchronous on purpose: ``run_sync`` takes a blocking callable and runs it on
    a worker thread. An ``async def`` here would be called and its coroutine
    returned unawaited, so the snapshot would be a coroutine object rather than a
    schema — and every comparison against it would fail for the wrong reason.

    Args:
        connection: A synchronous connection, as handed to ``run_sync``.

    Returns:
        Column names and index names per table, plus the primary key columns.
    """
    inspector = inspect(connection)
    snapshot: dict[str, Any] = {}
    for table in inspector.get_table_names():
        snapshot[table] = {
            "columns": {column["name"] for column in inspector.get_columns(table)},
            "indexes": {index["name"] for index in inspector.get_indexes(table)},
            "primary_key": set(inspector.get_pk_constraint(table)["constrained_columns"] or []),
        }
    return snapshot


async def test_the_database_is_migrated_to_head(database: Database, alembic_config: Config) -> None:
    """A test run against a stale database proves nothing about the current code.

    The test database is created by running the migrations, so this also fails
    loudly if a migration stops being applied rather than being silently skipped.
    """
    head = ScriptDirectory.from_config(alembic_config).get_current_head()

    async with database.engine.connect() as connection:
        applied = await connection.scalar(text(f"SELECT version_num FROM {_ALEMBIC_TABLE}"))

    assert applied == head, f"database is at {applied}, migrations head is {head}"


async def test_the_migrated_schema_has_every_table_the_models_declare(
    database: Database,
) -> None:
    """A model with no table is a table that will never exist in production."""
    async with database.engine.connect() as connection:
        live = await connection.run_sync(_inspect)

    expected = set(Base.metadata.tables) - {_ALEMBIC_TABLE}
    missing = sorted(expected - set(live))
    unexpected = sorted(set(live) - expected - _SUPPORTED_EXTRAS)

    assert not missing, f"models declare tables the migrations never create: {missing}"
    assert not unexpected, f"the database has tables the models do not declare: {unexpected}"

    # The checkpointer's tables are only there once something has opened a
    # checkpointer, which the API suite does and a bare migration run does not. So
    # they are allowed rather than required — requiring them would make this test
    # depend on which suite ran first. Either none or the whole set: a partial set
    # would mean ``setup()`` was interrupted, which is the difference between a
    # managed schema and half of one.
    present = _CHECKPOINTER_TABLES & set(live)
    assert present in (frozenset(), _CHECKPOINTER_TABLES), sorted(present)


async def test_every_migrated_table_has_the_columns_the_models_declare(
    database: Database,
) -> None:
    """Column drift is the failure this whole module exists to catch."""
    async with database.engine.connect() as connection:
        live = await connection.run_sync(_inspect)

    problems: list[str] = []
    for name, table in Base.metadata.tables.items():
        declared = {column.name for column in table.columns}
        actual = live.get(name, {}).get("columns", set())
        if declared - actual:
            problems.append(
                f"{name}: declared but missing from the database {sorted(declared - actual)}"
            )
        if actual - declared:
            problems.append(
                f"{name}: present in the database but not declared {sorted(actual - declared)}"
            )

    assert not problems, problems


async def test_primary_keys_agree_with_the_models(database: Database) -> None:
    """A table without the declared key cannot be addressed by the repository."""
    async with database.engine.connect() as connection:
        live = await connection.run_sync(_inspect)

    problems: list[str] = []
    for name, table in Base.metadata.tables.items():
        declared = {column.name for column in table.primary_key.columns}
        actual = live.get(name, {}).get("primary_key", set())
        if declared != actual:
            problems.append(f"{name}: model {sorted(declared)} vs database {sorted(actual)}")

    assert not problems, problems


async def test_every_declared_index_exists_in_the_database(database: Database) -> None:
    """Indexes are the part of a schema most often declared and never migrated.

    Only model-declared indexes are checked. Extra indexes in the database are
    allowed: a migration may add one deliberately for a query shape the model
    does not describe, and failing on that would be wrong.
    """
    async with database.engine.connect() as connection:
        live = await connection.run_sync(_inspect)

    problems: list[str] = []
    for name, table in Base.metadata.tables.items():
        for index in table.indexes:
            if index.name and index.name not in live.get(name, {}).get("indexes", set()):
                problems.append(f"{name}: index {index.name!r} is declared but not migrated")

    assert not problems, problems


async def test_the_search_path_is_not_relied_on_for_the_models(database: Database) -> None:
    """Every model must resolve through the default schema.

    A model bound to an explicit schema would work in the test database, which is
    built the same way, and fail wherever the deployment uses a different one.
    """
    schemas = {table.schema for table in Base.metadata.tables.values()}

    assert schemas in ({None}, set()), f"models pin an explicit schema: {schemas}"


def test_running_a_migration_in_process_does_not_silence_the_application() -> None:
    """Alembic's ``fileConfig`` disables loggers that already exist.

    ``fileConfig`` defaults to ``disable_existing_loggers=True``, which sets
    ``disabled = True`` on every logger not named in ``alembic.ini`` — so a process
    that runs migrations and *then* serves traffic emits no log lines at all, and
    nothing reports a problem. The deployment in ``docker-compose.yml`` is safe by
    accident, because it migrates in a separate process; anything that migrates in
    process was not.

    The logger is created before the upgrade deliberately: that is the condition
    ``disable_existing_loggers`` acts on.
    """
    logger = logging.getLogger("app.migration_probe")
    logger.disabled = False

    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", settings_database_url())

    command.upgrade(config, "head")

    assert logger.disabled is False, "running a migration disabled the application's loggers"
