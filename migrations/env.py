"""Alembic environment.

Two things differ from the generated default:

1. **The URL comes from the application.** :mod:`app.core.config` is the single
   definition of how to reach the database, so this environment imports it
   rather than reading ``alembic.ini``. That means a migration runs against the
   same target the application does, with no second copy of a credential.

2. **Migrations run on an async engine.** The application is async end to end,
   and the driver it uses (``asyncpg``) has no synchronous mode. Alembic drives
   the same engine the application will use, so a migration cannot succeed
   against a driver the application cannot actually open.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Importing the models module is what populates Base.metadata. Without it,
# autogenerate would compare against an empty schema and produce a migration
# that drops every table.
from app.core.config import get_settings
from app.database import models  # noqa: F401
from app.database.base import Base

config = context.config

if config.config_file_name is not None:
    # ``disable_existing_loggers=False`` is not the default and has to be asked
    # for. ``fileConfig`` otherwise sets ``disabled = True`` on every logger that
    # already exists and is not named in ``alembic.ini`` — which, when migrations
    # are run in the same process as the application, is every logger the
    # application owns. The result is a process that migrates, starts, serves
    # traffic, and emits no log lines at all, with nothing to say why.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _database_url() -> str:
    """Return the URL to migrate.

    Resolution order, most specific first:

    1. ``sqlalchemy.url`` set on the config object. This is how the test suite
       points migrations at a throwaway database without touching the
       environment of the process running the tests.
    2. ``alembic -x url=...`` on the command line, for a one-off run.
    3. The application settings, which is the normal path.
    """
    url = (
        config.get_main_option("sqlalchemy.url")
        or context.get_x_argument(as_dictionary=True).get("url")
        or get_settings().database_url
    )
    # ConfigParser treats '%' as interpolation. A password can contain one.
    return url.replace("%", "%%")


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting to a database.

    Used to review what a migration will do before it is run.
    """
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def _run_migrations(connection: Connection) -> None:
    """Configure the migration context against a live connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        # Render the deterministic constraint names declared in the metadata, so
        # a generated migration matches what the models say.
        render_as_batch=False,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations against a live database on an async engine."""
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = _database_url()

    engine = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with engine.connect() as connection:
        await connection.run_sync(_run_migrations)

    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
