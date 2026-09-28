"""Database engine, sessions, and connectivity checks.

The engine is built once per process and owned by a :class:`Database` instance
rather than a module-level global. Tests build their own against a test
database, and a process that never touches the database never opens a pool.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.exceptions import DatabaseError

__all__ = ["Database", "build_engine", "plain_dsn"]

logger = logging.getLogger(__name__)

#: Only these drivers are accepted. A synchronous driver in an async application
#: blocks the event loop on every query, which is a defect that hides until load.
_ASYNC_DRIVERS = ("postgresql+asyncpg", "sqlite+aiosqlite")


def plain_dsn(url: str) -> str:
    """Return ``url`` without its SQLAlchemy driver suffix.

    SQLAlchemy writes the driver into the scheme (``postgresql+asyncpg``);
    libpq-based clients such as psycopg do not recognise that form and reject
    the DSN. Stripping it here lets one configured URL serve both without a
    second copy of the credential.

    Args:
        url: A SQLAlchemy database URL.

    Returns:
        The same URL in plain ``postgresql://`` form.
    """
    scheme, separator, rest = url.partition("://")
    if not separator:
        return url
    return f"{scheme.split('+', 1)[0]}://{rest}"


def build_engine(
    url: str,
    *,
    pool_size: int = 10,
    max_overflow: int = 5,
    echo: bool = False,
    connect_args: Mapping[str, Any] | None = None,
) -> AsyncEngine:
    """Create an async engine for ``url``.

    Args:
        url: SQLAlchemy database URL. Must use an async driver.
        pool_size: Persistent connections to keep open.
        max_overflow: Additional connections allowed beyond the pool.
        echo: Whether to log every statement. Off outside debugging.
        connect_args: Extra arguments passed to the driver on connect. Only the
            database tool uses this, to pin a connection to read-only mode at
            the server rather than relying on a check in Python.

    Returns:
        A configured async engine.

    Raises:
        DatabaseError: If the URL does not name an async driver. Failing here is
            deliberate: the alternative is a blocking driver that appears to work
            in a single-request test and stalls under concurrency.
    """
    if not url.startswith(_ASYNC_DRIVERS):
        raise DatabaseError(
            "the database URL must use an async driver",
            detail=f"expected one of {', '.join(_ASYNC_DRIVERS)}",
        )

    options: dict[str, Any] = {"echo": echo, "pool_pre_ping": True}
    if url.startswith("postgresql"):
        # SQLite's default pool does not accept these, and a file-backed SQLite
        # test database has no use for them.
        options["pool_size"] = pool_size
        options["max_overflow"] = max_overflow
    if connect_args:
        options["connect_args"] = dict(connect_args)

    return create_async_engine(url, **options)


class Database:
    """Owns the engine and hands out sessions."""

    def __init__(
        self,
        url: str,
        *,
        pool_size: int = 10,
        max_overflow: int = 5,
        echo: bool = False,
    ) -> None:
        self.url = url
        self.engine = build_engine(url, pool_size=pool_size, max_overflow=max_overflow, echo=echo)
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            # Rows are read, handed to a repository, and returned; autoflush would
            # emit partial writes mid-read and has no benefit here.
            autoflush=False,
        )

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a session inside a transaction.

        Commits on success, rolls back on any exception, and always closes the
        session. A caller that forgets to commit cannot leak a half-written
        transaction back into the pool.
        """
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def ping(self) -> tuple[bool, str]:
        """Check that the database answers.

        Returns:
            ``(True, "ok")`` when reachable, otherwise ``(False, reason)``. The
            reason is a classification, never a connection string: a driver error
            can echo the DSN, and the DSN carries the password.
        """
        try:
            async with self.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except SQLAlchemyError as exc:
            logger.warning("database.ping_failed", extra={"error": type(exc).__name__})
            return False, f"unreachable ({type(exc).__name__})"
        except Exception as exc:
            # A driver can raise anything at all on a broken connection, and a
            # readiness probe must report rather than propagate.
            logger.warning("database.ping_failed", extra={"error": type(exc).__name__})
            return False, f"unreachable ({type(exc).__name__})"
        return True, "ok"

    async def dispose(self) -> None:
        """Close every pooled connection."""
        await self.engine.dispose()
