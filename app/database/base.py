"""Declarative base and shared column types.

The naming convention below is not cosmetic. Without it, Alembic cannot drop or
alter an index or constraint it did not name itself, because PostgreSQL
generates the name and Alembic has nothing stable to reference. Every constraint
in this schema therefore has a deterministic name.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, MetaData, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

__all__ = ["Base", "JsonColumn", "TimestampMixin", "utc_now"]

#: Deterministic names for every index and constraint, so migrations are
#: reversible and a review can see the intended name before it exists.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

#: Portable JSON that becomes JSONB on PostgreSQL. JSONB is used for the real
#: deployment because it is indexable and validates on write; the portable
#: variant keeps the models usable against a lightweight backend in a unit test.
JsonColumn = JSON().with_variant(JSONB(), "postgresql")


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for every ORM model."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    def as_dict(self) -> dict[str, Any]:
        """Return the row as a plain dictionary of column values.

        Convenience for logging and for the repositories' read paths. It does
        not walk relationships, so it cannot accidentally serialise an entire
        object graph.
        """
        return {column.key: getattr(self, column.key) for column in self.__table__.columns}


class TimestampMixin:
    """Adds server-side created/updated timestamps.

    ``server_default`` and ``onupdate`` are used rather than Python defaults so
    the timestamps are correct even when a row is written by something other
    than the application — a migration backfill, for example.
    """

    # Deliberately not indexed on its own. Every read that orders by creation
    # time also filters by an owner column first, so an index on ``created_at``
    # alone would be written on every insert and never chosen by the planner.
    # Each table declares the composite index its real query shape needs.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )


#: Primary-key column shared by every table: a UUID generated in Python, so an
#: entity has its identity before it is flushed and can be used as a LangGraph
#: thread id without a round trip.
PrimaryKey = Uuid(as_uuid=True)
