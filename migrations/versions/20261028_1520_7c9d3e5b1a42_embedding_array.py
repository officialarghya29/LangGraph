"""Migration 7c9d3e5b1a42: store embeddings as an array rather than JSON.

The column holds a list of floats and always has; JSONB was a convenient way to
write one, not a requirement of the value. Reading it back cost more than the rest
of a retrieval put together — 500 rows of 256 dimensions meant parsing 128 000
numbers out of text, about 95 ms — because JSON has to be tokenised before a
single similarity can be computed. ``double precision[]`` is decoded by the driver
in C, and ``double precision`` rather than ``real`` so the values round-trip
exactly: cosine similarity is computed in Python floats, and a narrower column
would move the last bits of every score and reorder exact ties.

The backfill deliberately does not trust the old column. A JSONB value could be a
string, an object, or an array containing something that is not a number, because
nothing enforced otherwise; those rows were already treated as having no vector by
the reader, so they keep that meaning rather than failing the upgrade.

Revision ID: 7c9d3e5b1a42
Revises: 91057e4e7d8d
Create Date: 2026-10-28 15:20:00.000000

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "7c9d3e5b1a42"
down_revision: str | None = "91057e4e7d8d"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Convert the embedding column from JSONB to a float array, keeping the data.

    Rows that are not an array of numbers are left NULL rather than failing the
    migration: the reader already treated them as having no vector, and a
    deployment should not be stuck between two schema versions over a row nothing
    could use anyway.
    """
    op.add_column(
        "memory_records",
        sa.Column("embedding_array", postgresql.ARRAY(sa.Float()), nullable=True),
    )
    op.execute(
        """
        UPDATE memory_records
        SET embedding_array = ARRAY(
            SELECT element::double precision
            FROM jsonb_array_elements(embedding) AS element
        )
        WHERE jsonb_typeof(embedding) = 'array'
          AND NOT EXISTS (
              SELECT 1 FROM jsonb_array_elements(embedding) AS element
              WHERE jsonb_typeof(element) <> 'number'
          )
        """
    )
    op.drop_column("memory_records", "embedding")
    op.alter_column("memory_records", "embedding_array", new_column_name="embedding")


def downgrade() -> None:
    """Convert the column back to JSONB.

    The reverse is lossless: every float in an array has an exact JSONB
    representation, and a NULL stays NULL.
    """
    op.add_column(
        "memory_records",
        sa.Column(
            "embedding_json",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=True,
        ),
    )
    op.execute(
        """
        UPDATE memory_records
        SET embedding_json = to_jsonb(embedding)
        WHERE embedding IS NOT NULL
        """
    )
    op.drop_column("memory_records", "embedding")
    op.alter_column("memory_records", "embedding_json", new_column_name="embedding")
