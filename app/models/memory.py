"""Memory domain models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

__all__ = ["ContextItem", "MemoryItem", "MemoryType"]


class MemoryType(StrEnum):
    """The four memory tiers."""

    SHORT_TERM = "short_term"
    WORKING = "working"
    LONG_TERM = "long_term"
    EXECUTION = "execution"

    @property
    def is_durable(self) -> bool:
        """Return whether this tier survives the process.

        Only durable tiers are worth the cost of evaluating relevance before
        writing, because only they accumulate.
        """
        return self in {MemoryType.LONG_TERM, MemoryType.EXECUTION}


class ContextItem(BaseModel):
    """A piece of external context retrieved during a run.

    Retrieved content is untrusted by construction. ``trusted`` defaults to
    ``False`` and must be set explicitly by the application for content that
    originated inside the system. Everything retrieved from a web page, a
    document, a repository, or a database row stays untrusted, and must never be
    allowed to override system instructions.
    """

    content: str
    source: str | None = None
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    trusted: bool = False


class MemoryItem(BaseModel):
    """A single memory record, tier-independent."""

    id: str
    content: str
    type: MemoryType = MemoryType.LONG_TERM
    user_id: str | None = None
    conversation_id: str | None = None
    source: str | None = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime | None = None

    @classmethod
    def should_persist(cls, importance: float, *, threshold: float = 0.5) -> bool:
        """Decide whether a candidate is worth durable storage.

        Evaluated before writing so the long-term store does not fill with
        conversational noise.

        Args:
            importance: Scored importance in the range 0.0 to 1.0.
            threshold: Minimum importance worth keeping.

        Returns:
            Whether the item should be persisted.
        """
        return importance >= threshold
