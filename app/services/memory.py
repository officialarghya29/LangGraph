"""Memory management across four tiers.

The tiers exist because the things worth remembering have different lifetimes.
A note about the current step is worthless in an hour; a fact the user stated
about themselves is worth keeping indefinitely. Collapsing them into one store
means either discarding things that should persist or persisting things that
should have expired, and there is no threshold that fixes both.

:class:`MemoryManager` is the single entry point. It owns four decisions that
would otherwise be repeated — and eventually disagreed about — at every call
site:

- **What is worth writing.** Durable tiers are gated on an importance estimate;
  volatile tiers are not, because a volatile record that is not kept costs
  nothing either way.
- **How a memory is found.** Retrieval blends embedding similarity, importance,
  and recency. Similarity alone surfaces recently-repeated trivia; recency alone
  surfaces whatever happened last regardless of relevance.
- **What is injected into a prompt.** Only the top few above a score floor, so a
  weak match is dropped rather than padding a prompt with noise.
- **What is expired.** Only the volatile tiers are pruned. Long-term memory is
  curated by importance, not by age.

Retrieved memory is marked **untrusted**. A memory may have been distilled from
a web page, a repository, or a document, so it carries the same injection risk
as any other retrieved content, and it must not be able to override system
instructions by having been stored once.
"""

from __future__ import annotations

import math
import re
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence, Set
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from app.core.config import Settings
from app.core.exceptions import EmbeddingError
from app.models.memory import ContextItem, MemoryItem, MemoryType
from app.services.embeddings import EmbeddingProvider

__all__ = [
    "InMemoryMemoryStore",
    "MemoryManager",
    "MemoryQuery",
    "MemoryStore",
    "PostgresMemoryStore",
    "build_memory_manager",
    "score_candidate",
]

_WORD_PATTERN = re.compile(r"[a-z0-9]+")

#: Relative weights for the blended relevance score. They sum to 1.0 so a score
#: stays comparable to the configured floor regardless of the store's size.
_SIMILARITY_WEIGHT = 0.7
_IMPORTANCE_WEIGHT = 0.2
_RECENCY_WEIGHT = 0.3

#: Half-life of the recency term, in days. A memory this old contributes half of
#: what a brand-new one does, all else equal.
_RECENCY_HALF_LIFE_DAYS = 14.0

#: Tokens that carry no discriminating signal, so lexical overlap computed
#: against them would rank everything as similar to everything.
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
    }
)


def _tokens(text: str) -> set[str]:
    """Return the discriminating lowercase tokens of ``text``."""
    return {word for word in _WORD_PATTERN.findall(text.lower()) if word not in _STOP_WORDS}


def _vector_norm(vector: Sequence[float]) -> float:
    """Return the Euclidean norm of ``vector``."""
    return math.sqrt(sum(value * value for value in vector))


def _cosine(embedding: Sequence[float], other: Sequence[float]) -> float:
    """Return cosine similarity, or ``0.0`` when it is undefined.

    Returns the negative part clipped to zero: two vectors pointing in opposite
    directions are unrelated for retrieval purposes, and letting a negative score
    through would let anti-correlated text displace genuinely relevant text.
    """
    return _cosine_prepared(embedding, _vector_norm(embedding), other)


def _cosine_prepared(
    embedding: Sequence[float], embedding_norm: float, other: Sequence[float]
) -> float:
    """Return cosine similarity when one side's norm is already known.

    The query embedding is the same for every candidate in a scan, so its norm
    is computed once and passed in rather than recomputed per row.

    Args:
        embedding: The first vector.
        embedding_norm: The Euclidean norm of ``embedding``.
        other: The second vector.

    Returns:
        Similarity in the range 0.0 to 1.0, or ``0.0`` when it is undefined —
        mismatched dimensions, an empty vector, or a zero norm.
    """
    if len(embedding) != len(other) or not embedding:
        return 0.0
    other_norm = _vector_norm(other)
    if embedding_norm == 0.0 or other_norm == 0.0:
        return 0.0
    dot = sum(a * b for a, b in zip(embedding, other, strict=True))
    return max(0.0, dot / (embedding_norm * other_norm))


def _jaccard(left: str, right: str) -> float:
    """Return lexical overlap between two strings, in the range 0.0 to 1.0.

    Used when one side has no embedding. A row written before embeddings were
    configured is still worth finding, and scoring it at zero would make it
    permanently invisible instead of merely ranked lower.
    """
    return _jaccard_tokens(frozenset(_tokens(left)), _tokens(right))


def _jaccard_tokens(left: frozenset[str], right: Set[str]) -> float:
    """Return overlap between two pre-tokenised sets.

    Split out from :func:`_jaccard` so the query side can be tokenised once per
    scan instead of once per candidate.
    """
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


@dataclass(frozen=True, slots=True)
class _QueryContext:
    """The parts of a query that do not change as candidates are scored.

    Tokenising the query text, taking the norm of its embedding, and reading the
    clock are each invariant across a scan. Retrieval scans up to
    ``memory_scan_limit`` rows, so leaving them inside the per-row loop turns a
    constant amount of work into linear work for no gain. This holds them so the
    loop body only does the part that genuinely differs per candidate.
    """

    text: str
    tokens: frozenset[str]
    vector: tuple[float, ...] | None
    norm: float
    now: datetime


def _prepare_query(
    text: str, embedding: Sequence[float] | None, *, now: datetime | None = None
) -> _QueryContext:
    """Return the invariant scoring context for a query.

    Args:
        text: The query text.
        embedding: The query embedding, or ``None`` when embeddings are absent.
        now: The instant to measure recency against. Injectable so tests can pin
            the clock instead of tolerating drift.

    Returns:
        A context reusable across every candidate of this query.
    """
    vector = tuple(embedding) if embedding else None
    return _QueryContext(
        text=text,
        tokens=frozenset(_tokens(text)),
        vector=vector,
        norm=_vector_norm(vector) if vector else 0.0,
        now=now or datetime.now(UTC),
    )


def score_candidate(
    context: _QueryContext, item: MemoryItem, embedding: Sequence[float] | None
) -> float:
    """Blend similarity, importance, and recency into one comparable score.

    Args:
        context: The prepared query context. May be reused across candidates.
        item: The stored memory being scored.
        embedding: The candidate's embedding, or ``None`` when it has none.

    Returns:
        A score comparable only against other scores from the same context.
    """
    if context.vector is not None and embedding is not None:
        similarity = _cosine_prepared(context.vector, context.norm, embedding)
    else:
        # One side has no vector. Lexical overlap is a weaker signal than cosine
        # similarity, so it is discounted rather than treated as equivalent.
        similarity = 0.8 * _jaccard_tokens(context.tokens, _tokens(item.content))

    age_days = max(0.0, (context.now - item.created_at).total_seconds() / 86_400)
    # Annotated because ``float.__pow__`` is typed as returning ``Any``: an
    # exponent may produce a complex number for some operand combinations.
    recency: float = 0.5 ** (age_days / _RECENCY_HALF_LIFE_DAYS)

    return (
        _SIMILARITY_WEIGHT * similarity
        + _IMPORTANCE_WEIGHT * item.importance
        + _RECENCY_WEIGHT * recency
    )


def estimate_importance(content: str, *, base: float = 0.5) -> float:
    """Estimate how much a piece of text is worth keeping.

    A heuristic, and deliberately a transparent one rather than a model call: it
    must be cheap enough to run on every candidate, deterministic so a test can
    assert on it, and auditable so an operator can see why something was kept.

    What raises the estimate is text that is *about* something durable — an
    explicit preference or decision, a concrete figure, a stated constraint.
    What lowers it is conversational filler.

    Args:
        content: The candidate text.
        base: Starting estimate for a neutral statement.

    Returns:
        An estimate clamped to the range 0.0 to 1.0.
    """
    text = content.strip()
    if not text:
        return 0.0

    lowered = text.lower()
    score = base

    # Length: a one-word note carries little, a paragraph usually carries some.
    words = len(_WORD_PATTERN.findall(lowered))
    if words <= 2:
        score -= 0.2
    elif words >= 25:
        score += 0.15

    # Durability markers: things a user states about themselves or their work.
    for marker in ("remember", "always", "never", "prefer", "my ", "our ", "must", "requires"):
        if marker in lowered:
            score += 0.25
            break

    # A concrete figure is usually a fact rather than an opinion.
    if any(character.isdigit() for character in lowered):
        score += 0.1

    # Interrogatives and filler are conversation, not memory.
    if lowered.endswith("?") or lowered.startswith(("hi ", "hello", "thanks", "ok")):
        score -= 0.25

    return max(0.0, min(1.0, score))


class MemoryQuery(BaseModel):
    """A request for memories relevant to some text."""

    text: str
    user_id: str
    conversation_id: str | None = None
    kinds: tuple[MemoryType, ...] = (MemoryType.LONG_TERM, MemoryType.EXECUTION)
    limit: int = Field(default=5, ge=1, le=50)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)


#: A stored row: the item plus whatever embedding accompanied it.
Candidate = tuple[MemoryItem, list[float] | None]


class MemoryStore(ABC):
    """Where memories live.

    Declared as a base class rather than a protocol so an incomplete
    implementation fails at import, and so every implementation is forced to
    state what it does about owner scoping: a store with no ``user_id`` filter
    is a cross-tenant leak, and that must be impossible to add by accident.
    """

    @abstractmethod
    async def add(
        self,
        item: MemoryItem,
        *,
        embedding: list[float] | None,
        embedding_model: str | None,
    ) -> MemoryItem:
        """Persist a memory and return it with its store-assigned id."""

    @abstractmethod
    async def candidates(
        self, *, user_id: str, kinds: Sequence[MemoryType], limit: int
    ) -> list[Candidate]:
        """Return this user's most recent memories of the given kinds."""

    @abstractmethod
    async def mark_accessed(self, memory_ids: Sequence[str]) -> None:
        """Record that memories were retrieved."""

    @abstractmethod
    async def delete(self, memory_id: str, *, user_id: str) -> bool:
        """Delete one of a user's memories. Returns whether a row was removed."""

    @abstractmethod
    async def prune(self, *, kind: MemoryType, older_than: datetime, limit: int) -> int:
        """Delete old memories of a volatile kind. Returns how many were removed."""

    @abstractmethod
    async def counts(self, user_id: str) -> dict[str, int]:
        """Return how many memories a user has of each kind."""


class InMemoryMemoryStore(MemoryStore):
    """A process-local store, for tests and single-process runs.

    Loses everything on restart, and says so rather than pretending: the durable
    postgres implementation is the one a deployment should select.
    """

    def __init__(self) -> None:
        self._items: dict[str, tuple[MemoryItem, list[float] | None, str | None]] = {}

    async def add(
        self,
        item: MemoryItem,
        *,
        embedding: list[float] | None,
        embedding_model: str | None,
    ) -> MemoryItem:
        """Persist a memory in this process."""
        self._items[item.id] = (item, embedding, embedding_model)
        return item

    async def candidates(
        self, *, user_id: str, kinds: Sequence[MemoryType], limit: int
    ) -> list[Candidate]:
        """Return the user's newest memories of the given kinds."""
        wanted = set(kinds)
        rows = [
            (item, embedding)
            for item, embedding, _ in self._items.values()
            if item.user_id == user_id and item.type in wanted
        ]
        rows.sort(key=lambda row: row[0].created_at, reverse=True)
        return rows[:limit]

    async def mark_accessed(self, memory_ids: Sequence[str]) -> None:
        """Record retrieval, which raises recency for the next ranking."""
        now = datetime.now(UTC)
        for memory_id in memory_ids:
            existing = self._items.get(memory_id)
            if existing is None:
                continue
            item, embedding, model = existing
            self._items[memory_id] = (
                item.model_copy(update={"updated_at": now}),
                embedding,
                model,
            )

    async def delete(self, memory_id: str, *, user_id: str) -> bool:
        """Delete a memory, but only its owner's."""
        existing = self._items.get(memory_id)
        if existing is None or existing[0].user_id != user_id:
            return False
        del self._items[memory_id]
        return True

    async def prune(self, *, kind: MemoryType, older_than: datetime, limit: int) -> int:
        """Delete old memories of a volatile kind."""
        if kind.is_durable:
            raise ValueError(f"refusing to prune durable memory kind {kind.value!r}")
        doomed = [
            memory_id
            for memory_id, (item, _, _) in self._items.items()
            if item.type is kind and item.created_at < older_than
        ][:limit]
        for memory_id in doomed:
            del self._items[memory_id]
        return len(doomed)

    async def counts(self, user_id: str) -> dict[str, int]:
        """Count a user's memories by kind."""
        counts: dict[str, int] = {}
        for item, _, _ in self._items.values():
            if item.user_id == user_id:
                counts[item.type.value] = counts.get(item.type.value, 0) + 1
        return counts


class PostgresMemoryStore(MemoryStore):
    """A durable store backed by the ``memory_records`` table."""

    def __init__(self, database: Any) -> None:
        self._database = database

    @staticmethod
    def _to_item(record: Any) -> MemoryItem:
        """Convert an ORM row into a domain item."""
        return MemoryItem(
            id=str(record.id),
            content=record.content,
            type=MemoryType(record.kind),
            user_id=str(record.user_id),
            conversation_id=None if record.conversation_id is None else str(record.conversation_id),
            source=record.source,
            importance=float(record.importance),
            metadata=dict(record.meta or {}),
            created_at=record.created_at,
            updated_at=record.accessed_at,
        )

    @staticmethod
    def _as_uuid(value: str) -> uuid.UUID | None:
        """Parse an identifier, returning ``None`` rather than raising."""
        try:
            return uuid.UUID(value)
        except (ValueError, AttributeError):
            return None

    async def _user_id(self, session: Any, external_id: str) -> uuid.UUID | None:
        """Resolve a principal to a database row, or ``None`` when unknown."""
        from app.database.repositories import UserRepository

        user = await UserRepository(session).get_by_external_id(external_id)
        return None if user is None else uuid.UUID(str(user.id))

    async def add(
        self,
        item: MemoryItem,
        *,
        embedding: list[float] | None,
        embedding_model: str | None,
    ) -> MemoryItem:
        """Persist a memory, provisioning the owning user on first sight."""
        from app.database.repositories import MemoryRepository, UserRepository

        async with self._database.session() as session:
            owner = await UserRepository(session).get_or_create(item.user_id or "anonymous")
            record = await MemoryRepository(session).create(
                user_id=uuid.UUID(str(owner.id)),
                kind=item.type.value,
                content=item.content,
                conversation_id=self._as_uuid(item.conversation_id or ""),
                embedding=embedding,
                embedding_model=embedding_model,
                importance=item.importance,
                source=item.source,
                meta=dict(item.metadata),
            )
            return self._to_item(record)

    async def candidates(
        self, *, user_id: str, kinds: Sequence[MemoryType], limit: int
    ) -> list[Candidate]:
        """Return this user's newest memories of the given kinds.

        The kinds are filtered in the query rather than after it, so a store with
        a large volatile tier does not push durable memories out of the scan
        window.
        """
        from app.database.repositories import MemoryRepository

        async with self._database.session() as session:
            owner = await self._user_id(session, user_id)
            if owner is None:
                return []
            repository = MemoryRepository(session)
            rows: list[Any] = []
            for kind in kinds:
                rows.extend(await repository.list_for_user(owner, kind=kind.value, limit=limit))
            rows.sort(key=lambda record: record.created_at, reverse=True)
            return [
                (self._to_item(record), _as_float_list(record.embedding)) for record in rows[:limit]
            ]

    async def mark_accessed(self, memory_ids: Sequence[str]) -> None:
        """Bump the access counter for retrieved memories."""
        from app.database.repositories import MemoryRepository

        keys = [key for key in (self._as_uuid(value) for value in memory_ids) if key is not None]
        if not keys:
            return
        async with self._database.session() as session:
            await MemoryRepository(session).mark_accessed(keys)

    async def delete(self, memory_id: str, *, user_id: str) -> bool:
        """Delete a memory, but only its owner's."""
        from app.database.repositories import MemoryRepository

        key = self._as_uuid(memory_id)
        if key is None:
            return False
        async with self._database.session() as session:
            owner = await self._user_id(session, user_id)
            if owner is None:
                return False
            return await MemoryRepository(session).delete(key, user_id=owner)

    async def prune(self, *, kind: MemoryType, older_than: datetime, limit: int) -> int:
        """Delete old memories of a volatile kind."""
        from app.database.repositories import MemoryRepository

        async with self._database.session() as session:
            return await MemoryRepository(session).prune(
                kind=kind.value, older_than=older_than, limit=limit
            )

    async def counts(self, user_id: str) -> dict[str, int]:
        """Count a user's memories by kind."""
        from app.database.repositories import MemoryRepository

        async with self._database.session() as session:
            owner = await self._user_id(session, user_id)
            if owner is None:
                return {}
            return await MemoryRepository(session).count_for_user(owner)


def _as_float_list(value: object) -> list[float] | None:
    """Coerce a stored JSON column back into a float vector, or ``None``.

    JSONB round-trips integers as integers, so a vector written as ``[0, 1]``
    returns ``[0, 1]`` rather than ``[0.0, 1.0]``. Anything that is not a list of
    numbers is treated as absent rather than partially trusted.
    """
    if not isinstance(value, list):
        return None
    try:
        return [float(component) for component in value]
    except (TypeError, ValueError):
        return None


class MemoryManager:
    """Reads, writes, and curates memory across the four tiers."""

    def __init__(
        self,
        store: MemoryStore,
        embeddings: EmbeddingProvider,
        settings: Settings,
    ) -> None:
        self._store = store
        self._embeddings = embeddings
        self._settings = settings

    @property
    def enabled(self) -> bool:
        """Return whether memory is configured on."""
        return self._settings.memory_enabled

    @property
    def store_name(self) -> str:
        """Return which store is in use, for readiness reporting.

        A durable and a volatile store look identical from the outside until a
        restart, so the difference is stated rather than left to be inferred.
        """
        return type(self._store).__name__

    async def _embed(self, text: str) -> list[float] | None:
        """Embed text, returning ``None`` rather than failing the caller.

        A memory that cannot be embedded is still worth storing and can still be
        found lexically. Propagating an embedding outage would turn a degraded
        retrieval into a failed task, which is the wrong trade for a subsystem
        whose entire contribution is optional context.
        """
        try:
            return await self._embeddings.embed(text)
        except EmbeddingError:
            return None

    async def remember(
        self,
        content: str,
        *,
        user_id: str,
        memory_type: MemoryType = MemoryType.LONG_TERM,
        importance: float | None = None,
        conversation_id: str | None = None,
        source: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> MemoryItem | None:
        """Store a memory, subject to the tier's retention rule.

        Args:
            content: What to remember. Blank content is refused.
            user_id: The principal the memory belongs to.
            memory_type: Which tier to write to.
            importance: Explicit importance. Estimated from the content when
                omitted.
            conversation_id: Conversation to attach the memory to.
            source: Where the content came from, for the audit trail.
            metadata: Anything else worth keeping alongside it.

        Returns:
            The stored item, or ``None`` when nothing was written — a blank
            string, a disabled manager, or a durable candidate below the
            importance threshold.
        """
        text = content.strip()
        if not self.enabled or not text:
            return None

        score = importance if importance is not None else estimate_importance(text)
        score = max(0.0, min(1.0, score))

        if memory_type.is_durable and not MemoryItem.should_persist(
            score, threshold=self._settings.memory_importance_threshold
        ):
            return None

        item = MemoryItem(
            id=str(uuid.uuid4()),
            content=text,
            type=memory_type,
            user_id=user_id,
            conversation_id=conversation_id,
            source=source,
            importance=score,
            metadata=metadata or {},
        )
        embedding = await self._embed(text)
        stored = await self._store.add(
            item,
            embedding=embedding,
            embedding_model=self._embeddings.name if embedding is not None else None,
        )
        return stored

    def _score(
        self,
        query: MemoryQuery,
        item: MemoryItem,
        embedding: list[float] | None,
        query_embedding: list[float] | None,
    ) -> float:
        """Score one candidate.

        Kept for callers scoring a single row; :meth:`recall` uses
        :func:`score_candidate` directly so the query is prepared once.
        """
        return score_candidate(_prepare_query(query.text, query_embedding), item, embedding)

    async def recall(self, query: MemoryQuery) -> list[MemoryItem]:
        """Return the memories most relevant to a query, best first.

        Only the top ``limit`` above the score floor are returned, and retrieval
        is recorded so a memory that keeps proving useful ranks higher next time.
        """
        if not self.enabled or not query.text.strip():
            return []

        floor = max(query.min_score, self._settings.memory_min_score)
        candidates = await self._store.candidates(
            user_id=query.user_id,
            kinds=query.kinds,
            limit=self._settings.memory_scan_limit,
        )
        if not candidates:
            return []

        query_embedding = await self._embed(query.text)
        # One tokenisation, one vector norm, and one clock reading for the whole
        # scan: all three are properties of the query, not of the candidate.
        context = _prepare_query(query.text, query_embedding)
        scored = [
            (score_candidate(context, item, embedding), item) for item, embedding in candidates
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        selected = [item for score, item in scored if score >= floor][: query.limit]
        if selected:
            await self._store.mark_accessed([item.id for item in selected])
        return selected

    async def context_for(self, query: MemoryQuery) -> list[ContextItem]:
        """Return recalled memories as prompt context.

        Marked untrusted, with the reason stated rather than assumed: a memory
        is text the system once wrote down, and it may have been distilled from
        a web page, a repository, or a document. Being stored is not a
        provenance guarantee, so it must not be able to override instructions.
        """
        recalled = await self.recall(query)
        return [
            ContextItem(
                content=item.content,
                source=item.source or f"memory:{item.type.value}",
                score=item.importance,
                trusted=False,
            )
            for item in recalled
        ]

    async def forget(self, memory_id: str, *, user_id: str) -> bool:
        """Delete one of a user's memories."""
        return await self._store.delete(memory_id, user_id=user_id)

    async def counts(self, user_id: str) -> dict[str, int]:
        """Return a user's memory counts by tier."""
        return await self._store.counts(user_id)

    async def consolidate(self, *, limit: int = 1000) -> dict[str, int]:
        """Expire volatile memories past their time-to-live.

        Only the volatile tiers are touched. Pruning long-term memory by age
        would discard exactly the things it exists to keep.

        Returns:
            How many rows were removed, per tier.
        """
        if not self.enabled:
            return {}

        cutoff = datetime.now(UTC) - timedelta(seconds=self._settings.memory_volatile_ttl_seconds)
        removed: dict[str, int] = {}
        for kind in (MemoryType.SHORT_TERM, MemoryType.WORKING):
            count = await self._store.prune(kind=kind, older_than=cutoff, limit=limit)
            if count:
                removed[kind.value] = count
        return removed


def build_memory_manager(
    settings: Settings,
    *,
    database: Any | None = None,
    embeddings: EmbeddingProvider | None = None,
) -> MemoryManager:
    """Construct the memory manager for a deployment.

    Args:
        settings: Validated application settings.
        database: A :class:`~app.database.connection.Database`, for durable
            storage. When absent the manager is process-local and volatile.
        embeddings: Embedding provider to use. Built from settings when omitted.

    Returns:
        A configured manager.
    """
    from app.services.embeddings import build_embedding_provider

    store: MemoryStore = (
        PostgresMemoryStore(database) if database is not None else InMemoryMemoryStore()
    )
    provider = embeddings if embeddings is not None else build_embedding_provider(settings)
    return MemoryManager(store, provider, settings)
