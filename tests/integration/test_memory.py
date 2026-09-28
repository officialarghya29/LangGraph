"""Integration tests for durable memory, against real PostgreSQL.

The manager's own decisions are covered by unit tests; what only a real server
can prove is here — that a vector survives a JSONB round trip without losing its
type, that retrieval never crosses an owner boundary at the SQL level, and that
pruning expires the volatile tiers without touching the curated one.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.database.connection import Database
from app.models.memory import MemoryType
from app.services.embeddings import LocalHashingEmbeddingProvider
from app.services.memory import (
    MemoryManager,
    MemoryQuery,
    PostgresMemoryStore,
    build_memory_manager,
)


def build_manager(database: Database, **overrides: object) -> MemoryManager:
    """Return a manager over the real store and deterministic embeddings."""
    settings = Settings(_env_file=None, **overrides)
    embeddings = LocalHashingEmbeddingProvider(dimensions=settings.embedding_dimensions)
    return MemoryManager(PostgresMemoryStore(database), embeddings, settings)


async def store_item(
    database: Database,
    *,
    user: str,
    content: str,
    memory_type: MemoryType = MemoryType.LONG_TERM,
    importance: float = 0.9,
    embedding: list[float] | None = None,
    age: timedelta | None = None,
) -> str:
    """Write one memory row, optionally backdated, and return its id."""
    from app.models.memory import MemoryItem

    store = PostgresMemoryStore(database)
    item = MemoryItem(
        id="pending",
        content=content,
        type=memory_type,
        user_id=user,
        importance=importance,
    )
    stored = await store.add(item, embedding=embedding, embedding_model="test")
    if age is not None:
        async with database.session() as session:
            await session.execute(
                text("UPDATE memory_records SET created_at = :when WHERE id = :id"),
                {"when": datetime.now(UTC) - age, "id": stored.id},
            )
    return stored.id


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #


async def test_a_memory_survives_a_round_trip(database: Database) -> None:
    memory_id = await store_item(
        database, user="alice", content="Alice prefers metric units.", importance=0.8
    )

    candidates = await PostgresMemoryStore(database).candidates(
        user_id="alice", kinds=(MemoryType.LONG_TERM,), limit=10
    )

    assert [item.id for item, _ in candidates] == [memory_id]
    item, _ = candidates[0]
    assert item.content == "Alice prefers metric units."
    assert item.type is MemoryType.LONG_TERM
    assert item.importance == 0.8
    assert item.user_id
    assert item.user_id != "alice"  # stored as the database id, not the principal


async def test_an_embedding_round_trips_as_floats(database: Database) -> None:
    """JSONB stores ``0`` as an integer; a vector must still come back numeric.

    A cosine similarity computed against mixed int and float components is
    correct in Python but the column type is not guaranteed, so the store
    coerces rather than trusting what the dialect hands back.
    """
    await store_item(database, user="alice", content="vector test", embedding=[0, 1, -1, 0.5])

    _, embedding = (
        await PostgresMemoryStore(database).candidates(
            user_id="alice", kinds=(MemoryType.LONG_TERM,), limit=1
        )
    )[0]

    assert embedding == [0.0, 1.0, -1.0, 0.5]
    assert all(isinstance(component, float) for component in embedding)


async def test_a_non_vector_embedding_is_treated_as_absent(
    database: Database, session: AsyncSession
) -> None:
    """A malformed stored value must degrade retrieval, not crash it.

    Written through the ORM rather than the repository-facing API, because the
    point is a value that no correct caller would produce but a hand-edit or an
    older schema version could have left behind.
    """
    from app.database.repositories import MemoryRepository, UserRepository

    owner = await UserRepository(session).get_or_create("alice")
    await MemoryRepository(session).create(
        user_id=owner.id,
        kind=MemoryType.LONG_TERM.value,
        content="malformed vector",
        embedding={"not": "a vector"},  # type: ignore[arg-type]
    )
    await session.commit()

    candidates = await PostgresMemoryStore(database).candidates(
        user_id="alice", kinds=(MemoryType.LONG_TERM,), limit=5
    )

    assert candidates
    assert all(embedding is None for _, embedding in candidates)


# --------------------------------------------------------------------------- #
# Owner scoping
# --------------------------------------------------------------------------- #


async def test_retrieval_never_crosses_an_owner_boundary(database: Database) -> None:
    await store_item(database, user="alice", content="alice's secret")
    await store_item(database, user="bob", content="bob's secret")

    store = PostgresMemoryStore(database)
    alice = [
        item.content
        for item, _ in await store.candidates(
            user_id="alice", kinds=(MemoryType.LONG_TERM,), limit=10
        )
    ]

    assert alice == ["alice's secret"]


async def test_an_unknown_principal_has_no_memories(database: Database) -> None:
    await store_item(database, user="alice", content="something")

    assert (
        await PostgresMemoryStore(database).candidates(
            user_id="nobody", kinds=(MemoryType.LONG_TERM,), limit=10
        )
        == []
    )


async def test_counts_are_scoped_to_one_owner(database: Database) -> None:
    await store_item(database, user="alice", content="one")
    await store_item(database, user="bob", content="two")

    store = PostgresMemoryStore(database)

    assert await store.counts("alice") == {"long_term": 1}
    assert await store.counts("nobody") == {}


async def test_deleting_another_users_memory_is_refused(database: Database) -> None:
    memory_id = await store_item(database, user="alice", content="alice's note")

    store = PostgresMemoryStore(database)

    assert await store.delete(memory_id, user_id="mallory") is False
    assert await store.counts("alice") == {"long_term": 1}
    assert await store.delete(memory_id, user_id="alice") is True
    assert await store.counts("alice") == {}


async def test_deleting_a_malformed_id_returns_false(database: Database) -> None:
    assert await PostgresMemoryStore(database).delete("not-a-uuid", user_id="alice") is False


# --------------------------------------------------------------------------- #
# Kinds, ordering, and access tracking
# --------------------------------------------------------------------------- #


async def test_retrieval_filters_by_tier(database: Database) -> None:
    await store_item(database, user="alice", content="durable", memory_type=MemoryType.LONG_TERM)
    await store_item(database, user="alice", content="volatile", memory_type=MemoryType.WORKING)

    store = PostgresMemoryStore(database)

    durable = await store.candidates(user_id="alice", kinds=(MemoryType.LONG_TERM,), limit=10)
    assert [item.content for item, _ in durable] == ["durable"]

    both = await store.candidates(
        user_id="alice", kinds=(MemoryType.LONG_TERM, MemoryType.WORKING), limit=10
    )
    assert {item.content for item, _ in both} == {"durable", "volatile"}


async def test_retrieval_is_newest_first(database: Database) -> None:
    await store_item(database, user="alice", content="older", age=timedelta(days=2))
    await store_item(database, user="alice", content="newer")

    candidates = await PostgresMemoryStore(database).candidates(
        user_id="alice", kinds=(MemoryType.LONG_TERM,), limit=10
    )

    assert [item.content for item, _ in candidates] == ["newer", "older"]


async def test_marking_accessed_records_the_retrieval(database: Database) -> None:
    memory_id = await store_item(database, user="alice", content="retrieved")

    store = PostgresMemoryStore(database)
    await store.mark_accessed([memory_id])
    await store.mark_accessed([memory_id])

    async with database.session() as session:
        result = await session.execute(
            text("SELECT access_count, accessed_at FROM memory_records WHERE id = :id"),
            {"id": memory_id},
        )
        count, accessed_at = result.one()

    assert count == 2
    assert accessed_at is not None


async def test_marking_nothing_accessed_is_harmless(database: Database) -> None:
    assert await PostgresMemoryStore(database).mark_accessed([]) is None


# --------------------------------------------------------------------------- #
# Pruning
# --------------------------------------------------------------------------- #


async def test_pruning_removes_only_old_volatile_memories(database: Database) -> None:
    await store_item(
        database,
        user="alice",
        content="stale thought",
        memory_type=MemoryType.SHORT_TERM,
        age=timedelta(days=2),
    )
    await store_item(
        database,
        user="alice",
        content="fresh thought",
        memory_type=MemoryType.SHORT_TERM,
        age=timedelta(minutes=5),
    )
    await store_item(
        database,
        user="alice",
        content="ancient but curated",
        memory_type=MemoryType.LONG_TERM,
        age=timedelta(days=365),
    )

    store = PostgresMemoryStore(database)
    removed = await store.prune(
        kind=MemoryType.SHORT_TERM,
        older_than=datetime.now(UTC) - timedelta(hours=1),
        limit=100,
    )

    assert removed == 1
    assert await store.counts("alice") == {"short_term": 1, "long_term": 1}


# --------------------------------------------------------------------------- #
# The manager over the real store
# --------------------------------------------------------------------------- #


async def test_the_manager_recalls_across_a_real_write_and_read(database: Database) -> None:
    manager = build_manager(database)
    await manager.remember(
        "The nightly ETL job writes to the warehouse schema at 02:00 UTC.",
        user_id="alice",
        importance=0.9,
    )
    await manager.remember(
        "Spare keyboards are kept in the supply cupboard.",
        user_id="alice",
        importance=0.9,
    )

    recalled = await manager.recall(
        MemoryQuery(text="when does the nightly ETL job run", user_id="alice")
    )

    assert recalled
    assert "ETL" in recalled[0].content


async def test_the_manager_is_owner_scoped_through_the_real_store(database: Database) -> None:
    manager = build_manager(database)
    await manager.remember("alice's private note about the vault", user_id="alice", importance=0.9)
    await manager.remember("bob's private note about the vault", user_id="bob", importance=0.9)

    recalled = await manager.recall(
        MemoryQuery(text="private note about the vault", user_id="alice")
    )

    assert recalled
    assert all(item.user_id != "bob" for item in recalled)


async def test_consolidation_against_real_rows(database: Database) -> None:
    manager = build_manager(database, memory_volatile_ttl_seconds=60)
    await store_item(
        database,
        user="alice",
        content="old working note",
        memory_type=MemoryType.WORKING,
        age=timedelta(hours=1),
    )
    await store_item(
        database,
        user="alice",
        content="old but durable note",
        memory_type=MemoryType.LONG_TERM,
        age=timedelta(hours=1),
    )

    removed = await manager.consolidate()

    assert removed == {"working": 1}
    assert await manager.counts("alice") == {"long_term": 1}


async def test_the_factory_selects_the_durable_store(database: Database) -> None:
    manager = build_memory_manager(Settings(_env_file=None), database=database)

    assert manager.store_name == "PostgresMemoryStore"
    assert await manager.counts("alice") == {}


async def test_the_factory_falls_back_to_a_volatile_store() -> None:
    manager = build_memory_manager(Settings(_env_file=None))

    assert manager.store_name == "InMemoryMemoryStore"
