"""Integration tests for the Redis cache, against a real server.

The behaviour worth proving here is not "get returns what set stored" — that a
mock could show. It is the parts that depend on Redis semantics: expiry,
atomicity of the counter, and what happens when the server is not there.
"""

from __future__ import annotations

import asyncio

import pytest
from redis.asyncio import Redis

from app.core.exceptions import CacheError
from app.services.cache import MAX_BUFFERED_EVENTS, Cache

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def cache(redis_url: str) -> Cache:
    """Return a cache in an isolated namespace, emptied before and after."""
    # A dedicated database index keeps these keys away from application data.
    client = Redis.from_url(f"{redis_url.rsplit('/', 1)[0]}/15", decode_responses=True)
    instance = Cache(prefix="testcache", default_ttl=60, client=client)
    await instance.clear_namespace()
    try:
        yield instance
    finally:
        await instance.clear_namespace()
        await instance.close()


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #


async def test_the_cache_answers_a_ping(cache: Cache) -> None:
    ok, detail = await cache.ping()

    assert ok is True
    assert detail == "ok"


async def test_an_unreachable_cache_reports_rather_than_raising() -> None:
    """A health probe must classify, not propagate."""
    dead = Cache("redis://127.0.0.1:1/0", prefix="dead")

    ok, detail = await dead.ping()

    assert ok is False
    assert detail.startswith("unreachable")
    await dead.close()


async def test_cache_info_reports_memory(cache: Cache) -> None:
    info = await cache.info()

    assert "used_memory_human" in info
    assert "maxmemory_policy" in info


# --------------------------------------------------------------------------- #
# Basic operations
# --------------------------------------------------------------------------- #


async def test_a_value_round_trips(cache: Cache) -> None:
    await cache.set(cache.key("greeting"), "hello")

    assert await cache.get(cache.key("greeting")) == "hello"


async def test_a_missing_key_returns_the_default(cache: Cache) -> None:
    assert await cache.get(cache.key("absent")) is None
    assert await cache.get(cache.key("absent"), default="fallback") == "fallback"


async def test_structured_values_round_trip(cache: Cache) -> None:
    payload = {"agent": "researcher", "sources": ["https://example.test"], "score": 0.9}

    await cache.set(cache.key("payload"), payload)

    assert await cache.get(cache.key("payload")) == payload


async def test_exists_reflects_reality(cache: Cache) -> None:
    key = cache.key("thing")

    assert await cache.exists(key) is False
    await cache.set(key, "value")
    assert await cache.exists(key) is True


async def test_delete_reports_how_many_it_removed(cache: Cache) -> None:
    first, second = cache.key("a"), cache.key("b")
    await cache.set(first, 1)
    await cache.set(second, 2)

    assert await cache.delete(first, second, cache.key("missing")) == 2


async def test_deleting_nothing_does_not_touch_redis(cache: Cache) -> None:
    assert await cache.delete() == 0


async def test_keys_are_namespaced(cache: Cache) -> None:
    """Two environments on one server must not read each other's keys."""
    other = Cache(prefix="othernamespace", client=cache.client)

    await cache.set("shared-name", "first")
    await other.set("shared-name", "second")

    assert await cache.get("shared-name") == "first"
    assert await other.get("shared-name") == "second"


async def test_a_caller_cannot_escape_the_namespace(cache: Cache) -> None:
    """A raw logical key must still land inside this cache's namespace.

    A caller that forgets to build the key through ``key()`` would otherwise
    write into another environment's keyspace.
    """
    await cache.set("plain", "value")

    physical = await cache.client.keys(f"{cache.prefix}:plain")
    assert physical == ["testcache:plain"]


# --------------------------------------------------------------------------- #
# Expiry
# --------------------------------------------------------------------------- #


async def test_a_value_expires(cache: Cache) -> None:
    key = cache.key("brief")
    await cache.set(key, "value", ttl=1)

    assert 0 < await cache.ttl(key) <= 1
    await asyncio.sleep(1.2)
    assert await cache.get(key) is None


async def test_the_default_ttl_applies_when_none_is_given(cache: Cache) -> None:
    key = cache.key("defaulted")
    await cache.set(key, "value")

    assert 0 < await cache.ttl(key) <= 60


async def test_ttl_distinguishes_a_missing_key_from_an_immortal_one(cache: Cache) -> None:
    """``-2`` and ``-1`` mean different things and must not be collapsed."""
    immortal = cache.key("immortal")
    await cache.set(immortal, "value", persist=True)

    assert await cache.ttl(immortal) == -1
    assert await cache.ttl(cache.key("absent")) == -2


async def test_persist_overrides_the_default_ttl(cache: Cache) -> None:
    """Asking for no expiry must be distinguishable from passing no ttl."""
    key = cache.key("deliberate")
    assert cache.default_ttl is not None

    await cache.set(key, "value", persist=True)

    assert await cache.ttl(key) == -1


async def test_expiry_can_be_extended(cache: Cache) -> None:
    key = cache.key("extendable")
    await cache.set(key, "value", ttl=5)

    assert await cache.expire(key, 120) is True
    assert await cache.ttl(key) == 120


# --------------------------------------------------------------------------- #
# Cache-aside
# --------------------------------------------------------------------------- #


async def test_get_or_set_computes_once_then_serves_from_cache(cache: Cache) -> None:
    calls = 0

    async def factory() -> str:
        nonlocal calls
        calls += 1
        return "computed"

    key = cache.key("expensive")
    assert await cache.get_or_set(key, factory) == "computed"
    assert await cache.get_or_set(key, factory) == "computed"

    assert calls == 1


async def test_get_or_set_survives_a_cache_outage() -> None:
    """A broken cache must not break the request; the value is just not cached."""
    dead = Cache("redis://127.0.0.1:1/0", prefix="dead")

    async def factory() -> str:
        return "computed anyway"

    assert await dead.get_or_set("k", factory) == "computed anyway"
    await dead.close()


# --------------------------------------------------------------------------- #
# Counters and rate limiting
# --------------------------------------------------------------------------- #


async def test_a_counter_increments(cache: Cache) -> None:
    key = cache.key("counter")

    assert await cache.increment(key, ttl=60) == 1
    assert await cache.increment(key, ttl=60) == 2
    assert await cache.increment(key, ttl=60, amount=5) == 7


async def test_a_counter_window_does_not_slide(cache: Cache) -> None:
    """The expiry is set once, so the window is fixed rather than rolling.

    If the TTL were refreshed on every increment, a client sending steadily
    would never be reset and the limit would never free up.
    """
    key = cache.key("window")
    await cache.increment(key, ttl=30)
    first_ttl = await cache.ttl(key)

    for _ in range(5):
        await cache.increment(key, ttl=30)

    assert await cache.ttl(key) == first_ttl


async def test_incrementing_is_atomic_under_concurrency(cache: Cache) -> None:
    """A non-atomic read-modify-write would lose increments under contention."""
    key = cache.key("concurrent")

    await asyncio.gather(*(cache.increment(key, ttl=60) for _ in range(50)))

    assert await cache.get(key) == 50


async def test_a_rate_limit_allows_up_to_its_limit(cache: Cache) -> None:
    for expected in range(3):
        allowed, remaining = await cache.check_rate_limit("user-a", limit=3, window_seconds=60)
        assert allowed is True
        assert remaining == 2 - expected

    allowed, remaining = await cache.check_rate_limit("user-a", limit=3, window_seconds=60)
    assert allowed is False
    assert remaining == 0


async def test_rate_limits_are_per_identity(cache: Cache) -> None:
    await cache.check_rate_limit("user-a", limit=1, window_seconds=60)
    blocked, _ = await cache.check_rate_limit("user-a", limit=1, window_seconds=60)
    allowed, _ = await cache.check_rate_limit("user-b", limit=1, window_seconds=60)

    assert blocked is False
    assert allowed is True


async def test_a_rate_limit_window_resets(cache: Cache) -> None:
    await cache.check_rate_limit("user-c", limit=1, window_seconds=1)
    blocked, _ = await cache.check_rate_limit("user-c", limit=1, window_seconds=1)
    assert blocked is False

    await asyncio.sleep(1.2)

    allowed, _ = await cache.check_rate_limit("user-c", limit=1, window_seconds=1)
    assert allowed is True


async def test_an_unreachable_cache_raises_on_rate_limit() -> None:
    """The middleware decides the policy; the cache reports the failure."""
    dead = Cache("redis://127.0.0.1:1/0", prefix="dead")

    with pytest.raises(CacheError):
        await dead.check_rate_limit("user", limit=1, window_seconds=60)
    await dead.close()


# --------------------------------------------------------------------------- #
# Execution state and event buffering
# --------------------------------------------------------------------------- #


async def test_task_state_round_trips(cache: Cache) -> None:
    await cache.set_task_state("task-1", {"status": "running", "events": 3})

    state = await cache.get_task_state("task-1")

    assert state == {"status": "running", "events": 3}


async def test_task_state_that_was_never_written_is_none(cache: Cache) -> None:
    assert await cache.get_task_state("unknown-task") is None


async def test_events_are_buffered_in_order(cache: Cache) -> None:
    for index in range(4):
        await cache.append_event("task-2", {"seq": index, "type": f"e{index}"})

    events = await cache.read_events("task-2")

    assert [event["seq"] for event in events] == [0, 1, 2, 3]


async def test_event_reads_can_be_limited_to_the_tail(cache: Cache) -> None:
    for index in range(10):
        await cache.append_event("task-3", {"seq": index})

    recent = await cache.read_events("task-3", limit=3)

    assert [event["seq"] for event in recent] == [7, 8, 9]


async def test_the_event_buffer_is_bounded(cache: Cache) -> None:
    """A live stream must not grow the cache without limit."""
    for index in range(MAX_BUFFERED_EVENTS + 20):
        await cache.append_event("task-4", {"seq": index})

    events = await cache.read_events("task-4", limit=MAX_BUFFERED_EVENTS + 50)

    assert len(events) == MAX_BUFFERED_EVENTS
    assert events[-1]["seq"] == MAX_BUFFERED_EVENTS + 19


async def test_events_are_scoped_to_their_task(cache: Cache) -> None:
    await cache.append_event("task-5", {"seq": 0})
    await cache.append_event("task-6", {"seq": 99})

    assert [event["seq"] for event in await cache.read_events("task-5")] == [0]


# --------------------------------------------------------------------------- #
# Locks
# --------------------------------------------------------------------------- #


async def test_a_lock_admits_only_one_holder(cache: Cache) -> None:
    assert await cache.acquire_lock("task-7", owner="worker-a") is True
    assert await cache.acquire_lock("task-7", owner="worker-b") is False


async def test_a_lock_can_be_released_and_reacquired(cache: Cache) -> None:
    await cache.acquire_lock("task-8", owner="worker-a")
    assert await cache.release_lock("task-8", owner="worker-a") is True
    assert await cache.acquire_lock("task-8", owner="worker-b") is True


async def test_a_lock_cannot_be_released_by_a_different_owner(cache: Cache) -> None:
    """Otherwise a worker whose lock expired could free somebody else's."""
    await cache.acquire_lock("task-9", owner="worker-a")

    assert await cache.release_lock("task-9", owner="worker-b") is False
    assert await cache.acquire_lock("task-9", owner="worker-c") is False


async def test_releasing_a_lock_that_does_not_exist_is_not_an_error(cache: Cache) -> None:
    assert await cache.release_lock("never-locked") is False


async def test_a_lock_expires_so_a_dead_worker_cannot_block_forever(cache: Cache) -> None:
    await cache.acquire_lock("task-10", ttl=1, owner="dying-worker")

    await asyncio.sleep(1.2)

    assert await cache.acquire_lock("task-10", ttl=30, owner="worker-b") is True


# --------------------------------------------------------------------------- #
# Maintenance
# --------------------------------------------------------------------------- #


async def test_clearing_the_namespace_leaves_other_namespaces_alone(cache: Cache) -> None:
    neighbour = Cache(prefix="neighbour", client=cache.client)
    await cache.set("mine", 1)
    await cache.set("also-mine", 2)
    await neighbour.set("theirs", 3)

    removed = await cache.clear_namespace()

    assert removed == 2
    assert await neighbour.get("theirs") == 3


async def test_an_outage_is_reported_as_a_cache_error() -> None:
    """Every operation classifies an outage rather than leaking a driver error."""
    dead = Cache("redis://127.0.0.1:1/0", prefix="dead")

    operations: dict[str, object] = {
        "get": dead.get("k"),
        "set": dead.set("k", "v"),
        "delete": dead.delete("k"),
        "exists": dead.exists("k"),
        "ttl": dead.ttl("k"),
        "expire": dead.expire("k", 10),
        "increment": dead.increment("k", ttl=10),
        "append_event": dead.append_event("t", {}),
        "read_events": dead.read_events("t"),
        "acquire_lock": dead.acquire_lock("l"),
        "release_lock": dead.release_lock("l"),
        "clear_namespace": dead.clear_namespace(),
        "info": dead.info(),
    }

    failures: list[str] = []
    for name, coroutine in operations.items():
        try:
            await coroutine
        except CacheError:
            continue
        except Exception as exc:
            failures.append(f"{name} raised {type(exc).__name__}")
        else:
            failures.append(f"{name} did not raise")

    assert failures == []
    await dead.close()
