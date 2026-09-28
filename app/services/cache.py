"""Cache and coordination, backed by Redis.

Redis is used for four things, and for nothing else:

- **Caching** read-heavy results that are safe to recompute.
- **Rate limiting**, where the counter must be shared across processes.
- **Short-lived coordination**: the execution facts a request needs while it is
  running, which are gone once it finishes.
- **Locks**, to stop two workers acting on the same task.

It is deliberately *not* the only place a durable fact lives. A cache eviction,
a restart, or a flush must never lose a task, an approval, or a memory. Every
method here is therefore written so that a miss is a normal, recoverable
outcome and an outage degrades the application rather than breaking it.

Keys are namespaced. One Redis instance can host several environments at once
without one run reading another's keys.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.exceptions import CacheError

__all__ = ["Cache", "build_cache"]

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Cache:
    """A namespaced, TTL-aware view of a Redis database."""

    def __init__(
        self,
        url: str | None = None,
        *,
        prefix: str = "langgraph",
        default_ttl: int | None = 300,
        client: Redis | None = None,
    ) -> None:
        """Create a cache.

        Args:
            url: Redis URL. Required unless ``client`` is supplied.
            prefix: Namespace applied to every key.
            default_ttl: Seconds a key survives unless a call overrides it. A
                default is set deliberately: an unbounded cache is a slow memory
                leak, and the usual cause is somebody forgetting an argument.
            client: Optional pre-built client, so a caller can point the cache at
                a specific database index or supply a stub.

        Raises:
            ValueError: If neither ``url`` nor ``client`` is given. Silently
                building a default client would quietly connect somewhere the
                caller did not ask for.
        """
        if client is None and not url:
            raise ValueError("a cache needs either a url or a client")

        self.prefix = prefix
        self.default_ttl = default_ttl
        self.client = client or Redis.from_url(url or "", decode_responses=True)
        self._client = self.client

    # ------------------------------------------------------------------ #
    # Keys
    # ------------------------------------------------------------------ #

    def key(self, *parts: object) -> str:
        """Build a logical key from ``parts``.

        The result is *not* prefixed. Every public method applies the namespace
        itself, so a caller cannot accidentally write a key outside this cache's
        namespace by forgetting to build it here — which would silently share
        state between two environments on one Redis instance.
        """
        return ":".join(str(part) for part in parts if part is not None)

    def _physical(self, key: str) -> str:
        """Return the namespaced key actually stored in Redis."""
        return f"{self.prefix}:{key}"

    @staticmethod
    def _serialise(value: Any) -> str:
        """Render a value as text, JSON-encoding anything that is not already."""
        if isinstance(value, str):
            return value
        return json.dumps(value, default=str, separators=(",", ":"))

    @staticmethod
    def _deserialise(raw: str | bytes | None) -> Any:
        """Parse a stored value, returning it untouched when it is not JSON.

        Accepts bytes as well as str: the driver returns bytes unless the client
        was built with ``decode_responses=True``, and a helper that only handled
        one of them would fail depending on how the client was constructed.
        """
        if raw is None:
            return None
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            return text

    # ------------------------------------------------------------------ #
    # Basic operations
    # ------------------------------------------------------------------ #

    async def get(self, key: str, *, default: T | None = None) -> Any | T | None:
        """Return a value, or ``default`` when it is absent.

        Raises:
            CacheError: Only on an actual outage. A missing key returns
                ``default``, because a miss is not an error.
        """
        try:
            raw = await self._client.get(self._physical(key))
        except RedisError as exc:
            raise CacheError("cache read failed", detail=type(exc).__name__) from exc
        if raw is None:
            return default
        return self._deserialise(raw)

    async def set(
        self,
        key: str,
        value: Any,
        *,
        ttl: int | None = None,
        persist: bool = False,
        only_if_absent: bool = False,
    ) -> bool:
        """Store a value.

        Args:
            key: Logical key, relative to this cache's namespace.
            value: Value to store; anything non-str is JSON-encoded.
            ttl: Seconds to live. When omitted, the cache's default applies.
            persist: Store without an expiry. Separate from ``ttl`` because
                ``None`` already means "use the default", and overloading it to
                also mean "never expire" would make forgetting an argument
                indistinguishable from asking for an immortal key.
            only_if_absent: Use ``SET NX`` semantics, which is what makes this
                usable as a lock.

        Returns:
            Whether the value was written. ``False`` with ``only_if_absent``
            means somebody else holds the key.

        Raises:
            CacheError: If Redis is unreachable.
        """
        effective_ttl = None if persist else (self.default_ttl if ttl is None else ttl)
        try:
            result = await self._client.set(
                self._physical(key),
                self._serialise(value),
                ex=effective_ttl,
                nx=only_if_absent,
            )
        except RedisError as exc:
            raise CacheError("cache write failed", detail=type(exc).__name__) from exc
        return bool(result)

    async def delete(self, *keys: str) -> int:
        """Delete keys, returning how many existed."""
        if not keys:
            return 0
        try:
            return int(await self._client.delete(*[self._physical(key) for key in keys]))
        except RedisError as exc:
            raise CacheError("cache delete failed", detail=type(exc).__name__) from exc

    async def exists(self, key: str) -> bool:
        """Return whether a key is present."""
        try:
            return bool(await self._client.exists(self._physical(key)))
        except RedisError as exc:
            raise CacheError("cache lookup failed", detail=type(exc).__name__) from exc

    async def ttl(self, key: str) -> int:
        """Return a key's remaining lifetime in seconds.

        Redis reports ``-1`` for a key with no expiry and ``-2`` for a key that
        does not exist; both are passed through unchanged rather than being
        collapsed, because the two mean different things to a caller.
        """
        try:
            return int(await self._client.ttl(self._physical(key)))
        except RedisError as exc:
            raise CacheError("cache ttl lookup failed", detail=type(exc).__name__) from exc

    async def expire(self, key: str, ttl: int) -> bool:
        """Set or replace a key's expiry."""
        try:
            return bool(await self._client.expire(self._physical(key), ttl))
        except RedisError as exc:
            raise CacheError("cache expiry failed", detail=type(exc).__name__) from exc

    # ------------------------------------------------------------------ #
    # Cache-aside
    # ------------------------------------------------------------------ #

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Awaitable[T]],
        *,
        ttl: int | None = None,
    ) -> T:
        """Return a cached value, computing and storing it on a miss.

        An outage is not fatal here: if Redis cannot be read, the value is
        computed and returned without being cached. Serving a correct answer
        slowly is better than failing because a cache was down.
        """
        try:
            cached = await self.get(key)
        except CacheError:
            logger.warning("cache.unavailable_on_read", extra={"key": key})
            return await factory()

        if cached is not None:
            return cached  # type: ignore[no-any-return]

        value = await factory()
        try:
            await self.set(key, value, ttl=ttl)
        except CacheError:
            logger.warning("cache.unavailable_on_write", extra={"key": key})
        return value

    # ------------------------------------------------------------------ #
    # Counters and rate limiting
    # ------------------------------------------------------------------ #

    async def increment(self, key: str, *, ttl: int, amount: int = 1) -> int:
        """Atomically increment a counter and return its new value.

        The expiry is applied only when the counter is created, so a fixed
        window does not slide forward on every request. That is what makes the
        window a window rather than a rolling one.
        """
        physical = self._physical(key)
        try:
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.incrby(physical, amount)
                pipe.ttl(physical)
                value, remaining = await pipe.execute()
            if int(remaining) < 0:
                await self._client.expire(physical, ttl)
            return int(value)
        except RedisError as exc:
            raise CacheError("counter increment failed", detail=type(exc).__name__) from exc

    async def check_rate_limit(
        self, identity: str, *, limit: int, window_seconds: int
    ) -> tuple[bool, int]:
        """Decide whether ``identity`` may proceed.

        Args:
            identity: Who is being limited, already namespaced by the caller —
                a user id, an IP, or a route plus one of those.
            limit: Requests permitted per window.
            window_seconds: Length of the fixed window.

        Returns:
            ``(allowed, remaining)``.

        Raises:
            CacheError: If Redis is unreachable. The caller decides what to do;
                the default in :class:`~app.api.middleware.RateLimitMiddleware`
                is to allow the request, because failing closed on a cache
                outage would turn a degraded cache into a total outage.
        """
        used = await self.increment(self.key("ratelimit", identity), ttl=window_seconds)
        return used <= limit, max(0, limit - used)

    # ------------------------------------------------------------------ #
    # Short-lived execution state
    # ------------------------------------------------------------------ #

    async def set_task_state(self, task_id: str, state: dict[str, Any], *, ttl: int = 900) -> None:
        """Record a task's coarse progress for the duration of a run.

        Not the task's record of truth — that is in PostgreSQL. This is what a
        streaming client polls without hitting the database on every tick.
        """
        await self.set(self.key("task", task_id), state, ttl=ttl)

    async def get_task_state(self, task_id: str) -> dict[str, Any] | None:
        """Return a task's coarse progress, if it is still live."""
        value = await self.get(self.key("task", task_id))
        return value if isinstance(value, dict) else None

    async def append_event(self, task_id: str, event: dict[str, Any], *, ttl: int = 3600) -> None:
        """Append an event to a short-lived stream buffer.

        Bounded by the TTL and by :data:`MAX_BUFFERED_EVENTS`: the durable copy
        is written to PostgreSQL, so this buffer exists only to make a live
        stream cheap, and must not grow without limit.
        """
        physical = self._physical(self.key("events", task_id))
        try:
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.rpush(physical, self._serialise(event))
                pipe.ltrim(physical, -MAX_BUFFERED_EVENTS, -1)
                pipe.expire(physical, ttl)
                await pipe.execute()
        except RedisError as exc:
            raise CacheError("event buffering failed", detail=type(exc).__name__) from exc

    async def read_events(self, task_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        """Return buffered events for a task, oldest first."""
        try:
            raw = await self._client.lrange(self._physical(self.key("events", task_id)), -limit, -1)
        except RedisError as exc:
            raise CacheError("event read failed", detail=type(exc).__name__) from exc
        return [self._deserialise(item) for item in raw]

    # ------------------------------------------------------------------ #
    # Coordination
    # ------------------------------------------------------------------ #

    async def acquire_lock(self, name: str, *, ttl: int = 30, owner: str = "") -> bool:
        """Acquire a short-lived lock.

        Returns:
            Whether the lock was acquired. The TTL is the safety net: a worker
            that dies holding a lock does not block the work forever.
        """
        return await self.set(self.key("lock", name), owner or "1", ttl=ttl, only_if_absent=True)

    async def release_lock(self, name: str, *, owner: str = "") -> bool:
        """Release a lock, but only if this caller still holds it.

        The owner check matters: without it, a worker whose lock had already
        expired could release a lock a different worker had since acquired.

        The compare-and-delete is done as a Lua script so it is atomic. Reading
        the holder and then deleting is two round trips, and a lock can expire
        between them.
        """
        physical = self._physical(self.key("lock", name))
        script = (
            "if redis.call('get', KEYS[1]) == ARGV[1] "
            "then return redis.call('del', KEYS[1]) else return 0 end"
        )
        try:
            deleted = await self._client.eval(script, 1, physical, owner or "1")
        except RedisError as exc:
            raise CacheError("lock release failed", detail=type(exc).__name__) from exc
        return bool(deleted)

    # ------------------------------------------------------------------ #
    # Maintenance and health
    # ------------------------------------------------------------------ #

    async def clear_namespace(self) -> int:
        """Delete every key in this cache's namespace.

        Uses ``SCAN`` rather than ``KEYS``: ``KEYS`` blocks the server for the
        whole scan, which on a shared instance stalls every other client.
        """
        removed = 0
        try:
            async for key in self._client.scan_iter(match=f"{self.prefix}:*", count=500):
                removed += int(await self._client.delete(key))
        except RedisError as exc:
            raise CacheError("namespace clear failed", detail=type(exc).__name__) from exc
        return removed

    async def ping(self) -> tuple[bool, str]:
        """Check that Redis answers.

        Returns:
            ``(True, "ok")`` or ``(False, reason)``. The reason is a
            classification, never the URL, which may carry a password.
        """
        try:
            await self._client.ping()
        except RedisError as exc:
            return False, f"unreachable ({type(exc).__name__})"
        except Exception as exc:
            return False, f"unreachable ({type(exc).__name__})"
        return True, "ok"

    async def info(self) -> dict[str, Any]:
        """Return a small, safe subset of the server's statistics."""
        try:
            raw = await self._client.info("memory")
        except RedisError as exc:
            raise CacheError("cache info failed", detail=type(exc).__name__) from exc
        return {
            "used_memory_human": raw.get("used_memory_human"),
            "maxmemory_human": raw.get("maxmemory_human"),
            "maxmemory_policy": raw.get("maxmemory_policy"),
        }

    async def close(self) -> None:
        """Close the client and its connection pool."""
        await self._client.aclose()


#: How many events are retained per task in the short-lived buffer.
MAX_BUFFERED_EVENTS = 500


def build_cache(
    url: str | None = None, *, prefix: str = "langgraph", default_ttl: int | None = 300
) -> Cache:
    """Create a cache for the given URL.

    Args:
        url: Redis URL.
        prefix: Namespace for every key.
        default_ttl: Default key lifetime in seconds.

    Returns:
        A configured :class:`Cache`.
    """
    return Cache(url, prefix=prefix, default_ttl=default_ttl)
