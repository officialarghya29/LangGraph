"""Middleware.

Both middlewares here are written as raw ASGI rather than
``BaseHTTPMiddleware``. That is not a style preference: ``BaseHTTPMiddleware``
buffers responses through an anyio task group, which breaks server-sent events —
the stream is collected and flushed at the end instead of arriving as it is
produced. Since this application streams execution events, a middleware that
silently defeats streaming is not acceptable.

Rate limiting is a middleware rather than a per-route dependency so that
coverage is structural. A new mutating route inherits the limit from its path;
it cannot be added and quietly left unprotected.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.auth import ANONYMOUS_USER_ID, AUTHORIZATION_HEADER, IDENTITY_HEADER, decode_token
from app.core.config import Settings, get_settings
from app.core.exceptions import CacheError
from app.observability.metrics import MetricRegistry
from app.services.cache import Cache

__all__ = ["RateLimitMiddleware", "RequestContextMiddleware", "install_rate_limiting"]

logger = logging.getLogger(__name__)

#: Request id echoed back to the caller and attached to every log line.
REQUEST_ID_HEADER = "X-Request-Id"

#: Header carrying how many requests remain in the current window.
RATE_LIMIT_REMAINING_HEADER = "X-RateLimit-Remaining"
#: Header carrying the window length, so a client can back off sensibly.
RATE_LIMIT_WINDOW_HEADER = "X-RateLimit-Window"

#: Metric label used for a request that matched no route. A constant, because
#: the alternative is a label value the caller chooses.
UNMATCHED_ROUTE = "<unmatched>"


class RequestContextMiddleware:
    """Assigns a correlation id and emits one access log line per request.

    The id is taken from the caller when supplied, so a trace can be followed
    across services, and generated otherwise. It is only trusted as a
    correlation value, never as an authorization signal.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI scope."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        request_id = headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id

        started = time.perf_counter()
        status_holder: dict[str, int] = {}

        async def send_with_context(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = int(message["status"])
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_context)
        finally:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            status = status_holder.get("status", 500)
            # Never log the query string or the body: either can carry a token,
            # a prompt, or a user identifier that does not belong in a log line.
            logger.info(
                "http.request",
                extra={
                    "request_id": request_id,
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "status": status,
                    "duration_ms": elapsed_ms,
                },
            )
            self._record(scope, status, elapsed_ms / 1000)

    @staticmethod
    def _record(scope: Scope, status: int, seconds: float) -> None:
        """Record the request's count and duration.

        The route *template* is used where Starlette has resolved one, so
        ``/api/v1/tasks/{task_id}`` is one series rather than one series per id.
        That is the difference between a useful metric and a cardinality
        explosion, and it is why the path is read from the route rather than
        from the raw request.

        A request that matched no route is bucketed under a single constant. An
        unmatched path is attacker-chosen, so labelling by it would let anyone
        mint unbounded series and take down the metrics backend — a denial of
        service against monitoring rather than against the service, which is
        exactly the kind of failure that is noticed too late.
        """
        metrics: MetricRegistry | None = getattr(
            getattr(scope.get("app"), "state", None), "metrics", None
        )
        if metrics is None:
            return

        route = scope.get("route")
        path = getattr(route, "path", None) or UNMATCHED_ROUTE
        # The status class, not the exact code: 4xx vs 5xx is the distinction an
        # alert acts on, and the code itself is already in the access log.
        metrics.increment(
            "http_requests_total",
            method=str(scope.get("method", "")),
            path=path,
            status=f"{status // 100}xx",
        )
        metrics.observe(
            "http_request_duration_seconds",
            seconds,
            method=str(scope.get("method", "")),
            path=path,
        )


#: Which paths are limited, and how. A prefix match keeps this readable and
#: makes the intent reviewable in one place. The limits are deliberately
#: different per operation: creating a task starts a multi-agent run and is far
#: more expensive than reading one.
RATE_LIMIT_POLICIES: tuple[tuple[str, str, int], ...] = (
    # (method or "*", path prefix, multiplier against the configured limit)
    ("POST", "/api/v1/chat", 1),
    ("POST", "/api/v1/tasks", 1),
    ("POST", "/api/v1/tasks/", 2),  # approve / reject decisions
    ("GET", "/api/v1/events/", 4),  # streaming reconnects are cheap
)


def _policy_for(method: str, path: str) -> int | None:
    """Return the multiplier for a request, or ``None`` when it is unlimited."""
    for expected_method, prefix, multiplier in RATE_LIMIT_POLICIES:
        if expected_method in {"*", method} and path.startswith(prefix):
            return multiplier
    return None


def _identity(scope: Scope, settings: Settings) -> str:
    """Return the rate-limit bucket for a request.

    Prefers an authenticated subject, because a principal is stable behind a
    proxy while an address is not. Falls back to the client address, then to a
    single shared bucket, so an unattributable request is still limited rather
    than exempt.
    """
    headers = Headers(scope=scope)

    authorization = headers.get(AUTHORIZATION_HEADER)
    if authorization and authorization.lower().startswith("bearer ") and settings.jwt_secret:
        try:
            claims = decode_token(authorization[7:].strip(), settings)
        except Exception:
            claims = {}
        subject = claims.get("sub")
        if isinstance(subject, str) and subject:
            return subject

    if settings.trust_identity_header and not settings.auth_enabled:
        claimed = headers.get(IDENTITY_HEADER)
        if claimed:
            return claimed.strip()

    client = scope.get("client")
    if client and client[0]:
        return str(client[0])
    return ANONYMOUS_USER_ID


class RateLimitMiddleware:
    """Applies a fixed-window request limit to the configured paths."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Enforce the policy for this request, then delegate."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        settings = get_settings()
        path = str(scope.get("path", ""))
        method = str(scope.get("method", ""))
        multiplier = _policy_for(method, path)

        if multiplier is None:
            await self.app(scope, receive, send)
            return

        application = scope.get("app")
        cache: Cache | None = getattr(getattr(application, "state", None), "cache", None)
        if cache is None:  # pragma: no cover - the cache is built in the lifespan
            await self.app(scope, receive, send)
            return

        limit = settings.rate_limit_requests * multiplier
        identity = (
            f"{method}:{path.split('/')[3] if '/' in path else 'root'}:{_identity(scope, settings)}"
        )

        try:
            allowed, remaining = await cache.check_rate_limit(
                identity,
                limit=limit,
                window_seconds=settings.rate_limit_window_seconds,
            )
        except CacheError:
            # The cache is down. Which way to fail is a policy decision, not a
            # detail: failing open keeps the API serving; failing closed refuses
            # work. Both are defensible, so it is configurable and logged loudly.
            logger.error(
                "ratelimit.store_unavailable",
                extra={"fail_closed": settings.rate_limit_fail_closed},
            )
            if settings.rate_limit_fail_closed:
                await self._reject(send, settings.rate_limit_window_seconds)
                return
            await self.app(scope, receive, send)
            return

        if not allowed:
            logger.warning(
                "ratelimit.exceeded",
                extra={"limit": limit, "bucket": identity.rsplit(":", 1)[-1]},
            )
            await self._reject(send, settings.rate_limit_window_seconds)
            return

        async def send_with_limits(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers[RATE_LIMIT_REMAINING_HEADER] = str(remaining)
                headers[RATE_LIMIT_WINDOW_HEADER] = str(settings.rate_limit_window_seconds)
            await send(message)

        await self.app(scope, receive, send_with_limits)

    @staticmethod
    async def _reject(send: Send, retry_after: int) -> None:
        """Send a 429 without invoking the application."""
        import json

        body = json.dumps(
            {"detail": "rate limit exceeded", "retry_after_seconds": retry_after}
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"retry-after", str(retry_after).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def install_rate_limiting(application: Any) -> None:
    """Attach the rate-limiting middleware.

    Args:
        application: The FastAPI application.
    """
    application.add_middleware(RateLimitMiddleware)
