"""Liveness and readiness endpoints.

``/health`` reports process liveness only. It performs no dependency checks, so
it stays cheap and never fails because a downstream service is degraded. It is
what a container orchestrator should use to decide whether to restart a process.

``/ready`` is the opposite: it reports whether the process can actually serve
work, and which dependencies are unavailable. It should be used to decide whether
to route traffic here.

Readiness is reported as a *status plus details*, not as a pass/fail. A degraded
dependency is not an error — returning 503 for a missing cache would take a
working API out of rotation over a performance problem. The status field is what
a caller keys on, and it distinguishes the two cases:

- ``ok`` — the provider, the graph, the database, and durable checkpoints are all
  available, so a task will be recorded and can be resumed.
- ``degraded`` — the application serves requests, but something is missing. The
  cache being down is degraded, because rate limiting and live streaming lose
  their fast path.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.core.config import get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Liveness payload."""

    status: str


class ReadyResponse(BaseModel):
    """Readiness payload."""

    status: str
    checks: dict[str, str] = Field(default_factory=dict)


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    """Report that the process is running and able to serve requests.

    Returns:
        ``{"status": "ok"}`` when the application is live.
    """
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse, summary="Readiness probe")
async def ready(request: Request) -> ReadyResponse:
    """Report whether the application can serve work, and what is missing.

    Checks are performed live rather than read from a startup flag: a database
    that was reachable an hour ago and is not now is exactly what this endpoint
    exists to reveal.
    """
    state = request.app.state
    settings = get_settings()
    checks: dict[str, str] = {}

    provider = getattr(state, "provider", None)
    checks["llm_provider"] = (
        "ok" if provider is not None else getattr(state, "provider_error", "unavailable")
    )
    checks["orchestration_graph"] = (
        "ok" if getattr(state, "graph", None) is not None else "unavailable"
    )

    registry = getattr(state, "tool_registry", None)
    checks["tool_registry"] = f"{len(registry)} tools" if registry is not None else "unavailable"

    agents = getattr(state, "agents", None) or {}
    checks["agents"] = f"{len(agents)} registered" if agents else "unavailable"

    database = getattr(state, "database", None)
    if database is None:
        checks["database"] = "unavailable"
    else:
        ok, detail = await database.ping()
        checks["database"] = "ok" if ok else detail

    checkpointer = getattr(state, "checkpointer", None)
    if checkpointer is None:
        checks["checkpoint_store"] = "unavailable"
    elif checkpointer.durable:
        checks["checkpoint_store"] = f"ok ({checkpointer.backend})"
    else:
        # Reported plainly, because a volatile checkpointer changes what a client
        # may assume: a run will not survive a restart.
        checks["checkpoint_store"] = f"volatile ({checkpointer.backend})"

    cache = getattr(state, "cache", None)
    if cache is None:
        checks["cache"] = "unavailable"
    else:
        cache_ok, cache_detail = await cache.ping()
        checks["cache"] = "ok" if cache_ok else f"{cache_detail} (rate limiting bypassed)"

    memory = getattr(state, "memory", None)
    if memory is None:
        checks["memory"] = "unavailable"
    elif not settings.memory_enabled:
        checks["memory"] = "disabled by configuration"
    else:
        checks["memory"] = f"ok ({memory.store_name})"

    checks["authentication"] = "enabled" if settings.auth_enabled else "disabled"
    checks["python_execution"] = (
        "enabled" if settings.python_execution_enabled else "disabled (no sandbox configured)"
    )

    # Serving means the things a task cannot do without. A missing cache degrades
    # performance; a missing database or checkpointer means work would be
    # accepted and silently lost.
    serving = (
        provider is not None
        and getattr(state, "graph", None) is not None
        and database is not None
        and checkpointer is not None
        and checkpointer.durable
    )
    return ReadyResponse(status="ok" if serving else "degraded", checks=checks)
