"""Liveness and readiness endpoints.

``/health`` reports process liveness only. It performs no dependency checks, so
it stays cheap and never fails because a downstream service is degraded. It is
what a container orchestrator should use to decide whether to restart a process.

``/ready`` is the opposite: it reports whether the process can actually serve
work, including which dependencies are unavailable. It should be used to decide
whether to route traffic here.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

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

    A degraded result is not an error: the process is healthy but lacks a
    dependency. Returning 200 with details lets an operator see *why* rather
    than only that it failed.
    """
    provider = getattr(request.app.state, "provider", None)
    graph = getattr(request.app.state, "graph", None)
    registry = getattr(request.app.state, "tool_registry", None)
    agents = getattr(request.app.state, "agents", {}) or {}

    checks: dict[str, str] = {
        "llm_provider": "ok" if provider is not None else "unavailable",
        "orchestration_graph": "ok" if graph is not None else "unavailable",
        "tool_registry": f"{len(registry)} tools" if registry is not None else "unavailable",
        "agents": f"{len(agents)} registered" if agents else "unavailable",
        # Not wired up yet. Reported honestly rather than omitted, so the
        # readiness output does not imply durability the system lacks.
        "checkpoint_store": "in-memory (not durable across restarts)",
        "database": "not configured",
        "cache": "not configured",
    }

    if provider is None:
        checks["llm_provider"] = getattr(request.app.state, "provider_error", "unavailable")

    serving = provider is not None and graph is not None
    return ReadyResponse(status="ok" if serving else "degraded", checks=checks)
