"""Liveness endpoints.

``/health`` reports process liveness only. It deliberately performs no
dependency checks, so it stays cheap and never fails because a downstream
service is degraded. Readiness (dependency-aware) probing arrives in Phase 23.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Liveness payload."""

    status: str


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    """Report that the process is running and able to serve requests.

    Returns:
        ``{"status": "ok"}`` when the application is live.
    """
    return HealthResponse(status="ok")
