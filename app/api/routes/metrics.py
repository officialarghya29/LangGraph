"""Metrics endpoint.

Serves the Prometheus text exposition format. This is the only route that
returns a non-JSON body, and the only one whose content type is prescribed by
something outside this project.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status

from app.core.config import get_settings
from app.observability.metrics import MetricRegistry

__all__ = ["router"]

router = APIRouter(tags=["observability"])

#: The content type Prometheus expects. Spelled out rather than left to
#: FastAPI's default, because the scraper is what decides whether this route is
#: useful and it does not negotiate.
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    response_class=Response,
    responses={200: {"content": {PROMETHEUS_CONTENT_TYPE: {}}}},
)
async def metrics(request: Request) -> Response:
    """Return every recorded metric series.

    Args:
        request: The incoming request, used to reach the application state.

    Returns:
        The exposition payload. Falls back to an explanatory comment rather than
        a 404 when the registry is absent, so a scraper sees a valid response
        and an operator sees why it is empty.

    Raises:
        HTTPException: 404 when ``METRICS_ENABLED`` is false. Until this check
            existed the setting changed what ``/ready`` reported and nothing
            else, so an operator who turned it off still served the endpoint.
    """
    if not get_settings().metrics_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")

    registry: MetricRegistry | None = getattr(request.app.state, "metrics", None)
    if registry is None:  # pragma: no cover - the registry is built in the lifespan
        return Response(
            content="# metrics registry is not configured\n",
            media_type=PROMETHEUS_CONTENT_TYPE,
        )
    return Response(content=registry.render(), media_type=PROMETHEUS_CONTENT_TYPE)
