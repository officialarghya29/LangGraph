"""Application entry point.

Exposes an application factory rather than a module-level singleton so tests can
build isolated instances. The module-level ``app`` object is the ASGI target
that uvicorn imports (``uvicorn app.main:app``).

Expensive collaborators — the LLM provider, the tool registry, the agent
workforce, and the compiled graph — are built once in the lifespan and stored on
``app.state``. Building the graph per request would rebuild every agent each
time and would discard the checkpointer's value entirely.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes.agents import router as discovery_router
from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.tasks import router as tasks_router
from app.core.config import get_settings
from app.core.exceptions import (
    AppError,
    ApprovalRequiredError,
    AuthenticationError,
    ConfigurationError,
    InputValidationError,
    NotFoundError,
    PermissionDeniedError,
)
from app.core.logging import configure_logging
from app.graph.builder import build_all_agents, build_graph
from app.services.llm import build_llm_provider
from app.services.tasks import InMemoryTaskStore
from app.tools.registry import build_default_registry

__all__ = ["app", "create_app"]

logger = logging.getLogger(__name__)

#: Maps each deliberate error type to the status code a caller should see.
_ERROR_STATUS: tuple[tuple[type[AppError], int], ...] = (
    (InputValidationError, 422),
    (NotFoundError, 404),
    (PermissionDeniedError, 403),
    (AuthenticationError, 401),
    (ApprovalRequiredError, 409),
)


async def app_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Convert a deliberate application error into a response.

    Only the exception's own message is returned. Tracebacks and chained causes
    stay in the logs, where they cannot leak internals to a caller.
    """
    del request
    if not isinstance(exc, AppError):  # pragma: no cover - registered for AppError only
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    status_code = next(
        (code for error_type, code in _ERROR_STATUS if isinstance(exc, error_type)),
        500,
    )
    body: dict[str, Any] = {"detail": str(exc)}
    if exc.detail:
        body["context"] = exc.detail
    return JSONResponse(status_code=status_code, content=body)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Build application singletons on startup.

    A missing credential is *not* fatal. The process still starts, ``/health``
    still reports liveness, and ``/ready`` explains what is missing. Refusing to
    boot would make a configuration problem look like a crash loop, and it would
    make the discovery endpoints unreachable for no reason.
    """
    settings = get_settings()
    configure_logging(settings)

    application.state.task_store = InMemoryTaskStore()
    registry = build_default_registry(settings)
    application.state.tool_registry = registry
    logger.info("startup.tools_registered", extra={"count": len(registry)})

    try:
        provider = build_llm_provider(settings)
    except ConfigurationError as exc:
        application.state.provider = None
        application.state.provider_error = exc.message
        application.state.graph = None
        application.state.agents = {}
        logger.warning("startup.provider_unavailable", extra={"reason": exc.message})
    else:
        application.state.provider = provider
        application.state.provider_error = None
        application.state.graph = build_graph(settings, provider, registry)
        application.state.agents = build_all_agents(provider, registry)
        logger.info("startup.graph_ready", extra={"agents": len(application.state.agents)})

    yield


def create_app() -> FastAPI:
    """Build and configure a FastAPI application instance.

    Returns:
        A fully configured :class:`fastapi.FastAPI` application.
    """
    application = FastAPI(
        title="LangGraph Multi-Agent System",
        version="0.1.0",
        description="Production-oriented multi-agent orchestration over LangGraph.",
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    application.add_exception_handler(AppError, app_error_handler)

    application.include_router(health_router)
    application.include_router(chat_router)
    application.include_router(tasks_router)
    application.include_router(discovery_router)

    return application


app = create_app()
