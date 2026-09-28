"""Application entry point.

Exposes an application factory rather than a module-level singleton so tests can
build isolated instances. The module-level ``app`` object is the ASGI target
that uvicorn imports (``uvicorn app.main:app``).

Expensive collaborators — the database, the cache, the checkpoint backend, the
LLM provider, the tool registry, the agent workforce, and the compiled graph —
are built once in the lifespan and stored on ``app.state``. Building the graph
per request would rebuild every agent each time and would discard the
checkpointer's value entirely.

Startup policy, which is the interesting decision here:

- A **missing LLM credential** is not fatal. The process still starts, ``/health``
  still reports liveness, and ``/ready`` explains what is missing. Refusing to
  boot would make a configuration gap look like a crash loop and would take the
  discovery endpoints down for no reason.
- A **missing database or checkpoint backend** *is* fatal. Every task, approval,
  and memory lives there, and the durability guarantee is the point of the
  system. Starting without them would mean serving requests that silently cannot
  be resumed, which is worse than not starting.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.middleware import RequestContextMiddleware, install_rate_limiting
from app.api.routes.agents import router as discovery_router
from app.api.routes.chat import router as chat_router
from app.api.routes.events import router as events_router
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
    RateLimitExceededError,
    UsageLimitExceededError,
)
from app.core.logging import configure_logging
from app.database.connection import Database
from app.database.sql_executor import SqlQueryExecutor
from app.graph.builder import build_all_agents, build_graph
from app.graph.checkpoints import CheckpointHandle, open_checkpointer
from app.services.cache import build_cache
from app.services.llm import build_llm_provider
from app.services.memory import build_memory_manager
from app.services.task_store import PostgresTaskStore
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
    (RateLimitExceededError, 429),
    (UsageLimitExceededError, 429),
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
    """Build application singletons on startup and release them on shutdown."""
    settings = get_settings()
    configure_logging(settings)
    logger.info(
        "startup.begin",
        extra={"app": settings.app_name, "env": settings.app_env},
    )

    database = Database(
        settings.database_url,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        echo=settings.database_echo,
    )
    application.state.database = database

    ready, detail = await database.ping()
    if not ready:
        # Refusing to start is deliberate. Every task, approval, and memory is
        # durable state; serving without it would mean accepting work that
        # cannot be recorded.
        await database.dispose()
        raise ConfigurationError("the database is unreachable", detail=detail)
    logger.info("startup.database_ready")

    cache = build_cache(
        settings.redis_url,
        prefix=settings.redis_key_prefix,
        default_ttl=settings.cache_default_ttl_seconds,
    )
    application.state.cache = cache
    cache_ready, cache_detail = await cache.ping()
    if cache_ready:
        logger.info("startup.cache_ready")
    else:
        # Not fatal. Redis holds no durable state, so the application degrades:
        # rate limiting is bypassed and live streaming falls back to the durable
        # event log, both of which are reported by /ready.
        logger.warning("startup.cache_unavailable", extra={"reason": cache_detail})

    application.state.task_store = PostgresTaskStore(database)

    try:
        application.state.memory = build_memory_manager(settings, database=database)
    except ConfigurationError as exc:
        # A missing embedding credential degrades memory rather than the whole
        # application: memory contributes optional context, so losing it must
        # not take down task execution.
        application.state.memory = None
        logger.warning("startup.memory_unavailable", extra={"reason": exc.message})

    # The database tool is registered only when a target is configured. Pointing
    # it at the application's own database would hand an agent every user row, so
    # the absence of a URL means absence of the tool rather than a default that
    # happens to be dangerous.
    sql_executor: SqlQueryExecutor | None = None
    if settings.database_tool_url is not None:
        sql_executor = SqlQueryExecutor(
            settings.database_tool_url,
            read_only=not settings.database_allow_writes,
            echo=settings.database_echo,
        )
    application.state.sql_executor = sql_executor

    registry = build_default_registry(settings, query_executor=sql_executor)
    application.state.tool_registry = registry

    checkpoint: CheckpointHandle = await open_checkpointer(settings)
    application.state.checkpointer = checkpoint
    if not checkpoint.durable:
        # Only reachable when the memory backend was chosen explicitly; a failed
        # durable backend raises above rather than degrading silently.
        logger.warning("startup.volatile_checkpoints")

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
        application.state.graph = build_graph(
            settings,
            provider,
            registry,
            checkpointer=checkpoint.saver,
            memory=getattr(application.state, "memory", None),
        )
        application.state.agents = build_all_agents(provider, registry)
        logger.info("startup.graph_ready", extra={"agents": len(application.state.agents)})

    logger.info("startup.complete")

    try:
        yield
    finally:
        logger.info("shutdown.begin")
        executor = getattr(application.state, "sql_executor", None)
        if executor is not None:
            await executor.close()
        await checkpoint.close()
        await cache.close()
        await database.dispose()
        logger.info("shutdown.complete")


def create_app() -> FastAPI:
    """Build and configure a FastAPI application instance.

    Returns:
        A fully configured :class:`fastapi.FastAPI` application.
    """
    application = FastAPI(
        title="LangGraph Multi-Agent System",
        version="1.0.0",
        description=(
            "Production-oriented multi-agent orchestration over LangGraph, with "
            "durable checkpointing, human approval, and a least-privilege tool "
            "pipeline."
        ),
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    application.add_exception_handler(AppError, app_error_handler)

    # Order matters: correlation ids are assigned first so every later log line
    # and error response carries one, and rate limiting runs before any handler
    # body does work.
    application.add_middleware(RequestContextMiddleware)
    install_rate_limiting(application)

    application.include_router(health_router)
    application.include_router(chat_router)
    application.include_router(tasks_router)
    application.include_router(events_router)
    application.include_router(discovery_router)

    return application


app = create_app()
