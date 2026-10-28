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

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.middleware import RequestContextMiddleware, install_rate_limiting
from app.api.routes.agents import router as discovery_router
from app.api.routes.chat import router as chat_router
from app.api.routes.dashboard import router as dashboard_router
from app.api.routes.events import router as events_router
from app.api.routes.health import router as health_router
from app.api.routes.metrics import router as metrics_router
from app.api.routes.tasks import router as tasks_router
from app.core.config import Settings, get_settings
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
from app.observability.metrics import MetricRegistry
from app.observability.tracing import build_tracer
from app.services.budget import BudgetedProvider
from app.services.cache import build_cache
from app.services.llm import (
    LLMProvider,
    build_llm_provider,
    catalogue_status,
    closest_model_names,
)
from app.services.memory import build_memory_manager
from app.services.resilience import RetryingProvider
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


async def _warn_if_model_is_missing(
    provider: LLMProvider,
    settings: Settings,
    metrics: MetricRegistry | None = None,
) -> None:
    """Log a warning when the configured model is no longer served.

    The commonest way a working deployment stops working is a model name that the
    vendor retired or renamed, and the failure arrives at the first inference —
    after being paid for — as a 404 the operator has to trace back to
    configuration. The catalogue costs one request and no tokens, so the name is
    checked once, in the background, and the warning names the nearest live
    models instead of leaving the operator to guess.

    The answer is also published as ``configured_model_available`` so an alert can
    watch it: a log line is only noticed by someone reading logs. The series is
    labelled with the provider and the model, both of which are low-cardinality
    and change once per deployment.

    Silence is the default in every other case: no catalogue, an unreachable
    endpoint, or a provider that does not publish one is not evidence about the
    model, and a startup that fails because a vendor was briefly down would be a
    worse bug than the one this catches. When the catalogue cannot be read the
    gauge is left unset rather than set to zero — an absent series means nobody
    has asked, where ``0`` would assert that the model is gone.

    Args:
        provider: The provider whose catalogue to read, as wrapped by the
            lifespan.
        settings: The configuration naming the model to look for.
        metrics: The registry to publish the verdict to, when one exists.
    """
    models = await catalogue_status(provider)
    if models is None:
        return

    available = settings.llm_model_name in models
    if metrics is not None:
        metrics.set_gauge(
            "configured_model_available",
            1.0 if available else 0.0,
            provider=settings.llm_provider,
            model=settings.llm_model_name,
        )
    if available:
        return
    logger.warning(
        "startup.model_missing",
        extra={
            "model": settings.llm_model_name,
            "provider": settings.llm_provider,
            "closest": ", ".join(closest_model_names(settings.llm_model_name, models)) or "none",
        },
    )


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Build application singletons on startup and release them on shutdown."""
    settings = get_settings()
    configure_logging(settings)
    logger.info(
        "startup.begin",
        extra={"app": settings.app_name, "env": settings.app_env},
    )

    # Built first: the request middleware reads it from the application state to
    # record every request, including the ones that fail during startup work.
    application.state.metrics = MetricRegistry()

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

    application.state.tracer = build_tracer(settings, metrics=application.state.metrics)
    application.state.metrics.set_gauge("registered_tools", len(registry.list()))

    checkpoint: CheckpointHandle = await open_checkpointer(settings)
    application.state.checkpointer = checkpoint
    if not checkpoint.durable:
        # Only reachable when the memory backend was chosen explicitly; a failed
        # durable backend raises above rather than degrading silently.
        logger.warning("startup.volatile_checkpoints")

    try:
        # Two decorators, in this order. Retrying outermost means a retried call
        # is retried *and charged*, since each attempt passes through the budget.
        # Reversed, the discarded attempts would be invisible in the totals.
        provider: LLMProvider = RetryingProvider(
            BudgetedProvider(build_llm_provider(settings)),
            # The provider's own retry budget, which is a different question from
            # ``max_retries``: that one bounds how many times a run may be
            # replanned after a failed verdict, while this bounds how many times
            # one model call may be attempted.
            max_retries=settings.llm_max_retries,
            # Retries are counted by kind, which is a bounded set of ten values,
            # so the series cannot be inflated by a misbehaving provider.
            on_retry=lambda attempt, kind, exc: application.state.metrics.increment(
                "llm_retries_total", kind=kind.value
            ),
        )
    except ConfigurationError as exc:
        application.state.provider = None
        application.state.provider_error = exc.message
        application.state.graph = None
        application.state.agents = {}
        logger.warning("startup.provider_unavailable", extra={"reason": exc.message})
    else:
        application.state.provider = provider
        application.state.provider_error = None
        # Which model this process will call is now derived from the provider when
        # LLM_MODEL is unset, so it is not visible in the environment. Logging it
        # is the only way an operator can confirm from the outside that the
        # intended model is the one running.
        logger.info(
            "startup.model_ready",
            extra={
                "provider": settings.llm_provider,
                "model": settings.llm_model_name,
                # Included because it is the one sampling setting that can be set
                # and still not travel: it is sent only to a model that reasons.
                # Logged next to the model, which is what decides that.
                "reasoning_effort": settings.llm_reasoning_effort or "vendor-default",
            },
        )
        application.state.graph = build_graph(
            settings,
            provider,
            registry,
            checkpointer=checkpoint.saver,
            memory=getattr(application.state, "memory", None),
        )
        application.state.agents = build_all_agents(provider, registry)
        application.state.metrics.set_gauge("registered_agents", len(application.state.agents))
        logger.info("startup.graph_ready", extra={"agents": len(application.state.agents)})

        # Asked in the background so a slow vendor cannot delay serving, and
        # outside development so the test suite never needs the network: the CI
        # live job is the place a model name is checked before it ships.
        if settings.app_env != "development":
            application.state.catalogue_check = asyncio.create_task(
                _warn_if_model_is_missing(provider, settings, application.state.metrics)
            )

    logger.info("startup.complete")

    try:
        yield
    finally:
        logger.info("shutdown.begin")
        # The provider holds a pooled HTTP client once it has made a call, which
        # is a resource the process owns rather than one it may abandon. Closed
        # before the database so an in-flight model call fails on its own timeout
        # rather than on a closed event loop.
        catalogue_check = getattr(application.state, "catalogue_check", None)
        if catalogue_check is not None and not catalogue_check.done():
            # Cancelled before the client closes, so a vendor request outstanding
            # at shutdown fails on this cancellation rather than on a closed pool.
            catalogue_check.cancel()
            with suppress(asyncio.CancelledError):
                await catalogue_check
        active_provider = getattr(application.state, "provider", None)
        if active_provider is not None:
            await active_provider.aclose()
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
    # Withheld in production unless it was asked for: the schema is an inventory
    # of every route, field, and error shape, which is useful while building
    # against the API and unnecessary exposure once it is deployed. Passing
    # ``None`` for ``openapi_url`` is what removes ``/openapi.json`` and, with it,
    # the interactive docs page.
    serve_docs = get_settings().serve_api_docs

    application = FastAPI(
        title="LangGraph Multi-Agent System",
        version="1.0.0",
        description=(
            "Production-oriented multi-agent orchestration over LangGraph, with "
            "durable checkpointing, human approval, and a least-privilege tool "
            "pipeline."
        ),
        docs_url="/docs" if serve_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if serve_docs else None,
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
    application.include_router(metrics_router)

    # Registered unconditionally; each route returns 404 unless
    # ``DASHBOARD_ENABLED`` is set. Deciding at request time rather than at
    # startup keeps the route table identical between environments, so a test can
    # assert both the enabled and the disabled behaviour without rebuilding the
    # application.
    application.include_router(dashboard_router)

    return application


app = create_app()
