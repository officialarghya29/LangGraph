"""Application entry point for the LangGraph multi-agent system.

Exposes an application factory rather than a module-level singleton so tests can
build isolated instances. The module-level ``app`` object is the ASGI target that
uvicorn imports (``uvicorn app.main:app``).
"""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes.health import router as health_router

__all__ = ["app", "create_app"]


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
    )

    application.include_router(health_router)

    return application


app = create_app()
