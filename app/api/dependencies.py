"""Shared FastAPI dependencies.

Heavy collaborators — the provider, the tool registry, the compiled graph — are
built once during application startup and read from ``app.state`` here. Building
a graph per request would rebuild the whole agent workforce every time, and
would make the graph's checkpointer pointless.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, status

from app.core.config import Settings, get_settings
from app.services.tasks import InMemoryTaskStore

__all__ = [
    "GraphDep",
    "RegistryDep",
    "SettingsDep",
    "TaskStoreDep",
    "get_graph",
    "get_provider",
    "get_registry",
    "get_settings_dep",
    "get_task_store",
    "require_configured",
]


def get_settings_dep() -> Settings:
    """Return the process settings."""
    return get_settings()


def get_task_store(request: Request) -> InMemoryTaskStore:
    """Return the application's task store."""
    store: InMemoryTaskStore = request.app.state.task_store
    return store


def get_provider(request: Request) -> Any:
    """Return the configured LLM provider.

    Raises:
        HTTPException: 503 if the provider could not be constructed at startup,
            which happens when credentials are missing. Failing loudly is
            correct: answering from a fabricated provider would be worse than
            refusing.
    """
    provider = getattr(request.app.state, "provider", None)
    if provider is None:
        reason = getattr(request.app.state, "provider_error", "no provider is configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"the language model provider is unavailable: {reason}",
        )
    return provider


def get_registry(request: Request) -> Any:
    """Return the application's tool registry."""
    return request.app.state.tool_registry


def get_graph(request: Request) -> Any:
    """Return the compiled orchestration graph."""
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        reason = getattr(request.app.state, "provider_error", "the graph is not available")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"the orchestration graph is unavailable: {reason}",
        )
    return graph


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
TaskStoreDep = Annotated[InMemoryTaskStore, Depends(get_task_store)]
GraphDep = Annotated[Any, Depends(get_graph)]
RegistryDep = Annotated[Any, Depends(get_registry)]


def require_configured(settings: Settings) -> None:
    """Raise if the application is missing configuration it needs to run tasks.

    Raises:
        HTTPException: 503, so a caller sees a service problem rather than a
            server error.
    """
    if settings.llm_provider in {"openai", "anthropic"} and settings.llm_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no LLM credential is configured, so tasks cannot be run",
        )
