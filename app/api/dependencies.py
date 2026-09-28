"""Shared FastAPI dependencies.

Heavy collaborators — the database, the cache, the provider, the tool registry,
the compiled graph — are built once during application startup and read from
``app.state`` here. Building a graph per request would rebuild the whole agent
workforce every time and would make the graph's checkpointer pointless.

Settings, identity, and the principal live in :mod:`app.core.auth` and are
re-exported here so a route module imports its dependencies from one place.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, status

from app.core.auth import Principal, PrincipalDep, SettingsDep, get_settings_dep
from app.database.connection import Database
from app.services.cache import Cache
from app.services.task_store import PostgresTaskStore
from app.services.tasks import InMemoryTaskStore

__all__ = [
    "CacheDep",
    "DatabaseDep",
    "GraphDep",
    "Principal",
    "PrincipalDep",
    "RegistryDep",
    "SettingsDep",
    "TaskStoreDep",
    "get_cache",
    "get_database",
    "get_graph",
    "get_provider",
    "get_registry",
    "get_settings_dep",
    "get_task_store",
    "require_configured",
]

#: Either store implementation. The API is coded against the shared surface, so
#: a route does not need to know which one a deployment selected.
AnyTaskStore = PostgresTaskStore | InMemoryTaskStore


def get_task_store(request: Request) -> AnyTaskStore:
    """Return the application's task store."""
    store: AnyTaskStore = request.app.state.task_store
    return store


def get_cache(request: Request) -> Cache:
    """Return the application's cache."""
    cache: Cache = request.app.state.cache
    return cache


def get_database(request: Request) -> Database:
    """Return the application's database handle."""
    database: Database = request.app.state.database
    return database


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
    """Return the compiled orchestration graph.

    Raises:
        HTTPException: 503 when the graph is unavailable, which means either the
            provider could not be built or the checkpoint backend did not open.
    """
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        reason = getattr(request.app.state, "provider_error", "the graph is not available")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"the orchestration graph is unavailable: {reason}",
        )
    return graph


TaskStoreDep = Annotated[AnyTaskStore, Depends(get_task_store)]
CacheDep = Annotated[Cache, Depends(get_cache)]
DatabaseDep = Annotated[Database, Depends(get_database)]
GraphDep = Annotated[Any, Depends(get_graph)]
RegistryDep = Annotated[Any, Depends(get_registry)]


def require_configured(settings: SettingsDep) -> None:
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
