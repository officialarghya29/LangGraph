"""Discovery endpoints.

Expose what the system can do: which agents exist and which tools are
registered, with their schemas and risk levels.

Both endpoints are strictly read-only and return no secrets. Tool descriptions
include the input schema, which callers need in order to construct a request,
but never a credential the tool holds.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from app.api.dependencies import RegistryDep

router = APIRouter(prefix="/api/v1", tags=["discovery"])


@router.get("/agents", summary="List the registered agents")
async def list_agents(request: Request) -> list[dict[str, Any]]:
    """Return a description of every agent the system can run."""
    agents = getattr(request.app.state, "agents", {}) or {}
    return [agents[name].describe() for name in sorted(agents)]


@router.get("/tools", summary="List the registered tools")
async def list_tools(registry: RegistryDep) -> list[dict[str, Any]]:
    """Return a description of every registered tool, including its risk level."""
    return list(registry.describe())
