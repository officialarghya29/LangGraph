"""Shared scaffolding for the adversarial suites.

The graph harness — scripted provider, stub tools, a registry that satisfies the
agents' required-tool contract — already exists for the execution tests. It is
imported from there rather than copied, because two copies of a harness drift,
and the copy that drifts is always the one asserting a security property.
"""

from __future__ import annotations

import json
from typing import Any

from app.core.config import Settings
from app.services.llm import FakeLLMProvider
from tests.graph.test_graph_execution import (
    DEFAULT_PLAN,
    SETTINGS,
    agent_output,
    build,
    make_provider,
    route_payload,
    run,
    verdict,
    workforce_registry,
)

__all__ = [
    "BUILTIN_AGENT_NAMES",
    "DEFAULT_PLAN",
    "SETTINGS",
    "agent_output",
    "build",
    "build_with_plan",
    "make_provider",
    "route_payload",
    "run",
    "verdict",
    "workforce_registry",
]

#: Every agent the application registers, so a test can assert that the
#: orchestrator roles are not dispatchable by a plan.
BUILTIN_AGENT_NAMES: tuple[str, ...] = (
    "analyst",
    "coder",
    "critic",
    "document",
    "executor",
    "planner",
    "researcher",
    "synthesizer",
)


def build_with_plan(plan: dict[str, Any], *, route: str = "research") -> Any:
    """Build a graph whose planner returns exactly ``plan``.

    Used to script a *hostile* plan — one a prompt injection would have to
    produce in order to widen its own authority — and assert the orchestrator
    refuses it.
    """
    return build(make_provider(route=route, plan=plan))


def plan_with(agent: str, tools: list[str] | None = None) -> dict[str, Any]:
    """Return a one-subtask plan naming ``agent`` and ``tools``."""
    return {
        "objective": "do the thing",
        "subtasks": [
            {
                "id": "a",
                "description": "do the thing",
                "agent": agent,
                "tools": tools or [],
                "expected_output": "out",
                "success_criteria": "ok",
            }
        ],
    }


def injection_provider(payload: str) -> FakeLLMProvider:
    """Return a provider whose every model response is ``payload``.

    Models an injection that successfully controls the model's output entirely —
    the worst case. Assertions written against this are assertions about the
    orchestrator, not about the prompt, which is the only kind that survives a
    model swap.
    """
    return FakeLLMProvider(responder=lambda messages: payload)


def scripted_settings(**overrides: Any) -> Settings:
    """Return settings with the scripted-test defaults plus overrides."""
    return Settings(_env_file=None, **overrides)


def as_json(payload: dict[str, Any]) -> str:
    """Serialise a scripted model response."""
    return json.dumps(payload)
