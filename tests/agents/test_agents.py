"""Tests for the agent contract and least-privilege tool access."""

from __future__ import annotations

import json

import pytest
from app.agents.analyst import DataAnalysisAgent
from app.agents.base import AgentContext
from app.agents.coder import CodingAgent
from app.agents.critic import CriticAgent
from app.agents.executor import ExecutorAgent
from app.agents.planner import PlannerAgent
from app.agents.researcher import ResearchAgent
from app.agents.synthesizer import SynthesizerAgent
from app.core.config import Settings
from app.core.exceptions import NotFoundError, ToolPermissionError
from app.models.tool import AccessMode
from app.services.llm import FakeLLMProvider
from app.tools.base import Tool, ToolContext
from app.tools.registry import ToolRegistry
from pydantic import BaseModel, ConfigDict

SETTINGS = Settings(_env_file=None)


class _Args(BaseModel):
    """Strict input, so arguments that no tool could accept are rejected."""

    model_config = ConfigDict(extra="forbid")

    query: str = ""


class _Out(BaseModel):
    ok: bool = True


async def _stub_run(self: object, payload: _Args, context: ToolContext) -> _Out:
    """Do nothing, successfully."""
    return _Out()


def stub_tool(name: str, mode: AccessMode = AccessMode.READ) -> Tool[_Args, _Out]:
    """Build a stub tool registered under ``name``.

    The tool is built as a class rather than configured per instance because
    ``effective_risk()`` and ``requires_approval()`` are classmethods: a risk
    mode set on an instance would be invisible to the approval gate, and the
    test would pass for the wrong reason. Putting ``run`` in the namespace is
    also what satisfies the abstract method.
    """
    namespace: dict[str, object] = {
        "name": name,
        "description": f"stub for {name}",
        "access_mode": mode,
        "input_model": _Args,
        "output_model": _Out,
        "run": _stub_run,
    }
    return type(f"Stub_{name}", (Tool,), namespace)()


ALL_TOOL_NAMES = ("web_search", "filesystem", "python_executor", "database", "github")


def full_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            stub_tool("web_search"),
            # Destructive so that it exercises the approval gate: a plain
            # write is MEDIUM risk and legitimately needs no approval.
            stub_tool("filesystem", AccessMode.DESTRUCTIVE),
            stub_tool("python_executor", AccessMode.WRITE),
            stub_tool("database", AccessMode.READ),
            stub_tool("github", AccessMode.WRITE),
        ]
    )


def provider(**kwargs: object) -> FakeLLMProvider:
    return FakeLLMProvider(**kwargs)  # type: ignore[arg-type]


def context() -> AgentContext:
    return AgentContext(settings=SETTINGS, user_id="u", task_id="t")


# --------------------------------------------------------------------------- #
# Least privilege
# --------------------------------------------------------------------------- #


def test_the_researcher_can_only_search() -> None:
    agent = ResearchAgent(provider(), full_registry())

    assert agent.tool_names == ("web_search",)


def test_the_analyst_gets_computation_and_read_only_data() -> None:
    agent = DataAnalysisAgent(provider(), full_registry())

    assert set(agent.tool_names) == {"python_executor", "database"}


def test_the_coder_gets_only_the_filesystem() -> None:
    agent = CodingAgent(provider(), full_registry())

    assert agent.tool_names == ("filesystem",)


def test_pler_agents_get_no_tools() -> None:
    """Planning, verification, and synthesis must not have side effects."""
    for agent in (
        PlannerAgent(provider(), full_registry()),
        CriticAgent(provider(), full_registry()),
        SynthesizerAgent(provider(), full_registry()),
    ):
        assert agent.tool_names == (), f"{agent.name} should have no tools"


def test_no_agent_can_reach_the_whole_registry() -> None:
    registry = full_registry()
    allowed = set(ALL_TOOL_NAMES)

    for agent in (
        ResearchAgent(provider(), registry),
        CodingAgent(provider(), registry),
        DataAnalysisAgent(provider(), registry),
        ExecutorAgent(provider(), registry),
    ):
        tools = set(agent.tool_names)
        assert tools < allowed, f"{agent.name} has unrestricted tool access"


def test_a_missing_tool_fails_loudly_at_wiring_time() -> None:
    """A typo in an agent's tool list must not silently remove capability."""
    agent = ResearchAgent(provider(), ToolRegistry())

    with pytest.raises(NotFoundError):
        agent.tools()


def test_allows_tool_is_exact() -> None:
    agent = ResearchAgent(provider(), full_registry())

    assert agent.allows_tool("web_search") is True
    assert agent.allows_tool("filesystem") is False
    assert agent.allows_tool("") is False


async def test_calling_a_forbidden_tool_is_refused() -> None:
    """An agent must not be able to invoke a tool outside its allow-list."""
    agent = ResearchAgent(provider(), full_registry())

    with pytest.raises(ToolPermissionError, match="not authorised"):
        await agent.call_tool("filesystem", {"query": "x"}, context())


async def test_calling_an_allowed_tool_succeeds() -> None:
    agent = ResearchAgent(provider(), full_registry())

    result = await agent.call_tool("web_search", {"query": "x"}, context())

    assert result.ok is True


async def test_arguments_are_still_validated_for_allowed_tools() -> None:
    agent = ResearchAgent(provider(), full_registry())

    result = await agent.call_tool("web_search", {"nonsense": 1}, context())

    assert result.ok is False


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #


def test_describe_is_client_safe_and_complete() -> None:
    described = ResearchAgent(provider(), full_registry()).describe()

    assert described["name"] == "researcher"
    assert described["tools"] == ["web_search"]
    assert "input_schema" in described
    assert "output_schema" in described


def test_every_agent_declares_a_distinct_name() -> None:
    agents = [
        ResearchAgent(provider(), full_registry()),
        CodingAgent(provider(), full_registry()),
        DataAnalysisAgent(provider(), full_registry()),
        PlannerAgent(provider(), full_registry()),
        CriticAgent(provider(), full_registry()),
        SynthesizerAgent(provider(), full_registry()),
        ExecutorAgent(provider(), full_registry()),
    ]

    names = [agent.name for agent in agents]

    assert len(names) == len(set(names))
    assert all(name for name in names)


def test_every_agent_declares_a_description_and_prompt() -> None:
    for agent in (
        ResearchAgent(provider(), full_registry()),
        CriticAgent(provider(), full_registry()),
    ):
        assert agent.description
        assert agent.system_prompt


def test_agents_do_not_leak_their_system_prompt_into_the_output_schema() -> None:
    described = CriticAgent(provider(), full_registry()).describe()

    assert "system_prompt" not in described


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #


async def test_the_researcher_labels_its_output() -> None:
    payload = json.dumps(
        {"agent": "ignored", "content": "found it", "summary": "s", "confidence": 0.5}
    )
    agent = ResearchAgent(FakeLLMProvider([payload]), full_registry())

    from app.agents.researcher import ResearchInput

    output = await agent.run(ResearchInput(description="q", subtask_id="task-1"), context())

    assert output.agent == "researcher"
    assert output.subtask_id == "task-1"


async def test_the_planner_sends_the_offered_agents_to_the_model() -> None:
    plan = json.dumps(
        {
            "objective": "o",
            "subtasks": [
                {
                    "id": "a",
                    "description": "d",
                    "agent": "researcher",
                    "tools": [],
                    "expected_output": "",
                    "success_criteria": "",
                }
            ],
        }
    )
    provider_instance = FakeLLMProvider([plan])
    agent = PlannerAgent(provider_instance, full_registry())

    from app.agents.planner import PlannerInput

    await agent.run(PlannerInput(user_request="do it", available_agents=["researcher"]), context())

    assert "researcher" in provider_instance.last_prompt()


async def test_the_executor_passes_approval_through_to_the_tool() -> None:
    """An approved action must arrive at the tool already approved."""
    agent = ExecutorAgent(provider(), full_registry())

    from app.agents.executor import ExecutorInput

    payload = ExecutorInput(action="write", tool="filesystem", arguments={"query": "x"})

    unapproved = await agent.run(payload, context())
    approved = await agent.run(payload, AgentContext(settings=SETTINGS, approved=True))

    assert unapproved.performed is False
    assert approved.performed is True
