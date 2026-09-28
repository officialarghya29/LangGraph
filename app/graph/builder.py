"""Graph construction.

Wires the nodes into a graph with explicit conditional edges. Every branch is
decided by a function over typed state, and every cycle is bounded:

- the dispatch loop is capped by ``MAX_AGENT_ITERATIONS``;
- the retry loop is capped by ``MAX_RETRIES``;
- every node that calls outward has a timeout from configuration.

There is no path through this graph that can run forever.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.analyst import DataAnalysisAgent
from app.agents.base import BaseAgent
from app.agents.coder import CodingAgent
from app.agents.critic import CriticAgent
from app.agents.executor import ExecutorAgent
from app.agents.planner import PlannerAgent
from app.agents.researcher import ResearchAgent
from app.agents.synthesizer import SynthesizerAgent
from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.graph.checkpoints import build_checkpointer
from app.graph.nodes import GraphDependencies, GraphNodes
from app.graph.router import IntentRouter
from app.graph.state import AgentState
from app.models.agent import VerificationResult
from app.models.approval import ApprovalStatus
from app.schemas.plans import Plan, Route
from app.services.llm import LLMProvider
from app.tools.registry import ToolRegistry

__all__ = [
    "build_all_agents",
    "build_dependencies",
    "build_graph",
    "build_workforce",
    "route_after_aggregate",
    "route_after_approval",
    "route_after_critic",
    "route_after_planning",
    "route_after_risk",
    "route_after_routing",
    "route_after_validation",
]


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #


def build_workforce(
    provider: LLMProvider, registry: ToolRegistry
) -> dict[str, BaseAgent[Any, Any]]:
    """Construct the specialist worker agents, keyed by name.

    Only the agents a plan can dispatch to. The planner, critic, synthesizer,
    and executor are orchestrator roles rather than dispatchable workers, so
    they are not in this map and a plan cannot name them as a subtask agent.

    Args:
        provider: The LLM provider every agent reasons with.
        registry: The tool registry agents resolve their allow-lists against.

    Returns:
        A mapping of agent name to agent instance.

    Raises:
        ConfigurationError: If a worker declares a required tool that is not
            registered. Checked here, once, at composition time. A misnamed
            tool is a code defect, and failing loudly at startup is far better
            than discovering mid-run that an agent quietly has fewer
            capabilities than the plan assumed.
    """
    agents: list[BaseAgent[Any, Any]] = [
        ResearchAgent(provider, registry),
        CodingAgent(provider, registry),
        DataAnalysisAgent(provider, registry),
    ]

    missing = sorted(
        f"{agent.name}:{name}"
        for agent in agents
        for name in agent.allowed_tools
        if not registry.has(name)
    )
    if missing:
        raise ConfigurationError(
            "agents declare tools that are not registered",
            detail=", ".join(missing),
        )

    return {agent.name: agent for agent in agents}


def build_all_agents(
    provider: LLMProvider, registry: ToolRegistry
) -> dict[str, BaseAgent[Any, Any]]:
    """Construct every agent, including the orchestrator roles.

    Used for discovery and the API's agent listing. Dispatch uses
    :func:`build_workforce`, which is a strict subset.
    """
    agents: list[BaseAgent[Any, Any]] = [
        *build_workforce(provider, registry).values(),
        PlannerAgent(provider, registry),
        CriticAgent(provider, registry),
        SynthesizerAgent(provider, registry),
        ExecutorAgent(provider, registry),
    ]
    return {agent.name: agent for agent in agents}


def build_dependencies(
    settings: Settings,
    provider: LLMProvider,
    registry: ToolRegistry,
) -> GraphDependencies:
    """Assemble the graph's collaborators."""
    return GraphDependencies(
        settings=settings,
        provider=provider,
        registry=registry,
        router=IntentRouter(provider),
        planner=PlannerAgent(provider, registry),
        critic=CriticAgent(provider, registry),
        synthesizer=SynthesizerAgent(provider, registry),
        executor=ExecutorAgent(provider, registry),
        workers=build_workforce(provider, registry),
    )


# --------------------------------------------------------------------------- #
# Conditional routing
# --------------------------------------------------------------------------- #


def route_after_validation(state: AgentState) -> str:
    """Fail an unusable request, otherwise continue."""
    return "fail" if state.get("errors") else "continue"


def route_after_routing(state: AgentState) -> str:
    """Dispatch on the router's decision."""
    decision = state.get("route")
    if decision is None:
        return "direct"

    route = decision.route
    if route is Route.DIRECT:
        return "direct"
    if route is Route.HUMAN_APPROVAL:
        return "approval"
    return "plan"


def route_after_planning(state: AgentState) -> str:
    """Continue to execution only with a usable plan."""
    plan = state.get("plan")
    return "execute" if isinstance(plan, Plan) else "fail"


def route_after_aggregate(state: AgentState) -> str:
    """Loop back for outstanding subtasks, or move on to verification.

    This is the dispatch loop, and both its exit conditions live here: every
    subtask finished, or the iteration ceiling was reached.
    """
    plan = state.get("plan")
    if not isinstance(plan, Plan):
        return "critic"

    completed = set(state.get("completed_subtasks") or [])

    if completed >= set(plan.subtask_ids):
        return "critic"
    if (state.get("iteration_count") or 0) >= _coalesce(state.get("iteration_limit"), 10):
        return "critic"
    return "continue"


def _coalesce(value: int | None, default: int) -> int:
    """Return a configured ceiling, or the default when it is unset.

    An explicit ``None`` check rather than ``value or default``: a legitimate
    ceiling of zero is falsy, and the ``or`` form would silently substitute the
    default and let a loop run that should have been stopped.
    """
    return default if value is None else value


def route_after_critic(state: AgentState) -> str:
    """Decide what a failed verification means.

    Passing always proceeds. Failing retries while budget remains, and then
    proceeds anyway: the synthesizer is told about the unresolved criticism, so
    the user gets an answer with its caveats rather than a bare failure.
    """
    verdict = state.get("verification_result")
    if not isinstance(verdict, VerificationResult):
        return "synthesize"
    if verdict.passed:
        return "synthesize"

    retries = state.get("retry_count") or 0
    if retries < _coalesce(state.get("retry_limit"), 3):
        return "retry"
    return "synthesize"


def route_after_risk(state: AgentState) -> str:
    """Send gated work to a human, everything else straight to output."""
    return "approval" if state.get("requires_human_approval") else "finalize"


def route_after_approval(state: AgentState) -> str:
    """Execute an approved action, or cancel a rejected one."""
    return "execute" if state.get("approval_status") is ApprovalStatus.APPROVED else "cancel"


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def build_graph(
    settings: Settings,
    provider: LLMProvider,
    registry: ToolRegistry,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build and compile the orchestration graph.

    Args:
        settings: Application settings, including every execution ceiling.
        provider: The LLM provider used by the router and all agents.
        registry: Tool registry the agents resolve allow-lists against.
        checkpointer: Optional checkpoint saver. Defaults to the configured one.

    Returns:
        A compiled graph ready for ``ainvoke``.
    """
    deps = build_dependencies(settings, provider, registry)
    nodes = GraphNodes(deps)

    graph = StateGraph(AgentState)

    graph.add_node("validate_input", nodes.validate_input)
    graph.add_node("route_request", nodes.route_request)
    graph.add_node("direct_response", nodes.direct_response)
    graph.add_node("planner", nodes.plan)
    graph.add_node("validate_plan", nodes.validate_plan)
    graph.add_node("agent_execution", nodes.agent_execution)
    graph.add_node("aggregate_results", nodes.aggregate_results)
    graph.add_node("critic", nodes.critic)
    graph.add_node("retry_or_replan", nodes.retry_or_replan)
    graph.add_node("synthesizer", nodes.synthesizer)
    graph.add_node("risk_check", nodes.risk_check)
    graph.add_node("human_approval", nodes.human_approval)
    graph.add_node("execute_approved_action", nodes.execute_approved_action)
    graph.add_node("cancel_task", nodes.cancel_task)
    graph.add_node("fail_task", nodes.fail_task)
    graph.add_node("finalize", nodes.finalize)

    graph.add_edge(START, "validate_input")
    graph.add_conditional_edges(
        "validate_input",
        route_after_validation,
        {"continue": "route_request", "fail": "fail_task"},
    )
    graph.add_conditional_edges(
        "route_request",
        route_after_routing,
        {"direct": "direct_response", "plan": "planner", "approval": "risk_check"},
    )
    graph.add_edge("direct_response", "risk_check")
    graph.add_edge("planner", "validate_plan")
    graph.add_conditional_edges(
        "validate_plan",
        route_after_planning,
        {"execute": "agent_execution", "fail": "fail_task"},
    )
    graph.add_edge("agent_execution", "aggregate_results")
    graph.add_conditional_edges(
        "aggregate_results",
        route_after_aggregate,
        {"continue": "agent_execution", "critic": "critic"},
    )
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"synthesize": "synthesizer", "retry": "retry_or_replan"},
    )
    graph.add_edge("retry_or_replan", "agent_execution")
    graph.add_edge("synthesizer", "risk_check")
    graph.add_conditional_edges(
        "risk_check",
        route_after_risk,
        {"approval": "human_approval", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {"execute": "execute_approved_action", "cancel": "cancel_task"},
    )
    graph.add_edge("execute_approved_action", "finalize")
    graph.add_edge("cancel_task", "finalize")
    graph.add_edge("fail_task", "finalize")
    graph.add_edge("finalize", END)

    saver = checkpointer if checkpointer is not None else build_checkpointer(settings)
    return graph.compile(checkpointer=saver)
