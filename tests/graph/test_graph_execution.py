"""End-to-end graph execution tests.

Every test runs the real compiled graph against a deterministic fake provider,
so the whole orchestration is exercised offline: routing, planning, parallel
dispatch, criticism, synthesis, approval, and resume.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest
from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.graph.builder import build_graph
from app.graph.checkpoints import thread_config
from app.graph.state import initial_state
from app.models.tool import AccessMode
from app.services.llm import FakeLLMProvider
from app.tools.base import Tool, ToolContext
from app.tools.registry import ToolRegistry
from langgraph.types import Command
from pydantic import BaseModel

SETTINGS = Settings(_env_file=None)


class _NoArgs(BaseModel):
    """Empty input, so the executor's argument-free call validates."""


class _WriteResult(BaseModel):
    written: bool = True


class WriteFileTool(Tool[_NoArgs, _WriteResult]):
    """A stand-in for the real write tool, so approval resume is testable."""

    name = "write_file"
    description = "test double for the write tool"
    access_mode = AccessMode.WRITE
    input_model = _NoArgs
    output_model = _WriteResult

    async def run(self, payload: _NoArgs, context: ToolContext) -> _WriteResult:
        assert context.approved is True, "the executor must pass approval through"
        return _WriteResult()


async def _inert_run(self: object, payload: Any, context: ToolContext) -> _WriteResult:
    """Do nothing, successfully."""
    return _WriteResult()


#: Tool names the worker agents declare as required, and the mode each is
#: registered with. The graph cannot be built without satisfying this contract,
#: which is the point: these tests exercise orchestration against the same
#: wiring rules as production, not against a registry that could never exist.
_WORKER_TOOLS: dict[str, AccessMode] = {
    "web_search": AccessMode.READ,
    "read_file": AccessMode.READ,
    "list_directory": AccessMode.READ,
    "write_file": AccessMode.WRITE,
    "python_executor": AccessMode.WRITE,
}


def stub_tool(name: str, mode: AccessMode = AccessMode.READ) -> Tool[Any, Any]:
    """Build an inert tool registered under ``name``.

    Built as a class rather than configured per instance because
    ``effective_risk()`` reads a class attribute; a mode set on an instance
    would be invisible to the approval gate and the test would pass for the
    wrong reason.
    """
    namespace: dict[str, object] = {
        "name": name,
        "description": f"stub for {name}",
        "access_mode": mode,
        "input_model": _NoArgs,
        "output_model": _WriteResult,
        "run": _inert_run,
    }
    return type(f"Stub_{name}", (Tool,), namespace)()


def workforce_registry() -> ToolRegistry:
    """A registry that satisfies the worker agents' required-tool contract."""
    return ToolRegistry(
        [
            WriteFileTool() if name == "write_file" else stub_tool(name, mode)
            for name, mode in _WORKER_TOOLS.items()
        ]
    )


# --------------------------------------------------------------------------- #
# Scripted provider
# --------------------------------------------------------------------------- #


def route_payload(
    route: str,
    *,
    approval: bool = False,
    required_tools: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "route": route,
        "complexity": "complex" if route not in {"direct", "human_approval"} else "simple",
        "intent": "scripted",
        "required_capabilities": [],
        "required_agents": [],
        "required_tools": required_tools or [],
        "requires_planning": False,
        "requires_approval": approval,
        "reasoning_summary": "scripted decision",
    }


DEFAULT_PLAN: dict[str, Any] = {
    "objective": "compare two stores",
    "subtasks": [
        {
            "id": "a",
            "description": "research postgres",
            "agent": "researcher",
            "tools": ["web_search"],
            "expected_output": "notes",
            "success_criteria": "sourced",
        },
        {
            "id": "b",
            "description": "research redis",
            "agent": "researcher",
            "tools": ["web_search"],
            "expected_output": "notes",
            "success_criteria": "sourced",
        },
    ],
}


def agent_output(subtask_id: str) -> dict[str, Any]:
    return {
        "agent": "researcher",
        "subtask_id": subtask_id,
        "content": f"finding for {subtask_id}",
        "summary": f"summary for {subtask_id}",
        "sources": [f"https://example.test/{subtask_id}"],
        "confidence": 0.8,
    }


def verdict(passed: bool, issues: list[str] | None = None) -> dict[str, Any]:
    return {
        "passed": passed,
        "confidence": 0.9,
        "issues": issues or [],
        "missing_requirements": [],
        "corrections": [],
        "verification_summary": "scripted verdict",
    }


SYNTHESIS = {
    "answer": "Postgres and Redis differ.",
    "sources": ["https://example.test/a"],
    "caveats": [],
}


class SlowResearchProvider(FakeLLMProvider):
    """Delays only worker agents, so dispatch concurrency is measurable.

    The delay is an ``await``, not a blocking sleep. A blocking sleep would
    occupy the event loop and serialise the very work the test is trying to
    observe, making a correct implementation look broken.
    """

    def __init__(self, delay: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._delay = delay

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        joined = "\n".join(m.content for m in messages)
        if "You are a research agent" in joined:
            await asyncio.sleep(self._delay)
        return await super().ainvoke(messages, **kwargs)


def responder_for(
    *,
    route: str = "direct",
    approval: bool = False,
    required_tools: list[str] | None = None,
    plan: dict[str, Any] | None = None,
    verdicts: list[dict[str, Any]] | None = None,
) -> Any:
    """Build a role-keyed responder.

    Responses are keyed on the role prompt rather than on call order, so the
    test stays correct even when two agents run concurrently.
    """
    pending = list(verdicts or [verdict(True)])

    def responder(messages: Any) -> str:
        joined = "\n".join(m.content for m in messages)

        if "You route requests" in joined:
            return json.dumps(
                route_payload(route, approval=approval, required_tools=required_tools)
            )
        if "You are a planning agent" in joined:
            return json.dumps(plan or DEFAULT_PLAN)
        if "You are a research agent" in joined:
            return json.dumps(agent_output("a" if "postgres" in joined else "b"))
        if "You are a verification agent" in joined:
            return json.dumps(pending.pop(0) if pending else verdict(True))
        if "You are a synthesis agent" in joined:
            return json.dumps(SYNTHESIS)
        return "direct answer"

    return responder


def make_provider(**kwargs: Any) -> FakeLLMProvider:
    """Build a deterministic provider for the scripted roles."""
    return FakeLLMProvider(responder=responder_for(**kwargs))


def build(provider: FakeLLMProvider, registry: ToolRegistry | None = None) -> Any:
    return build_graph(
        SETTINGS, provider, registry if registry is not None else workforce_registry()
    )


def run(graph: Any, request: str = "compare two stores", **state_overrides: Any) -> dict[str, Any]:
    state = initial_state(
        request,
        iteration_limit=SETTINGS.max_agent_iterations,
        retry_limit=SETTINGS.max_retries,
        **state_overrides,
    )
    return asyncio.run(graph.ainvoke(state, thread_config(state["task_id"])))


# --------------------------------------------------------------------------- #
# Simple path
# --------------------------------------------------------------------------- #


def test_a_simple_request_is_answered_directly() -> None:
    provider = make_provider(route="direct")

    result = run(build(provider), "hello")

    assert result["final_answer"]
    assert result["route"].route.value == "direct"


def test_a_direct_request_skips_planning_and_agents() -> None:
    provider = make_provider(route="direct")

    result = run(build(provider), "hello")

    assert result.get("plan") is None
    assert result["agent_outputs"] == []


def test_a_direct_request_does_not_require_approval() -> None:
    provider = make_provider(route="direct")

    result = run(build(provider), "hello")

    assert result["requires_human_approval"] is False


# --------------------------------------------------------------------------- #
# Complex path
# --------------------------------------------------------------------------- #


def test_a_complex_request_is_planned_and_executed() -> None:
    provider = make_provider(route="research")

    result = run(build(provider))

    assert sorted(result["completed_subtasks"]) == ["a", "b"]
    assert len(result["agent_outputs"]) == 2
    assert result["verification_result"].passed is True
    assert result["final_answer"] == "Postgres and Redis differ."


def test_independent_subtasks_run_concurrently() -> None:
    """Two 0.25s subtasks should finish in about 0.25s, not 0.5s.

    Only the worker agents are delayed, so the measured time is the dispatch
    cost rather than routing, planning, verification, and synthesis.
    """
    provider = SlowResearchProvider(0.25, responder=responder_for(route="research"))
    graph = build(provider)
    state = initial_state("compare two stores", iteration_limit=10, retry_limit=3)

    started = time.perf_counter()
    asyncio.run(graph.ainvoke(state, thread_config(state["task_id"])))
    elapsed = time.perf_counter() - started

    assert elapsed < 0.45, f"dispatch looks serial ({elapsed:.2f}s)"


def test_parallel_dispatch_respects_the_configured_ceiling() -> None:
    """Concurrency is bounded, not unlimited."""
    settings = Settings(_env_file=None, max_parallel_tasks=1)
    provider = SlowResearchProvider(0.15, responder=responder_for(route="research"))
    graph = build_graph(settings, provider, workforce_registry())
    state = initial_state("compare two stores", iteration_limit=10, retry_limit=3)

    started = time.perf_counter()
    asyncio.run(graph.ainvoke(state, thread_config(state["task_id"])))
    elapsed = time.perf_counter() - started

    # With a ceiling of one, the two delayed subtasks must serialise.
    assert elapsed > 0.3, f"ceiling of 1 did not serialise dispatch ({elapsed:.2f}s)"


def test_a_failing_verdict_triggers_a_retry_then_succeeds() -> None:
    provider = make_provider(
        route="research",
        verdicts=[verdict(False, ["unsupported claim"]), verdict(True)],
    )

    result = run(build(provider))

    assert result["retry_count"] == 1
    assert result["final_answer"] == "Postgres and Redis differ."


def test_retrying_discards_the_rejected_outputs() -> None:
    """A retry must not re-verify work already known to be wrong."""
    provider = make_provider(
        route="research",
        verdicts=[verdict(False, ["bad"]), verdict(True)],
    )

    result = run(build(provider))

    assert len(result["agent_outputs"]) == 2


def test_exhausted_retries_still_deliver_an_answer() -> None:
    provider = make_provider(route="research", verdicts=[verdict(False, ["still wrong"])] * 10)

    result = run(build(provider))

    assert result["final_answer"] == "Postgres and Redis differ."
    assert result["retry_count"] == SETTINGS.max_retries


def test_retry_loop_terminates() -> None:
    """Guards against the retry path becoming unbounded."""
    provider = make_provider(route="research", verdicts=[verdict(False, ["nope"])] * 50)

    result = run(build(provider))

    assert result["retry_count"] <= SETTINGS.max_retries


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #


def test_an_empty_request_fails_cleanly() -> None:
    provider = make_provider(route="direct")
    graph = build(provider)
    state = initial_state("x", iteration_limit=10, retry_limit=3)
    state["user_request"] = "   "

    result = asyncio.run(graph.ainvoke(state, thread_config(state["task_id"])))

    assert "could not be processed" in (result["final_answer"] or "")
    assert result["errors"]


def test_a_plan_that_cannot_be_validated_fails_cleanly() -> None:
    """A plan naming an agent that does not exist must not dispatch."""
    bad_plan = {
        "objective": "do something impossible",
        "subtasks": [
            {
                "id": "a",
                "description": "summon a wizard",
                "agent": "wizard",
                "tools": [],
                "expected_output": "",
                "success_criteria": "",
            }
        ],
    }
    provider = make_provider(route="research", plan=bad_plan)

    result = run(build(provider))

    assert result["final_answer"]
    assert result["agent_outputs"] == []


def test_routing_failure_falls_back_to_a_direct_answer() -> None:
    """A broken router degrades to the cheapest safe path, not to a crash."""
    provider = FakeLLMProvider(default="not valid json at all")

    result = run(build(provider), "hello")

    assert result["final_answer"]
    assert result["route"] is not None


# --------------------------------------------------------------------------- #
# Approval
# --------------------------------------------------------------------------- #


def test_a_gated_request_pauses_for_approval() -> None:
    provider = make_provider(route="human_approval", approval=True)
    graph = build(provider)
    state = initial_state("delete the production database", iteration_limit=10, retry_limit=3)

    asyncio.run(graph.ainvoke(state, thread_config(state["task_id"])))

    snapshot = graph.get_state(thread_config(state["task_id"]))
    assert snapshot.next, "the graph should be suspended awaiting input"


def approval_graph(**provider_kwargs: Any) -> tuple[Any, Any]:
    """Build a graph whose executor has a usable tool."""
    provider = make_provider(route="human_approval", approval=True, **provider_kwargs)
    return build(provider), provider


def test_approving_resumes_and_executes() -> None:
    graph, _ = approval_graph()
    state = initial_state("delete the production database", iteration_limit=10, retry_limit=3)
    config = thread_config(state["task_id"])

    asyncio.run(graph.ainvoke(state, config))
    result = asyncio.run(graph.ainvoke(Command(resume="approve"), config))

    assert result["approval_status"].value == "approved"


def test_an_approved_action_is_actually_performed() -> None:
    graph, _ = approval_graph()
    state = initial_state("delete the production database", iteration_limit=10, retry_limit=3)
    config = thread_config(state["task_id"])

    asyncio.run(graph.ainvoke(state, config))
    result = asyncio.run(graph.ainvoke(Command(resume="approve"), config))

    assert "completed" in (result["final_answer"] or "").lower()


def test_rejecting_cancels_without_executing() -> None:
    graph, _ = approval_graph()
    state = initial_state("delete the production database", iteration_limit=10, retry_limit=3)
    config = thread_config(state["task_id"])

    asyncio.run(graph.ainvoke(state, config))
    result = asyncio.run(graph.ainvoke(Command(resume="reject"), config))

    assert result["approval_status"].value == "rejected"
    assert "rejected" in (result["final_answer"] or "").lower()


def test_an_unregistered_execution_tool_fails_cleanly() -> None:
    """An approved action naming a tool that does not exist must not crash."""
    provider = make_provider(
        route="human_approval",
        approval=True,
        required_tools=["no_such_tool"],
    )
    graph = build(provider)
    state = initial_state("delete the production database", iteration_limit=10, retry_limit=3)
    config = thread_config(state["task_id"])

    asyncio.run(graph.ainvoke(state, config))
    result = asyncio.run(graph.ainvoke(Command(resume="approve"), config))

    assert "could not be performed" in (result["final_answer"] or "")


def test_a_workforce_declaring_an_unknown_tool_cannot_be_built() -> None:
    """A misnamed tool must fail at wiring time, not silently reduce capability."""
    provider = make_provider(route="direct")

    with pytest.raises(ConfigurationError, match="not registered"):
        build(provider, ToolRegistry())


def test_resume_is_checkpointed_under_a_stable_thread() -> None:
    """The same task id must always resume the same run."""
    graph, _ = approval_graph()
    state = initial_state("delete the production database", iteration_limit=10, retry_limit=3)
    config = thread_config(state["task_id"])

    asyncio.run(graph.ainvoke(state, config))
    before = graph.get_state(config)
    asyncio.run(graph.ainvoke(Command(resume="approve"), config))
    after = graph.get_state(config)

    assert before.values["task_id"] == after.values["task_id"]
    assert before.next and not after.next


# --------------------------------------------------------------------------- #
# State integrity
# --------------------------------------------------------------------------- #


def test_every_run_records_execution_metadata() -> None:
    provider = make_provider(route="direct")

    result = run(build(provider), "hello")

    assert result["execution_metadata"] is not None
    assert result["execution_metadata"].finished_at is not None


def test_a_successful_run_records_no_errors() -> None:
    provider = make_provider(route="research")

    result = run(build(provider))

    assert result["errors"] == []


def test_the_iteration_ceiling_bounds_the_dispatch_loop() -> None:
    """Even if no subtask ever completes, the graph must terminate."""
    provider = make_provider(route="research")
    graph = build(provider)
    state = initial_state("compare two stores", iteration_limit=2, retry_limit=1)

    result = asyncio.run(graph.ainvoke(state, thread_config(state["task_id"])))

    assert result["iteration_count"] <= 4


@pytest.mark.parametrize("route", ["research", "coding", "data_analysis", "multi_agent"])
def test_every_complex_route_reaches_a_final_answer(route: str) -> None:
    provider = make_provider(route=route)

    result = run(build(provider))

    assert result["final_answer"]
