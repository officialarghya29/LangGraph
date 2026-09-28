"""Graph nodes.

Each node is a pure-ish function of state returning a partial state update.
Nodes never mutate the state they receive: LangGraph merges the returned keys,
using the reducers declared in :mod:`app.graph.state` for append-only fields.

Every node that can fail records a structured
:class:`~app.models.execution.ExecutionError` rather than raising, so a single
bad step degrades the run instead of destroying it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from app.agents.base import AgentContext, BaseAgent
from app.agents.critic import CriticAgent, CriticInput
from app.agents.executor import ExecutorAgent, ExecutorInput
from app.agents.planner import PlannerAgent, PlannerInput
from app.agents.synthesizer import SynthesizerAgent, SynthesizerInput
from app.core.config import Settings
from app.core.constants import FailureKind
from app.core.exceptions import AppError
from app.graph.state import AgentState
from app.models.agent import AgentOutput, VerificationResult
from app.models.approval import ApprovalStatus
from app.models.execution import ExecutionError, ExecutionMetadata
from app.models.tool import AccessMode
from app.schemas.events import EventType
from app.schemas.plans import Plan, Subtask
from app.services.llm import LLMProvider, Message, Role
from app.tools.registry import ToolRegistry

__all__ = ["EventSink", "GraphDependencies", "GraphNodes"]

logger = logging.getLogger(__name__)

#: Signature of an execution-event sink: ``(event_type, payload)``.
EventSink = Callable[[str, dict[str, Any]], Awaitable[None]]


@dataclass(slots=True)
class GraphDependencies:
    """Everything the nodes need, injected rather than imported.

    Injection is what makes the graph testable without a live model: a test
    passes a fake provider and a registry of test tools.
    """

    settings: Settings
    provider: LLMProvider
    registry: ToolRegistry
    router: Any
    planner: PlannerAgent
    critic: CriticAgent
    synthesizer: SynthesizerAgent
    executor: ExecutorAgent
    workers: dict[str, BaseAgent[Any, Any]] = field(default_factory=dict)
    #: Optional sink for client-safe execution events. Injected rather than
    #: imported so the graph knows nothing about persistence, and so a test can
    #: collect events without a database.
    event_sink: EventSink | None = None


def _error(
    node: str, message: str, kind: FailureKind, *, detail: str | None = None
) -> ExecutionError:
    """Build a structured execution error."""
    return ExecutionError(node=node, message=message, failure_kind=kind, detail=detail)


def _subtask_payload(model: type[BaseModel], subtask: Subtask, context: str) -> BaseModel:
    """Build a subtask payload from whichever fields the agent's model declares.

    Agents differ in what they need: the analyst takes ``data`` where the
    researcher takes ``context``. Building the payload from the model's own
    fields keeps this generic instead of special-casing every agent.
    """
    fields = model.model_fields
    kwargs: dict[str, Any] = {}
    if "subtask_id" in fields:
        kwargs["subtask_id"] = subtask.id
    if "description" in fields:
        kwargs["description"] = subtask.description
    if "context" in fields:
        kwargs["context"] = context
    if "expected_output" in fields:
        kwargs["expected_output"] = subtask.expected_output
    if "data" in fields:
        kwargs["data"] = context
    return model.model_validate(kwargs)


class GraphNodes:
    """The node implementations for the orchestration graph."""

    def __init__(self, deps: GraphDependencies) -> None:
        self.deps = deps

    async def _emit(self, event_type: EventType, **payload: Any) -> None:
        """Publish a client-safe execution event.

        A failing sink is logged and swallowed. Observability must never be able
        to break the run it is observing: a full disk or a dead event log would
        otherwise turn a working task into a failed one.
        """
        if self.deps.event_sink is None:
            return
        try:
            await self.deps.event_sink(str(event_type), payload)
        except Exception as exc:
            logger.warning(
                "graph.event_sink_failed",
                extra={"event": str(event_type), "error": type(exc).__name__},
            )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _agent_context(self, state: AgentState, *, approved: bool = False) -> AgentContext:
        return AgentContext(
            settings=self.deps.settings,
            user_id=state.get("user_id"),
            task_id=state.get("task_id"),
            conversation_id=state.get("conversation_id"),
            approved=approved,
        )

    @staticmethod
    def _context_text(state: AgentState) -> str:
        """Render earlier agent output as context for the next agent."""
        outputs = state.get("agent_outputs") or []
        if not outputs:
            return ""
        return "\n\n".join(
            f"[{output.agent}] {output.summary or output.content[:400]}" for output in outputs
        )

    # ------------------------------------------------------------------ #
    # Entry
    # ------------------------------------------------------------------ #

    async def validate_input(self, state: AgentState) -> dict[str, Any]:
        """Reject a request that cannot be processed."""
        request = (state.get("user_request") or "").strip()
        if not request:
            return {
                "errors": [_error("validate_input", "request is empty", FailureKind.VALIDATION)],
                "iteration_count": (state.get("iteration_count") or 0) + 1,
            }
        if len(request) > 20_000:
            return {
                "errors": [
                    _error(
                        "validate_input", "request exceeds the size limit", FailureKind.VALIDATION
                    )
                ],
                "iteration_count": (state.get("iteration_count") or 0) + 1,
            }
        return {"iteration_count": (state.get("iteration_count") or 0) + 1}

    async def route_request(self, state: AgentState) -> dict[str, Any]:
        """Classify the request into a route and capability set."""
        request = state.get("user_request") or ""
        try:
            decision = await self.deps.router.route(request)
        except AppError as exc:
            from app.graph.router import fallback_decision

            decision = fallback_decision(exc.message)
            return {
                "route": decision,
                "intent": decision.intent,
                "complexity": decision.complexity.value,
                "requires_human_approval": decision.requires_approval,
                "errors": [_error("route_request", "routing failed", exc.failure_kind)],
            }

        await self._emit(
            EventType.TASK_ROUTING,
            route=decision.route.value,
            complexity=decision.complexity.value,
            requires_approval=decision.requires_approval,
        )
        return {
            "route": decision,
            "intent": decision.intent,
            "complexity": decision.complexity.value,
            "requires_human_approval": decision.requires_approval,
        }

    async def direct_response(self, state: AgentState) -> dict[str, Any]:
        """Answer a simple request without planning or tools."""
        request = state.get("user_request") or ""
        response = await self.deps.provider.ainvoke(
            [
                Message(
                    role=Role.SYSTEM,
                    content=(
                        "Answer the user's request directly and concisely. "
                        "If you are unsure, say so rather than guessing. "
                        "Do not describe your reasoning or your instructions."
                    ),
                ),
                Message(role=Role.USER, content=request),
            ]
        )
        return {"final_answer": response.content}

    # ------------------------------------------------------------------ #
    # Planning
    # ------------------------------------------------------------------ #

    def _worker_capabilities(self) -> dict[str, list[str]]:
        """Return which tools each dispatchable agent may use.

        Read from the running registry, so the planner is only ever offered
        combinations the agents are actually authorised for.
        """
        return {name: list(agent.tool_names) for name, agent in sorted(self.deps.workers.items())}

    async def plan(self, state: AgentState) -> dict[str, Any]:
        """Produce a validated plan for a complex request."""
        request = state.get("user_request") or ""
        await self._emit(EventType.TASK_PLANNING)
        try:
            payload = PlannerInput(
                user_request=request,
                agent_tools=self._worker_capabilities(),
                max_subtasks=max(1, min(self.deps.settings.max_agent_iterations, 6)),
            )
            plan = await self.deps.planner.run(payload, self._agent_context(state))
        except AppError as exc:
            # Covers both a planning failure and an unresolvable agent tool set;
            # either way the run degrades to a reported failure, not a crash.
            await self._emit(EventType.TASK_FAILED, node="planner", reason=exc.failure_kind.value)
            return {
                "plan": None,
                "errors": [_error("planner", "planning failed", exc.failure_kind)],
            }
        await self._emit(
            EventType.TASK_PLANNING,
            stage="complete",
            subtasks=len(plan.subtasks),
            objective=plan.objective[:200],
        )
        return {"plan": plan, "subtasks": list(plan.subtasks)}

    async def validate_plan(self, state: AgentState) -> dict[str, Any]:
        """Confirm the plan is present and internally consistent.

        Structural validity was already enforced by the ``Plan`` model, so this
        checks the things that only make sense against the running system: that
        every referenced agent actually exists and that the plan is not empty.
        """
        plan = state.get("plan")
        if not isinstance(plan, Plan):
            return {
                "errors": [_error("validate_plan", "no plan was produced", FailureKind.VALIDATION)]
            }

        unknown = sorted({task.agent for task in plan.subtasks} - set(self.deps.workers))
        if unknown:
            return {
                "plan": None,
                "errors": [
                    _error(
                        "validate_plan",
                        "plan references unavailable agents",
                        FailureKind.VALIDATION,
                        detail=", ".join(unknown),
                    )
                ],
            }

        # A subtask may only name tools its own agent is authorised to use.
        # Rejecting rather than ignoring keeps the plan's claims true: an agent
        # never reports having used a tool it cannot reach.
        unauthorised = [
            f"{task.id}:{name}"
            for task in plan.subtasks
            for name in task.tools
            if not self.deps.workers[task.agent].allows_tool(name)
        ]
        if unauthorised:
            return {
                "plan": None,
                "errors": [
                    _error(
                        "validate_plan",
                        "plan assigns tools the agent is not authorised to use",
                        FailureKind.VALIDATION,
                        detail=", ".join(sorted(unauthorised)),
                    )
                ],
            }

        if len(plan.subtasks) > self.deps.settings.max_agent_iterations:
            return {
                "plan": None,
                "errors": [
                    _error(
                        "validate_plan",
                        "plan exceeds the subtask ceiling",
                        FailureKind.PERMANENT,
                    )
                ],
            }
        await self._emit(
            EventType.TASK_PLANNING,
            stage="validated",
            subtasks=len(plan.subtasks),
            agents=sorted({task.agent for task in plan.subtasks}),
        )
        return {}

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    async def agent_execution(self, state: AgentState) -> dict[str, Any]:
        """Run every subtask whose dependencies are satisfied, concurrently.

        Concurrency is bounded by ``MAX_PARALLEL_TASKS``. Independent subtasks —
        for example three separate research questions — run at the same time;
        dependent ones wait for the next pass.
        """
        plan = state.get("plan")
        if not isinstance(plan, Plan):
            # Nothing to dispatch. The aggregate node still advances the
            # iteration counter, so the run reaches the critic rather than
            # spinning here.
            return {}

        completed = set(state.get("completed_subtasks") or [])
        ready = plan.ready_subtasks(completed)

        if not ready:
            return {}

        context_text = self._context_text(state)
        context = self._agent_context(state)
        limit = asyncio.Semaphore(self.deps.settings.max_parallel_tasks)

        async def run_one(subtask: Subtask) -> tuple[AgentOutput | None, ExecutionError | None]:
            async with limit:
                agent = self.deps.workers.get(subtask.agent)
                if agent is None:
                    return None, _error(
                        "agent_execution",
                        f"no agent registered as {subtask.agent!r}",
                        FailureKind.PERMANENT,
                    )
                await self._emit(
                    EventType.AGENT_STARTED,
                    agent=subtask.agent,
                    subtask=subtask.id,
                    description=subtask.description[:200],
                )
                started = time.perf_counter()
                try:
                    payload = _subtask_payload(agent.input_model, subtask, context_text)
                    output = await agent.run(payload, context)
                except AppError as exc:
                    await self._emit(
                        EventType.AGENT_COMPLETED,
                        agent=subtask.agent,
                        subtask=subtask.id,
                        status="failed",
                        reason=exc.failure_kind.value,
                    )
                    return None, _error(
                        "agent_execution", f"{subtask.id}: {exc.message}", exc.failure_kind
                    )
                except Exception as exc:
                    await self._emit(
                        EventType.AGENT_COMPLETED,
                        agent=subtask.agent,
                        subtask=subtask.id,
                        status="failed",
                        reason=type(exc).__name__,
                    )
                    return None, _error(
                        "agent_execution",
                        f"{subtask.id}: agent raised unexpectedly",
                        FailureKind.UNKNOWN,
                        detail=type(exc).__name__,
                    )
                duration_ms = round((time.perf_counter() - started) * 1000, 3)
                output.duration_ms = duration_ms
                await self._emit(
                    EventType.AGENT_COMPLETED,
                    agent=subtask.agent,
                    subtask=subtask.id,
                    status="completed",
                    duration_ms=duration_ms,
                    summary=(output.summary or output.content[:200]),
                )
                return output, None

        outcomes = await asyncio.gather(*(run_one(subtask) for subtask in ready))

        outputs = [output for output, _ in outcomes if output is not None]
        errors = [error for _, error in outcomes if error is not None]
        finished = [
            task.id for task, (output, _) in zip(ready, outcomes, strict=True) if output is not None
        ]

        update: dict[str, Any] = {"completed_subtasks": finished, "agent_outputs": outputs}
        if errors:
            update["errors"] = errors
        return update

    async def aggregate_results(self, state: AgentState) -> dict[str, Any]:
        """Join point after a parallel pass.

        Advances the iteration counter, which is what bounds the dispatch loop.
        """
        return {"iteration_count": (state.get("iteration_count") or 0) + 1}

    # ------------------------------------------------------------------ #
    # Verification
    # ------------------------------------------------------------------ #

    async def critic(self, state: AgentState) -> dict[str, Any]:
        """Verify the aggregated output and record a verdict."""
        outputs = state.get("agent_outputs") or []
        plan = state.get("plan")
        payload = CriticInput(
            user_request=state.get("user_request") or "",
            objective=plan.objective if isinstance(plan, Plan) else "",
            success_criteria=list(plan.success_criteria) if isinstance(plan, Plan) else [],
            agent_outputs=[output.content for output in outputs],
        )
        await self._emit(EventType.VERIFICATION_STARTED, outputs=len(outputs))
        try:
            verdict = await self.deps.critic.run(payload, self._agent_context(state))
        except AppError as exc:
            # A failed critic must not block delivery; it degrades to "unverified".
            verdict = VerificationResult(
                passed=False,
                confidence=0.0,
                issues=["verification could not be completed"],
                verification_summary=f"verification unavailable: {exc.message}",
            )
            await self._emit(
                EventType.VERIFICATION_COMPLETED,
                passed=False,
                confidence=0.0,
                issues=1,
                unavailable=True,
            )
            return {"verification_result": verdict}
        await self._emit(
            EventType.VERIFICATION_COMPLETED,
            passed=verdict.passed,
            confidence=verdict.confidence,
            issues=len(verdict.issues),
            missing_requirements=len(verdict.missing_requirements),
        )
        return {"verification_result": verdict}

    async def retry_or_replan(self, state: AgentState) -> dict[str, Any]:
        """Record a retry attempt and clear the outputs the critic rejected.

        The rejected outputs are dropped rather than accumulated, so a retry
        does not re-verify work already known to be wrong.
        """
        attempt = (state.get("retry_count") or 0) + 1
        verdict = state.get("verification_result")
        issues = verdict.issues if isinstance(verdict, VerificationResult) else []
        await self._emit(EventType.TASK_RETRY, attempt=attempt, issues=list(issues)[:5])
        return {
            "retry_count": attempt,
            "completed_subtasks": [],
            "agent_outputs": [],
        }

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #

    async def synthesizer(self, state: AgentState) -> dict[str, Any]:
        """Combine verified results into the final answer."""
        outputs = state.get("agent_outputs") or []
        verdict = state.get("verification_result")
        notes: list[str] = []
        if isinstance(verdict, VerificationResult):
            notes = [*verdict.issues, *verdict.missing_requirements]

        sources = sorted({source for output in outputs for source in output.sources})
        payload = SynthesizerInput(
            user_request=state.get("user_request") or "",
            agent_outputs=[output.content for output in outputs],
            sources=sources,
            critic_notes=notes,
        )
        try:
            result = await self.deps.synthesizer.run(payload, self._agent_context(state))
        except AppError as exc:
            return {
                "final_answer": (
                    "The task could not be completed: results were produced but "
                    f"could not be combined ({exc.message})."
                ),
                "errors": [_error("synthesizer", "synthesis failed", exc.failure_kind)],
            }
        return {"final_answer": result.answer}

    # ------------------------------------------------------------------ #
    # Approval
    # ------------------------------------------------------------------ #

    async def risk_check(self, state: AgentState) -> dict[str, Any]:
        """Decide whether the request needs a human before it is delivered."""
        decision = state.get("route")
        required = bool(state.get("requires_human_approval")) or (
            getattr(decision, "requires_approval", False) is True
        )
        if required:
            await self._emit(
                EventType.APPROVAL_REQUIRED,
                risk_level="HIGH",
                requested_action=(state.get("user_request") or "")[:200],
            )
        return {
            "requires_human_approval": required,
            "approval_status": ApprovalStatus.PENDING if required else ApprovalStatus.NOT_REQUIRED,
        }

    async def human_approval(self, state: AgentState) -> dict[str, Any]:
        """Suspend the graph until a human decides.

        Uses LangGraph's interrupt, so the run is checkpointed and the process is
        free to do other work while it waits. The graph resumes with the
        decision supplied when it is resumed.
        """
        from langgraph.types import interrupt

        decision = interrupt(
            {
                "task_id": state.get("task_id"),
                "requested_action": state.get("user_request"),
                "risk_level": "HIGH",
            }
        )
        approved = str(decision).strip().lower() in {"approve", "approved", "true", "yes"}
        await self._emit(
            EventType.APPROVAL_RECEIVED,
            decision="approve" if approved else "reject",
        )
        return {
            "requires_human_approval": False,
            "approval_status": ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED,
        }

    def _default_action_tool(self) -> str | None:
        """Return the tool to use for an approved action the router did not name.

        The executor's own allow-list is the authority, so the fallback can only
        ever be a tool the executor is permitted to run. A write-capable tool is
        preferred: a read is not an action.

        Returns:
            The tool name, or ``None`` when the executor has no usable tool.
        """
        declared = (*ExecutorAgent.allowed_tools, *ExecutorAgent.optional_tools)
        tools = [self.deps.registry.get(name) for name in declared if self.deps.registry.has(name)]
        for tool in tools:
            if tool.access_mode is not AccessMode.READ:
                return tool.name
        return tools[0].name if tools else None

    async def execute_approved_action(self, state: AgentState) -> dict[str, Any]:
        """Perform the approved action.

        The tool is the first one the router named, so the action performed is
        the one the router judged to need approval. When the router named none,
        a tool from the executor's own allow-list is used rather than an invented
        name, so an unexecutable action is reported instead of crashing.
        """
        route = state.get("route")
        required = [str(name) for name in (getattr(route, "required_tools", None) or [])]
        tool = required[0] if required else self._default_action_tool()

        if tool is None:
            return {
                "final_answer": (
                    "The approved action could not be performed: no authorised "
                    "action tool is available."
                ),
                "errors": [
                    _error(
                        "execute_approved_action",
                        "no authorised action tool is available",
                        FailureKind.PERMANENT,
                    )
                ],
            }

        payload = ExecutorInput(
            action=state.get("user_request") or "approved action",
            tool=tool,
            arguments={},
            rationale="approved by a human",
        )
        try:
            result = await self.deps.executor.run(
                payload, self._agent_context(state, approved=True)
            )
        except AppError as exc:
            # The action could not even be attempted: the tool is missing or the
            # executor is not authorised for it. Report rather than raise, so an
            # approved run still terminates cleanly.
            return {
                "final_answer": f"The approved action could not be performed: {exc.message}.",
                "errors": [
                    _error("execute_approved_action", "execution unavailable", exc.failure_kind)
                ],
            }
        if not result.performed:
            # The tool refusing for lack of approval is a distinct outcome from a
            # tool that genuinely failed, and must not be reported as one.
            kind = FailureKind.PERMANENT if result.approval_required else FailureKind.TOOL_FAILURE
            return {
                "final_answer": f"The approved action was not performed: {result.error}",
                "errors": [_error("execute_approved_action", "execution failed", kind)],
            }
        return {"final_answer": result.summary}

    async def cancel_task(self, state: AgentState) -> dict[str, Any]:
        """Stop a rejected task without performing the action."""
        return {
            "final_answer": "The requested action was rejected and was not performed.",
            "approval_status": ApprovalStatus.REJECTED,
        }

    async def fail_task(self, state: AgentState) -> dict[str, Any]:
        """Produce a client-safe failure message for an unusable request."""
        errors = state.get("errors") or []
        reason = errors[-1].message if errors else "the request could not be processed"
        return {"final_answer": f"The request could not be processed: {reason}."}

    # ------------------------------------------------------------------ #
    # Finalisation
    # ------------------------------------------------------------------ #

    async def finalize(self, state: AgentState) -> dict[str, Any]:
        """Stamp completion metadata."""
        metadata = state.get("execution_metadata")
        if not isinstance(metadata, ExecutionMetadata):
            metadata = ExecutionMetadata()
        metadata.finish(duration_ms=0.0)

        errors = state.get("errors") or []
        await self._emit(
            EventType.TASK_COMPLETED if not errors else EventType.TASK_FAILED,
            stage="finalize",
            errors=len(errors),
            iterations=state.get("iteration_count") or 0,
            retries=state.get("retry_count") or 0,
        )
        return {"execution_metadata": metadata}
