"""Planner agent.

The planner produces a plan and nothing else. It is deliberately given no tools,
so it cannot act on the world while planning; its only output is a validated
:class:`~app.schemas.plans.Plan`.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from app.agents.base import AgentContext, BaseAgent
from app.schemas.plans import Plan

__all__ = ["PlannerAgent", "PlannerInput"]


class PlannerInput(BaseModel):
    """What the planner is asked to plan.

    ``agent_tools`` maps each dispatchable agent to the tools it may actually
    use. The capabilities are given per agent rather than as one flat list,
    because a union lets the planner assign a tool to an agent that is not
    authorised to run it — a plan that looks valid and silently does less than
    it claims.
    """

    user_request: str = Field(min_length=1)
    agent_tools: dict[str, list[str]] = Field(default_factory=dict)
    max_subtasks: int = Field(default=6, ge=1, le=20)


class PlannerAgent(BaseAgent[PlannerInput, Plan]):
    """Decomposes a complex request into an ordered, dependency-aware plan."""

    name = "planner"
    description = "Decomposes a request into subtasks with explicit dependencies"
    #: Intentionally empty: planning must not have side effects.
    allowed_tools: ClassVar[tuple[str, ...]] = ()
    input_model: ClassVar[type[BaseModel]] = PlannerInput
    output_model: ClassVar[type[BaseModel]] = Plan

    system_prompt = (
        "You are a planning agent. Decompose the user's request into a minimal "
        "set of subtasks that together satisfy it.\n"
        "Rules:\n"
        "- Every subtask must name exactly one agent from the available list.\n"
        "- Use dependencies to express ordering. Independent subtasks must not "
        "depend on each other, so they can run in parallel.\n"
        "- Ask for the fewest subtasks that still satisfy the request. Do not "
        "invent work that was not requested.\n"
        "- Give each subtask concrete, checkable success criteria.\n"
        "- Never include a subtask that requires a capability it was not offered. "
        "A subtask may only use tools listed for the agent it names.\n"
        "- Assume nothing about the user's environment or data."
    )

    async def run(self, payload: PlannerInput, context: AgentContext) -> Plan:
        """Produce a validated plan.

        Args:
            payload: The request and the capabilities that may be used.
            context: Runtime context. Unused: planning has no side effects.

        Returns:
            A plan whose subtasks reference only the offered agents and tools.
        """
        prompt = self._render(payload)
        return await self.complete_structured(prompt, Plan)

    def _render(self, payload: PlannerInput) -> str:
        """Render the planning request."""
        lines = [
            f"- {agent}: {', '.join(tools) if tools else 'no tools'}"
            for agent, tools in sorted(payload.agent_tools.items())
        ]
        capabilities = "\n".join(lines) or "(none)"
        return (
            f"Request:\n{payload.user_request}\n\n"
            f"Available agents and the tools each may use:\n{capabilities}\n\n"
            f"Maximum subtasks: {payload.max_subtasks}\n\n"
            "Produce the plan."
        )
