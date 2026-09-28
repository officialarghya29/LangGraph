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
    """What the planner is asked to plan."""

    user_request: str = Field(min_length=1)
    available_agents: list[str] = Field(default_factory=list)
    available_tools: list[str] = Field(default_factory=list)
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
        "- Never include a subtask that requires a capability it was not offered.\n"
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
        agents = ", ".join(payload.available_agents) or "(none)"
        tools = ", ".join(payload.available_tools) or "(none)"
        return (
            f"Request:\n{payload.user_request}\n\n"
            f"Available agents: {agents}\n"
            f"Available tools: {tools}\n"
            f"Maximum subtasks: {payload.max_subtasks}\n\n"
            "Produce the plan."
        )
