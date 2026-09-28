"""Critic agent.

The critic verifies and reports. It never rewrites the work it is judging, so a
result cannot be silently altered between production and delivery. The
orchestrator reads the verdict and decides whether to pass, retry, replan, or
fail.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from app.agents.base import AgentContext, BaseAgent
from app.models.agent import VerificationResult

__all__ = ["CriticAgent", "CriticInput"]


class CriticInput(BaseModel):
    """The material the critic must verify."""

    user_request: str = Field(min_length=1)
    objective: str = ""
    success_criteria: list[str] = Field(default_factory=list)
    agent_outputs: list[str] = Field(default_factory=list)


class CriticAgent(BaseAgent[CriticInput, VerificationResult]):
    """Independently verifies whether the output satisfies the request."""

    name = "critic"
    description = "Verifies outputs against requirements and reports issues"
    #: Empty by design: verification must not modify state or call tools.
    allowed_tools: ClassVar[tuple[str, ...]] = ()
    input_model: ClassVar[type[BaseModel]] = CriticInput
    output_model: ClassVar[type[BaseModel]] = VerificationResult

    system_prompt = (
        "You are a verification agent. Judge whether the provided work actually "
        "satisfies the user's request.\n"
        "Be specific and sceptical.\n"
        "- Report concrete defects, not stylistic preferences.\n"
        "- Separate unsupported claims from errors. A claim with no evidence is "
        "a missing requirement, not necessarily a falsehood.\n"
        "- Report contradictions between the outputs.\n"
        "- List anything the request asked for that is absent.\n"
        "- Set passed=true only when the work genuinely satisfies the request.\n"
        "- Do not rewrite the work. Report problems; do not fix them.\n"
        "- Confidence is your certainty in your own verdict, not in the work."
    )

    async def run(self, payload: CriticInput, context: AgentContext) -> VerificationResult:
        """Verify the aggregated output.

        Args:
            payload: The request, its criteria, and the work to judge.
            context: Runtime context. Unused: verification is side-effect free.

        Returns:
            The critic's verdict.
        """
        return await self.complete_structured(self._render(payload), VerificationResult)

    def _render(self, payload: CriticInput) -> str:
        """Render the verification request."""
        criteria = "\n".join(f"- {item}" for item in payload.success_criteria) or "(none stated)"
        outputs = "\n\n".join(
            f"--- output {index + 1} ---\n{text}"
            for index, text in enumerate(payload.agent_outputs)
        )
        return (
            f"User request:\n{payload.user_request}\n\n"
            f"Plan objective:\n{payload.objective or '(not stated)'}\n\n"
            f"Success criteria:\n{criteria}\n\n"
            f"Work to verify:\n{outputs or '(nothing produced)'}\n\n"
            "Return your verdict."
        )
