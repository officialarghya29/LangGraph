"""Coding agent.

Generates, analyses, and debugs code. It is given filesystem access, which is
not write-safe on its own, but the filesystem tool is confined to an allow-listed
root and any write is written-through as a gated, audited operation.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from app.agents.base import AgentContext, BaseAgent
from app.models.agent import AgentOutput

__all__ = ["CodingAgent", "CodingInput"]


class CodingInput(BaseModel):
    """A coding assignment."""

    subtask_id: str | None = None
    description: str = Field(min_length=1)
    context: str = ""
    language: str = ""
    expected_output: str = ""


class CodingAgent(BaseAgent[CodingInput, AgentOutput]):
    """Produces, reviews, and debugs code."""

    name = "coder"
    description = "Generates code, debugs failures, and analyses implementations"
    allowed_tools: ClassVar[tuple[str, ...]] = ("filesystem",)
    input_model: ClassVar[type[BaseModel]] = CodingInput
    output_model: ClassVar[type[BaseModel]] = AgentOutput

    system_prompt = (
        "You are a coding agent. Produce correct, minimal code that solves the "
        "assigned problem.\n"
        "Rules:\n"
        "- Match the conventions of the surrounding code. Do not introduce a "
        "dependency the project does not already use.\n"
        "- Handle the error cases that can actually occur. Do not add "
        "speculative abstraction.\n"
        "- Never invent an API or a library function that may not exist.\n"
        "- When debugging, identify the root cause rather than patching the "
        "symptom.\n"
        "- State any assumption the code depends on.\n"
        "- Set confidence to your genuine certainty that the code is correct."
    )

    async def run(self, payload: CodingInput, context: AgentContext) -> AgentOutput:
        """Complete the coding assignment.

        Args:
            payload: The coding assignment.
            context: Runtime context.

        Returns:
            The proposed implementation with assumptions noted.
        """
        prompt = (
            f"Coding assignment:\n{payload.description}\n\n"
            f"Language or stack:\n{payload.language or '(not specified)'}\n\n"
            f"Context from earlier steps:\n{payload.context or '(none)'}\n\n"
            f"Expected output:\n"
            f"{payload.expected_output or '(working code with a brief rationale)'}"
        )
        result = await self.complete_structured(prompt, AgentOutput)
        result.agent = self.name
        result.subtask_id = payload.subtask_id
        return result
