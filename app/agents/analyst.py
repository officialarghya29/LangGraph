"""Data analysis agent.

Inspects datasets, performs calculations, and reports structured results. Its
tools are computation and read-only database access.

The Python execution tool is disabled by default in configuration, because
arbitrary code execution without a real isolation boundary is not safe. When
execution is unavailable the agent must still be able to reason about the data
it is given, and must say so rather than pretend it computed something.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from app.agents.base import UNTRUSTED_CONTENT_RULE, AgentContext, BaseAgent
from app.models.agent import AgentOutput

__all__ = ["AnalysisInput", "DataAnalysisAgent"]


class AnalysisInput(BaseModel):
    """An analysis assignment."""

    subtask_id: str | None = None
    description: str = Field(min_length=1)
    data: str = ""
    context: str = ""
    expected_output: str = ""


class DataAnalysisAgent(BaseAgent[AnalysisInput, AgentOutput]):
    """Analyses data and reports computed results."""

    name = "analyst"
    description = "Inspects datasets, performs calculations, and reports structured results"
    #: Computation and read-only data access. No write path exists.
    allowed_tools: ClassVar[tuple[str, ...]] = ("python_executor",)
    #: Available only when a query executor is wired up.
    optional_tools: ClassVar[tuple[str, ...]] = ("database",)
    input_model: ClassVar[type[BaseModel]] = AnalysisInput
    output_model: ClassVar[type[BaseModel]] = AgentOutput

    system_prompt = (
        "You are a data analysis agent. Inspect the data and report what it "
        "actually shows.\n"
        "Rules:\n"
        "- Never invent a number. If you did not compute a value, do not state it.\n"
        "- If you cannot run a calculation, say so and report only what the raw "
        "data supports.\n"
        "- State the method behind every derived figure so it can be checked.\n"
        "- Distinguish correlation from causation explicitly.\n"
        "- Report the sample size and any obvious selection bias.\n"
        "- Give the uncertainty, not just a point estimate.\n"
        "- Set confidence to your genuine certainty in the result.\n"
        f"- {UNTRUSTED_CONTENT_RULE}"
    )

    async def run(self, payload: AnalysisInput, context: AgentContext) -> AgentOutput:
        """Perform the analysis.

        Args:
            payload: The analysis assignment and its data.
            context: Runtime context.

        Returns:
            Computed results with the method behind each figure.
        """
        prompt = (
            f"Analysis assignment:\n{payload.description}\n\n"
            f"Data:\n{payload.data or '(no data supplied)'}\n\n"
            f"Context from earlier steps:\n{payload.context or '(none)'}\n\n"
            f"Expected output:\n"
            f"{payload.expected_output or '(figures with the method and uncertainty for each)'}"
        )
        result = await self.complete_structured(prompt, AgentOutput)
        result.agent = self.name
        result.subtask_id = payload.subtask_id
        return result
