"""Executor agent.

Performs actions that change the world. It is only ever invoked after the risk
check has decided that approval is required and a human has granted it, so the
call it makes is explicit and already authorised.

The executor decides nothing about risk. It runs exactly the action it was
handed, and reports what happened.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from app.agents.base import AgentContext, BaseAgent

__all__ = ["ExecutionOutput", "ExecutorAgent", "ExecutorInput"]


class ExecutorInput(BaseModel):
    """A concrete, already-approved action to perform."""

    action: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    arguments: dict[str, object] = Field(default_factory=dict)
    rationale: str = ""


class ExecutionOutput(BaseModel):
    """The outcome of an approved action."""

    performed: bool
    summary: str = Field(min_length=1)
    tool: str
    error: str | None = None
    #: Set when the tool refused because no human had approved the action. Lets
    #: the caller distinguish "needs approval" from "genuinely failed".
    approval_required: bool = False


class ExecutorAgent(BaseAgent[ExecutorInput, ExecutionOutput]):
    """Carries out an approved action through the tool pipeline."""

    name = "executor"
    description = "Executes approved actions through authorised tools"
    #: The action-capable tools. The write and execute tools are gated by the
    #: approval check in the tool pipeline; the read tools are here so the
    #: executor can confirm what it is about to change.
    allowed_tools: ClassVar[tuple[str, ...]] = (
        "read_file",
        "list_directory",
        "write_file",
        "python_executor",
    )
    #: Available only when the corresponding credential or store is configured.
    optional_tools: ClassVar[tuple[str, ...]] = (
        "database",
        "github_repository",
        "github_create_issue",
    )
    input_model: ClassVar[type[BaseModel]] = ExecutorInput
    output_model: ClassVar[type[BaseModel]] = ExecutionOutput

    system_prompt = (
        "You are an execution agent. Carry out the approved action exactly as "
        "specified. Do not extend it, generalise it, or perform additional work."
    )

    async def run(self, payload: ExecutorInput, context: AgentContext) -> ExecutionOutput:
        """Run the approved action.

        Args:
            payload: The action and the tool that performs it.
            context: Runtime context. ``approved`` is honoured, so an unapproved
                call is refused by the tool pipeline rather than by this agent.

        Returns:
            Whether the action was performed, and a summary.
        """
        result = await self.call_tool(
            payload.tool,
            payload.arguments,
            context,
            approved=context.approved,
        )

        if not result.ok:
            return ExecutionOutput(
                performed=False,
                summary=f"action not performed: {result.error}",
                tool=payload.tool,
                error=result.error,
                approval_required=result.approval_required,
            )

        return ExecutionOutput(
            performed=True,
            summary=f"{payload.action} completed via {payload.tool}",
            tool=payload.tool,
        )
