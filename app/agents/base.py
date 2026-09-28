"""Base agent contract.

Agents operate through controlled interfaces only. They do not touch the
database, they do not manipulate graph internals, and they cannot reach a tool
that is not on their own allow-list.

Each agent declares:

- ``name`` and ``description`` for discovery and the audit trail;
- ``system_prompt`` for its role;
- ``allowed_tools`` for least-privilege access;
- ``input_model`` / ``output_model`` for a typed, validated contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel

from app.core.config import Settings
from app.core.exceptions import ToolPermissionError
from app.services.llm import LLMProvider, Message, Role
from app.tools.base import ToolContext, ToolRequest, ToolResult
from app.tools.registry import AnyTool, ToolRegistry

__all__ = ["AgentContext", "BaseAgent"]


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Runtime context handed to an agent for one invocation."""

    settings: Settings
    user_id: str | None = None
    task_id: str | None = None
    conversation_id: str | None = None
    approved: bool = False


class BaseAgent[In: BaseModel, Out: BaseModel](ABC):
    """Base class for every agent, generic over its input and output models."""

    #: Stable agent name used in routing, events, and audit records.
    name: ClassVar[str] = ""
    #: One-line description shown to the router and to operators.
    description: ClassVar[str] = ""
    #: Role instruction sent as the system message.
    system_prompt: ClassVar[str] = ""
    #: Exactly the tools this agent may use. Nothing else is visible to it.
    allowed_tools: ClassVar[tuple[str, ...]] = ()

    #: Declared ``ClassVar`` so subclasses can assign concrete models in the
    #: class body. The generic parameters are used by ``run``; the schemas
    #: themselves are only ever introspected, never constructed from ``Out``.
    input_model: ClassVar[type[BaseModel]]
    output_model: ClassVar[type[BaseModel]]

    def __init__(self, provider: LLMProvider, registry: ToolRegistry) -> None:
        self._provider = provider
        self._registry = registry

    @property
    def provider(self) -> LLMProvider:
        """Return the LLM provider this agent reasons with."""
        return self._provider

    def describe(self) -> dict[str, object]:
        """Return a client-safe description of the agent."""
        return {
            "name": self.name,
            "description": self.description,
            "tools": list(self.tool_names),
            "input_schema": self.input_model.model_json_schema(),
            "output_schema": self.output_model.model_json_schema(),
        }

    # ------------------------------------------------------------------ #
    # Least-privilege tool access
    # ------------------------------------------------------------------ #

    def tools(self) -> tuple[AnyTool, ...]:
        """Resolve this agent's tool set from the registry.

        Raises:
            NotFoundError: If an allowed tool is not registered. A missing
                dependency is a wiring bug and must surface immediately.
        """
        return self._registry.get_allowed_tools(self.allowed_tools)

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Return the names of the tools this agent may use."""
        return tuple(tool.name for tool in self.tools())

    def allows_tool(self, name: str) -> bool:
        """Return whether this agent is authorised to use a tool."""
        return name in self.allowed_tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, object],
        context: AgentContext,
        *,
        approved: bool = False,
    ) -> ToolResult:
        """Run an authorised tool through the full security pipeline.

        Args:
            tool_name: Tool to run. Must be on this agent's allow-list.
            arguments: Raw arguments, validated by the tool.
            context: The agent's runtime context.
            approved: Whether a human has approved a gated action.

        Returns:
            The tool result, including failure results.

        Raises:
            ToolPermissionError: If the tool is not on this agent's allow-list.
        """
        if not self.allows_tool(tool_name):
            raise ToolPermissionError(f"agent {self.name!r} is not authorised to use {tool_name!r}")

        tool = self._registry.get(tool_name)
        tool_context = ToolContext(
            settings=context.settings,
            user_id=context.user_id,
            task_id=context.task_id,
            agent=self.name,
            approved=approved,
        )
        request = ToolRequest(
            tool=tool_name,
            arguments=arguments,
            task_id=context.task_id,
            agent=self.name,
            user_id=context.user_id,
        )
        return await tool.execute(request, tool_context)

    # ------------------------------------------------------------------ #
    # Reasoning
    # ------------------------------------------------------------------ #

    def build_messages(self, prompt: str) -> list[Message]:
        """Build the system-plus-user message pair for a call."""
        return [
            Message(role=Role.SYSTEM, content=self.system_prompt),
            Message(role=Role.USER, content=prompt),
        ]

    async def complete_structured(self, prompt: str, schema: type[Out]) -> Out:
        """Ask the model for output conforming to ``schema``.

        Raises:
            StructuredOutputError: If the model cannot produce valid output
                within the provider's attempt budget.
        """
        return await self._provider.astructured_output(self.build_messages(prompt), schema)

    async def complete_text(self, prompt: str) -> str:
        """Ask the model for free-form text."""
        response = await self._provider.ainvoke(self.build_messages(prompt))
        return response.content

    # ------------------------------------------------------------------ #
    # Contract
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def run(self, payload: In, context: AgentContext) -> Out:
        """Perform this agent's work."""
