"""Least-privilege tool registry.

An agent never sees the whole toolbox. It declares the tools it needs, and the
registry resolves exactly that set. Names that do not exist raise rather than
being ignored, because a typo in an agent's tool list should fail loudly at
wiring time instead of silently removing a capability.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from app.core.exceptions import NotFoundError
from app.models.tool import RiskLevel
from app.tools.base import Tool

__all__ = ["AnyTool", "ToolRegistry"]

#: Tools are heterogeneous in their input and output models, so the registry
#: stores them at their most general type.
AnyTool = Tool[Any, Any]


class ToolRegistry:
    """A registry of available tools."""

    def __init__(self, tools: Iterable[AnyTool] = ()) -> None:
        self._tools: dict[str, AnyTool] = {}
        for tool in tools:
            self.register(tool)

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def register(self, tool: AnyTool) -> None:
        """Add a tool to the registry.

        Args:
            tool: The tool instance to register.

        Raises:
            ValueError: If the tool has no name, or the name is already taken.
        """
        name = tool.name
        if not name:
            raise ValueError("a tool must declare a non-empty name")
        if name in self._tools:
            raise ValueError(f"tool {name!r} is already registered")
        self._tools[name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool.

        Raises:
            NotFoundError: If no such tool is registered.
        """
        if name not in self._tools:
            raise NotFoundError(f"tool {name!r} is not registered")
        del self._tools[name]

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #

    def get(self, name: str) -> AnyTool:
        """Return a tool by name.

        Raises:
            NotFoundError: If no such tool is registered.
        """
        try:
            return self._tools[name]
        except KeyError:
            raise NotFoundError(f"tool {name!r} is not registered") from None

    def try_get(self, name: str) -> AnyTool | None:
        """Return a tool by name, or ``None`` if it is absent."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Return whether a tool is registered."""
        return name in self._tools

    def list(self) -> tuple[AnyTool, ...]:
        """Return every registered tool, ordered by name."""
        return tuple(self._tools[name] for name in sorted(self._tools))

    def names(self) -> tuple[str, ...]:
        """Return every registered tool name, ordered."""
        return tuple(sorted(self._tools))

    def __len__(self) -> int:
        """Return how many tools are registered."""
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        """Return whether a name is registered."""
        return isinstance(name, str) and name in self._tools

    # ------------------------------------------------------------------ #
    # Least privilege
    # ------------------------------------------------------------------ #

    def get_allowed_tools(self, allowed: Sequence[str]) -> tuple[AnyTool, ...]:
        """Resolve an explicit allow-list of tool names.

        This is the only way an agent should obtain tools. Anything not named is
        invisible to it.

        Args:
            allowed: Tool names the caller is permitted to use.

        Returns:
            The resolved tools, in the order given.

        Raises:
            NotFoundError: If any name is not registered.
        """
        unknown = [name for name in allowed if name not in self._tools]
        if unknown:
            raise NotFoundError(f"unknown tools requested: {sorted(set(unknown))}")
        return tuple(self._tools[name] for name in allowed)

    def filter_by_max_risk(self, maximum: RiskLevel) -> tuple[AnyTool, ...]:
        """Return tools whose effective risk does not exceed ``maximum``.

        Useful for restricting a whole agent or role to safe operations without
        enumerating every tool by name.
        """
        return tuple(tool for tool in self.list() if tool.effective_risk() <= maximum)

    def describe(self) -> Sequence[dict[str, Any]]:
        """Return client-safe descriptions of every registered tool.

        Annotated as ``Sequence`` rather than ``list`` because the ``list``
        method above shadows the builtin inside this class body.
        """
        return [tool.describe() for tool in self.list()]
