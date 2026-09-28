"""Routing and planning schemas.

These are the contracts the router and planner produce. They are validated
strictly, because a malformed plan is worse than a rejected one: it would
dispatch work that cannot succeed.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "Complexity",
    "Plan",
    "Route",
    "RouteDecision",
    "Subtask",
]


class Route(StrEnum):
    """Where the router sends a request."""

    DIRECT = "direct"
    RESEARCH = "research"
    CODING = "coding"
    DATA_ANALYSIS = "data_analysis"
    DOCUMENT = "document"
    MULTI_AGENT = "multi_agent"
    HUMAN_APPROVAL = "human_approval"

    @property
    def needs_planning(self) -> bool:
        """Return whether this route goes through the planner."""
        return self not in {Route.DIRECT, Route.HUMAN_APPROVAL}


class Complexity(StrEnum):
    """How much work a request implies."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class RouteDecision(BaseModel):
    """The router's structured verdict.

    ``reasoning_summary`` is a short, safe justification suitable for showing to
    a user. It is not the model's internal reasoning.
    """

    route: Route
    complexity: Complexity = Complexity.SIMPLE
    intent: str = ""
    required_capabilities: list[str] = Field(default_factory=list)
    required_agents: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    requires_planning: bool = False
    requires_approval: bool = False
    reasoning_summary: str = ""

    @model_validator(mode="after")
    def _reconcile_flags(self) -> RouteDecision:
        """Force the derived flags to agree with the chosen route.

        A model cannot choose the approval route while leaving the approval flag
        off, nor claim a complex route needs no planning. Deriving these from the
        route rather than trusting the model removes a whole class of silent
        failure where the request is misclassified because one field was wrong.
        """
        self.requires_planning = self.route.needs_planning
        if self.route is Route.HUMAN_APPROVAL:
            self.requires_approval = True
        return self


class Subtask(BaseModel):
    """One unit of work assigned to one agent."""

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    agent: str = Field(min_length=1)
    dependencies: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    expected_output: str = ""
    success_criteria: str = ""


class Plan(BaseModel):
    """A validated plan for a complex request."""

    objective: str = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    subtasks: list[Subtask] = Field(min_length=1)
    dependencies: list[str] = Field(default_factory=list)
    required_agents: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_structure(self) -> Plan:
        """Reject plans that cannot be executed.

        Checks three things a planner can plausibly get wrong: duplicate
        subtask ids, dependencies pointing at subtasks that do not exist, and a
        dependency cycle.

        Raises:
            ValueError: If the plan is not dispatchable.
        """
        ids = [subtask.id for subtask in self.subtasks]
        if len(ids) != len(set(ids)):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate subtask ids: {duplicates}")

        known = set(ids)
        for subtask in self.subtasks:
            unknown = [dep for dep in subtask.dependencies if dep not in known]
            if unknown:
                raise ValueError(f"subtask {subtask.id!r} depends on unknown {unknown}")

        self._assert_acyclic()
        return self

    def _assert_acyclic(self) -> None:
        """Raise if the dependency graph contains a cycle."""
        graph = {subtask.id: list(subtask.dependencies) for subtask in self.subtasks}
        state: dict[str, int] = {}

        def visit(node: str) -> None:
            colour = state.get(node, 0)
            if colour == 1:
                raise ValueError(f"dependency cycle detected at subtask {node!r}")
            if colour == 2:
                return
            state[node] = 1
            for neighbour in graph.get(node, ()):
                visit(neighbour)
            state[node] = 2

        for node in graph:
            visit(node)

    def ready_subtasks(self, completed: set[str]) -> list[Subtask]:
        """Return subtasks whose dependencies are all satisfied.

        Args:
            completed: Ids of subtasks that have finished.

        Returns:
            Subtasks that are runnable now and not yet completed, in plan order.
        """
        return [
            subtask
            for subtask in self.subtasks
            if subtask.id not in completed
            and all(dependency in completed for dependency in subtask.dependencies)
        ]

    @property
    def subtask_ids(self) -> list[str]:
        """Return every subtask id in plan order."""
        return [subtask.id for subtask in self.subtasks]
