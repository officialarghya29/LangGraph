"""Intent router.

Routing decisions come from validated structured output, never from
string-matching a model's prose. Free-form text parsing is how routers acquire
silent failure modes: a reworded response quietly changes the route.

The router classifies; it does not act. It is given no tools.
"""

from __future__ import annotations

from app.core.exceptions import StructuredOutputError
from app.schemas.plans import Complexity, Route, RouteDecision
from app.services.llm import LLMProvider, Message, Role

__all__ = ["IntentRouter"]


class IntentRouter:
    """Classifies a request into a route, a complexity, and a capability set."""

    #: Routes that may be selected. Used to constrain and to validate output.
    ROUTES: tuple[Route, ...] = tuple(Route)

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    @property
    def system_prompt(self) -> str:
        """Return the routing instruction."""
        routes = "\n".join(f"- {route.value}" for route in self.ROUTES)
        return (
            "You route requests. Choose exactly one route.\n\n"
            f"Available routes:\n{routes}\n\n"
            "Guidance:\n"
            "- direct: a trivial request answerable without tools or planning. "
            "Greetings, definitions, short transformations.\n"
            "- research: needs external evidence or sources.\n"
            "- coding: produces, reviews, or debugs code.\n"
            "- data_analysis: computes over a dataset or reasons about numbers.\n"
            "- document: extracts or generates a structured document.\n"
            "- multi_agent: genuinely needs several of the above working together.\n"
            "- human_approval: the request asks for a destructive or otherwise "
            "irreversible action.\n\n"
            "Rules:\n"
            "- Prefer the simplest route that satisfies the request. Do not "
            "escalate a trivial request to multi_agent.\n"
            "- Set complexity to reflect the real work, not surface wording.\n"
            "- List only capabilities the request actually needs.\n"
            "- Set requires_approval when the request would delete data, drop a "
            "table, or otherwise act irreversibly.\n"
            "- reasoning_summary is a single short sentence a user may read. It "
            "is not a place to think out loud."
        )

    async def route(self, user_request: str) -> RouteDecision:
        """Classify a request.

        Args:
            user_request: The raw request text.

        Returns:
            A validated routing decision.

        Raises:
            StructuredOutputError: If the model cannot produce a valid decision.
        """
        decision = await self._provider.astructured_output(
            [
                Message(role=Role.SYSTEM, content=self.system_prompt),
                Message(
                    role=Role.USER, content=f"Request:\n{user_request}\n\nReturn the decision."
                ),
            ],
            RouteDecision,
        )

        # The planner flag is derived from the route by the schema validator, so
        # the only reconciliation left is complexity versus route.
        if decision.route is Route.DIRECT:
            decision.complexity = Complexity.SIMPLE
        return decision


def fallback_decision(reason: str) -> RouteDecision:
    """Return the decision used when routing itself fails.

    Fails closed onto the cheapest safe path: a direct response rather than an
    unplanned multi-agent run, and an explicit note of why.

    Args:
        reason: Short, client-safe explanation.

    Returns:
        A direct-route decision.
    """
    return RouteDecision(
        route=Route.DIRECT,
        complexity=Complexity.SIMPLE,
        intent="unrouted",
        reasoning_summary=f"routing unavailable, answering directly ({reason})",
    )


__all__ += ["StructuredOutputError", "fallback_decision"]
