"""Per-run token accounting and budget enforcement.

Every model call returns a token count, and before this existed those counts were
parsed, carried on the response, and then dropped. The task record has
``prompt_tokens`` and ``completion_tokens`` columns that were therefore always
zero, and ``max_token_budget`` was a setting that bounded nothing — the exact
failure this project claims not to have, where a configured limit is mistaken for
an enforced one.

Two pieces, deliberately separate:

- :class:`TokenBudget` is the accounting. It accumulates usage and knows when the
  allowance is spent.
- :class:`BudgetedProvider` is the enforcement. It wraps any provider, so every
  call is counted — including the router's and the critic's, which are easy to
  forget when counting is done by hand at each call site.

The budget travels in a :class:`~contextvars.ContextVar` rather than on the
provider, for the same reason the event sink does: the compiled graph is built
once and invoked concurrently, so anything stored on it would pool two runs'
token counts into one number. The context is set per run, next to the state whose
metadata it writes into.

**Exhaustion is checked before a call, not after.** A call that has already been
paid for should not have its result thrown away: overshooting by at most one call
is the honest behaviour, and the alternative discards a completion that the
provider has already billed. What is prevented is starting another one.

The same call is charged twice over, to two different questions: the run's total,
and the invoking agent's own figure. The second one needs to know *which* agent is
spending, which is what :func:`agent_scope` establishes — and it is established in
a context variable for the same concurrency reason as the budget itself.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from app.core.exceptions import UsageLimitExceededError
from app.models.execution import TokenUsageSummary
from app.services.llm import LLMProvider, LLMResponse, Message

__all__ = [
    "BudgetedProvider",
    "TokenBudget",
    "agent_scope",
    "budget_for",
    "current_budget",
    "token_budget",
]


class TokenBudget:
    """A token allowance for one run, and the usage recorded against it."""

    def __init__(self, *, limit: int, usage: TokenUsageSummary | None = None) -> None:
        """Create a budget.

        Args:
            limit: The total tokens the run may spend. At least one.
            usage: The accumulator to record into. Pass the run's own
                ``ExecutionMetadata.usage`` so the totals reach the task record;
                omit it to keep the accounting private.

        Raises:
            ValueError: If ``limit`` is less than one. A zero budget would refuse
                the first call, which is indistinguishable from a broken
                provider, so it is rejected rather than accepted silently.
        """
        if limit < 1:
            raise ValueError("a token budget must allow at least one token")
        self.limit = limit
        self.usage = usage if usage is not None else TokenUsageSummary()
        #: Calls counted so far. Useful for explaining a budget failure: "eight
        #: calls, 104,000 tokens" is actionable, "over budget" is not.
        self.calls = 0

    @property
    def spent(self) -> int:
        """Return the total tokens recorded so far."""
        return self.usage.total_tokens

    @property
    def remaining(self) -> int:
        """Return how many tokens are left, never below zero."""
        return max(0, self.limit - self.spent)

    @property
    def exhausted(self) -> bool:
        """Return whether the allowance is spent."""
        return self.spent >= self.limit

    def record(self, *, prompt: int, completion: int) -> None:
        """Accumulate one call's usage, against the run and against its agent.

        Args:
            prompt: Tokens sent.
            completion: Tokens generated.
        """
        self.calls += 1
        sent, generated = max(0, prompt), max(0, completion)
        self.usage.add(prompt=sent, completion=generated)
        # Charged to the invoking agent as well, when there is one in scope. The
        # run total and the per-agent figure are recorded from the same call, so
        # they cannot disagree.
        scope = _CURRENT_AGENT.get()
        if scope is not None:
            scope[1].add(prompt=sent, completion=generated)

    def check(self) -> None:
        """Refuse further work if the allowance is spent.

        Raises:
            UsageLimitExceededError: If nothing is left.
        """
        if self.exhausted:
            raise UsageLimitExceededError(
                "the run's token budget is spent",
                detail=f"{self.spent} of {self.limit} tokens after {self.calls} call(s)",
            )

    def summary(self) -> str:
        """Return a one-line description of the budget's state."""
        return (
            f"{self.spent}/{self.limit} tokens over {self.calls} call(s), "
            f"{self.remaining} remaining"
        )


#: The budget in scope for the current run, if any. ``None`` means unbounded,
#: which is the behaviour outside a graph run — a direct call to a provider from
#: a script should not need a budget to be arranged first.
_CURRENT: ContextVar[TokenBudget | None] = ContextVar("token_budget", default=None)

#: The agent currently spending in this context, paired with the accumulator for
#: its current invocation. One tuple rather than two variables, so the name and
#: its usage can never be restored out of step with each other.
_CURRENT_AGENT: ContextVar[tuple[str, TokenUsageSummary] | None] = ContextVar(
    "current_agent", default=None
)


@contextmanager
def agent_scope(agent: str) -> Iterator[TokenUsageSummary]:
    """Attribute every model call inside the block to ``agent``.

    Yields a *fresh* accumulator rather than one per agent name, so an agent that
    runs twice — a subtask the critic sends back, for example — reports each
    invocation's own cost instead of a running total that makes the second run
    look twice as expensive as it was.

    The scope travels in a context variable, which is what makes the attribution
    correct under concurrency: ``asyncio.gather`` runs each coroutine in a task
    with its own copy of the context, so four agents dispatching at once cannot
    see, charge, or credit each other's calls. A shared counter around the outside
    of the dispatch loop would measure whatever finished in the window.

    Args:
        agent: The agent name to charge.

    Yields:
        The usage this invocation accumulates.
    """
    usage = TokenUsageSummary()
    token = _CURRENT_AGENT.set((agent, usage))
    try:
        yield usage
    finally:
        _CURRENT_AGENT.reset(token)


def budget_for(state: Mapping[str, Any], limit: int) -> TokenBudget:
    """Return a budget bound to a run's own metadata, sized from settings.

    Both entry points into the graph — the asynchronous task path and the
    synchronous chat path — call this, so a limit configured for one cannot go
    unenforced in the other. Binding it to ``state["execution_metadata"].usage``
    is what carries the totals back out to the caller and into the task record.

    Args:
        state: The graph state the run was built from.
        limit: The configured allowance, in tokens.

    Returns:
        A budget that writes into that state's usage summary.
    """
    metadata = state["execution_metadata"]
    return TokenBudget(limit=limit, usage=metadata.usage)


def current_budget() -> TokenBudget | None:
    """Return the budget in scope for this context, or ``None``.

    Returns:
        The active budget.
    """
    return _CURRENT.get()


@contextmanager
def token_budget(budget: TokenBudget) -> Iterator[TokenBudget]:
    """Scope ``budget`` to the current run.

    Args:
        budget: The budget every provider call in this context is charged to.

    Yields:
        The budget, so the caller can report on it afterwards.
    """
    token = _CURRENT.set(budget)
    try:
        yield budget
    finally:
        _CURRENT.reset(token)


class BudgetedProvider(LLMProvider):
    """A provider that charges every call to the run's token budget.

    A decorator rather than a base class, so it composes with any provider —
    hosted, self-hosted, or the deterministic fake — and so the accounting cannot
    be forgotten by a provider author who never read this module.
    """

    def __init__(self, inner: LLMProvider) -> None:
        """Wrap ``inner``.

        Args:
            inner: The provider to delegate to.
        """
        super().__init__(
            default_model=inner.default_model,
            # Read leniently, as in the retry decorator: a provider that declares
            # neither still composes, and neither is consulted for a decision.
            default_temperature=getattr(inner, "default_temperature", 0.0),
            default_max_tokens=getattr(inner, "default_max_tokens", None),
        )
        self.inner = inner
        # Instance attribute shadowing the class default, so a log line names the
        # real provider rather than "provider".
        self.name = f"budgeted:{inner.name}"

    async def ainvoke(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Charge the call to the budget, then delegate.

        Returns:
            The provider's response.

        Raises:
            UsageLimitExceededError: If the budget is already spent.
        """
        budget = _CURRENT.get()
        if budget is not None:
            budget.check()

        response = await self.inner.ainvoke(
            messages, model=model, temperature=temperature, max_tokens=max_tokens
        )

        if budget is not None:
            budget.record(
                prompt=response.usage.prompt_tokens,
                completion=response.usage.completion_tokens,
            )
        return response
