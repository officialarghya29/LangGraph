"""Retrying model calls with the shared retry policy.

The classification table, the backoff curve, and the retry loop all existed and
were all tested, and none of them was connected to anything: no caller in the
application invoked :func:`~app.services.execution.retry_async`, so a transient
429 or a timeout failed the whole run while a retry policy sat beside it unused.
The documentation said retries happened. They did not.

This module is the missing wire. It is a decorator, so it composes with any
provider, and it is applied *outside* the budget decorator:

    RetryingProvider(BudgetedProvider(inner))

With the budget inside, the allowance is checked between attempts rather than
once per logical call, so a run whose budget is spent stops retrying instead of
finishing the loop and discovering the problem afterwards.

What that ordering does *not* do is change the totals. An attempt that raises
never yields usage, so it is never billed — a failed HTTP call costs nothing to
account for, whatever order the decorators are in. Only a request that came back
carries a count, and each of those is charged once.

Composition with the graph's own retries is intentional and bounded: a
verification failure re-runs an agent, and each of those model calls may itself
be retried. The ceilings multiply rather than nest, which is worth knowing and is
why the attempt counts are reported to the caller's callback.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from app.core.constants import FailureKind
from app.services.execution import RetryExhaustedError, RetryPolicy, retry_async
from app.services.llm import LLMProvider, LLMResponse, Message

__all__ = ["RetryingProvider"]

#: Called after each failed attempt, with the 1-based attempt number, the
#: classified kind, and the exception. Used to count retries without this module
#: having to know about metrics.
RetryObserver = Callable[[int, FailureKind, BaseException], None]


class RetryingProvider(LLMProvider):
    """A provider that retries failures the classification says are retryable.

    A permanent failure — an invalid key, a validation error, a spent budget —
    fails on the first attempt. Retrying those changes nothing and spends the
    caller's time to prove it.
    """

    def __init__(
        self,
        inner: LLMProvider,
        *,
        max_retries: int = 3,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        on_retry: RetryObserver | None = None,
    ) -> None:
        """Wrap ``inner``.

        Args:
            inner: The provider to call.
            max_retries: How many *additional* attempts a retryable failure earns.
            sleep: Injectable sleep, so tests do not wait out real backoff.
            on_retry: Optional observer, called once per failed attempt.
        """
        super().__init__(default_model=inner.default_model)
        self.inner = inner
        self.max_retries = max_retries
        self._sleep = sleep
        self._on_retry = on_retry
        self.name = f"retrying:{inner.name}"

    @property
    def policy(self) -> RetryPolicy:
        """Return the retry policy applied to every call."""
        return RetryPolicy(max_retries=self.max_retries)

    async def ainvoke(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Call the provider, retrying failures worth retrying.

        The original exception is re-raised once the budget is spent, rather than
        the retry loop's own wrapper type: callers classify what actually went
        wrong, and ``LLMTimeoutError`` says more than "exhausted" does. The
        exhausted wrapper stays attached as the chain's cause for the logs.

        Returns:
            The provider's response from the first attempt that succeeds.

        Raises:
            Exception: The last failure, if every permitted attempt failed.
        """

        async def attempt() -> LLMResponse:
            return await self.inner.ainvoke(
                messages, model=model, temperature=temperature, max_tokens=max_tokens
            )

        try:
            return await retry_async(
                attempt,
                policy=self.policy,
                on_retry=self._on_retry,
                sleep=self._sleep,
            )
        except RetryExhaustedError as exc:
            raise exc.cause from exc
