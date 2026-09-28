"""The chat endpoint.

Runs the graph and returns the answer in the same response. Suitable for
interactive use where the caller is waiting; long or parallel work should go
through ``POST /api/v1/tasks`` instead, which returns immediately.

The response contains the answer, the route taken, and whether a human is
needed. It contains no reasoning, no prompt, and no tool arguments.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.api.dependencies import GraphDep, PrincipalDep, SettingsDep, require_configured
from app.core.exceptions import AppError
from app.graph.checkpoints import thread_config
from app.graph.state import initial_state
from app.models.approval import ApprovalStatus
from app.services.budget import budget_for, token_budget
from app.services.limits import limits_for, run_limits

router = APIRouter(prefix="/api/v1", tags=["chat"])

#: Cap the request size here as well as in the schema, so an oversized body is
#: rejected before it reaches the graph.
MAX_MESSAGE_LENGTH = 20_000


class ChatRequest(BaseModel):
    """A conversational request."""

    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    """The result of a chat turn."""

    task_id: str
    answer: str
    route: str | None = None
    requires_approval: bool = False
    approval_status: str | None = None


@router.post("/chat", response_model=ChatResponse, summary="Run a request to completion")
async def chat(
    payload: ChatRequest,
    graph: GraphDep,
    settings: SettingsDep,
    principal: PrincipalDep,
) -> ChatResponse:
    """Answer a request synchronously.

    Args:
        payload: The message and optional conversation to continue.
        graph: The compiled orchestration graph.
        settings: Application settings, for the execution ceilings.
        principal: The resolved caller, whose identity scopes the run.

    Returns:
        The answer and coarse execution metadata.

    Raises:
        HTTPException: 503 if the provider is not configured, 500 if the run
            fails for any other reason.
    """
    require_configured(settings)

    state = initial_state(
        payload.message,
        user_id=principal.user_id,
        conversation_id=payload.conversation_id,
        iteration_limit=settings.max_agent_iterations,
        retry_limit=settings.max_retries,
    )

    try:
        # The same allowances the asynchronous path enforces: one token budget, and
        # one set of execution ceilings. A limit that applied to one of two entry
        # points would not be a limit.
        budget = budget_for(state, limit=settings.max_token_budget)
        with token_budget(budget), run_limits(limits_for(settings)):
            async with asyncio.timeout(settings.max_execution_time):
                result: dict[str, Any] = await graph.ainvoke(state, thread_config(state["task_id"]))
    except AppError:
        # A deliberate refusal — a spent token budget, a spent tool-call
        # allowance, a provider rejection, an input the graph rejected — keeps its
        # own meaning. Converting it to a 500 here would hide a 429 behind a
        # server error and tell the caller to retry something that will fail
        # identically.
        raise
    except TimeoutError as exc:
        # The wall-clock ceiling, which is not a caller error. 504 rather than 500:
        # the server gave up on work it had accepted, and the caller must not be
        # told their request was malformed.
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(f"the run exceeded its {settings.max_execution_time} second execution ceiling"),
        ) from exc
    except Exception as exc:
        # The graph is designed not to raise, so this is genuinely unexpected.
        # The class name only: a message could carry a prompt fragment or a URL.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"the run failed with {type(exc).__name__}",
        ) from exc

    decision = result.get("route")
    approval_status = result.get("approval_status") or ApprovalStatus.NOT_REQUIRED

    return ChatResponse(
        task_id=state["task_id"],
        answer=result.get("final_answer") or "",
        route=decision.route.value if decision is not None else None,
        requires_approval=bool(result.get("requires_human_approval")),
        approval_status=ApprovalStatus(approval_status).value,
    )
