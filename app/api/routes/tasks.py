"""Task endpoints.

``POST /api/v1/tasks`` returns immediately with a task id and runs the graph in
the background. ``GET`` endpoints report progress. The approval endpoints resume
a suspended graph with a human's decision.

Every handler takes a :class:`~app.core.auth.Principal` rather than reading an
identity header itself. Identity is resolved in exactly one place, so a handler
cannot accidentally trust a caller-supplied value.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError

from app.api.dependencies import (
    GraphDep,
    MetricsDep,
    PrincipalDep,
    SettingsDep,
    TaskStoreDep,
    TracerDep,
    require_configured,
)
from app.core.exceptions import DatabaseError
from app.graph.checkpoints import thread_config
from app.graph.nodes import event_sink_for
from app.models.approval import ApprovalDecision, ApprovalStatus
from app.models.execution import TaskStatus
from app.services.limits import limits_for, run_limits
from app.services.task_store import TaskRecord, execute_task_with_store, task_event_sink

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])

#: Mirrors the request schema, so an oversized body is rejected before the graph.
MAX_REQUEST_LENGTH = 20_000


class CreateTaskRequest(BaseModel):
    """A task to run asynchronously."""

    request: str = Field(min_length=1, max_length=MAX_REQUEST_LENGTH)
    conversation_id: str | None = Field(default=None, max_length=64)


class TaskResponse(BaseModel):
    """The client-visible state of a task.

    Carries no prompt beyond the request the caller supplied, no reasoning, and no
    tool arguments.
    """

    task_id: str
    request: str
    status: str
    answer: str | None = None
    failure_reason: str | None = None
    route: str | None = None
    approval_status: str
    created_at: str
    updated_at: str

    @classmethod
    def from_record(cls, record: TaskRecord) -> TaskResponse:
        """Build a response from a stored record."""
        return cls(
            task_id=record.task_id,
            request=record.request,
            status=record.status.value,
            answer=record.answer,
            failure_reason=record.failure_reason,
            route=record.route,
            approval_status=record.approval_status.value,
            created_at=record.created_at.isoformat(),
            updated_at=record.updated_at.isoformat(),
        )


class TaskStatusResponse(BaseModel):
    """A minimal status payload."""

    task_id: str
    status: str
    approval_status: str
    has_answer: bool


class ApprovalRequest(BaseModel):
    """A human's decision on a suspended task."""

    note: str | None = Field(default=None, max_length=1000)


class ApprovalResponse(BaseModel):
    """The outcome of recording a decision."""

    task_id: str
    approval_status: str
    status: str
    answer: str | None = None


async def _owned_task(store: TaskStoreDep, task_id: str, user_id: str) -> TaskRecord:
    """Fetch a task, enforcing ownership.

    Raises:
        HTTPException: 404, whether the task is absent or owned by somebody else.
            Distinguishing the two would confirm that another user's task exists.
    """
    record = await store.get_for_user(task_id, user_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such task")
    return record


def _storage_unavailable(exc: Exception) -> HTTPException:
    """Translate a storage failure into a service-unavailable response."""
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"task storage is unavailable ({type(exc).__name__})",
    )


@router.post("", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_task(
    payload: CreateTaskRequest,
    background: BackgroundTasks,
    store: TaskStoreDep,
    graph: GraphDep,
    settings: SettingsDep,
    principal: PrincipalDep,
    tracer: TracerDep,
    metrics: MetricsDep,
) -> TaskResponse:
    """Create a task and start it in the background.

    Returns 202 as soon as the task is recorded, so the caller is not held open
    for the duration of a multi-agent run.

    Raises:
        HTTPException: 503 if the provider is unconfigured or storage is down.
    """
    require_configured(settings)

    record = TaskRecord(request=payload.request, user_id=principal.user_id)
    try:
        await store.create(record)
    except (DatabaseError, SQLAlchemyError) as exc:
        raise _storage_unavailable(exc) from exc

    background.add_task(
        execute_task_with_store,
        record.task_id,
        graph=graph,
        store=store,
        settings=settings,
        # Captured here because a background task outlives the request that
        # started it: reading them from ``request`` later would read a scope
        # that no longer exists.
        tracer=tracer,
        metrics=metrics,
    )

    return TaskResponse.from_record(record)


@router.get("", response_model=list[TaskResponse])
async def list_tasks(
    store: TaskStoreDep,
    principal: PrincipalDep,
    limit: int = 50,
) -> list[TaskResponse]:
    """List the caller's own tasks, newest first.

    Scoped to the caller with no opt-out: there is no parameter that widens this
    to another principal's tasks.
    """
    records = await store.list(user_id=principal.user_id, limit=max(1, min(limit, 200)))
    return [TaskResponse.from_record(record) for record in records]


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str,
    store: TaskStoreDep,
    principal: PrincipalDep,
) -> TaskResponse:
    """Fetch a task's full visible state."""
    return TaskResponse.from_record(await _owned_task(store, task_id, principal.user_id))


@router.get("/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    store: TaskStoreDep,
    principal: PrincipalDep,
) -> TaskStatusResponse:
    """Fetch just a task's status."""
    record = await _owned_task(store, task_id, principal.user_id)
    return TaskStatusResponse(
        task_id=record.task_id,
        status=record.status.value,
        approval_status=record.approval_status.value,
        has_answer=record.answer is not None,
    )


async def _decide(
    *,
    task_id: str,
    decision: ApprovalDecision,
    note: str | None,
    principal_user: str,
    graph: Any,
    store: Any,
    settings: Any,
) -> ApprovalResponse:
    """Record a human decision and resume the suspended graph.

    The decision is written to the durable record *before* the graph is resumed.
    If the resume then fails, the audit trail still shows who decided what and
    when, which is the fact that cannot be reconstructed.

    The resumed run gets fresh execution ceilings. It is a new segment of work,
    and it is bounded like any other: before this the resume was the one way to
    invoke the graph with no ceiling at all. The clock deliberately restarts — a
    human may take hours to decide, and charging that wait to the run's execution
    time would fail every approved task.

    Its events are routed to the store, as the first segment's are. Without a
    sink a resumed run published nothing, so a client following a gated task saw
    it fall silent at exactly the point where the action was performed.
    """
    record = await store.get_for_user(task_id, principal_user)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such task")

    if record.approval_status is not ApprovalStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="this task is not awaiting approval",
        )

    await store.decide_approval(
        task_id,
        decision=decision.value,
        decided_by=principal_user,
        note=note,
    )

    from langgraph.types import Command

    limits = limits_for(settings)
    publish = task_event_sink(task_id, store)
    try:
        with event_sink_for(publish), run_limits(limits):
            async with asyncio.timeout(settings.max_execution_time):
                result = await graph.ainvoke(Command(resume=decision.value), thread_config(task_id))
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                f"the resumed run exceeded its {settings.max_execution_time} second "
                "execution ceiling"
            ),
        ) from exc

    final_answer = result.get("final_answer")
    await store.update(
        task_id,
        status=TaskStatus.COMPLETED,
        approval_status=(
            ApprovalStatus.APPROVED
            if decision is ApprovalDecision.APPROVE
            else ApprovalStatus.REJECTED
        ),
        answer=final_answer,
        # Added to whatever the earlier segment recorded, not written over it.
        tool_call_count=record.tool_call_count + limits.tool_calls,
    )

    updated = await store.get(task_id)
    final_status = updated.status.value if updated is not None else TaskStatus.COMPLETED.value

    return ApprovalResponse(
        task_id=task_id,
        approval_status=(
            ApprovalStatus.APPROVED.value
            if decision is ApprovalDecision.APPROVE
            else ApprovalStatus.REJECTED.value
        ),
        status=final_status,
        answer=final_answer,
    )


@router.post("/{task_id}/approve", response_model=ApprovalResponse)
async def approve_task(
    task_id: str,
    payload: ApprovalRequest,
    graph: GraphDep,
    store: TaskStoreDep,
    principal: PrincipalDep,
    settings: SettingsDep,
) -> ApprovalResponse:
    """Approve a gated action and resume the run.

    Raises:
        HTTPException: 404 if the task is absent or not the caller's, 409 if it
            is not awaiting a decision, 504 if the resumed run outlives its
            execution ceiling.
    """
    return await _decide(
        task_id=task_id,
        decision=ApprovalDecision.APPROVE,
        note=payload.note,
        principal_user=principal.user_id,
        graph=graph,
        store=store,
        settings=settings,
    )


@router.post("/{task_id}/reject", response_model=ApprovalResponse)
async def reject_task(
    task_id: str,
    payload: ApprovalRequest,
    graph: GraphDep,
    store: TaskStoreDep,
    principal: PrincipalDep,
    settings: SettingsDep,
) -> ApprovalResponse:
    """Reject a gated action. The action is never performed.

    Raises:
        HTTPException: 404 if the task is absent or not the caller's, 409 if it
            is not awaiting a decision, 504 if the run outlives its execution
            ceiling.
    """
    return await _decide(
        task_id=task_id,
        decision=ApprovalDecision.REJECT,
        note=payload.note,
        principal_user=principal.user_id,
        graph=graph,
        store=store,
        settings=settings,
    )


@router.post("/{task_id}/cancel", response_model=TaskResponse)
async def cancel_task(
    task_id: str,
    store: TaskStoreDep,
    principal: PrincipalDep,
) -> TaskResponse:
    """Mark a task as cancelled.

    Limitation, stated rather than hidden: this cancels the record. It cannot
    interrupt a graph run that is already executing, because the run holds the
    event loop rather than polling a cancellation token. Cooperative
    cancellation is a separate piece of work, and :mod:`docs/DEVELOPMENT_PLAN.md`
    tracks it.

    Raises:
        HTTPException: 404 if the task is absent or not the caller's, 409 if it
            has already finished.
    """
    record = await _owned_task(store, task_id, principal.user_id)

    if record.status.is_terminal:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"the task is already {record.status.value}",
        )

    updated = await store.update(task_id, status=TaskStatus.CANCELLED)
    if updated is None:  # pragma: no cover - the record was just fetched above
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="the task could not be updated",
        )
    return TaskResponse.from_record(updated)
