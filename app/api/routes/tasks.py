"""Task endpoints.

``POST /api/v1/tasks`` returns immediately with a task id and runs the graph in
the background. ``GET`` endpoints report progress. The approval endpoints resume
a suspended graph with a human's decision.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.api.dependencies import GraphDep, SettingsDep, TaskStoreDep, require_configured
from app.graph.checkpoints import thread_config
from app.models.approval import ApprovalDecision, ApprovalStatus
from app.models.execution import TaskStatus
from app.services.tasks import TaskRecord, execute_task

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])

MAX_REQUEST_LENGTH = 20_000


class CreateTaskRequest(BaseModel):
    """A task to run asynchronously."""

    request: str = Field(min_length=1, max_length=MAX_REQUEST_LENGTH)
    conversation_id: str | None = None


class TaskResponse(BaseModel):
    """The client-visible state of a task."""

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


def _route_name(decision: object) -> str | None:
    """Return a routing decision's route name, if the run got as far as routing.

    The decision arrives from graph state as an object rather than a typed value,
    so this reads it defensively instead of assuming routing succeeded.
    """
    route = getattr(decision, "route", None)
    if route is None:
        return None
    value = getattr(route, "value", route)
    return str(value)


async def _owned_task(store: TaskStoreDep, task_id: str, user_id: str) -> TaskRecord:
    """Fetch a task, enforcing ownership.

    Raises:
        HTTPException: 404, whether the task is absent or owned by somebody
            else. Distinguishing the two would confirm that another user's task
            exists.
    """
    record = await store.get_for_user(task_id, user_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such task")
    return record


@router.post("", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_task(
    payload: CreateTaskRequest,
    background: BackgroundTasks,
    store: TaskStoreDep,
    graph: GraphDep,
    settings: SettingsDep,
    x_user_id: Annotated[str, Header()] = "anonymous",
) -> TaskResponse:
    """Create a task and start it in the background.

    Returns 202 as soon as the task is recorded, so the caller is not held open
    for the duration of a multi-agent run.
    """
    require_configured(settings)

    record = TaskRecord(request=payload.request, user_id=x_user_id)
    await store.create(record)

    background.add_task(
        execute_task,
        record.task_id,
        graph=graph,
        store=store,
        settings=settings,
    )

    return TaskResponse.from_record(record)


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str,
    store: TaskStoreDep,
    x_user_id: Annotated[str, Header()] = "anonymous",
) -> TaskResponse:
    """Fetch a task's full visible state."""
    return TaskResponse.from_record(await _owned_task(store, task_id, x_user_id))


@router.get("/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    store: TaskStoreDep,
    x_user_id: Annotated[str, Header()] = "anonymous",
) -> TaskStatusResponse:
    """Fetch just a task's status."""
    record = await _owned_task(store, task_id, x_user_id)
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
    user_id: str,
    graph: Any,
    store: Any,
) -> ApprovalResponse:
    """Record a human decision and resume the suspended graph."""
    record = await store.get_for_user(task_id, user_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such task")

    if record.approval_status is not ApprovalStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="this task is not awaiting approval",
        )

    from langgraph.types import Command

    result = await graph.ainvoke(Command(resume=decision.value), thread_config(task_id))

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
    x_user_id: Annotated[str, Header()] = "anonymous",
) -> ApprovalResponse:
    """Approve a gated action and resume the run."""
    return await _decide(
        task_id=task_id,
        decision=ApprovalDecision.APPROVE,
        note=payload.note,
        user_id=x_user_id,
        graph=graph,
        store=store,
    )


@router.post("/{task_id}/reject", response_model=ApprovalResponse)
async def reject_task(
    task_id: str,
    payload: ApprovalRequest,
    graph: GraphDep,
    store: TaskStoreDep,
    x_user_id: Annotated[str, Header()] = "anonymous",
) -> ApprovalResponse:
    """Reject a gated action. The action is never performed."""
    return await _decide(
        task_id=task_id,
        decision=ApprovalDecision.REJECT,
        note=payload.note,
        user_id=x_user_id,
        graph=graph,
        store=store,
    )


@router.post("/{task_id}/cancel", response_model=TaskResponse)
async def cancel_task(
    task_id: str,
    request: Request,
    store: TaskStoreDep,
    x_user_id: Annotated[str, Header()] = "anonymous",
) -> TaskResponse:
    """Mark a task as cancelled.

    Limitation: this marks the record. It does not interrupt a graph run that is
    already executing, because the run holds the event loop rather than polling
    a cancellation token. Cooperative cancellation arrives with the durable
    checkpoint store.
    """
    record = await _owned_task(store, task_id, x_user_id)

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
