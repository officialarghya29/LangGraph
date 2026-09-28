"""Typed LangGraph state.

The state is a :class:`TypedDict` rather than a loose ``dict[str, Any]`` for
three reasons:

1. every field is discoverable and type-checked;
2. every value is a Pydantic model, a primitive, or ``None``, so the whole state
   round-trips through the checkpoint serializer;
3. fields written by concurrent branches declare an explicit reducer
   (``operator.add``) so parallel writes merge predictably instead of racing.

``total=False`` lets a node return only the keys it changed; LangGraph merges
the partial update into the running state.
"""

from __future__ import annotations

import operator
import uuid
from typing import Annotated, TypedDict

from app.models.agent import AgentOutput, VerificationResult
from app.models.approval import ApprovalStatus
from app.models.execution import ExecutionError, ExecutionMetadata
from app.models.memory import ContextItem, MemoryItem
from app.models.tool import ToolCallRecord
from app.schemas.plans import Plan, RouteDecision, Subtask
from app.services.llm import Message

__all__ = ["AgentState", "initial_state"]

#: Keys written by more than one node in the same superstep. Each concatenates
#: rather than overwrites, so no branch can silently discard another's work.
_APPEND = Annotated[list, operator.add]


class AgentState(TypedDict, total=False):
    """The full state of one task execution."""

    # --- Identity ---------------------------------------------------------- #
    task_id: str
    conversation_id: str
    user_id: str

    # --- Conversation ------------------------------------------------------ #
    user_request: str
    messages: Annotated[list[Message], operator.add]

    # --- Routing ----------------------------------------------------------- #
    route: RouteDecision | None
    intent: str | None
    complexity: str | None

    # --- Planning ---------------------------------------------------------- #
    plan: Plan | None
    subtasks: list[Subtask]
    active_subtask: str | None
    completed_subtasks: Annotated[list[str], operator.add]

    # --- Results ----------------------------------------------------------- #
    agent_outputs: Annotated[list[AgentOutput], operator.add]
    tool_results: Annotated[list[ToolCallRecord], operator.add]
    retrieved_context: Annotated[list[ContextItem], operator.add]
    memory_context: list[MemoryItem]

    # --- Control ----------------------------------------------------------- #
    errors: Annotated[list[ExecutionError], operator.add]
    retry_count: int
    iteration_count: int
    tool_call_count: int

    # --- Approval ---------------------------------------------------------- #
    requires_human_approval: bool
    approval_status: ApprovalStatus
    approval_id: str | None

    # --- Verification and output ------------------------------------------- #
    verification_result: VerificationResult | None
    final_answer: str | None
    execution_metadata: ExecutionMetadata

    # --- Events ------------------------------------------------------------ #
    events: Annotated[list[str], operator.add]
    event_sequence: int

    # --- Limits ------------------------------------------------------------ #
    # Copied from settings at run start so the conditional-edge routing
    # functions stay pure functions of state, and are therefore testable
    # without constructing a graph.
    iteration_limit: int
    retry_limit: int


def initial_state(
    user_request: str,
    *,
    user_id: str = "anonymous",
    conversation_id: str | None = None,
    task_id: str | None = None,
    iteration_limit: int = 10,
    retry_limit: int = 3,
) -> AgentState:
    """Build a fully populated initial state.

    Every key is present so downstream nodes can read without guarding against
    missing keys. Starting from a sparse state is a common source of subtle
    graph bugs.

    Args:
        user_request: The user's request text.
        user_id: Owning user identifier.
        conversation_id: Conversation to attach the task to. Generated if absent.
        task_id: Task identifier. Generated if absent.
        iteration_limit: Ceiling for the dispatch loop.
        retry_limit: Ceiling for the verification retry loop.

    Returns:
        A state ready to be passed to the graph.

    Raises:
        ValueError: If ``user_request`` is empty or whitespace only, or either
            limit is not positive.
    """
    if iteration_limit < 1:
        raise ValueError("iteration_limit must be at least 1")
    if retry_limit < 0:
        raise ValueError("retry_limit must not be negative")
    if not user_request or not user_request.strip():
        raise ValueError("user_request must not be empty")

    return AgentState(
        task_id=task_id or str(uuid.uuid4()),
        conversation_id=conversation_id or str(uuid.uuid4()),
        user_id=user_id,
        user_request=user_request.strip(),
        messages=[Message(role="user", content=user_request.strip())],
        route=None,
        intent=None,
        complexity=None,
        plan=None,
        subtasks=[],
        active_subtask=None,
        completed_subtasks=[],
        agent_outputs=[],
        tool_results=[],
        retrieved_context=[],
        memory_context=[],
        errors=[],
        retry_count=0,
        iteration_count=0,
        tool_call_count=0,
        requires_human_approval=False,
        approval_status=ApprovalStatus.NOT_REQUIRED,
        approval_id=None,
        verification_result=None,
        final_answer=None,
        execution_metadata=ExecutionMetadata(),
        events=[],
        event_sequence=0,
        iteration_limit=iteration_limit,
        retry_limit=retry_limit,
    )
