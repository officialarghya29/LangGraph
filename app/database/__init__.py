"""Persistence: engine, sessions, ORM models, and repositories.

Importing this package is enough to register every mapped class with
``Base.metadata``, which is what Alembic's autogenerate compares against.
"""

from __future__ import annotations

from app.database.base import Base, utc_now
from app.database.connection import Database, build_engine
from app.database.models import (
    AgentRun,
    Approval,
    Conversation,
    ExecutionEvent,
    MemoryRecord,
    Task,
    TaskStep,
    ToolCall,
    User,
)
from app.database.repositories import (
    AgentRunRepository,
    ApprovalRepository,
    ConversationRepository,
    EventRepository,
    MemoryRepository,
    TaskRepository,
    TaskStepRepository,
    ToolCallRepository,
    UserRepository,
)

__all__ = [
    "AgentRun",
    "AgentRunRepository",
    "Approval",
    "ApprovalRepository",
    "Base",
    "Conversation",
    "ConversationRepository",
    "Database",
    "EventRepository",
    "ExecutionEvent",
    "MemoryRecord",
    "MemoryRepository",
    "Task",
    "TaskRepository",
    "TaskStep",
    "TaskStepRepository",
    "ToolCall",
    "ToolCallRepository",
    "User",
    "UserRepository",
    "build_engine",
    "utc_now",
]
