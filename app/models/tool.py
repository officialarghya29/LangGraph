"""Tool-related domain models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import IntEnum, StrEnum

from pydantic import BaseModel, Field

__all__ = ["AccessMode", "RiskLevel", "ToolCallRecord"]


class RiskLevel(IntEnum):
    """How dangerous a tool operation is.

    An ``IntEnum`` so risk can be compared ordinally, which is how the approval
    policy is expressed: anything at or above :attr:`HIGH` is gated.
    """

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @property
    def label(self) -> str:
        """Return the level's display name."""
        return self.name

    @property
    def requires_approval(self) -> bool:
        """Return whether this level must be approved by a human first."""
        return self >= RiskLevel.HIGH


class AccessMode(StrEnum):
    """What a tool does to the outside world.

    Tools that reach outward declare their mode so the registry can apply a
    consistent least-privilege default rather than each tool inventing one.
    """

    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"

    @property
    def risk_level(self) -> RiskLevel:
        """Return the minimum risk this access mode implies."""
        if self is AccessMode.DESTRUCTIVE:
            return RiskLevel.CRITICAL
        if self is AccessMode.WRITE:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW


class ToolCallRecord(BaseModel):
    """An auditable record of one tool invocation."""

    request_id: str
    task_id: str | None = None
    agent: str | None = None
    tool: str
    arguments: dict[str, object] = Field(default_factory=dict)
    ok: bool
    error: str | None = None
    risk_level: RiskLevel = RiskLevel.LOW
    approved: bool = False
    duration_ms: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
