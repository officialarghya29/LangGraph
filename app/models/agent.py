"""Agent-related domain models."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

__all__ = ["AgentOutput", "VerificationResult"]


class AgentOutput(BaseModel):
    """What one agent produced for one subtask.

    ``summary`` is a short, client-safe restatement of the result. It is not the
    agent's reasoning, and it is what gets surfaced in events and the API.
    """

    agent: str
    subtask_id: str | None = None
    content: str
    summary: str = ""
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    tool_calls: int = 0
    duration_ms: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VerificationResult(BaseModel):
    """The critic's verdict on the aggregated agent output.

    The critic reports; it does not rewrite. The orchestrator decides what to do
    with this verdict, so the critic can never silently alter a result.
    """

    passed: bool
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    corrections: list[str] = Field(default_factory=list)
    verification_summary: str = ""

    @property
    def is_actionable(self) -> bool:
        """Return whether the verdict carries something the orchestrator can act on."""
        return bool(self.issues or self.missing_requirements or self.corrections)
