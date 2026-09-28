"""Human approval domain models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from app.models.tool import RiskLevel

__all__ = ["ApprovalDecision", "ApprovalRecord", "ApprovalStatus"]


class ApprovalDecision(StrEnum):
    """A human's answer to an approval request."""

    PENDING = "pending"
    APPROVE = "approve"
    REJECT = "reject"

    @property
    def is_resolved(self) -> bool:
        """Return whether a decision has been made."""
        return self is not ApprovalDecision.PENDING


class ApprovalStatus(StrEnum):
    """Where a task stands with respect to approval."""

    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalRecord(BaseModel):
    """A persisted request for a human to approve or reject an action."""

    approval_id: str
    task_id: str
    requested_action: str
    risk_level: RiskLevel = RiskLevel.HIGH
    requested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    decision: ApprovalDecision = ApprovalDecision.PENDING
    decided_at: datetime | None = None
    decided_by: str | None = None
    note: str | None = None

    @property
    def is_resolved(self) -> bool:
        """Return whether a human has answered this request."""
        return self.decision.is_resolved

    def resolve(
        self,
        decision: ApprovalDecision,
        *,
        decided_by: str,
        note: str | None = None,
    ) -> ApprovalRecord:
        """Record a decision and return this instance.

        Args:
            decision: The human's answer. Must be approve or reject.
            decided_by: Identifier of the deciding principal.
            note: Optional rationale.

        Returns:
            This record, updated in place.

        Raises:
            ValueError: If ``decision`` is still pending.
        """
        if not decision.is_resolved:
            raise ValueError("cannot resolve an approval with a pending decision")
        self.decision = decision
        self.decided_by = decided_by
        self.decided_at = datetime.now(UTC)
        self.note = note
        return self
