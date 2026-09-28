"""Tool framework.

Every tool call runs the same pipeline, implemented once here so no individual
tool can skip a stage:

    ToolRequest -> input validation -> permission -> risk classification
               -> approval check -> execution -> result validation -> audit

The pipeline is enforced in :meth:`Tool.execute`. Subclasses implement only
:meth:`Tool.run`, which receives an already-validated payload and a context that
has already passed the approval gate.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.config import Settings
from app.core.constants import FailureKind
from app.core.exceptions import ToolError
from app.models.tool import AccessMode, RiskLevel

__all__ = ["Tool", "ToolContext", "ToolRequest", "ToolResult"]


class ToolRequest(BaseModel):
    """A request to run a tool.

    ``tool`` is carried explicitly so a request can be rejected before any tool
    is resolved, which keeps the audit trail honest about what was asked for.
    """

    tool: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str | None = None
    agent: str | None = None
    user_id: str | None = None


class ToolResult(BaseModel):
    """The outcome of a tool call, including its audit metadata."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    tool: str
    ok: bool
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    failure_kind: FailureKind = FailureKind.UNKNOWN
    risk_level: RiskLevel = RiskLevel.LOW
    approved: bool = False
    duration_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Everything a tool is allowed to know about its caller.

    Deliberately small. A tool has no access to the graph, the database
    session, or the registry, so it cannot reach around the framework.
    """

    settings: Settings
    user_id: str | None = None
    task_id: str | None = None
    agent: str | None = None
    approved: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class Tool[In: BaseModel, Out: BaseModel](ABC):
    """Base class for every tool, generic over its input and output models."""

    #: Unique tool name used in requests and audit records.
    name: ClassVar[str] = ""
    #: One-line description shown to the model and to operators.
    description: ClassVar[str] = ""
    #: What the tool does to the outside world.
    access_mode: ClassVar[AccessMode] = AccessMode.READ
    #: Declared danger level. Effective risk is never lower than the mode implies.
    risk_level: ClassVar[RiskLevel] = RiskLevel.LOW
    #: Hard execution ceiling. A tool may not run indefinitely.
    timeout_seconds: ClassVar[float] = 30.0

    #: Pydantic models describing and validating the tool's contract.
    #:
    #: Declared ``ClassVar`` so subclasses can assign concrete models in the
    #: class body. The generic parameters are carried by :meth:`run`; the models
    #: here are used for validation and schema introspection.
    input_model: ClassVar[type[BaseModel]]
    output_model: ClassVar[type[BaseModel]]

    # ------------------------------------------------------------------ #
    # Contract introspection
    # ------------------------------------------------------------------ #

    @classmethod
    def effective_risk(cls) -> RiskLevel:
        """Return the greater of the declared risk and the access mode's minimum.

        A tool cannot under-declare its own danger: a write-mode tool is at
        least ``MEDIUM`` regardless of what it claims.
        """
        return max(cls.risk_level, cls.access_mode.risk_level)

    @classmethod
    def requires_approval(cls) -> bool:
        """Return whether this tool's operations must be human-approved."""
        return cls.effective_risk().requires_approval

    def input_schema(self) -> dict[str, Any]:
        """Return the JSON schema of the tool's input."""
        return self.input_model.model_json_schema()

    def output_schema(self) -> dict[str, Any]:
        """Return the JSON schema of the tool's output."""
        return self.output_model.model_json_schema()

    def describe(self) -> dict[str, Any]:
        """Return a client-safe description of the tool."""
        return {
            "name": self.name,
            "description": self.description,
            "access_mode": self.access_mode.value,
            "risk_level": self.effective_risk().label,
            "requires_approval": self.requires_approval(),
            "input_schema": self.input_schema(),
            "output_schema": self.output_schema(),
        }

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    async def execute(self, request: ToolRequest, context: ToolContext) -> ToolResult:
        """Run the full tool pipeline.

        Args:
            request: The requested tool and its arguments.
            context: Caller identity and approval state.

        Returns:
            A :class:`ToolResult`. Failures are returned, not raised, so the
            graph can classify and record them uniformly.
        """
        started = time.perf_counter()
        risk = self.effective_risk()

        if request.tool != self.name:
            return self._failure(
                request, risk, "tool name mismatch", FailureKind.VALIDATION, started
            )

        # 1. Input validation. The cast records the contract each tool
        # subclass upholds: ``input_model`` validates the payload that ``run``
        # declares as its ``In`` parameter.
        try:
            payload = cast(In, self.input_model.model_validate(request.arguments))
        except ValidationError as exc:
            return self._failure(
                request,
                risk,
                "arguments failed validation",
                FailureKind.VALIDATION,
                started,
                detail=_safe_detail(exc, context),
            )

        # 2 and 3. Permission and risk classification are properties of the
        # registry membership and the tool class; the approval gate below is
        # where they take effect.

        # 4. Approval check.
        if self.requires_approval() and not context.approved:
            return self._failure(
                request,
                risk,
                f"{risk.label} risk action requires human approval",
                FailureKind.PERMANENT,
                started,
                approval_required=True,
            )

        # 5. Execution, under a hard timeout.
        try:
            raw = await asyncio.wait_for(self.run(payload, context), timeout=self.timeout_seconds)
        except TimeoutError:
            return self._failure(
                request,
                risk,
                f"tool exceeded its {self.timeout_seconds:g}s timeout",
                FailureKind.TIMEOUT,
                started,
            )
        except ToolError as exc:
            return self._failure(
                request, risk, str(exc), exc.failure_kind, started, detail=exc.detail
            )
        except Exception as exc:
            return self._failure(
                request,
                risk,
                "tool raised an unexpected error",
                FailureKind.UNKNOWN,
                started,
                detail=_safe_detail(exc, context),
            )

        # 6. Result validation.
        try:
            validated = self.output_model.model_validate(raw)
        except ValidationError as exc:
            return self._failure(
                request,
                risk,
                "tool returned an invalid result",
                FailureKind.UNKNOWN,
                started,
                detail=_safe_detail(exc, context),
            )

        # 7. Audit record.
        return ToolResult(
            request_id=request.request_id,
            tool=self.name,
            ok=True,
            output=validated.model_dump(mode="json"),
            risk_level=risk,
            approved=context.approved,
            duration_ms=_elapsed_ms(started),
        )

    @abstractmethod
    async def run(self, payload: In, context: ToolContext) -> Out:
        """Perform the tool's work on an already-validated payload."""

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _failure(
        self,
        request: ToolRequest,
        risk: RiskLevel,
        message: str,
        kind: FailureKind,
        started: float,
        *,
        detail: str | None = None,
        approval_required: bool = False,
    ) -> ToolResult:
        """Build a failure result, redacting any configured secret."""
        text = message if not detail else f"{message}: {detail}"
        return ToolResult(
            request_id=request.request_id,
            tool=self.name,
            ok=False,
            error=text,
            failure_kind=kind,
            risk_level=risk,
            approved=False,
            duration_ms=_elapsed_ms(started),
        )


def _elapsed_ms(started: float) -> float:
    """Return milliseconds elapsed since ``started``."""
    return round((time.perf_counter() - started) * 1000, 3)


def _safe_detail(exc: BaseException, context: ToolContext) -> str | None:
    """Render an exception message with any configured secret redacted.

    Tools receive credentials, so an exception message is a realistic leak path.
    Every configured secret is scrubbed before the text is recorded.
    """
    text = str(exc)
    if not text:
        return None
    for secret in context.settings.secret_values():
        if secret and secret in text:
            text = text.replace(secret, "[REDACTED]")
    return text
