"""Python execution tool.

Executing model-authored Python inside the API process is remote code execution
by design. This tool therefore does not execute anything.

It is modelled as a real tool with a real security posture rather than left
unimplemented, so that:

- the capability appears in the registry and can be reasoned about;
- it is classified ``HIGH`` risk, so an unapproved call is refused by the
  standard approval gate before it ever reaches :meth:`run`;
- an *approved* call still refuses, because this host has no isolation boundary.

A container or micro-VM boundary is required before this can do anything else.
Until such a boundary exists, refusing is the correct behaviour: an
implementation that ran code here would be a vulnerability, not a feature.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.core.constants import FailureKind
from app.core.exceptions import AppError
from app.models.tool import AccessMode, RiskLevel
from app.tools.base import Tool, ToolContext

__all__ = ["ExecutionRefusedError", "PythonExecutionTool", "PythonInput", "PythonOutput"]

logger = logging.getLogger(__name__)


class ExecutionRefusedError(AppError):
    """Raised when execution cannot be performed safely.

    Permanent: retrying changes nothing until an operator changes the
    environment, so this must never be retried.
    """

    failure_kind = FailureKind.PERMANENT


class PythonInput(BaseModel):
    """A request to run Python.

    ``code`` is accepted so the request validates and the refusal can be
    specific, but it is never executed.
    """

    code: str = Field(min_length=1, max_length=100_000)
    timeout_seconds: int = Field(default=10, ge=1, le=120)


class PythonOutput(BaseModel):
    """Output of an execution. Only produced if execution were permitted."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


class PythonExecutionTool(Tool[PythonInput, PythonOutput]):
    """Refuses to execute code until an isolation boundary exists.

    ``HIGH`` risk, so the approval gate applies: an unapproved request fails
    with "requires human approval" before reaching :meth:`run`.
    """

    name = "python_executor"
    description = "Execute Python in an isolated sandbox (unavailable in this environment)"
    access_mode = AccessMode.WRITE
    risk_level = RiskLevel.HIGH
    timeout_seconds = 120.0
    input_model = PythonInput
    output_model = PythonOutput

    async def run(self, payload: PythonInput, context: ToolContext) -> PythonOutput:
        """Refuse to execute.

        Args:
            payload: The requested code. Deliberately unused.
            context: Caller context, including whether approval was granted.

        Raises:
            ExecutionRefusedError: Always, unless a sandbox is configured.
        """
        del payload

        if not context.settings.python_execution_enabled:
            logger.warning(
                "python_execution.refused",
                extra={"reason": "disabled by configuration", "task_id": context.task_id},
            )
            raise ExecutionRefusedError(
                "code execution is disabled",
                detail="set PYTHON_EXECUTION_ENABLED only alongside a real sandbox",
            )

        # Execution is switched on, but there is still nowhere safe to run the
        # code. Enabling the flag alone does not create an isolation boundary.
        logger.warning(
            "python_execution.refused",
            extra={"reason": "no isolation boundary", "task_id": context.task_id},
        )
        raise ExecutionRefusedError(
            "no isolation boundary is available, so the code was not executed",
            detail="a container or micro-VM runtime is required",
        )
