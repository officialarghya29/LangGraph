"""Tests for the Python execution tool.

The tool's contract is that it does not execute. These tests pin that down, so
that a future change cannot quietly turn it into an RCE surface without a
failing test standing in the way.
"""

from __future__ import annotations

from app.core.config import Settings
from app.models.tool import RiskLevel
from app.tools.base import ToolContext, ToolRequest
from app.tools.python_executor import PythonExecutionTool


def context_for(*, approved: bool = False, **settings_overrides: object) -> ToolContext:
    """Build a context.

    ``approved`` belongs to :class:`ToolContext`, not to ``Settings``. It is
    named explicitly because ``Settings`` ignores unknown keys, so routing it
    through would silently drop the approval and make every refusal look like
    the approval gate rather than the real cause.
    """
    settings = Settings(_env_file=None, **settings_overrides)  # type: ignore[arg-type]
    return ToolContext(settings=settings, approved=approved)


def request(code: str = "print('hi')") -> ToolRequest:
    return ToolRequest(tool="python_executor", arguments={"code": code})


def test_execution_is_high_risk() -> None:
    """High risk means the approval gate applies before run() is reached."""
    assert PythonExecutionTool.effective_risk() is RiskLevel.HIGH
    assert PythonExecutionTool.requires_approval() is True


async def test_an_unapproved_call_is_stopped_by_the_approval_gate() -> None:
    result = await PythonExecutionTool().execute(request(), context_for())

    assert result.ok is False
    assert "approval" in (result.error or "").lower()


async def test_execution_is_refused_even_when_approved_and_disabled() -> None:
    result = await PythonExecutionTool().execute(request(), context_for(approved=True))

    assert result.ok is False
    assert "disabled" in (result.error or "")


async def test_execution_is_still_refused_when_the_flag_is_on() -> None:
    """Enabling the flag does not create a sandbox, so the refusal must stand."""
    result = await PythonExecutionTool().execute(
        request(), context_for(approved=True, python_execution_enabled=True)
    )

    assert result.ok is False
    assert "isolation boundary" in (result.error or "")


async def test_no_output_is_produced_by_a_refused_call() -> None:
    result = await PythonExecutionTool().execute(
        request(), context_for(approved=True, python_execution_enabled=True)
    )

    assert result.output == {}


async def test_the_refusal_is_marked_permanent_so_it_is_never_retried() -> None:
    from app.core.constants import FailureKind

    result = await PythonExecutionTool().execute(
        request(), context_for(approved=True, python_execution_enabled=True)
    )

    assert result.failure_kind is FailureKind.PERMANENT


async def test_empty_code_is_rejected_before_the_tool_runs() -> None:
    result = await PythonExecutionTool().execute(
        ToolRequest(tool="python_executor", arguments={"code": ""}), context_for(approved=True)
    )

    assert result.ok is False
