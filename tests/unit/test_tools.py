"""Tests for the tool framework and registry."""

from __future__ import annotations

import asyncio

import pytest
from app.core.config import Settings
from app.core.constants import FailureKind
from app.core.exceptions import NotFoundError, ToolError
from app.models.tool import AccessMode, RiskLevel
from app.tools.base import Tool, ToolContext, ToolRequest
from app.tools.registry import ToolRegistry
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Fixtures: concrete tools
# --------------------------------------------------------------------------- #


class EchoInput(BaseModel):
    text: str


class EchoOutput(BaseModel):
    echoed: str


class EchoTool(Tool[EchoInput, EchoOutput]):
    """A well-behaved read-only tool."""

    name = "echo"
    description = "Echo the provided text"
    input_model = EchoInput
    output_model = EchoOutput

    async def run(self, payload: EchoInput, context: ToolContext) -> EchoOutput:
        return EchoOutput(echoed=payload.text)


class WriteTool(Tool[EchoInput, EchoOutput]):
    """A write-mode tool that declares no risk of its own."""

    name = "writer"
    description = "Write text"
    access_mode = AccessMode.WRITE
    risk_level = RiskLevel.LOW
    input_model = EchoInput
    output_model = EchoOutput

    async def run(self, payload: EchoInput, context: ToolContext) -> EchoOutput:
        return EchoOutput(echoed=payload.text)


class DangerousTool(Tool[EchoInput, EchoOutput]):
    """A high-risk tool that must be approved."""

    name = "dangerous"
    description = "Delete everything"
    access_mode = AccessMode.DESTRUCTIVE
    risk_level = RiskLevel.CRITICAL
    input_model = EchoInput
    output_model = EchoOutput

    async def run(self, payload: EchoInput, context: ToolContext) -> EchoOutput:
        return EchoOutput(echoed=payload.text)


class SlowTool(Tool[EchoInput, EchoOutput]):
    """A tool that outlives its timeout."""

    name = "slow"
    description = "Sleep"
    timeout_seconds = 0.05
    input_model = EchoInput
    output_model = EchoOutput

    async def run(self, payload: EchoInput, context: ToolContext) -> EchoOutput:
        await asyncio.sleep(5)
        return EchoOutput(echoed="never")


class CrashingTool(Tool[EchoInput, EchoOutput]):
    """A tool that raises something unexpected."""

    name = "crash"
    description = "Crash"
    input_model = EchoInput
    output_model = EchoOutput

    async def run(self, payload: EchoInput, context: ToolContext) -> EchoOutput:
        raise RuntimeError("unexpected internal failure")


class FailingTool(Tool[EchoInput, EchoOutput]):
    """A tool that raises a deliberate tool error."""

    name = "failing"
    description = "Fail deliberately"
    input_model = EchoInput
    output_model = EchoOutput

    async def run(self, payload: EchoInput, context: ToolContext) -> EchoOutput:
        raise ToolError("upstream refused", detail="status 503")


class LeakyTool(Tool[EchoInput, EchoOutput]):
    """A tool whose error message contains a configured secret."""

    name = "leaky"
    description = "Leak a secret"
    input_model = EchoInput
    output_model = EchoOutput

    async def run(self, payload: EchoInput, context: ToolContext) -> EchoOutput:
        raise RuntimeError("auth failed with token llm-secret-value")


def make_context(**overrides: object) -> ToolContext:
    settings = Settings(_env_file=None, llm_api_key="llm-secret-value")
    return ToolContext(settings=settings, **overrides)  # type: ignore[arg-type]


def make_request(tool: str = "echo", **arguments: object) -> ToolRequest:
    return ToolRequest(tool=tool, arguments=dict(arguments))


# --------------------------------------------------------------------------- #
# Risk classification
# --------------------------------------------------------------------------- #


def test_effective_risk_never_understates_a_write() -> None:
    """A tool cannot label a write as LOW and escape the audit trail."""
    assert WriteTool.effective_risk() is RiskLevel.MEDIUM


def test_effective_risk_never_understates_a_destructive_action() -> None:
    assert DangerousTool.effective_risk() is RiskLevel.CRITICAL


def test_effective_risk_respects_a_higher_declared_level() -> None:
    assert EchoTool.effective_risk() is RiskLevel.LOW


def test_only_high_and_above_require_approval() -> None:
    assert RiskLevel.LOW.requires_approval is False
    assert RiskLevel.MEDIUM.requires_approval is False
    assert RiskLevel.HIGH.requires_approval is True
    assert RiskLevel.CRITICAL.requires_approval is True


def test_tools_requiring_approval_are_identified() -> None:
    assert DangerousTool.requires_approval() is True
    assert WriteTool.requires_approval() is False


def test_describe_is_client_safe() -> None:
    described = EchoTool().describe()

    assert described["name"] == "echo"
    assert described["risk_level"] == "LOW"
    assert described["requires_approval"] is False
    assert "input_schema" in described


# --------------------------------------------------------------------------- #
# Execution pipeline
# --------------------------------------------------------------------------- #


async def test_a_valid_call_succeeds() -> None:
    result = await EchoTool().execute(make_request(text="hi"), make_context())

    assert result.ok is True
    assert result.output == {"echoed": "hi"}
    assert result.duration_ms >= 0
    assert result.risk_level is RiskLevel.LOW


async def test_invalid_arguments_are_rejected_before_execution() -> None:
    result = await EchoTool().execute(make_request(wrong_field="x"), make_context())

    assert result.ok is False
    assert result.failure_kind is FailureKind.VALIDATION
    assert "validation" in (result.error or "")


async def test_a_missing_required_argument_is_rejected() -> None:
    result = await EchoTool().execute(make_request(), make_context())

    assert result.ok is False
    assert result.failure_kind is FailureKind.VALIDATION


async def test_the_tool_name_must_match_the_request() -> None:
    result = await EchoTool().execute(make_request("hijacked", text="x"), make_context())

    assert result.ok is False
    assert result.failure_kind is FailureKind.VALIDATION


async def test_high_risk_tools_are_gated_behind_approval() -> None:
    result = await DangerousTool().execute(make_request("dangerous", text="x"), make_context())

    assert result.ok is False
    assert "approval" in (result.error or "").lower()
    assert result.failure_kind is FailureKind.PERMANENT


async def test_approval_unlocks_a_high_risk_tool() -> None:
    context = make_context(approved=True)

    result = await DangerousTool().execute(make_request("dangerous", text="x"), context)

    assert result.ok is True
    assert result.approved is True


async def test_timeouts_are_reported_as_timeouts() -> None:
    result = await SlowTool().execute(make_request("slow", text="x"), make_context())

    assert result.ok is False
    assert result.failure_kind is FailureKind.TIMEOUT
    assert "timeout" in (result.error or "")


async def test_an_unexpected_exception_does_not_escape_the_tool() -> None:
    """A broken tool must not crash the graph."""
    result = await CrashingTool().execute(make_request("crash", text="x"), make_context())

    assert result.ok is False
    assert result.failure_kind is FailureKind.UNKNOWN


async def test_tool_errors_carry_their_own_failure_kind() -> None:
    result = await FailingTool().execute(make_request("failing", text="x"), make_context())

    assert result.ok is False
    assert result.failure_kind is FailureKind.TOOL_FAILURE
    assert "upstream refused" in (result.error or "")


async def test_secrets_are_redacted_from_error_messages() -> None:
    """A tool holds credentials, so its error text is a realistic leak path."""
    result = await LeakyTool().execute(make_request("leaky", text="x"), make_context())

    assert "llm-secret-value" not in (result.error or "")
    assert "[REDACTED]" in (result.error or "")


async def test_every_result_carries_its_request_id() -> None:
    request = make_request(text="hi")

    result = await EchoTool().execute(request, make_context())

    assert result.request_id == request.request_id


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_registry_registers_and_lists() -> None:
    registry = ToolRegistry([EchoTool(), WriteTool()])

    assert registry.names() == ("echo", "writer")
    assert len(registry) == 2


def test_registry_rejects_a_duplicate_name() -> None:
    registry = ToolRegistry([EchoTool()])

    with pytest.raises(ValueError, match="already registered"):
        registry.register(EchoTool())


def test_registry_rejects_an_unnamed_tool() -> None:
    class Nameless(Tool[EchoInput, EchoOutput]):
        name = ""
        description = "no name"
        input_model = EchoInput
        output_model = EchoOutput

        async def run(self, payload: EchoInput, context: ToolContext) -> EchoOutput:
            return EchoOutput(echoed="")

    with pytest.raises(ValueError, match="non-empty name"):
        ToolRegistry([Nameless()])


def test_registry_get_raises_for_an_unknown_tool() -> None:
    registry = ToolRegistry([EchoTool()])

    with pytest.raises(NotFoundError, match="not registered"):
        registry.get("ghost")


def test_registry_try_get_returns_none() -> None:
    registry = ToolRegistry([EchoTool()])

    assert registry.try_get("ghost") is None
    assert registry.try_get("echo") is not None


def test_registry_supports_membership_tests() -> None:
    registry = ToolRegistry([EchoTool()])

    assert "echo" in registry
    assert "ghost" not in registry


def test_registry_unregisters() -> None:
    registry = ToolRegistry([EchoTool(), WriteTool()])

    registry.unregister("echo")

    assert registry.names() == ("writer",)


def test_registry_unregister_raises_for_an_unknown_tool() -> None:
    with pytest.raises(NotFoundError):
        ToolRegistry().unregister("ghost")


def test_get_allowed_tools_resolves_only_the_named_set() -> None:
    """Least privilege: anything not named is invisible."""
    registry = ToolRegistry([EchoTool(), WriteTool(), DangerousTool()])

    allowed = registry.get_allowed_tools(["echo"])

    assert [tool.name for tool in allowed] == ["echo"]


def test_get_allowed_tools_raises_for_an_unknown_name() -> None:
    """A typo in an agent's tool list must fail loudly, not silently drop a tool."""
    registry = ToolRegistry([EchoTool()])

    with pytest.raises(NotFoundError, match="unknown tools"):
        registry.get_allowed_tools(["echo", "typo"])


def test_get_allowed_tools_with_an_empty_list_returns_nothing() -> None:
    registry = ToolRegistry([EchoTool()])

    assert registry.get_allowed_tools([]) == ()


def test_filter_by_max_risk_excludes_dangerous_tools() -> None:
    registry = ToolRegistry([EchoTool(), WriteTool(), DangerousTool()])

    safe = registry.filter_by_max_risk(RiskLevel.MEDIUM)

    assert {tool.name for tool in safe} == {"echo", "writer"}


def test_registry_describe_returns_every_tool() -> None:
    registry = ToolRegistry([EchoTool(), WriteTool()])

    described = registry.describe()

    assert len(described) == 2
    assert {entry["name"] for entry in described} == {"echo", "writer"}
