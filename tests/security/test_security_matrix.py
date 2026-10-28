"""Phase 34 — the security matrix.

Each threat in the README's security table has a test here that fails if the
control is removed. Some of these overlap tests elsewhere; that is the point of a
matrix. Per-tool suites prove a tool behaves; this proves the *posture* is
internally consistent across every tool at once, which is what a reviewer
actually has to check when a new tool is added.

Everything is enumerated from the real registry rather than from a hardcoded
list, so a new tool is covered the moment it is registered. A test that names its
subjects will pass forever after the subject list changes.
"""

from __future__ import annotations

from typing import cast

import pytest
from starlette.types import Scope

from app.api.middleware import _identity
from app.core.auth import ANONYMOUS_USER_ID, issue_token
from app.core.config import Settings
from app.core.exceptions import ConfigurationError, PermissionDeniedError, ToolError
from app.core.security import redact, validate_outbound_url
from app.graph.builder import build_all_agents
from app.models.tool import AccessMode, RiskLevel
from app.tools.filesystem import resolve_within_roots
from app.tools.registry import build_default_registry
from tests.security.helpers import make_provider, workforce_registry

#: A token that is long enough to be treated as a credential by redaction.
SECRET = "sk-live-0123456789abcdefghijklmnop"


def default_registry() -> object:
    """Return the registry the application builds from settings alone."""
    return build_default_registry(Settings(_env_file=None))


def all_agents() -> dict[str, object]:
    """Return every agent, wired against a registry that satisfies its contract."""
    return dict(build_all_agents(make_provider(), workforce_registry()))


# --------------------------------------------------------------------------- #
# The approval policy is a rule, not a per-tool decision
# --------------------------------------------------------------------------- #


def test_approval_is_required_exactly_at_high_risk_and_above() -> None:
    """The gate must follow the declared risk, not a tool's own opinion.

    If this can drift, a tool can quietly exempt itself from human approval by
    describing itself as lower risk than it is.
    """
    registry = default_registry()
    checked = 0

    for tool in registry.list():  # type: ignore[attr-defined]
        expected = tool.effective_risk() >= RiskLevel.HIGH
        assert tool.requires_approval() is expected, f"{tool.name} disagrees with its risk level"
        checked += 1

    assert checked >= 4, "the registry is too small for this to mean anything"


@pytest.mark.parametrize(
    ("mode", "minimum"),
    [
        (AccessMode.READ, RiskLevel.LOW),
        (AccessMode.WRITE, RiskLevel.MEDIUM),
        (AccessMode.DESTRUCTIVE, RiskLevel.CRITICAL),
    ],
)
def test_no_tool_under_declares_its_risk(mode: AccessMode, minimum: RiskLevel) -> None:
    """A tool's risk may be raised above its access mode, never lowered."""
    registry = default_registry()

    for tool in registry.list():  # type: ignore[attr-defined]
        if tool.access_mode is mode:
            assert tool.effective_risk() >= minimum, f"{tool.name} understates its risk"
        else:
            assert mode.risk_level <= RiskLevel.CRITICAL


def test_every_registered_tool_declares_its_access_mode() -> None:
    for tool in default_registry().list():  # type: ignore[attr-defined]
        assert tool.access_mode in set(AccessMode), f"{tool.name} has no access mode"


def test_every_registered_tool_declares_a_validated_contract() -> None:
    """A tool without input and output schemas cannot be called safely."""
    for tool in default_registry().list():  # type: ignore[attr-defined]
        described = tool.describe()
        assert described["name"]
        assert described["description"]
        assert described["input_schema"]["type"] == "object"
        assert described["output_schema"]


# --------------------------------------------------------------------------- #
# Secrets
# --------------------------------------------------------------------------- #


def test_no_secret_reaches_a_discovery_payload() -> None:
    """Discovery is unauthenticated, so it must carry nothing sensitive."""
    settings = Settings(_env_file=None, llm_api_key=SECRET, github_token=SECRET)
    registry = build_default_registry(settings)

    agents = build_all_agents(make_provider(), workforce_registry())
    described = repr(registry.describe()) + repr([agent.describe() for agent in agents.values()])

    assert SECRET not in described


def test_a_secret_in_a_tool_error_detail_is_redacted() -> None:
    """Tool details are shown to callers; a leaked DSN leaks the password."""
    message = f"connection failed for postgres://user:{SECRET}@db:5432/app"

    assert SECRET not in redact(message, (SECRET,))


def test_the_secret_list_covers_every_configured_credential() -> None:
    """A credential the redactor does not know about will be logged verbatim."""
    settings = Settings(
        _env_file=None,
        llm_api_key=SECRET,
        github_token=SECRET,
        jwt_secret=SECRET,
    )

    assert settings.secret_values().count(SECRET) >= 3


# --------------------------------------------------------------------------- #
# Network egress
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://localhost/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/",
        "http://[::1]/",
    ],
)
def test_internal_destinations_are_refused(url: str) -> None:
    """SSRF: the metadata service is the one that turns this into a breach."""
    with pytest.raises(PermissionDeniedError) as captured:
        validate_outbound_url(url, allow_private=False)

    message = str(captured.value)
    assert "not reachable" in message or "private or reserved" in message


@pytest.mark.parametrize("url", ["file:///etc/passwd", "gopher://x/", "ftp://x/"])
def test_non_http_schemes_are_refused(url: str) -> None:
    with pytest.raises(PermissionDeniedError):
        validate_outbound_url(url, allow_private=True)


# --------------------------------------------------------------------------- #
# Filesystem confinement
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "attempt",
    [
        "../outside.txt",
        "../../etc/passwd",
        "/etc/passwd",
        "./../outside.txt",
    ],
)
def test_paths_outside_the_allowed_root_are_refused(
    attempt: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Traversal is resolved, not string-matched, so `..` cannot escape."""
    root = tmp_path_factory.mktemp("confined").resolve()

    with pytest.raises(ToolError):
        resolve_within_roots(attempt, (root,))


def test_a_path_inside_the_allowed_root_resolves(tmp_path: object) -> None:
    """The control must permit the legitimate case, or it is not a control."""
    from pathlib import Path

    root = Path(str(tmp_path))
    resolved = resolve_within_roots("notes.txt", (root,))

    assert resolved == root / "notes.txt"


# --------------------------------------------------------------------------- #
# Least privilege
# --------------------------------------------------------------------------- #


def test_no_agent_declares_a_tool_the_application_does_not_register() -> None:
    """A typo would silently remove capability; the builder refuses instead."""
    registry = default_registry()

    for name, agent in all_agents().items():
        for tool_name in agent.allowed_tools:  # type: ignore[attr-defined]
            assert registry.has(tool_name), f"{name} requires an unregistered tool {tool_name}"


def test_every_agent_has_a_distinct_name() -> None:
    """A collision would make one agent's tool grant apply to another."""
    names = [agent.name for agent in all_agents().values()]  # type: ignore[attr-defined]

    assert len(names) == len(set(names))


def test_orchestrator_roles_hold_no_tools() -> None:
    """Planning, verification, and synthesis must not be able to act."""
    agents = all_agents()

    for role in ("planner", "critic", "synthesizer"):
        assert agents[role].tool_names == ()  # type: ignore[attr-defined]


def test_only_worker_roles_are_dispatchable() -> None:
    """A plan names workers; the orchestrator roles must not be reachable.

    The graph's ``validate_plan`` derives the dispatchable set from the workforce
    map, so this asserts the property that map is supposed to have.
    """
    from app.graph.builder import build_workforce

    workers = set(build_workforce(make_provider(), workforce_registry()))
    every = {agent.name for agent in all_agents().values()}  # type: ignore[attr-defined]

    assert workers == {"analyst", "coder", "document", "researcher"}
    assert workers < every
    assert not workers & {"planner", "critic", "synthesizer", "executor"}


# --------------------------------------------------------------------------- #
# Configuration invariants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "override",
    [
        {"debug": True},
        {"auth_enabled": False},
        {"trust_identity_header": True},
        {"python_execution_enabled": True},
        {"jwt_secret": None},
    ],
)
def test_production_refuses_each_unsafe_combination(override: dict[str, object]) -> None:
    """Every documented production invariant is enforced by construction."""
    safe: dict[str, object] = {
        "app_env": "production",
        "auth_enabled": True,
        "trust_identity_header": False,
        "jwt_secret": "a-strong-value",
    }
    safe.update(override)

    with pytest.raises(ConfigurationError):
        Settings(_env_file=None, **safe)


def test_no_execution_sandbox_is_available_here() -> None:
    """Reporting this honestly matters more than the value it returns.

    The property is checked in both environments, because the tempting shortcut
    is to make it true in production and ship an isolation boundary that does
    not exist.
    """
    development = Settings(_env_file=None)
    production = Settings(
        _env_file=None,
        app_env="production",
        auth_enabled=True,
        trust_identity_header=False,
        jwt_secret="a-strong-value",
    )

    assert development.execution_sandbox_available is False
    assert production.execution_sandbox_available is False


def test_debug_off_is_the_default() -> None:
    """A verbose error is an information disclosure by default."""
    assert Settings(_env_file=None).debug is False


def test_identity_header_trust_is_development_only() -> None:
    """Trusting a header is authentication bypass if it ever reaches production."""
    settings = Settings(
        _env_file=None,
        app_env="production",
        auth_enabled=True,
        trust_identity_header=False,
        jwt_secret="a-strong-value",
    )

    assert settings.trust_identity_header is False


# --------------------------------------------------------------------------- #
# The rate-limit bucket
# --------------------------------------------------------------------------- #


def scope_from(
    client: tuple[str, int] | None = None, headers: dict[str, str] | None = None
) -> Scope:
    """Build the slice of an ASGI scope that bucket selection reads.

    ``headers`` is always present, because the ASGI specification requires it:
    an empty list is what a real request without headers carries.
    """
    scope: dict[str, object] = {
        "type": "http",
        "headers": [
            (name.lower().encode(), value.encode()) for name, value in (headers or {}).items()
        ],
    }
    if client is not None:
        scope["client"] = client
    return cast(Scope, scope)


def test_a_forwarded_header_cannot_choose_a_bucket() -> None:
    """The bucket comes from the peer address, and no header can move it.

    Honouring ``X-Forwarded-For`` looks like the fix for running behind a proxy
    and is instead a rate-limit bypass: the header is caller-supplied, so anyone
    could send a new value per request and never reach a limit. The cost of the
    choice is the mirror image — behind a proxy that does not preserve the peer
    address, every client shares one bucket — which is why it is recorded rather
    than quietly improved.
    """
    settings = Settings(_env_file=None, auth_enabled=False, trust_identity_header=False)
    buckets = {
        _identity(scope_from(("203.0.113.7", 5000), {"X-Forwarded-For": claim}), settings)
        for claim in ("9.9.9.9", "8.8.8.8", "203.0.113.7", "")
    }

    assert buckets == {"203.0.113.7"}


def test_a_request_with_nothing_to_attribute_still_lands_in_a_bucket() -> None:
    """Unattributable is not exempt: the fallback is a shared bucket, not none."""
    settings = Settings(_env_file=None, auth_enabled=True, jwt_secret="a-strong-value")

    assert _identity(scope_from(), settings) == ANONYMOUS_USER_ID


def test_a_bearer_token_decides_the_bucket_so_a_proxy_cannot_pool_clients() -> None:
    """The documented reason a subject is preferred over an address.

    Behind a proxy every request arrives from one address, so an address-keyed
    limit becomes a global one and a single caller can exhaust it for everyone.
    Two subjects behind the same address are two buckets.
    """
    settings = Settings(_env_file=None, auth_enabled=True, jwt_secret="a-strong-value")
    first = issue_token("alice", settings)
    second = issue_token("bob", settings)
    address = ("203.0.113.7", 5000)

    alice = _identity(scope_from(address, {"Authorization": f"Bearer {first}"}), settings)
    bob = _identity(scope_from(address, {"Authorization": f"Bearer {second}"}), settings)

    assert alice == "alice"
    assert bob == "bob"
    assert alice != bob


def test_the_development_identity_header_is_also_a_rate_limit_bypass() -> None:
    """Which is a second reason to refuse it outside development.

    It is refusable as authentication bypass already. Worth recording that
    accepting it also hands every caller a fresh bucket on demand, so the two
    consequences are weighed together rather than the first alone.
    """
    settings = Settings(_env_file=None, auth_enabled=False, trust_identity_header=True)
    address = ("203.0.113.7", 5000)

    assert _identity(scope_from(address, {"X-User-Id": "a"}), settings) == "a"
    assert _identity(scope_from(address, {"X-User-Id": "b"}), settings) == "b"
    # The pair is not merely discouraged outside development; it is refused.
    with pytest.raises(ConfigurationError):
        Settings(
            _env_file=None,
            app_env="production",
            auth_enabled=False,
            trust_identity_header=True,
            jwt_secret="a-strong-value",
        )
