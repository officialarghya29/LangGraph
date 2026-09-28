"""Tests for the typed configuration layer."""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from app.core.config import (
    DEFAULT_EMBEDDING_MODELS,
    DEFAULT_LLM_MODELS,
    EmbeddingProviderName,
    LLMProviderName,
    Settings,
    get_settings,
    reset_settings_cache,
)
from app.core.exceptions import ConfigurationError

#: Every application module, for the "is this setting read anywhere?" checks below.
#: Reading source text is a blunt instrument, but it is the only one that catches
#: the failure it is aimed at: a setting that is parsed, validated, documented, and
#: then never consulted, which no behavioural test can observe.
_APP_SOURCES = sorted(Path("app").rglob("*.py"))


def make_settings(**overrides: object) -> Settings:
    """Build settings that ignore any local .env file."""
    return Settings(_env_file=None, **overrides)


def test_defaults_are_sane() -> None:
    settings = make_settings()

    assert settings.app_env == "development"
    assert settings.debug is False
    assert settings.max_agent_iterations == 10
    assert settings.max_tool_calls == 25
    assert settings.python_execution_enabled is False
    assert settings.github_tool_allow_writes is False


def test_every_provider_has_a_default_model() -> None:
    """A provider with no entry would raise ``KeyError`` on the first call.

    The resolution is a dictionary lookup keyed by provider, so adding a value to
    ``LLMProviderName`` without adding a model would move the failure from startup
    to the first request — which is the thing this configuration layer exists to
    prevent. Checking the keys against the literal keeps the two in step.
    """
    assert set(DEFAULT_LLM_MODELS) == set(get_args(LLMProviderName))
    assert set(DEFAULT_EMBEDDING_MODELS) == set(get_args(EmbeddingProviderName))


def test_an_unset_model_resolves_to_the_provider_default() -> None:
    """Which is the whole point of keeping it unset."""
    settings = make_settings()

    assert settings.llm_model is None
    assert settings.llm_model_name == DEFAULT_LLM_MODELS["openai"]


def test_switching_provider_switches_the_model() -> None:
    """Otherwise ``LLM_PROVIDER=anthropic`` alone sends a GPT name to Anthropic.

    That is a real request-time failure — a model-not-found error that names the
    model rather than the configuration, from a setting the operator never
    touched.
    """
    anthropic = make_settings(llm_provider="anthropic", llm_api_key="key")

    assert anthropic.llm_model is None
    assert anthropic.llm_model_name == DEFAULT_LLM_MODELS["anthropic"]
    assert "gpt" not in anthropic.llm_model_name


def test_a_named_model_still_wins() -> None:
    """The default is a fallback, not a policy."""
    settings = make_settings(llm_model="gpt-5.6-sol")

    assert settings.llm_model_name == "gpt-5.6-sol"


@pytest.mark.parametrize("field", ["llm_temperature", "llm_max_output_tokens"])
def test_the_sampling_settings_are_read_by_the_code(field: str) -> None:
    """A declared setting nobody reads is a setting that does nothing.

    ``LLM_MAX_RETRIES`` and ``TRACE_HISTORY_SIZE`` were both declared and unread
    until they were wired up; this is the cheap check that catches the next one.
    """
    readers = [path for path in _APP_SOURCES if f"settings.{field}" in path.read_text()]

    assert readers, f"{field} is declared but never read"


def test_a_blank_ceiling_means_the_vendor_default() -> None:
    """``LLM_MAX_OUTPUT_TOKENS=`` is how the template ships it.

    It is not parseable as an integer, so before the tri-state handling it stopped
    the process from starting — over the value that means "do not cap this".
    """
    settings = make_settings(llm_max_output_tokens="  ")

    assert settings.llm_max_output_tokens is None


def test_the_temperature_is_bounded() -> None:
    """A temperature outside the vendor's range is a 400, so it fails at startup."""
    with pytest.raises(ValidationError):
        make_settings(llm_temperature=2.5)

    with pytest.raises(ValidationError):
        make_settings(llm_temperature=-0.1)


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_NAME", "custom-service")
    monkeypatch.setenv("MAX_AGENT_ITERATIONS", "42")
    monkeypatch.setenv("DEBUG", "true")

    settings = make_settings()

    assert settings.app_name == "custom-service"
    assert settings.max_agent_iterations == 42
    assert settings.debug is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_agent_iterations", 0),
        ("max_tool_calls", 0),
        ("max_parallel_tasks", 0),
        ("max_execution_time", 0),
        ("max_retries", -1),
        ("llm_timeout_seconds", 0),
    ],
)
def test_execution_limits_reject_non_positive_values(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        make_settings(**{field: value})


def test_execution_limits_reject_absurd_values() -> None:
    with pytest.raises(ValidationError):
        make_settings(max_agent_iterations=10_000)


def test_allowed_roots_parses_comma_separated_paths() -> None:
    settings = make_settings(filesystem_allowed_roots="./workspace, /tmp/sandbox")

    roots = settings.allowed_roots

    assert len(roots) == 2
    assert all(isinstance(root, Path) for root in roots)
    assert all(root.is_absolute() for root in roots)


def test_allowed_roots_ignores_blank_entries() -> None:
    settings = make_settings(filesystem_allowed_roots=" ./workspace ,, ")

    assert len(settings.allowed_roots) == 1


def test_secret_values_collects_only_configured_secrets() -> None:
    settings = make_settings(llm_api_key="llm-secret", jwt_secret="jwt-secret")

    values = settings.secret_values()

    assert set(values) == {"llm-secret", "jwt-secret"}


def test_secrets_are_not_exposed_by_repr() -> None:
    settings = make_settings(llm_api_key="super-secret-value")

    assert "super-secret-value" not in repr(settings)


def safe_production(**overrides: object) -> Settings:
    """Build a production configuration that satisfies every safety invariant.

    Each production test below changes exactly one value, so a failure names the
    invariant that broke rather than an unrelated one that happened to be
    defaulted wrong.
    """
    base: dict[str, object] = {
        "app_env": "production",
        "auth_enabled": True,
        "trust_identity_header": False,
        "jwt_secret": "strong-value",
    }
    base.update(overrides)
    return make_settings(**base)


def test_is_production() -> None:
    assert safe_production().is_production is True
    assert make_settings().is_production is False


def test_api_docs_are_served_outside_production() -> None:
    """While building against the API, the schema is a convenience."""
    assert make_settings(app_env="development").serve_api_docs is True
    assert make_settings(app_env="staging").serve_api_docs is True


def test_api_docs_are_withheld_in_production_by_default() -> None:
    """Deployed, the schema describes the attack surface, not the product."""
    assert safe_production().serve_api_docs is False


def test_api_docs_can_be_enabled_explicitly_in_any_environment() -> None:
    """A deployment that fronts its docs with a gateway can still expose them."""
    assert safe_production(api_docs_enabled=True).serve_api_docs is True
    assert make_settings(app_env="development", api_docs_enabled=False).serve_api_docs is False


def test_a_blank_secret_is_treated_as_absent() -> None:
    """``LLM_API_KEY=`` in a .env file is how a person configures nothing.

    pydantic reads it as the empty string, not as absent, and that difference
    produced real bugs: a provider was built with an empty key and failed at the
    transport layer with "Illegal header value b'Bearer '", which reads like a
    client bug rather than a missing credential.
    """
    assert make_settings(llm_api_key="").llm_api_key is None
    assert make_settings(llm_api_key="   ").llm_api_key is None
    assert make_settings(jwt_secret="").jwt_secret is None
    assert make_settings(github_token="\n").github_token is None
    assert make_settings(search_api_key="").search_api_key is None
    assert make_settings(embedding_api_key="").embedding_api_key is None


def test_a_secret_is_trimmed_but_never_altered() -> None:
    """A key pasted with a trailing newline would make a header a server rejects."""
    configured = make_settings(llm_api_key="  sk-abc123  ")

    assert configured.llm_api_key is not None
    assert configured.llm_api_key.get_secret_value() == "sk-abc123"


def test_a_blank_url_is_treated_as_unset() -> None:
    """The same failure one layer on: an empty target is not a target.

    ``DATABASE_TOOL_URL=`` would have registered the database tool, because the
    deciding check is an ``is not None`` test.
    """
    assert make_settings(database_tool_url="").database_tool_url is None
    assert make_settings(llm_base_url="  ").llm_base_url is None
    assert make_settings(search_api_url="").search_api_url is None
    assert (
        make_settings(database_tool_url="postgresql://x/y").database_tool_url == "postgresql://x/y"
    )


def test_production_refuses_a_blank_signing_key() -> None:
    """The hole this closed: an empty secret is truthy as an object.

    ``not SecretStr("")`` is false, so the validator's ``is None`` guard passed
    for an empty string and a deployment could start believing it had a key.
    """
    with pytest.raises(ConfigurationError):
        Settings(
            _env_file=None,
            app_env="production",
            auth_enabled=True,
            trust_identity_header=False,
            jwt_secret="",
        )


def test_production_refuses_a_whitespace_signing_key() -> None:
    """Padding is not a key either."""
    with pytest.raises(ConfigurationError):
        Settings(
            _env_file=None,
            app_env="production",
            auth_enabled=True,
            trust_identity_header=False,
            jwt_secret="    ",
        )


def test_the_operator_console_is_off_unless_it_is_asked_for() -> None:
    """A control surface must not be one environment variable away from live.

    The default is the shipped behaviour, so it is asserted directly rather than
    inferred from whatever the developer's ``.env`` happens to contain.
    """
    assert make_settings().dashboard_enabled is False
    assert make_settings(dashboard_enabled=True).dashboard_enabled is True


# --------------------------------------------------------------------------- #
# Production safety
# --------------------------------------------------------------------------- #


def test_production_rejects_debug() -> None:
    with pytest.raises(ConfigurationError, match="DEBUG"):
        safe_production(debug=True)


def test_production_rejects_disabled_auth() -> None:
    with pytest.raises(ConfigurationError, match="AUTH_ENABLED"):
        safe_production(auth_enabled=False)


def test_production_requires_a_jwt_secret() -> None:
    with pytest.raises(ConfigurationError, match="JWT_SECRET"):
        make_settings(app_env="production", trust_identity_header=False)


def test_production_rejects_a_trusted_identity_header() -> None:
    """A self-asserted identity header would defeat every ownership check."""
    with pytest.raises(ConfigurationError, match="TRUST_IDENTITY_HEADER"):
        safe_production(trust_identity_header=True)


def test_production_rejects_execution_without_a_sandbox() -> None:
    with pytest.raises(ConfigurationError, match="sandbox"):
        safe_production(python_execution_enabled=True)


def test_no_environment_offers_a_code_execution_sandbox() -> None:
    """The property must not be optimistic in any environment."""
    assert make_settings().execution_sandbox_available is False
    assert safe_production().execution_sandbox_available is False


def test_production_accepts_a_safe_configuration() -> None:
    settings = safe_production()

    assert settings.is_production is True
    assert settings.auth_enabled is True
    assert settings.trust_identity_header is False


def test_development_tolerates_unsafe_settings() -> None:
    settings = make_settings(debug=True, auth_enabled=False)

    assert settings.debug is True


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #


def test_get_settings_is_cached() -> None:
    reset_settings_cache()

    assert get_settings() is get_settings()

    reset_settings_cache()


def test_reset_settings_cache_forces_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_settings_cache()
    first = get_settings()

    monkeypatch.setenv("APP_NAME", "reloaded")
    reset_settings_cache()
    second = get_settings()

    assert first is not second
    assert second.app_name == "reloaded"

    reset_settings_cache()
