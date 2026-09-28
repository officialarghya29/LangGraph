"""Tests that the environment template and the settings layer agree.

The settings layer ignores keys it does not know. That is the right behaviour for
a deployed process — an unrelated variable in the environment must not stop the
application from starting — but it makes ``.env.example`` unverifiable by reading
it: a typo in a key name is accepted in silence, and the setting it meant to
configure stays at its default while the file says otherwise. The same drift runs
the other way too, and that direction is worse in practice, because the settings
that go undocumented are the ones an operator never learns exist — among them
``TRUST_IDENTITY_HEADER``, ``RATE_LIMIT_FAIL_CLOSED``, and
``ALLOW_PRIVATE_NETWORK_EGRESS``.

These tests fail in all three directions: a key that is not a setting, a setting
with no key, and a template that does not parse as valid configuration.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.core.config import DEFAULT_EMBEDDING_MODELS, DEFAULT_LLM_MODELS, Settings

TEMPLATE = Path(".env.example")

#: ``KEY=value``, ignoring comments and blank lines.
_KEY = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", re.MULTILINE)


def documented_keys() -> set[str]:
    """Return the setting names the template mentions."""
    return {match.group(1).upper() for match in _KEY.finditer(TEMPLATE.read_text())}


def settings_names() -> set[str]:
    """Return every setting the application reads."""
    return {name.upper() for name in Settings.model_fields}


def test_the_template_exists() -> None:
    """It is the file the README tells an operator to copy."""
    assert TEMPLATE.is_file()


def test_every_documented_key_is_a_real_setting() -> None:
    """A typo here would be swallowed by ``extra="ignore"`` and configure nothing."""
    unknown = sorted(documented_keys() - settings_names())

    assert unknown == [], f"template documents settings that do not exist: {unknown}"


def test_every_setting_is_documented() -> None:
    """An undocumented setting is one nobody can discover or turn off."""
    missing = sorted(settings_names() - documented_keys())

    assert missing == [], f"settings missing from the template: {missing}"


def test_the_documented_defaults_match_the_code() -> None:
    """A template whose numbers disagree with the defaults is a trap.

    Only the values that the template states are checked, and only where the code
    declares the same default: a template that sets a value deliberately different
    from the default (a non-standard port for the local verification instance, for
    example) is exactly what it is for.
    """
    settings = Settings(_env_file=None)
    body = TEMPLATE.read_text()

    for name in ("MAX_TOOL_CALLS", "MAX_EXECUTION_TIME", "MAX_TOKEN_BUDGET"):
        stated = re.search(rf"^{name}=(\S+)", body, re.MULTILINE)
        assert stated is not None, name
        assert int(stated.group(1)) == getattr(settings, name.lower()), name


def test_the_template_parses_as_configuration() -> None:
    """Copying it must produce a working configuration, not a startup error."""
    loaded = Settings(_env_file=str(TEMPLATE))

    assert loaded.app_env == "development"
    assert loaded.checkpoint_backend == "postgres"


def test_a_blank_value_falls_back_to_the_default() -> None:
    """``LLM_MODEL=`` must not become an empty model name.

    It is valid input for a ``str`` field, so it used to be accepted, and the
    provider was then called with no model at all — an error about the request
    rather than about the configuration, from a file that looked correctly filled
    in.
    """
    loaded = Settings(_env_file=str(TEMPLATE), llm_model="   ", embedding_model="")

    assert loaded.llm_model is None
    assert loaded.embedding_model is None
    # ``None`` is not the answer that reaches the wire: the provider's own default
    # is resolved in its place.
    assert loaded.llm_model_name == DEFAULT_LLM_MODELS["openai"]
    assert loaded.embedding_model_name == DEFAULT_EMBEDDING_MODELS["local"]


def test_a_blank_secret_is_absent_rather_than_empty() -> None:
    """The template ships every secret blank, and blank must mean unset."""
    loaded = Settings(
        _env_file=str(TEMPLATE),
        llm_api_key="",
        github_token="   ",
        search_api_key="",
    )

    assert loaded.llm_api_key is None
    assert loaded.github_token is None
    assert loaded.search_api_key is None


@pytest.mark.parametrize("name", ["EMBEDDING_PROVIDER", "LLM_PROVIDER", "CHECKPOINT_BACKEND"])
def test_the_template_offers_only_values_the_code_accepts(name: str) -> None:
    """The comments are the documentation, so they have to be true.

    ``EMBEDDING_PROVIDER=local  # cloud | local`` was wrong for a while: ``cloud``
    was never an accepted value, and an operator following the comment would have
    found out at startup.
    """
    body = TEMPLATE.read_text()
    line = next(line for line in body.splitlines() if line.startswith(f"{name}="))

    chosen = line.split("=", 1)[1].split()[0]
    assert chosen in {"local", "openai", "postgres", "memory", "anthropic"}

    accepted = line.split("#", 1)[1].split() if "#" in line else []
    for candidate in accepted:
        if candidate == "|":
            continue
        assert candidate in {"local", "openai", "postgres", "memory", "anthropic"}, (
            f"{name} comment advertises {candidate!r}, which the code does not accept"
        )
