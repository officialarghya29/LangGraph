"""Tests that the deployment artefacts agree with the application.

Neither the ``Dockerfile`` nor ``docker-compose.yml`` can be executed on this
machine: there is no container runtime. That is a reason to check what *can* be
checked statically rather than a reason to check nothing, and the highest-value
static check is the one that catches a typo.

The settings layer ignores unknown environment keys, so
``TRUST_IDENTITIY_HEADER=false`` in a compose file is accepted and does nothing —
leaving the trusted identity header on while the file says otherwise. Everything
below is derived from the code rather than from a second copy of the truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from app.core.config import Settings

COMPOSE = Path("docker-compose.yml")
DOCKERFILE = Path("Dockerfile")
DOCKERIGNORE = Path(".dockerignore")


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    """Return the parsed compose document."""
    return yaml.safe_load(COMPOSE.read_text())


def api_environment(compose: dict[str, Any]) -> dict[str, str]:
    """Return the environment the API service is given."""
    return {str(k): str(v) for k, v in compose["services"]["api"]["environment"].items()}


def test_the_compose_file_parses() -> None:
    """A syntax error here is a deployment that cannot start."""
    document = yaml.safe_load(COMPOSE.read_text())

    assert document["services"]["api"]
    assert document["services"]["postgres"]
    assert document["services"]["redis"]


def test_compose_sets_only_settings_that_exist(compose: dict[str, Any]) -> None:
    """A misspelled key is silently ignored, so it has to be caught here."""
    known = {name.upper() for name in Settings.model_fields}
    unknown = sorted(set(api_environment(compose)) - known)

    assert unknown == [], f"compose sets variables that are not settings: {unknown}"


def test_compose_does_not_claim_to_be_production_without_the_invariants(
    compose: dict[str, Any],
) -> None:
    """If it ever says ``production``, it has to satisfy what the code requires.

    The compose file is a local stack and says so. This test exists so that
    changing that one line cannot quietly ship an unauthenticated deployment:
    switching to production without the other three settings raises at startup.
    """
    environment = api_environment(compose)
    if environment.get("APP_ENV") != "production":
        pytest.skip("the local stack declares development")

    assert environment.get("AUTH_ENABLED") == "true"
    assert environment.get("TRUST_IDENTITY_HEADER") == "false"
    assert environment.get("JWT_SECRET")


def test_compose_points_at_the_services_it_starts(compose: dict[str, Any]) -> None:
    """A hostname that is not a service name is a connection error at boot."""
    environment = api_environment(compose)
    names = set(compose["services"])

    for key in ("DATABASE_URL", "REDIS_URL"):
        host = environment[key].split("@")[-1].split("/")[0].split(":")[0]
        assert host in names, f"{key} points at {host!r}, which compose does not define"


def test_the_api_waits_for_its_dependencies(compose: dict[str, Any]) -> None:
    """Starting first means failing the readiness check, or migrating into nothing."""
    depends = compose["services"]["api"]["depends_on"]

    assert depends["postgres"]["condition"] == "service_healthy"
    assert depends["redis"]["condition"] == "service_healthy"


def test_the_image_does_not_copy_the_environment_file() -> None:
    """A real ``.env`` in the image is a credential inside an image layer."""
    ignored = DOCKERIGNORE.read_text().splitlines()

    assert ".env" in ignored
    assert ".env.*" in ignored


def test_the_image_copies_what_the_start_up_command_needs() -> None:
    """The compose command runs migrations, so the migrations must be in the image."""
    body = DOCKERFILE.read_text()

    assert "COPY --chown=appuser:appuser migrations ./migrations" in body
    assert "COPY --chown=appuser:appuser alembic.ini pyproject.toml ./" in body
    assert "alembic upgrade head && uvicorn" in COMPOSE.read_text()


def test_the_image_runs_as_a_non_root_user() -> None:
    """Nothing in the application writes to its own filesystem."""
    body = DOCKERFILE.read_text()

    assert "USER appuser" in body
    assert "useradd" in body


def test_the_image_health_check_reads_readiness() -> None:
    """A liveness-only check reports healthy while the database is unreachable."""
    body = DOCKERFILE.read_text()

    assert "HEALTHCHECK" in body
    assert "/ready" in body


def test_the_image_defaults_to_production() -> None:
    """The unsafe default is the one that ships to production by accident."""
    assert "APP_ENV=production" in DOCKERFILE.read_text()
