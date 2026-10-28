"""The startup catalogue check is wired, not merely written.

The unit tests prove what the check says. These prove it is asked: that a
deployment environment schedules it, that development does not — the test suite
must never need a vendor endpoint — and that the task is awaited or cancelled
cleanly rather than left to be destroyed with the loop.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.core.config import reset_settings_cache
from app.main import create_app

#: Everything a non-development environment needs to boot here. Staging is used
#: rather than production because the production invariants (real auth, secret
#: keys) are the subject of their own tests and would obscure this one.
_STAGING_ENV = {
    "APP_ENV": "staging",
    "AUTH_ENABLED": "true",
    "TRUST_IDENTITY_HEADER": "false",
    "JWT_SECRET": "test-secret-value-for-the-startup-check-only",
    "LLM_API_KEY": "test-key",
    "CHECKPOINT_BACKEND": "postgres",
}


@pytest.fixture
def environment(test_database_url: str) -> Iterator[dict[str, Any]]:
    """Point the process at a deployment-like environment for one test.

    Settings are cached, so the cache is cleared on both sides: a staging value
    left behind would silently change what every later test sees.
    """
    del test_database_url
    previous = {key: os.environ.get(key) for key in _STAGING_ENV}
    recorded: dict[str, Any] = {"asked": asyncio.Event()}

    async def catalogue(provider: Any) -> tuple[str, ...]:
        recorded["asked"].set()
        recorded["provider"] = provider
        return ("gpt-6.1-sol",)

    original = main_module.catalogue_status
    main_module.catalogue_status = catalogue  # type: ignore[assignment]
    try:
        yield recorded
    finally:
        main_module.catalogue_status = original  # type: ignore[assignment]
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_settings_cache()


async def test_a_deployment_asks_the_catalogue_once_at_startup(
    environment: dict[str, Any],
) -> None:
    """Staging is a deployment: it must ask, and it must not block serving."""
    for key, value in _STAGING_ENV.items():
        os.environ[key] = value
    reset_settings_cache()

    application = create_app()
    with TestClient(application) as client:
        assert application.state.catalogue_check is not None
        # Serving begins before the answer arrives: the request below would fail
        # on a startup that awaited a vendor.
        assert client.get("/health").status_code in {200, 503}
        await asyncio.wait_for(environment["asked"].wait(), timeout=5)

    assert application.state.catalogue_check.done()


async def test_development_never_calls_a_vendor_at_startup(
    environment: dict[str, Any],
) -> None:
    """The suite runs offline; a development boot must not reach for a catalogue."""
    for key in _STAGING_ENV:
        os.environ.pop(key, None)
    os.environ["APP_ENV"] = "development"
    reset_settings_cache()

    application = create_app()
    with TestClient(application) as client:
        assert client.get("/health").status_code in {200, 503}

    task = getattr(application.state, "catalogue_check", None)
    assert task is None or task.done()
    assert not environment["asked"].is_set()


async def test_a_catalogue_that_cannot_answer_does_not_stop_the_app(
    environment: dict[str, Any],
) -> None:
    """An unreadable catalogue completes the task and leaves the app serving."""
    for key, value in _STAGING_ENV.items():
        os.environ[key] = value
    os.environ["LLM_BASE_URL"] = "http://127.0.0.1:9/v1"  # nothing listens here
    reset_settings_cache()

    try:
        application = create_app()
        with TestClient(application) as client:
            task = application.state.catalogue_check
            assert task is not None
            await asyncio.wait_for(task, timeout=10)
            assert task.exception() is None, "a refused catalogue must not raise out of the task"
            assert client.get("/health").status_code in {200, 503}
    finally:
        os.environ.pop("LLM_BASE_URL", None)
        reset_settings_cache()
