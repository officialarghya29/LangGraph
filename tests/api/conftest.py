"""Fixtures for the HTTP API tests.

These are end-to-end tests, not handler unit tests: they build the application
through its real factory, run its real lifespan, and talk to a real PostgreSQL
and Redis. That is the only way to test what this project actually promises —
that a task is recorded durably, that an approval survives, that a stream
resumes — so the cost of requiring the services is accepted deliberately.

What is replaced: the language model. A scripted provider keeps the suite
offline and deterministic. Everything else is real.

The database is the throwaway one from :mod:`tests.conftest`, so a run cannot
damage development data.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.auth import issue_token
from app.core.config import Settings, reset_settings_cache
from app.graph.builder import build_all_agents, build_graph
from app.main import create_app
from app.services.budget import BudgetedProvider
from app.services.llm import FakeLLMProvider
from app.services.resilience import RetryingProvider
from app.services.tasks import InMemoryTaskStore
from tests.conftest import TRUNCATE_ALL

#: Redis keys this suite creates all live under this prefix. Redis outlives a
#: test run, so without a namespace of its own the suite would inherit
#: rate-limit counters from whatever ran last.
_REDIS_PREFIX = "langgraph-test"

#: Environment the API tests run under. Set explicitly rather than inherited, so
#: a developer's ``.env`` cannot change what the suite asserts.
_API_ENV = {
    # The settings layer accepts only development, staging, or production.
    # Production is refused on purpose: it enforces the safety invariants that
    # the trusted-header tests rely on.
    "APP_ENV": "development",
    "AUTH_ENABLED": "false",
    "TRUST_IDENTITY_HEADER": "true",
    "CHECKPOINT_BACKEND": "postgres",
    "REDIS_KEY_PREFIX": _REDIS_PREFIX,
    # Generous by default; the rate-limit tests set their own limit.
    "RATE_LIMIT_REQUESTS": "10000",
    "RATE_LIMIT_WINDOW_SECONDS": "60",
    "LLM_API_KEY": "test-key",
    # Blank, and set explicitly rather than inherited, because the registry is
    # built from what is configured: without this the tools under test would
    # depend on whether the machine running the suite happens to hold a GitHub
    # token. A blank value is also the case worth pinning down, since an empty
    # token must not register a tool that can only fail when it is called.
    "GITHUB_TOKEN": "",
    "SEARCH_API_URL": "",
}


def purge_test_cache() -> None:
    """Delete every Redis key this suite could have created.

    Scoped by scan-and-delete rather than ``FLUSHDB``, because the suite shares
    a server with development. Deleting only this suite's namespace cannot
    touch anything a developer is working on.
    """
    import redis as redis_client

    from app.core.config import Settings

    client = redis_client.Redis.from_url(Settings().redis_url)
    try:
        for key in client.scan_iter(match=f"{_REDIS_PREFIX}:*", count=500):
            client.delete(key)
    finally:
        client.close()


@pytest.fixture(scope="session", autouse=True)
def _isolated_cache(test_database_url: str) -> Iterator[None]:
    """Give the suite a Redis namespace of its own, emptied before and after."""
    del test_database_url
    previous = os.environ.get("REDIS_KEY_PREFIX")
    os.environ["REDIS_KEY_PREFIX"] = _REDIS_PREFIX
    reset_settings_cache()
    purge_test_cache()
    try:
        yield
    finally:
        purge_test_cache()
        if previous is None:
            os.environ.pop("REDIS_KEY_PREFIX", None)
        else:
            os.environ["REDIS_KEY_PREFIX"] = previous
        reset_settings_cache()


@pytest.fixture(autouse=True)
def _api_environment(test_database_url: str) -> Iterator[None]:
    """Apply the API test environment for the duration of one test.

    Function-scoped, not session-scoped, so that a setting this suite needs —
    an API key, disabled authentication — cannot leak into the other suites and
    quietly change what they are testing.
    """
    del test_database_url
    original = {key: os.environ.get(key) for key in _API_ENV}
    os.environ.update(_API_ENV)
    reset_settings_cache()
    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_settings_cache()


class ApiHarness:
    """A running application, its client, and the pieces a test needs to steer."""

    def __init__(self, client: TestClient, app: Any, provider: FakeLLMProvider) -> None:
        self.client = client
        self.app = app
        self.provider = provider

    def rewire(self, responder: Any) -> None:
        """Replace the scripted provider behaviour without rebuilding the app.

        The compiled graph holds a reference to the provider, so the behaviour is
        swapped on that same object rather than by replacing it — replacing it
        would leave the graph pointing at the original.
        """
        self.provider.responder = responder

    @staticmethod
    def token_for(user_id: str, *, scopes: tuple[str, ...] = ()) -> str:
        """Mint a valid bearer token for a principal."""
        return issue_token(user_id, Settings(), scopes=frozenset(scopes))

    @staticmethod
    def for_user(user_id: str) -> dict[str, str]:
        """Return the headers identifying ``user_id`` in development mode."""
        return {"X-User-Id": user_id}


@pytest.fixture
def harness(sql: Callable[..., list[tuple]]) -> Iterator[ApiHarness]:
    """Build the application for real and hand it to a test."""
    # Start from an empty cache: a fixed-window rate-limit counter left by an
    # earlier test — or an earlier run — would otherwise decide this one.
    purge_test_cache()

    app = create_app()
    provider = FakeLLMProvider(responder=lambda messages: "{}")
    # Wrapped exactly as the lifespan wraps the real provider, so token accounting
    # and budget enforcement are exercised rather than bypassed by the harness.
    # The unwrapped provider stays on the harness for rewiring, since the wrapper
    # delegates to the same object.
    # The same two decorators the lifespan applies, in the same order, so the API
    # suite exercises the production model path rather than a simplification of
    # it. The scripted provider never fails, so the retry loop costs nothing here.
    budgeted = RetryingProvider(BudgetedProvider(provider), max_retries=Settings().max_retries)

    with TestClient(app) as client:
        # The lifespan built a real graph against the real provider path.
        # Replace only the model, so orchestration, persistence, and streaming
        # remain the genuine article.
        app.state.provider = budgeted
        app.state.provider_error = None
        app.state.graph = build_graph(
            Settings(),
            budgeted,
            app.state.tool_registry,
            checkpointer=app.state.checkpointer.saver,
            # The real, PostgreSQL-backed manager, so a recalled memory in a test
            # is a row that was genuinely written and read back.
            memory=getattr(app.state, "memory", None),
        )
        app.state.agents = build_all_agents(budgeted, app.state.tool_registry)

        sql(TRUNCATE_ALL)
        yield ApiHarness(client, app, provider)

    app.dependency_overrides.clear()


@pytest.fixture
def client(harness: ApiHarness) -> TestClient:
    """Return just the HTTP client."""
    return harness.client


@pytest.fixture
def memory_store() -> InMemoryTaskStore:
    """Return a process-local store, for exercising the shared contract."""
    return InMemoryTaskStore()
