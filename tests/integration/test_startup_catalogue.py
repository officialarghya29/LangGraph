"""The startup catalogue check is wired, not merely written.

The unit tests prove what the check says. These prove it is asked, through the
same two decorators the lifespan wraps around the provider — which is the part
that is easy to get wrong, because a decorator that does not delegate inherits the
interface's empty catalogue and makes the check silently useless. A stub vendor
serves the answer, so nothing here touches a real endpoint.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import reset_settings_cache
from app.main import create_app

#: Everything a non-development environment needs to boot here. Staging is used
#: rather than production because the production invariants (real auth, secret
#: keys) are the subject of their own tests and would obscure this one.
_DEPLOYMENT_ENV = {
    "APP_ENV": "staging",
    "AUTH_ENABLED": "true",
    "TRUST_IDENTITY_HEADER": "false",
    "JWT_SECRET": "test-secret-value-for-the-startup-check-only",
    "LLM_API_KEY": "test-key",
    "LLM_PROVIDER": "openai",
    "LLM_MODEL": "gpt-6.1-retired",
    "CHECKPOINT_BACKEND": "postgres",
}


class _Capture(logging.Handler):
    """Collect the application's records.

    Attached to ``app.main`` rather than to the root logger, because the lifespan
    calls :func:`configure_logging`, which removes every root handler — including
    the one ``caplog`` installs. Capturing on the emitting logger survives that.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Keep the record."""
        self.records.append(record)


@pytest.fixture
def capture() -> Iterator[_Capture]:
    """Capture the startup logger's records for one test."""
    handler = _Capture()
    target = logging.getLogger("app.main")
    target.addHandler(handler)
    try:
        yield handler
    finally:
        target.removeHandler(handler)


def wait_until(condition: Callable[[], bool], *, timeout: float = 10.0) -> None:
    """Wait for a background effect, without touching the application's loop.

    The lifespan runs on the TestClient's own loop in another thread, so the
    check's task cannot be awaited from here — awaiting it across loops tears down
    the application's connections. Polling for the effect is the honest boundary:
    what matters is that the request was made and the warning logged, not which
    coroutine was awaited.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError(f"condition not met within {timeout}s")


class _Vendor(BaseHTTPRequestHandler):
    """A stub that answers the model catalogue and counts what it was asked."""

    served: tuple[str, ...] = ("gpt-6.1-sol", "gpt-6-luna")
    requests: int = 0

    def do_GET(self) -> None:
        """Answer every read with the catalogue."""
        type(self).requests += 1
        body = json.dumps({"data": [{"id": name} for name in type(self).served]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        """Stay quiet: this server is an implementation detail of a test."""


@pytest.fixture
def vendor() -> Iterator[str]:
    """Serve the model catalogue on a port the test process owns.

    Yields:
        The base URL to point ``LLM_BASE_URL`` at.
    """
    _Vendor.requests = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Vendor)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def environment(test_database_url: str, vendor: str) -> Iterator[None]:
    """Run the process as a deployment pointed at the stub vendor.

    Settings are cached, so the cache is cleared on both sides: a staging value
    left behind would silently change what every later test sees.
    """
    del test_database_url
    previous = {key: os.environ.get(key) for key in (*_DEPLOYMENT_ENV, "LLM_BASE_URL")}
    os.environ.update(_DEPLOYMENT_ENV)
    os.environ["LLM_BASE_URL"] = vendor
    reset_settings_cache()
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_settings_cache()


async def test_a_retired_model_is_reported_through_the_wrappers(
    environment: None, capture: _Capture
) -> None:
    """The answer reaches the log when it travels through the decorators.

    The wrappers are the part this test exists for: the provider the lifespan
    checks is a retrying budgeted provider, and a decorator that inherited the
    interface's empty catalogue instead of delegating would make every deployment
    believe its model had vanished — or, as it was first written, never speak at
    all.
    """
    application = create_app()
    with TestClient(application) as client:
        # Serving starts before the vendor answers: an awaited check would
        # make this request wait on the network.
        assert client.get("/health").status_code in {200, 503}
        task = application.state.catalogue_check
        assert task is not None
        wait_until(lambda: _Vendor.requests >= 1 and task.done())

    assert _Vendor.requests >= 1, "the catalogue was never requested"
    warnings = [r for r in capture.records if r.getMessage() == "startup.model_missing"]
    assert warnings, [r.getMessage() for r in capture.records]
    assert warnings[0].__dict__["model"] == "gpt-6.1-retired"
    assert "gpt-6.1-sol" in warnings[0].__dict__["closest"]


async def test_development_never_calls_a_vendor_at_startup(environment: None) -> None:
    """The suite runs offline; a development boot must not reach for a catalogue."""
    os.environ["APP_ENV"] = "development"
    reset_settings_cache()

    application = create_app()
    with TestClient(application) as client:
        assert client.get("/health").status_code in {200, 503}

    task = getattr(application.state, "catalogue_check", None)
    assert task is None or task.done()
    assert _Vendor.requests == 0


async def test_a_catalogue_that_cannot_answer_does_not_stop_the_app(environment: None) -> None:
    """An unreachable vendor completes the task and leaves the app serving."""
    os.environ["LLM_BASE_URL"] = "http://127.0.0.1:9/v1"  # nothing listens here
    reset_settings_cache()

    application = create_app()
    with TestClient(application) as client:
        task = application.state.catalogue_check
        assert task is not None
        wait_until(task.done)
        assert task.exception() is None, "a refused catalogue must not raise out of the task"
        assert client.get("/health").status_code in {200, 503}
