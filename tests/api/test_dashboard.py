"""Tests for the operator console.

The console is a page with approve, reject, and cancel buttons, so most of what
is worth testing about it is not whether it renders — it is whether it is off
when it should be, and whether it is safe to render task text that came from a
user.

The XSS guard is asserted structurally rather than by reading the script: the
page builds rows from request text and answers, and ``innerHTML`` with that data
is how an operator console becomes a stored-XSS delivery vehicle.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.routes.dashboard import API_PATHS, CSP
from app.core.config import reset_settings_cache
from app.main import create_app
from tests.api.conftest import ApiHarness


@pytest.fixture
def console(monkeypatch: pytest.MonkeyPatch, sql: object) -> Iterator[TestClient]:
    """Build an application with the console explicitly enabled."""
    del sql
    monkeypatch.setenv("DASHBOARD_ENABLED", "true")
    reset_settings_cache()
    try:
        with TestClient(create_app()) as client:
            yield client
    finally:
        reset_settings_cache()


# --------------------------------------------------------------------------- #
# Off by default
# --------------------------------------------------------------------------- #


def test_the_console_is_disabled_by_default() -> None:
    """A control surface that is on unless someone turns it off is not a default.

    Asserted through the settings object rather than through a request, because
    the point is the shipped default, not the behaviour of one environment.
    """
    from app.core.config import Settings

    assert Settings().dashboard_enabled is False


def test_a_disabled_console_is_not_found(harness: ApiHarness) -> None:
    """Every one of its routes is absent when it is off."""
    for path in ("/dashboard", "/dashboard/app.js", "/dashboard/app.css"):
        response = harness.client.get(path)
        assert response.status_code == 404, path


def test_a_disabled_console_does_not_advertise_itself(harness: ApiHarness) -> None:
    """404 rather than 403: the setting is not a permission the caller can hold."""
    response = harness.client.get("/dashboard")

    assert response.json() == {"detail": "Not Found"}


# --------------------------------------------------------------------------- #
# Enabled
# --------------------------------------------------------------------------- #


def test_the_console_serves_its_page(console: TestClient) -> None:
    """The page is HTML and says what it is."""
    response = console.get("/dashboard")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Agent control console" in response.text


def test_the_page_and_its_assets_all_resolve(console: TestClient) -> None:
    """A page whose script 404s is a page that does nothing.

    The failure mode this catches is a renamed asset path: the HTML would still
    serve 200 and the console would be silently inert in a browser.
    """
    page = console.get("/dashboard").text

    for asset, content_type in (
        ("/dashboard/app.css", "text/css"),
        ("/dashboard/app.js", "text/javascript"),
    ):
        assert f'href="{asset}"' in page or f'src="{asset}"' in page
        response = console.get(asset)
        assert response.status_code == 200, asset
        assert response.headers["content-type"].startswith(content_type), asset


def test_the_page_has_no_inline_script(console: TestClient) -> None:
    """Every script tag must load a file, or the CSP cannot forbid inline code."""
    page = console.get("/dashboard").text

    assert page.count("<script") == page.count("<script src=")


def test_the_content_security_policy_forbids_inline_and_remote_code(console: TestClient) -> None:
    """The policy is explicit, so a future CDN link fails closed."""
    policy = console.get("/dashboard").headers["content-security-policy"]

    assert policy == CSP
    assert "unsafe-inline" not in policy
    assert "unsafe-eval" not in policy
    assert "default-src 'none'" in policy
    assert "frame-ancestors 'none'" in policy
    assert "connect-src 'self'" in policy


def test_every_response_carries_the_security_headers(console: TestClient) -> None:
    """The assets are rendered by browsers too, so they need the same headers."""
    for path in ("/dashboard", "/dashboard/app.js", "/dashboard/app.css"):
        headers = console.get(path).headers
        assert headers["x-content-type-options"] == "nosniff", path
        assert headers["referrer-policy"] == "no-referrer", path
        assert headers["cache-control"] == "no-store", path


def test_the_console_is_not_indexed(console: TestClient) -> None:
    """An internal console should not appear in search results."""
    assert 'name="robots" content="noindex, nofollow"' in console.get("/dashboard").text


def _strip_javascript_comments(source: str) -> str:
    """Return ``source`` with ``//`` and ``/* */`` comments removed.

    Comments are stripped before the guard below runs, because a guard that
    scans prose fails on the first comment that warns against the very thing it
    forbids. Checking code rather than text is the difference between a rule the
    next author keeps and one they delete.

    Args:
        source: JavaScript source.

    Returns:
        The source with comments removed. String literals are not modelled, so a
        ``//`` inside a URL would be treated as a comment; that can only make the
        guard stricter, never weaker.
    """
    without_blocks = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", without_blocks)


def test_the_script_never_builds_html_from_data() -> None:
    """The XSS guard, asserted structurally.

    The page renders request text, answers, failure reasons, and event payloads.
    Assigning any of those to ``innerHTML`` would execute whatever a user put in
    a task. ``textContent`` cannot.
    """
    from app.api.routes.dashboard import _SCRIPT

    code = _strip_javascript_comments(_SCRIPT)

    assert "innerHTML" not in code
    assert "outerHTML" not in code
    assert "insertAdjacentHTML" not in code
    assert "document.write" not in code
    assert "eval(" not in code
    assert "textContent" in code

    # The guard must be able to fail. Without this, a typo in the stripping above
    # would turn the whole test into a tautology that always passes.
    assert "innerHTML" in _strip_javascript_comments("x.innerHTML = y;")


def test_the_script_calls_only_routes_that_exist(console: TestClient) -> None:
    """A renamed endpoint should break a test, not a button."""
    page = console.get("/dashboard").text
    script = console.get("/dashboard/app.js").text
    document = page + script

    for path in API_PATHS:
        assert path in document, path

    schema = console.get("/openapi.json").json()
    for path in API_PATHS:
        assert path in schema["paths"], f"{path} is called by the console but is not routed"


def test_the_console_sends_identity_the_same_way_any_client_would() -> None:
    """It holds no privileges of its own, and the script proves it."""
    from app.api.routes.dashboard import _SCRIPT

    assert "'X-User-Id'" in _SCRIPT
    assert "'Authorization'" in _SCRIPT
    assert "'Bearer '" in _SCRIPT


# --------------------------------------------------------------------------- #
# Deployment exposure
# --------------------------------------------------------------------------- #


def test_the_schema_can_be_withheld(monkeypatch: pytest.MonkeyPatch) -> None:
    """The OpenAPI schema is an inventory of the attack surface.

    It is useful while building against the API and unnecessary once deployed, so
    it must be possible to turn it off without touching the code. Both the schema
    and the interactive page that reads it go away together — leaving ``/docs``
    without a schema would serve a broken page rather than none.
    """
    monkeypatch.setenv("API_DOCS_ENABLED", "false")
    reset_settings_cache()
    try:
        with TestClient(create_app()) as client:
            assert client.get("/openapi.json").status_code == 404
            assert client.get("/docs").status_code == 404
    finally:
        reset_settings_cache()
