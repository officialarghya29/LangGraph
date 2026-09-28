"""Tests for the liveness endpoint."""

from __future__ import annotations

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_returns_ok() -> None:
    """GET /health responds 200 with the documented payload."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_response_is_json() -> None:
    """The health endpoint advertises a JSON content type."""
    response = client.get("/health")

    assert response.headers["content-type"].startswith("application/json")
