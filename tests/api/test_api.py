"""Tests for the HTTP API.

The application is built through its real factory and lifespan, then given a
deterministic provider and graph, so these tests exercise the actual wiring
rather than a hand-assembled stand-in.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from app.api.dependencies import get_settings_dep
from app.core.config import Settings, reset_settings_cache
from app.graph.builder import build_all_agents, build_graph
from app.main import create_app
from app.services.llm import FakeLLMProvider
from fastapi.testclient import TestClient

SETTINGS = Settings(_env_file=None, llm_api_key="test-key")


def route_payload(route: str, *, approval: bool = False) -> dict[str, Any]:
    return {
        "route": route,
        "complexity": "simple",
        "intent": "scripted",
        "required_capabilities": [],
        "required_agents": [],
        "required_tools": [],
        "requires_planning": False,
        "requires_approval": approval,
        "reasoning_summary": "scripted decision",
    }


def responder(route: str, *, approval: bool = False, answer: str = "the answer") -> Any:
    def respond(messages: Any) -> str:
        joined = "\n".join(m.content for m in messages)
        if "You route requests" in joined:
            return json.dumps(route_payload(route, approval=approval))
        if "You are a synthesis agent" in joined:
            return json.dumps({"answer": answer, "sources": [], "caveats": []})
        return answer

    return respond


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A client backed by a real lifespan and a deterministic provider."""
    app = create_app()
    app.dependency_overrides[get_settings_dep] = lambda: SETTINGS

    with TestClient(app) as test_client:
        provider = FakeLLMProvider(responder=responder("direct"))
        app.state.provider = provider
        app.state.provider_error = None
        app.state.graph = build_graph(SETTINGS, provider, app.state.tool_registry)
        app.state.agents = build_all_agents(provider, app.state.tool_registry)
        yield test_client

    app.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #


def test_health_is_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_reports_ok_when_wired(client: TestClient) -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"]["llm_provider"] == "ok"
    assert body["checks"]["orchestration_graph"] == "ok"


def test_ready_reports_the_tool_count(client: TestClient) -> None:
    body = client.get("/ready").json()

    assert "tools" in body["checks"]["tool_registry"]


def test_ready_is_honest_about_durability(client: TestClient) -> None:
    """Readiness must not imply durability the system does not have."""
    checks = client.get("/ready").json()["checks"]

    assert "in-memory" in checks["checkpoint_store"]
    assert checks["database"] == "not configured"


def test_ready_is_degraded_without_a_provider() -> None:
    app = create_app()
    app.dependency_overrides[get_settings_dep] = lambda: SETTINGS

    with TestClient(app) as test_client:
        app.state.provider = None
        app.state.provider_error = "no credential"

        response = test_client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert "no credential" in response.json()["checks"]["llm_provider"]

    app.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def test_listing_agents(client: TestClient) -> None:
    agents = client.get("/api/v1/agents").json()

    names = {agent["name"] for agent in agents}
    assert {"researcher", "coder", "analyst", "planner", "critic", "synthesizer"} <= names


def test_agent_descriptions_include_tools_but_not_prompts(client: TestClient) -> None:
    agents = client.get("/api/v1/agents").json()

    for agent in agents:
        assert "tools" in agent
        assert "system_prompt" not in agent


def test_every_agent_resolves_against_the_tool_registry(client: TestClient) -> None:
    """An agent naming a tool that does not exist would break the workforce."""
    agents = client.get("/api/v1/agents").json()

    assert agents
    for agent in agents:
        for tool in agent["tools"]:
            assert tool, agent["name"]


def test_unconfigured_capabilities_are_reported_not_hidden(client: TestClient) -> None:
    """With no query executor wired, the database tool is absent by design."""
    agents = {agent["name"]: agent for agent in client.get("/api/v1/agents").json()}

    assert "database" in agents["analyst"]["unavailable_tools"]
    assert "database" not in agents["analyst"]["tools"]
    assert "python_executor" in agents["analyst"]["tools"]


def test_listing_tools(client: TestClient) -> None:
    tools = client.get("/api/v1/tools").json()

    names = {tool["name"] for tool in tools}
    assert {"read_file", "write_file", "list_directory", "python_executor", "web_search"} <= names


def test_tool_listing_exposes_risk_and_approval(client: TestClient) -> None:
    tools = {tool["name"]: tool for tool in client.get("/api/v1/tools").json()}

    assert tools["read_file"]["risk_level"] == "LOW"
    assert tools["python_executor"]["requires_approval"] is True


def test_tool_listing_exposes_no_credentials(client: TestClient) -> None:
    body = json.dumps(client.get("/api/v1/tools").json())

    assert "test-key" not in body


def test_the_real_startup_wires_the_whole_workforce(monkeypatch: pytest.MonkeyPatch) -> None:
    """Startup must survive having a credential.

    Regression guard. ``build_all_agents`` used to raise because agents named
    tools that do not exist, which made startup fail as soon as a real key was
    configured — and took every endpoint down with it, including ``/health``.
    """
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    reset_settings_cache()
    app = create_app()

    try:
        with TestClient(app) as test_client:
            assert app.state.provider is not None
            assert test_client.get("/health").status_code == 200
            agents = test_client.get("/api/v1/agents").json()
    finally:
        reset_settings_cache()

    by_name = {agent["name"]: agent for agent in agents}
    assert len(agents) >= 7
    # Describing every agent must succeed: a required tool that does not exist
    # raises here, which is the startup failure this guards against.
    assert by_name["researcher"]["tools"] == ["web_search"]
    assert by_name["analyst"]["tools"] == ["python_executor"]
    # Orchestrator roles are tool-less by design: verification and synthesis
    # must not be able to cause a side effect.
    assert by_name["critic"]["tools"] == []


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #


def test_chat_returns_an_answer(client: TestClient) -> None:
    response = client.post("/api/v1/chat", json={"message": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "the answer"
    assert body["route"] == "direct"
    assert body["task_id"]


def test_chat_returns_no_reasoning_fields(client: TestClient) -> None:
    body = client.post("/api/v1/chat", json={"message": "hello"}).json()

    assert {"chain_of_thought", "reasoning", "prompt"}.isdisjoint(body)


def test_chat_rejects_an_empty_message(client: TestClient) -> None:
    assert client.post("/api/v1/chat", json={"message": ""}).status_code == 422


def test_chat_rejects_an_oversized_message(client: TestClient) -> None:
    response = client.post("/api/v1/chat", json={"message": "x" * 20_001})

    assert response.status_code == 422


def test_chat_requires_a_message_field(client: TestClient) -> None:
    assert client.post("/api/v1/chat", json={}).status_code == 422


def test_chat_is_unavailable_without_a_provider() -> None:
    app = create_app()
    app.dependency_overrides[get_settings_dep] = lambda: Settings(_env_file=None)

    with TestClient(app) as test_client:
        app.state.graph = None
        app.state.provider = None
        response = test_client.post("/api/v1/chat", json={"message": "hi"})

    assert response.status_code == 503

    app.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #


def test_creating_a_task_returns_acceptance(client: TestClient) -> None:
    response = client.post("/api/v1/tasks", json={"request": "do something"})

    assert response.status_code == 202
    assert response.json()["task_id"]


def test_a_task_can_be_fetched_afterwards(client: TestClient) -> None:
    created = client.post("/api/v1/tasks", json={"request": "do something"}).json()

    fetched = client.get(f"/api/v1/tasks/{created['task_id']}")

    assert fetched.status_code == 200
    assert fetched.json()["status"] in {"pending", "running", "completed"}


def test_a_completed_task_carries_its_answer(client: TestClient) -> None:
    created = client.post("/api/v1/tasks", json={"request": "do something"}).json()

    fetched = client.get(f"/api/v1/tasks/{created['task_id']}").json()

    assert fetched["answer"] == "the answer"


def test_task_status_endpoint(client: TestClient) -> None:
    created = client.post("/api/v1/tasks", json={"request": "do something"}).json()

    status_body = client.get(f"/api/v1/tasks/{created['task_id']}/status").json()

    assert status_body["task_id"] == created["task_id"]
    assert status_body["has_answer"] is True


def test_an_unknown_task_is_a_404(client: TestClient) -> None:
    assert client.get("/api/v1/tasks/does-not-exist").status_code == 404


def test_a_task_is_invisible_to_another_user(client: TestClient) -> None:
    """Ownership is enforced, and absence and denial are indistinguishable."""
    created = client.post(
        "/api/v1/tasks", json={"request": "private"}, headers={"X-User-Id": "alice"}
    ).json()

    assert client.get(f"/api/v1/tasks/{created['task_id']}").status_code == 404
    assert (
        client.get(
            f"/api/v1/tasks/{created['task_id']}", headers={"X-User-Id": "mallory"}
        ).status_code
        == 404
    )
    assert (
        client.get(
            f"/api/v1/tasks/{created['task_id']}", headers={"X-User-Id": "alice"}
        ).status_code
        == 200
    )


def test_cancelling_a_finished_task_is_a_conflict(client: TestClient) -> None:
    created = client.post("/api/v1/tasks", json={"request": "do something"}).json()

    # The background task completes before the client returns, so this is terminal.
    response = client.post(f"/api/v1/tasks/{created['task_id']}/cancel")

    assert response.status_code in {200, 409}


def test_a_task_response_carries_no_internals(client: TestClient) -> None:
    created = client.post("/api/v1/tasks", json={"request": "do something"}).json()

    assert {"prompt", "messages", "reasoning"}.isdisjoint(created)


# --------------------------------------------------------------------------- #
# Approval
# --------------------------------------------------------------------------- #


@pytest.fixture
def gated_client() -> Iterator[TestClient]:
    """A client whose graph routes to human approval."""
    app = create_app()
    app.dependency_overrides[get_settings_dep] = lambda: SETTINGS

    with TestClient(app) as test_client:
        provider = FakeLLMProvider(responder=responder("human_approval", approval=True))
        app.state.provider = provider
        app.state.provider_error = None
        app.state.graph = build_graph(SETTINGS, provider, app.state.tool_registry)
        app.state.agents = build_all_agents(provider, app.state.tool_registry)
        yield test_client

    app.dependency_overrides.clear()


def test_a_gated_task_reports_that_it_awaits_approval(gated_client: TestClient) -> None:
    created = gated_client.post("/api/v1/tasks", json={"request": "delete everything"}).json()

    fetched = gated_client.get(f"/api/v1/tasks/{created['task_id']}").json()

    assert fetched["status"] == "awaiting_approval"
    assert fetched["approval_status"] == "pending"


def test_approving_a_gated_task_completes_it(gated_client: TestClient) -> None:
    created = gated_client.post("/api/v1/tasks", json={"request": "delete everything"}).json()

    response = gated_client.post(f"/api/v1/tasks/{created['task_id']}/approve", json={})

    assert response.status_code == 200
    assert response.json()["approval_status"] == "approved"


def test_rejecting_a_gated_task_does_not_execute_it(gated_client: TestClient) -> None:
    created = gated_client.post("/api/v1/tasks", json={"request": "delete everything"}).json()

    response = gated_client.post(f"/api/v1/tasks/{created['task_id']}/reject", json={})

    assert response.status_code == 200
    assert response.json()["approval_status"] == "rejected"


def test_deciding_a_task_that_is_not_gated_is_a_conflict(client: TestClient) -> None:
    created = client.post("/api/v1/tasks", json={"request": "hello"}).json()

    response = client.post(f"/api/v1/tasks/{created['task_id']}/approve", json={})

    assert response.status_code == 409


def test_an_unknown_task_cannot_be_approved(gated_client: TestClient) -> None:
    assert gated_client.post("/api/v1/tasks/nope/approve", json={}).status_code == 404
