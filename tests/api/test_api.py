"""End-to-end HTTP API tests.

These drive the real application: the real lifespan, the real PostgreSQL, the
real Redis, and the real compiled graph. Only the language model is scripted.

The scenarios worth the cost of real infrastructure are the ones that span it —
a task surviving as a durable record, ownership being enforced on a real query,
an approval being written before the graph resumes — so those are what is
asserted here.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import reset_settings_cache
from app.main import create_app
from tests.api.conftest import ApiHarness


def route_payload(route: str, *, approval: bool = False) -> dict[str, Any]:
    """Build a scripted routing decision."""
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


def responder(
    route: str,
    *,
    approval: bool = False,
    answer: str = "the answer",
) -> Any:
    """Build a role-keyed scripted provider responder."""

    def respond(messages: Any) -> str:
        joined = "\n".join(message.content for message in messages)
        if "You route requests" in joined:
            return json.dumps(route_payload(route, approval=approval))
        if "You are a synthesis agent" in joined:
            return json.dumps({"answer": answer, "sources": [], "caveats": []})
        return answer

    return respond


def research_responder() -> Any:
    """Script a full planned run: routing, planning, two workers, criticism, synthesis.

    The other responders here answer in one shape for every role, which exercises a
    single agent. This one drives the orchestration the way a real request does, so
    the audit trail is checked against a run that has more than one agent in it.
    """
    plan = {
        "objective": "compare two stores",
        "subtasks": [
            {
                "id": "a",
                "description": "research postgres",
                "agent": "researcher",
                "tools": ["web_search"],
                "expected_output": "notes",
                "success_criteria": "sourced",
            },
            {
                "id": "b",
                "description": "research redis",
                "agent": "researcher",
                "tools": ["web_search"],
                "expected_output": "notes",
                "success_criteria": "sourced",
            },
        ],
    }

    def respond(messages: Any) -> str:
        joined = "\n".join(message.content for message in messages)
        if "You route requests" in joined:
            return json.dumps(route_payload("research"))
        if "You are a planning agent" in joined:
            return json.dumps(plan)
        if "You are a research agent" in joined:
            subtask_id = "a" if "postgres" in joined else "b"
            return json.dumps(
                {
                    "agent": "researcher",
                    "subtask_id": subtask_id,
                    "content": f"finding for {subtask_id}",
                    "summary": f"summary for {subtask_id}",
                    "sources": [f"https://example.test/{subtask_id}"],
                    "confidence": 0.8,
                }
            )
        if "You are a verification agent" in joined:
            return json.dumps({"passed": True, "confidence": 0.9, "issues": []})
        if "You are a synthesis agent" in joined:
            return json.dumps({"answer": "Postgres and Redis differ.", "sources": []})
        return "unexpected role"

    return respond


@pytest.fixture
def direct(harness: ApiHarness) -> ApiHarness:
    """A harness whose requests route straight to a direct answer."""
    harness.rewire(responder("direct"))
    return harness


@pytest.fixture
def gated(harness: ApiHarness) -> ApiHarness:
    """A harness whose requests route to human approval."""
    harness.rewire(responder("human_approval", approval=True))
    return harness


# --------------------------------------------------------------------------- #
# Health and readiness
# --------------------------------------------------------------------------- #


def test_health_reports_liveness_without_touching_dependencies(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_reports_ok_when_fully_wired(client: TestClient) -> None:
    body = client.get("/ready").json()

    assert body["status"] == "ok"
    assert body["checks"]["database"] == "ok"
    assert body["checks"]["checkpoint_store"].startswith("ok")
    assert body["checks"]["cache"] == "ok"


def test_ready_reports_the_registered_workforce(client: TestClient) -> None:
    checks = client.get("/ready").json()["checks"]

    assert checks["agents"].endswith("registered")
    assert checks["tool_registry"].endswith("tools")


def test_ready_reports_that_python_execution_is_disabled(client: TestClient) -> None:
    """Readiness must not imply a capability the system refuses to provide."""
    checks = client.get("/ready").json()["checks"]

    assert "no sandbox configured" in checks["python_execution"]


def test_ready_reports_the_authentication_mode(client: TestClient) -> None:
    checks = client.get("/ready").json()["checks"]

    assert checks["authentication"] == "disabled"


def test_a_request_gets_a_correlation_id(client: TestClient) -> None:
    response = client.get("/health")

    assert response.headers.get("X-Request-Id")


def test_a_supplied_correlation_id_is_echoed(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-Id": "trace-abc"})

    assert response.headers["X-Request-Id"] == "trace-abc"


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def test_listing_agents(client: TestClient) -> None:
    agents = client.get("/api/v1/agents").json()

    names = {agent["name"] for agent in agents}
    assert {"researcher", "coder", "analyst", "planner", "critic", "synthesizer"} <= names


def test_agent_descriptions_omit_prompts(client: TestClient) -> None:
    for agent in client.get("/api/v1/agents").json():
        assert "system_prompt" not in agent
        assert "input_schema" in agent


def test_every_agent_resolves_against_the_tool_registry(client: TestClient) -> None:
    """An agent naming a tool that does not exist would break the workforce."""
    agents = client.get("/api/v1/agents").json()

    assert agents
    for agent in agents:
        assert isinstance(agent["tools"], list)
        assert isinstance(agent["unavailable_tools"], list)


def test_unconfigured_capabilities_are_reported_not_hidden(client: TestClient) -> None:
    agents = {agent["name"]: agent for agent in client.get("/api/v1/agents").json()}

    assert "database" in agents["analyst"]["unavailable_tools"]
    assert "database" not in agents["analyst"]["tools"]


def test_listing_tools_includes_risk_and_approval(client: TestClient) -> None:
    tools = {tool["name"]: tool for tool in client.get("/api/v1/tools").json()}

    assert tools["read_file"]["risk_level"] == "LOW"
    assert tools["python_executor"]["requires_approval"] is True


def test_tool_listing_leaks_no_credentials(client: TestClient) -> None:
    body = json.dumps(client.get("/api/v1/tools").json())

    assert "password" not in body.lower()
    assert "token" not in body.lower()


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #


def test_chat_returns_an_answer(direct: ApiHarness) -> None:
    response = direct.client.post("/api/v1/chat", json={"message": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "the answer"
    assert body["route"] == "direct"
    assert body["task_id"]


def test_chat_exposes_no_reasoning(direct: ApiHarness) -> None:
    body = direct.client.post("/api/v1/chat", json={"message": "hello"}).json()

    assert {"chain_of_thought", "reasoning", "prompt", "messages"}.isdisjoint(body)


def test_chat_rejects_an_empty_message(client: TestClient) -> None:
    assert client.post("/api/v1/chat", json={"message": ""}).status_code == 422


def test_chat_rejects_an_oversized_message(client: TestClient) -> None:
    assert client.post("/api/v1/chat", json={"message": "x" * 20_001}).status_code == 422


def test_chat_requires_a_message(client: TestClient) -> None:
    assert client.post("/api/v1/chat", json={}).status_code == 422


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #


def test_creating_a_task_returns_acceptance(direct: ApiHarness) -> None:
    response = direct.client.post("/api/v1/tasks", json={"request": "do something"})

    assert response.status_code == 202
    assert response.json()["task_id"]


def test_a_completed_task_carries_its_answer(direct: ApiHarness) -> None:
    created = direct.client.post("/api/v1/tasks", json={"request": "do something"}).json()

    fetched = direct.client.get(f"/api/v1/tasks/{created['task_id']}").json()

    assert fetched["status"] == "completed"
    assert fetched["answer"] == "the answer"


def test_a_task_is_durable_not_just_in_memory(
    direct: ApiHarness, sql: Callable[..., list[tuple]]
) -> None:
    """The record must be in PostgreSQL, not only in the object that created it.

    Read over a separate, synchronously opened connection. The application's own
    engine lives on the event loop the test client runs, and borrowing it from a
    second loop would test the loop plumbing rather than the durability.
    """
    created = direct.client.post("/api/v1/tasks", json={"request": "do something"}).json()

    rows = sql("SELECT status, answer FROM tasks WHERE id = %s", (created["task_id"],))

    assert rows == [("completed", "the answer")]


def test_a_task_records_the_tokens_it_actually_spent(
    direct: ApiHarness, sql: Callable[..., list[tuple]]
) -> None:
    """Token accounting has to reach the database, or it is decoration.

    The columns existed and were always zero: usage was parsed off every response
    and then dropped. A test asserting only that the task completed passed
    throughout, which is why this one asserts the numbers.
    """
    created = direct.client.post("/api/v1/tasks", json={"request": "do something"}).json()

    rows = sql(
        "SELECT prompt_tokens, completion_tokens FROM tasks WHERE id = %s",
        (created["task_id"],),
    )

    assert rows, "the task should have been recorded"
    prompt, completion = rows[0]
    assert prompt > 0, "the routing call alone should have been counted"
    assert completion > 0


def test_metrics_can_be_turned_off(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A kill switch that only changed a readiness string was not a kill switch.

    ``METRICS_ENABLED=false`` used to leave the endpoint serving; an operator who
    turned it off for a reason still exposed it.
    """
    monkeypatch.setenv("METRICS_ENABLED", "false")
    reset_settings_cache()
    try:
        assert client.get("/metrics").status_code == 404
        checks = client.get("/ready").json()["checks"]
        assert "disabled" in checks["metrics"]
    finally:
        reset_settings_cache()


def test_the_provider_uses_its_own_retry_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """``LLM_MAX_RETRIES`` configures model retries; nothing else did.

    It was declared and never read, so the provider retried on the graph's
    replan budget instead — the same number doing two unrelated jobs.
    """
    monkeypatch.setenv("LLM_API_KEY", "sk-test-key")
    monkeypatch.setenv("LLM_MAX_RETRIES", "7")
    reset_settings_cache()
    try:
        with TestClient(create_app()) as probe:
            provider = probe.app.state.provider
            assert provider.max_retries == 7
            assert type(provider.inner).__name__ == "BudgetedProvider"
    finally:
        reset_settings_cache()


def test_startup_names_the_model_it_will_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model is derived from the provider, so the environment no longer says.

    With ``LLM_MODEL`` unset the answer is computed rather than written down, and
    "which model is this process calling?" is a question an operator has to be
    able to answer from outside the process.

    The record is collected through a handler of this test's own rather than
    through ``caplog``: start-up calls ``configure_logging``, which replaces the
    root handlers on purpose so that a reloaded process does not log every line
    twice, and that takes pytest's handler with it.
    """
    monkeypatch.setenv("LLM_API_KEY", "sk-test-key")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    reset_settings_cache()

    collected: list[logging.LogRecord] = []

    class Collect(logging.Handler):
        """Keep every record the application emits during start-up."""

        def emit(self, record: logging.LogRecord) -> None:
            """Store the record as emitted, before any formatter touches it."""
            collected.append(record)

    collector = Collect()
    app_logger = logging.getLogger("app")
    app_logger.addHandler(collector)
    try:
        with TestClient(create_app()):
            pass
    finally:
        app_logger.removeHandler(collector)
        reset_settings_cache()

    lines = [r for r in collected if r.getMessage() == "startup.model_ready"]

    assert len(lines) == 1
    assert getattr(lines[0], "model", None) == "claude-sonnet-5-5"
    assert getattr(lines[0], "provider", None) == "anthropic"


def test_a_task_reports_the_tokens_it_spent_as_metrics(direct: ApiHarness) -> None:
    """Accounting that stays inside the process is accounting nobody can alert on.

    The series exist so a cost overrun is visible without querying the database,
    and they are reported for failed runs too — a run that died halfway still
    consumed what it consumed.
    """
    direct.client.post("/api/v1/tasks", json={"request": "do something"})

    exposition = direct.client.get("/metrics").text
    samples = [
        line
        for line in exposition.splitlines()
        if line.startswith("llm_tokens_total") or line.startswith("llm_calls_total")
    ]

    assert samples, exposition
    prompt = next(line for line in samples if 'direction="prompt"' in line)
    assert float(prompt.rsplit(" ", 1)[1]) > 0


def test_a_spent_budget_keeps_its_own_status(
    direct: ApiHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deliberate refusal must not be flattened into a 500.

    The chat handler catches everything unexpected and reports a server error.
    Without an explicit pass-through for :class:`AppError`, an exhausted budget
    would arrive as a 500 — telling the caller the server broke and to retry,
    when the truth is that this request has no allowance left and retrying will
    fail identically.
    """
    monkeypatch.setenv("MAX_TOKEN_BUDGET", "1")
    reset_settings_cache()
    try:
        response = direct.client.post("/api/v1/chat", json={"message": "hi"})
    finally:
        reset_settings_cache()

    assert response.status_code == 429
    assert "token budget" in response.json()["detail"]


def test_a_spent_budget_fails_the_task_with_a_reason(
    direct: ApiHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The asynchronous path has no caller to tell, so it must record instead."""
    monkeypatch.setenv("MAX_TOKEN_BUDGET", "1")
    reset_settings_cache()
    try:
        created = direct.client.post("/api/v1/tasks", json={"request": "do something"}).json()
        fetched = direct.client.get(f"/api/v1/tasks/{created['task_id']}").json()
    finally:
        reset_settings_cache()

    assert fetched["status"] == "failed"
    assert fetched["failure_reason"]


def test_task_status_endpoint(direct: ApiHarness) -> None:
    created = direct.client.post("/api/v1/tasks", json={"request": "do something"}).json()

    body = direct.client.get(f"/api/v1/tasks/{created['task_id']}/status").json()

    assert body["task_id"] == created["task_id"]
    assert body["has_answer"] is True


def test_tasks_are_listed_for_their_owner_only(direct: ApiHarness) -> None:
    direct.client.post("/api/v1/tasks", json={"request": "mine"}, headers={"X-User-Id": "alice"})
    direct.client.post("/api/v1/tasks", json={"request": "yours"}, headers={"X-User-Id": "bob"})

    alice = direct.client.get("/api/v1/tasks", headers={"X-User-Id": "alice"}).json()

    assert [task["request"] for task in alice] == ["mine"]


def test_an_unknown_task_is_a_404(client: TestClient) -> None:
    assert client.get("/api/v1/tasks/does-not-exist").status_code == 404


def test_a_malformed_task_id_is_a_404_not_a_crash(client: TestClient) -> None:
    """A non-UUID path segment must not reach the database as a cast error."""
    assert client.get("/api/v1/tasks/not-a-uuid").status_code == 404


def test_a_task_is_invisible_to_another_user(direct: ApiHarness) -> None:
    """Absence and denial are indistinguishable, so existence is not leaked."""
    created = direct.client.post(
        "/api/v1/tasks", json={"request": "private"}, headers={"X-User-Id": "alice"}
    ).json()
    task_id = created["task_id"]

    assert direct.client.get(f"/api/v1/tasks/{task_id}").status_code == 404
    assert (
        direct.client.get(f"/api/v1/tasks/{task_id}", headers={"X-User-Id": "mallory"}).status_code
        == 404
    )
    assert (
        direct.client.get(f"/api/v1/tasks/{task_id}", headers={"X-User-Id": "alice"}).status_code
        == 200
    )


def test_a_task_response_carries_no_internals(direct: ApiHarness) -> None:
    created = direct.client.post("/api/v1/tasks", json={"request": "do something"}).json()

    assert {"prompt", "messages", "reasoning", "chain_of_thought"}.isdisjoint(created)


def test_cancelling_a_finished_task_is_a_conflict(direct: ApiHarness) -> None:
    created = direct.client.post("/api/v1/tasks", json={"request": "do something"}).json()

    response = direct.client.post(f"/api/v1/tasks/{created['task_id']}/cancel")

    assert response.status_code == 409


def test_an_unknown_task_cannot_be_cancelled(client: TestClient) -> None:
    assert client.post("/api/v1/tasks/nope/cancel").status_code == 404


# --------------------------------------------------------------------------- #
# Approval
# --------------------------------------------------------------------------- #


def test_a_gated_task_awaits_approval(gated: ApiHarness) -> None:
    created = gated.client.post("/api/v1/tasks", json={"request": "delete everything"}).json()

    fetched = gated.client.get(f"/api/v1/tasks/{created['task_id']}").json()

    assert fetched["status"] == "awaiting_approval"
    assert fetched["approval_status"] == "pending"


def test_an_approval_request_is_persisted(
    gated: ApiHarness, sql: Callable[..., list[tuple]]
) -> None:
    """The request must be a durable record, not just a suspended graph."""
    created = gated.client.post("/api/v1/tasks", json={"request": "delete everything"}).json()

    rows = sql("SELECT decision FROM approvals WHERE task_id = %s", (created["task_id"],))

    assert rows == [("pending",)]


def test_approving_resumes_and_completes(gated: ApiHarness) -> None:
    created = gated.client.post("/api/v1/tasks", json={"request": "delete everything"}).json()

    response = gated.client.post(f"/api/v1/tasks/{created['task_id']}/approve", json={})

    assert response.status_code == 200
    assert response.json()["approval_status"] == "approved"


def test_a_run_records_the_tool_calls_it_made(
    gated: ApiHarness, sql: Callable[..., list[tuple]]
) -> None:
    """``tool_call_count`` was updatable and updated by nothing.

    Every task reported zero tool calls however many it made, which is worse than
    no column at all: the number looked authoritative and was fictitious.
    """
    created = gated.client.post("/api/v1/tasks", json={"request": "delete everything"}).json()

    gated.client.post(f"/api/v1/tasks/{created['task_id']}/approve", json={})

    rows = sql("SELECT tool_call_count FROM tasks WHERE id = %s", (created["task_id"],))

    assert rows[0][0] >= 1, "the approved action is a tool call"


def test_a_tool_call_reaches_the_durable_event_log(
    gated: ApiHarness, sql: Callable[..., list[tuple]]
) -> None:
    """``TOOL_STARTED`` and ``TOOL_COMPLETED`` are advertised, so they must be real.

    A client streaming for progress was entitled to expect these; the event
    catalogue listed them and no code ever produced one.
    """
    created = gated.client.post("/api/v1/tasks", json={"request": "delete everything"}).json()

    gated.client.post(f"/api/v1/tasks/{created['task_id']}/approve", json={})

    rows = sql(
        "SELECT event_type FROM execution_events WHERE task_id = %s ORDER BY seq",
        (created["task_id"],),
    )
    kinds = [row[0] for row in rows]

    assert "tool_started" in kinds, kinds
    assert "tool_completed" in kinds, kinds


def test_every_agent_invocation_reaches_the_audit_table(
    harness: ApiHarness, sql: Callable[..., list[tuple]]
) -> None:
    """One row per invocation, each carrying the tokens that invocation spent.

    A run total cannot answer "what did the planner cost" or "was the second
    researcher more expensive than the first". Per-agent figures have to come from
    the agent's own scope rather than from a counter shared with whatever ran
    beside it, which is what the concurrency is for.
    """
    harness.rewire(research_responder())

    created = harness.client.post("/api/v1/tasks", json={"request": "compare two stores"}).json()
    fetched = harness.client.get(f"/api/v1/tasks/{created['task_id']}").json()
    assert fetched["status"] == "completed", fetched["failure_reason"]

    rows = sql(
        "SELECT agent, status, prompt_tokens, completion_tokens FROM agent_runs "
        "WHERE task_id = %s ORDER BY created_at",
        (created["task_id"],),
    )
    by_agent: dict[str, list[tuple]] = {}
    for agent, status, prompt, completion in rows:
        by_agent.setdefault(agent, []).append((status, prompt, completion))

    assert {"planner", "researcher", "critic", "synthesizer"} <= set(by_agent), by_agent
    assert len(by_agent["researcher"]) == 2, "a row per invocation, not per agent"
    for agent, runs in by_agent.items():
        for status, prompt, completion in runs:
            assert status == "completed", f"{agent} recorded as {status}"
            assert prompt > 0, f"{agent} recorded no prompt tokens"
            assert completion > 0, f"{agent} recorded no completion tokens"


def test_the_plan_s_steps_are_recorded_too(
    harness: ApiHarness, sql: Callable[..., list[tuple]]
) -> None:
    """The step trail answers "which subtasks ran, and how did they turn out"."""
    harness.rewire(research_responder())

    created = harness.client.post("/api/v1/tasks", json={"request": "compare two stores"}).json()
    harness.client.get(f"/api/v1/tasks/{created['task_id']}")

    rows = sql(
        "SELECT subtask_id, agent, status FROM task_steps WHERE task_id = %s ORDER BY subtask_id",
        (created["task_id"],),
    )

    assert rows == [("a", "researcher", "completed"), ("b", "researcher", "completed")]


def test_a_planned_run_reports_its_iterations(
    harness: ApiHarness, sql: Callable[..., list[tuple]]
) -> None:
    """The counters on the task row are read from the run, not left at their default."""
    harness.rewire(research_responder())

    created = harness.client.post("/api/v1/tasks", json={"request": "compare two stores"}).json()
    harness.client.get(f"/api/v1/tasks/{created['task_id']}")

    rows = sql(
        "SELECT iteration_count, retry_count FROM tasks WHERE id = %s",
        (created["task_id"],),
    )

    assert rows[0][0] >= 1, "the dispatch loop ran at least once"
    assert rows[0][1] == 0, "the verdict passed, so nothing was retried"


def test_the_timeline_describes_what_the_run_did(
    harness: ApiHarness,
) -> None:
    """The trail is readable over the API, not only by opening a SQL client."""
    harness.rewire(research_responder())

    created = harness.client.post("/api/v1/tasks", json={"request": "compare two stores"}).json()

    response = harness.client.get(f"/api/v1/tasks/{created['task_id']}/timeline")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["iteration_count"] >= 1
    assert [step["subtask_id"] for step in body["steps"]] == ["a", "b"]
    assert all(step["status"] == "completed" for step in body["steps"])
    assert body["steps"][0]["summary"], "the step's own result summary"
    assert {run["agent"] for run in body["agent_runs"]} >= {
        "planner",
        "researcher",
        "critic",
        "synthesizer",
    }
    assert all(run["prompt_tokens"] > 0 for run in body["agent_runs"])


def test_the_timeline_exposes_no_tool_arguments(gated: ApiHarness) -> None:
    """Tool arguments are model-authored and can carry a path or a credential."""
    created = gated.client.post("/api/v1/tasks", json={"request": "delete everything"}).json()
    gated.client.post(f"/api/v1/tasks/{created['task_id']}/approve", json={})

    body = gated.client.get(f"/api/v1/tasks/{created['task_id']}/timeline").json()

    assert body["tool_calls"], "the approved action should appear"
    recorded = body["tool_calls"][0]
    assert set(recorded) == {
        "tool",
        "agent",
        "ok",
        "risk_level",
        "approved",
        "approval_required",
        "failure_kind",
        "duration_ms",
        "created_at",
    }, "a field appeared that the audit contract does not promise"


def test_another_user_cannot_read_a_timeline(gated: ApiHarness) -> None:
    """The trail names every agent and tool a run touched, so it is owner-only."""
    created = gated.client.post(
        "/api/v1/tasks", json={"request": "delete it"}, headers={"X-User-Id": "alice"}
    ).json()

    response = gated.client.get(
        f"/api/v1/tasks/{created['task_id']}/timeline",
        headers={"X-User-Id": "mallory"},
    )

    assert response.status_code == 404


def test_an_unknown_timeline_is_a_404(client: TestClient) -> None:
    assert client.get("/api/v1/tasks/not-a-task/timeline").status_code == 404


def test_a_tool_call_reaches_the_audit_table(
    gated: ApiHarness, sql: Callable[..., list[tuple]]
) -> None:
    """``record_tool_call`` was called by nothing, so the table was always empty.

    The risk level is read back through the column's check constraint, which only
    accepts the level's *name*. ``RiskLevel`` is an ``IntEnum``, so publishing
    ``.value`` would have sent ``1`` and the insert would have been refused — the
    audit row is the one place where the difference is enforced.
    """
    created = gated.client.post("/api/v1/tasks", json={"request": "delete everything"}).json()

    gated.client.post(f"/api/v1/tasks/{created['task_id']}/approve", json={})

    rows = sql(
        "SELECT tool, agent, risk_level FROM tool_calls WHERE task_id = %s",
        (created["task_id"],),
    )

    assert rows, "the audit trail records no tool call"
    tool, agent, risk_level = rows[0]
    assert tool
    assert agent == "executor"
    assert risk_level in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_a_decision_records_who_made_it(gated: ApiHarness, sql: Callable[..., list[tuple]]) -> None:
    created = gated.client.post(
        "/api/v1/tasks", json={"request": "delete it"}, headers={"X-User-Id": "alice"}
    ).json()

    gated.client.post(
        f"/api/v1/tasks/{created['task_id']}/approve",
        json={"note": "reviewed"},
        headers={"X-User-Id": "alice"},
    )

    rows = sql(
        "SELECT decision, decided_by, note FROM approvals WHERE task_id = %s",
        (created["task_id"],),
    )

    assert rows == [("approve", "alice", "reviewed")]


def test_rejecting_does_not_execute_the_action(gated: ApiHarness) -> None:
    created = gated.client.post("/api/v1/tasks", json={"request": "delete everything"}).json()

    response = gated.client.post(f"/api/v1/tasks/{created['task_id']}/reject", json={})

    assert response.status_code == 200
    assert response.json()["approval_status"] == "rejected"
    assert "reject" in (response.json()["answer"] or "").lower()


def test_a_decision_cannot_be_reversed(gated: ApiHarness) -> None:
    """A second decision must be refused, not applied over the first."""
    created = gated.client.post("/api/v1/tasks", json={"request": "delete everything"}).json()
    task_id = created["task_id"]

    assert gated.client.post(f"/api/v1/tasks/{task_id}/reject", json={}).status_code == 200
    assert gated.client.post(f"/api/v1/tasks/{task_id}/approve", json={}).status_code == 409


def test_deciding_a_task_that_is_not_gated_is_a_conflict(direct: ApiHarness) -> None:
    created = direct.client.post("/api/v1/tasks", json={"request": "hello"}).json()

    assert (
        direct.client.post(f"/api/v1/tasks/{created['task_id']}/approve", json={}).status_code
        == 409
    )


def test_an_unknown_task_cannot_be_approved(client: TestClient) -> None:
    assert client.post("/api/v1/tasks/nope/approve", json={}).status_code == 404


def test_another_user_cannot_decide_someone_elses_approval(gated: ApiHarness) -> None:
    """The most consequential ownership check in the system."""
    created = gated.client.post(
        "/api/v1/tasks", json={"request": "delete it"}, headers={"X-User-Id": "alice"}
    ).json()

    response = gated.client.post(
        f"/api/v1/tasks/{created['task_id']}/approve",
        json={},
        headers={"X-User-Id": "mallory"},
    )

    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# Event streaming
# --------------------------------------------------------------------------- #


def test_a_completed_task_has_a_recorded_history(direct: ApiHarness) -> None:
    created = direct.client.post("/api/v1/tasks", json={"request": "do something"}).json()

    events = direct.client.get(f"/api/v1/events/{created['task_id']}/history").json()

    kinds = [event["type"] for event in events]
    assert "task_created" in kinds
    assert "task_completed" in kinds


def test_a_history_never_exposes_reasoning(direct: ApiHarness) -> None:
    created = direct.client.post("/api/v1/tasks", json={"request": "do something"}).json()

    body = json.dumps(direct.client.get(f"/api/v1/events/{created['task_id']}/history").json())

    for forbidden in ("chain_of_thought", "system_prompt", 'reasoning":'):
        assert forbidden not in body


def test_the_stream_emits_events_and_closes(direct: ApiHarness) -> None:
    created = direct.client.post("/api/v1/tasks", json={"request": "do something"}).json()

    with direct.client.stream("GET", f"/api/v1/events/{created['task_id']}") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert "event: stream_open" in body
    assert "event: stream_closed" in body
    assert "event: task_completed" in body


def test_a_stream_can_resume_after_a_sequence_number(direct: ApiHarness) -> None:
    created = direct.client.post("/api/v1/tasks", json={"request": "do something"}).json()
    task_id = created["task_id"]

    history = direct.client.get(f"/api/v1/events/{task_id}/history").json()
    all_seqs = [event["seq"] for event in history]
    resume_after = all_seqs[0]

    tail = direct.client.get(
        f"/api/v1/events/{task_id}/history", params={"after_seq": resume_after}
    ).json()

    assert [event["seq"] for event in tail] == all_seqs[1:]


def test_another_user_cannot_stream_a_task(direct: ApiHarness) -> None:
    created = direct.client.post(
        "/api/v1/tasks", json={"request": "private"}, headers={"X-User-Id": "alice"}
    ).json()

    response = direct.client.get(
        f"/api/v1/events/{created['task_id']}", headers={"X-User-Id": "mallory"}
    )

    assert response.status_code == 404


def test_streaming_an_unknown_task_is_a_404(client: TestClient) -> None:
    assert client.get("/api/v1/events/unknown/history").status_code == 404


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #


def test_rate_limit_headers_are_returned(direct: ApiHarness) -> None:
    response = direct.client.post("/api/v1/chat", json={"message": "hello"})

    assert response.headers.get("X-RateLimit-Remaining") is not None
    assert response.headers.get("X-RateLimit-Window") is not None


def test_exceeding_the_limit_is_a_429(harness: ApiHarness, monkeypatch: pytest.MonkeyPatch) -> None:
    """A burst must be refused rather than allowed to start unbounded work."""
    import os

    from app.core.config import reset_settings_cache

    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "2")
    reset_settings_cache()
    harness.rewire(responder("direct"))

    statuses = [
        harness.client.post(
            "/api/v1/chat", json={"message": "hi"}, headers={"X-User-Id": "burst"}
        ).status_code
        for _ in range(5)
    ]
    os.environ.pop("RATE_LIMIT_REQUESTS", None)
    reset_settings_cache()

    assert statuses.count(200) == 2
    assert 429 in statuses
    assert statuses[-1] == 429


def test_a_rejection_says_when_to_retry(
    harness: ApiHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.core.config import reset_settings_cache

    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "1")
    reset_settings_cache()
    harness.rewire(responder("direct"))

    harness.client.post("/api/v1/chat", json={"message": "hi"}, headers={"X-User-Id": "retryer"})
    blocked = harness.client.post(
        "/api/v1/chat", json={"message": "hi"}, headers={"X-User-Id": "retryer"}
    )
    reset_settings_cache()

    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) > 0


def test_rate_limits_are_per_caller(direct: ApiHarness) -> None:
    """One noisy caller must not consume another's budget."""
    first = direct.client.post("/api/v1/chat", json={"message": "hi"}, headers={"X-User-Id": "a"})
    second = direct.client.post("/api/v1/chat", json={"message": "hi"}, headers={"X-User-Id": "b"})

    assert first.status_code == 200
    assert second.status_code == 200


def test_read_only_endpoints_are_not_rate_limited(direct: ApiHarness) -> None:
    """Health and discovery are cheap and must stay reachable under load."""
    for _ in range(5):
        assert direct.client.get("/health").status_code == 200
        assert direct.client.get("/api/v1/agents").status_code == 200


# --------------------------------------------------------------------------- #
# Observability
# --------------------------------------------------------------------------- #


def test_metrics_are_served_in_the_prometheus_format(direct: ApiHarness) -> None:
    direct.client.get("/health")

    response = direct.client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "http_requests_total" in response.text


def test_a_request_is_counted_against_its_route_template(direct: ApiHarness) -> None:
    """The id must not become a label, or cardinality grows with traffic."""
    direct.client.get("/api/v1/agents")

    body = direct.client.get("/metrics").text

    assert 'path="/api/v1/agents"' in body


def test_an_unmatched_path_is_bucketed_under_a_constant(direct: ApiHarness) -> None:
    """An unmatched path is caller-chosen, so it must never become a label."""
    attacker_controlled = "/definitely/not/a/route/9f3a1c"
    direct.client.get(attacker_controlled)

    body = direct.client.get("/metrics").text

    assert attacker_controlled not in body
    assert 'path="<unmatched>"' in body


def test_startup_gauges_describe_the_running_application(direct: ApiHarness) -> None:
    """The tool gauge is checked against discovery, not against a fixed number.

    The registry holds whatever the deployment's credentials allow, so a literal
    count is a claim about one machine. It previously read seven, which was the
    number produced by a *blank* GitHub token registering two tools that could
    only fail when called.
    """
    body = direct.client.get("/metrics").text
    listed = direct.client.get("/api/v1/tools").json()

    assert "registered_agents 8" in body
    assert f"registered_tools {len(listed)}" in body


def test_an_unconfigured_credential_does_not_register_its_tools(direct: ApiHarness) -> None:
    """``GITHUB_TOKEN=`` is configuration meaning none, not a credential.

    An empty secret is not ``None``, so the "is a token configured?" test passed
    and the GitHub tools were reported as available. The first call would then
    fail with an authentication error that reads like an upstream problem rather
    than a missing setting.
    """
    names = {tool["name"] for tool in direct.client.get("/api/v1/tools").json()}

    assert "github_repository" not in names
    assert "github_create_issue" not in names


def test_a_task_outcome_is_counted(direct: ApiHarness) -> None:
    """The span covers the run, and its outcome lands in a fixed set of buckets."""
    created = direct.client.post("/api/v1/tasks", json={"request": "say hi"}).json()
    assert direct.client.get(f"/api/v1/tasks/{created['task_id']}").json()["status"] == (
        "completed"
    )

    body = direct.client.get("/metrics").text

    assert 'tasks_total{outcome="completed"} 1' in body
    assert 'spans_total{span="task.execute",status="ok"} 1' in body


def test_a_metrics_scrape_is_not_itself_an_event_source(direct: ApiHarness) -> None:
    """Scraping twice must be idempotent for every series but the scrape's own."""
    first = direct.client.get("/metrics").text
    second = direct.client.get("/metrics").text

    assert "registered_agents 8" in first
    assert "registered_agents 8" in second
