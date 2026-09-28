"""The provider adapters, over a real socket.

Every other test of these adapters replaces the transport, so they have never
actually opened a connection. That leaves a class of bug untested by
construction: a wrong base path, a header that never gets sent, a body that
serialises to something a server would reject, a connection that is opened and
never closed. Mocking the transport tests the *parsing*; it cannot test the
*request*.

So this module runs a small HTTP server on loopback, scripts what it answers, and
points a provider at it. It is not a substitute for a real vendor call — that
still needs a credential and a network — but it is the difference between
"the parsing logic works" and "the adapter speaks HTTP".

The server binds port 0 and reports the port it was given, so a busy machine
cannot make these tests flaky.
"""

from __future__ import annotations

import http.server
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from app.core.exceptions import LLMError, LLMRateLimitError
from app.services.llm import (
    AnthropicProvider,
    Message,
    OpenAICompatibleProvider,
    Role,
)

MESSAGES = [Message(role=Role.SYSTEM, content="be brief"), Message(role=Role.USER, content="hi")]

#: A key distinctive enough that a leak is unmistakable in an assertion.
API_KEY = "sk-test-DO-NOT-LEAK-0123456789"


class _Scripted(http.server.BaseHTTPRequestHandler):
    """Answers each request from a script, and records what it received."""

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        """Read the request, record it, and answer with the next scripted reply."""
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body: Any = json.loads(raw or b"{}")
        except ValueError:
            body = {"unparseable": raw.decode("utf-8", "replace")}

        server: Any = self.server
        server.received.append(
            {
                "path": self.path,
                "headers": {key.lower(): value for key, value in self.headers.items()},
                "body": body,
            }
        )

        status, extra_headers, payload = server.script.pop(0)
        encoded = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        for key, value in extra_headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args: Any) -> None:
        """Silence the default access log; this is a test fixture."""


@contextmanager
def scripted(*replies: tuple[int, dict[str, str], Any]) -> Iterator[Any]:
    """Run a loopback server that answers with ``replies`` in order.

    Args:
        *replies: ``(status, headers, body)`` triples, consumed one per request.

    Yields:
        The server, with a ``base_url`` and the list of requests it received.
    """
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Scripted)
    server.script = list(replies)  # type: ignore[attr-defined]
    server.received = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        server.base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"  # type: ignore[attr-defined]
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def completion(content: str = "hello there", *, model: str = "gpt-test") -> dict[str, Any]:
    """Return a well-formed OpenAI chat-completions payload."""
    return {
        "model": model,
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def anthropic_message(content: str = "hello there") -> dict[str, Any]:
    """Return a well-formed Anthropic Messages payload."""
    return {
        "model": "claude-test",
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 13, "output_tokens": 5},
    }


def openai(server: Any) -> OpenAICompatibleProvider:
    """Return an OpenAI-compatible provider pointed at ``server``."""
    return OpenAICompatibleProvider(api_key=API_KEY, model="gpt-test", base_url=server.base_url)


def anthropic(server: Any) -> AnthropicProvider:
    """Return an Anthropic provider pointed at ``server``."""
    return AnthropicProvider(api_key=API_KEY, model="claude-test", base_url=server.base_url)


# --------------------------------------------------------------------------- #
# The request
# --------------------------------------------------------------------------- #


async def test_a_completion_returns_parsed_content_and_usage() -> None:
    """The happy path, over a socket rather than a stub transport."""
    with scripted((200, {}, completion("a real answer"))) as server:
        response = await openai(server).ainvoke(MESSAGES)

    assert response.content == "a real answer"
    assert response.model == "gpt-test"
    assert response.finish_reason == "stop"
    assert response.usage.prompt_tokens == 11
    assert response.usage.completion_tokens == 7
    assert response.usage.total_tokens == 18


async def test_the_request_reaches_the_documented_endpoint() -> None:
    """A wrong path is exactly what a mocked transport cannot catch."""
    with scripted((200, {}, completion())) as server:
        await openai(server).ainvoke(MESSAGES)

    assert server.received[0]["path"] == "/v1/chat/completions"


async def test_the_request_carries_a_bearer_token_and_json() -> None:
    """Without these the vendor returns 401 in production and nothing runs."""
    with scripted((200, {}, completion())) as server:
        await openai(server).ainvoke(MESSAGES)

    headers = server.received[0]["headers"]

    assert headers["authorization"] == f"Bearer {API_KEY}"
    assert headers["content-type"].startswith("application/json")


async def test_the_body_keeps_the_roles_and_the_model() -> None:
    """A dropped system message changes behaviour without failing anything."""
    with scripted((200, {}, completion())) as server:
        await openai(server).ainvoke(MESSAGES, temperature=0.2, max_tokens=64)

    body = server.received[0]["body"]

    assert body["model"] == "gpt-test"
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 64
    assert body["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]


async def test_a_reasoning_model_reaches_the_socket_without_a_temperature() -> None:
    """The shape that used to be rejected, sent over a real connection.

    OpenAI fails these requests outright rather than ignoring the field, and the
    shipped default model is one of them — so this is the difference between the
    system working and every call failing.
    """
    with scripted((200, {}, completion(model="gpt-5.6-terra"))) as server:
        provider = OpenAICompatibleProvider(
            api_key=API_KEY, model="gpt-5.6-terra", base_url=server.base_url
        )
        await provider.ainvoke(MESSAGES, temperature=0.3, max_tokens=256)

    body = server.received[0]["body"]

    assert body["model"] == "gpt-5.6-terra"
    assert "temperature" not in body
    assert body["max_completion_tokens"] == 256
    assert "max_tokens" not in body


async def test_an_anthropic_call_uses_its_own_shape() -> None:
    """Different vendor, different payload and different usage field names."""
    with scripted((200, {}, anthropic_message("an answer"))) as server:
        provider = anthropic(server)
        response = await provider.ainvoke(MESSAGES)

    assert server.received[0]["path"] == "/v1/messages"
    assert response.content == "an answer"
    assert response.usage.prompt_tokens == 13
    assert response.usage.completion_tokens == 5
    assert response.usage.total_tokens == 18
    assert server.received[0]["headers"]["x-api-key"] == API_KEY
    # Anthropic requires an explicit ceiling on every request.
    assert server.received[0]["body"]["max_tokens"] > 0


async def test_the_key_never_appears_in_an_error() -> None:
    """Errors are logged; a credential in one is a credential in a log file."""
    with (
        scripted((500, {}, {"error": "upstream exploded"})) as server,
        pytest.raises(LLMError) as captured,
    ):
        await openai(server).ainvoke(MESSAGES)

    assert API_KEY not in str(captured.value)
    assert API_KEY not in repr(captured.value.detail)


# --------------------------------------------------------------------------- #
# The response
# --------------------------------------------------------------------------- #


async def test_a_throttle_is_classified_with_its_hint() -> None:
    """End to end: the header has to survive a real connection."""
    with (
        scripted((429, {"Retry-After": "17"}, {"error": "slow down"})) as server,
        pytest.raises(LLMRateLimitError) as captured,
    ):
        await openai(server).ainvoke(MESSAGES)

    assert captured.value.retry_after == 17.0


async def test_a_server_error_is_an_llm_error_not_a_crash() -> None:
    """A 500 must become a classified failure the retry policy can reason about."""
    with scripted((500, {}, {"error": "boom"})) as server, pytest.raises(LLMError):
        await openai(server).ainvoke(MESSAGES)


async def test_a_client_error_is_not_retried_as_a_throttle() -> None:
    """400 is a bad request, and no amount of backoff fixes it."""
    with (
        scripted((400, {}, {"error": "bad model"})) as server,
        pytest.raises(LLMError) as captured,
    ):
        await openai(server).ainvoke(MESSAGES)

    assert not isinstance(captured.value, LLMRateLimitError)
    assert "400" in (captured.value.detail or "")


async def test_a_malformed_body_is_reported_as_such() -> None:
    """A 200 with the wrong shape is a provider bug, not a crash."""
    with (
        scripted((200, {}, {"unexpected": "shape"})) as server,
        pytest.raises(LLMError) as captured,
    ):
        await openai(server).ainvoke(MESSAGES)

    assert "unexpected payload" in captured.value.message


async def test_a_body_that_is_not_json_is_reported_as_such() -> None:
    """A proxy returning an HTML error page must not raise a JSON decode error."""
    with (
        scripted((200, {}, b"<html>not json</html>")) as server,
        pytest.raises(LLMError) as captured,
    ):
        await openai(server).ainvoke(MESSAGES)

    assert "unexpected payload" in captured.value.message


@pytest.mark.parametrize("content", ["", "   "])
async def test_an_empty_completion_is_an_empty_string_not_none(content: str) -> None:
    """Callers concatenate this; ``None`` would turn a quiet reply into a crash."""
    with scripted((200, {}, completion(content))) as server:
        response = await openai(server).ainvoke(MESSAGES)

    assert response.content == content.strip() or response.content == content


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #


async def test_a_transient_failure_survives_the_decorators() -> None:
    """The whole stack — retry outside budget — against a real server.

    The server fails once and then succeeds, which is the scenario the retry
    decorator exists for, and the only test here that exercises both decorators
    plus the transport together.
    """
    from app.services.budget import BudgetedProvider, TokenBudget, token_budget
    from app.services.resilience import RetryingProvider

    async def no_sleep(seconds: float) -> None:
        del seconds

    with scripted(
        (500, {}, {"error": "transient"}),
        (200, {}, completion("second time lucky")),
    ) as server:
        provider = RetryingProvider(BudgetedProvider(openai(server)), max_retries=2, sleep=no_sleep)
        budget = TokenBudget(limit=1_000_000)
        with token_budget(budget):
            response = await provider.ainvoke(MESSAGES)

    assert response.content == "second time lucky"
    assert len(server.received) == 2
    assert budget.calls == 1, "only the successful attempt reported usage"


async def test_an_exhausted_budget_never_reaches_the_network() -> None:
    """The refusal happens before a request is made, which is the point of it."""
    from app.services.budget import BudgetedProvider, TokenBudget, token_budget
    from app.services.resilience import RetryingProvider

    async def no_sleep(seconds: float) -> None:
        del seconds

    with scripted((200, {}, completion())) as server:
        provider = RetryingProvider(BudgetedProvider(openai(server)), sleep=no_sleep)
        budget = TokenBudget(limit=1)
        with token_budget(budget):
            await provider.ainvoke(MESSAGES)
            with pytest.raises(Exception, match="token budget"):
                await provider.ainvoke(MESSAGES)

    assert len(server.received) == 1
