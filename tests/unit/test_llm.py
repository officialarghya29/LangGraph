"""Tests for the LLM provider abstraction."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from app.core.config import Settings
from app.core.exceptions import (
    ConfigurationError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    StructuredOutputError,
)
from app.services.llm import (
    AnthropicProvider,
    FakeLLMProvider,
    Message,
    OpenAICompatibleProvider,
    Role,
    _extract_json,
    build_llm_provider,
)


class Verdict(BaseModel):
    """Small schema used to exercise structured output."""

    passed: bool
    reason: str


def user(text: str) -> list[Message]:
    return [Message(role=Role.USER, content=text)]


# --------------------------------------------------------------------------- #
# Fake provider
# --------------------------------------------------------------------------- #


async def test_fake_provider_returns_scripted_responses_in_order() -> None:
    provider = FakeLLMProvider(["first", "second"])

    assert (await provider.ainvoke(user("a"))).content == "first"
    assert (await provider.ainvoke(user("b"))).content == "second"


async def test_fake_provider_falls_back_to_default() -> None:
    provider = FakeLLMProvider(default="fallback")

    assert (await provider.ainvoke(user("a"))).content == "fallback"


async def test_fake_provider_supports_a_responder_callable() -> None:
    provider = FakeLLMProvider(responder=lambda messages: messages[-1].content.upper())

    assert (await provider.ainvoke(user("hello"))).content == "HELLO"


async def test_fake_provider_records_calls() -> None:
    provider = FakeLLMProvider()

    await provider.ainvoke(user("remember me"))

    assert provider.call_count == 1
    assert "remember me" in provider.last_prompt()


async def test_fake_provider_reports_pending_responses() -> None:
    provider = FakeLLMProvider(["a", "b"])

    await provider.ainvoke(user("x"))

    assert provider.pending == 1


def test_fake_provider_sync_wrapper_works() -> None:
    provider = FakeLLMProvider(["sync"])

    assert provider.invoke(user("x")).content == "sync"


async def test_fake_provider_reports_deterministic_usage() -> None:
    provider = FakeLLMProvider(["four"])

    usage = (await provider.ainvoke(user("12345678"))).usage

    assert usage.prompt_tokens == 2
    assert usage.completion_tokens == 1
    assert usage.total_tokens == 3


# --------------------------------------------------------------------------- #
# Structured output
# --------------------------------------------------------------------------- #


async def test_structured_output_validates_and_returns_the_model() -> None:
    payload = json.dumps({"passed": True, "reason": "looks right"})
    provider = FakeLLMProvider([payload])

    verdict = await provider.astructured_output(user("judge this"), Verdict)

    assert verdict.passed is True
    assert verdict.reason == "looks right"


async def test_structured_output_instructs_the_model_with_the_schema() -> None:
    provider = FakeLLMProvider([json.dumps({"passed": True, "reason": "ok"})])

    await provider.astructured_output(user("judge"), Verdict)

    prompt = provider.last_prompt()
    assert "JSON" in prompt
    assert "passed" in prompt


async def test_structured_output_strips_markdown_fences() -> None:
    fenced = '```json\n{"passed": false, "reason": "nope"}\n```'
    provider = FakeLLMProvider([fenced])

    verdict = await provider.astructured_output(user("judge"), Verdict)

    assert verdict.passed is False


async def test_structured_output_retries_after_invalid_json() -> None:
    provider = FakeLLMProvider(["not json at all", '{"passed": true, "reason": "fixed"}'])

    verdict = await provider.astructured_output(user("judge"), Verdict)

    assert verdict.passed is True
    assert provider.call_count == 2


async def test_structured_output_feeds_the_error_back_to_the_model() -> None:
    provider = FakeLLMProvider(['{"passed": true}', '{"passed": true, "reason": "now valid"}'])

    await provider.astructured_output(user("judge"), Verdict)

    assert "reason" in provider.last_prompt()


async def test_structured_output_raises_after_exhausting_attempts() -> None:
    provider = FakeLLMProvider(["garbage"])

    with pytest.raises(StructuredOutputError):
        await provider.astructured_output(user("judge"), Verdict, max_attempts=2)

    assert provider.call_count == 2


async def test_structured_output_rejects_a_zero_attempt_budget() -> None:
    provider = FakeLLMProvider()

    with pytest.raises(ValueError, match="max_attempts"):
        await provider.astructured_output(user("judge"), Verdict, max_attempts=0)


def test_structured_output_sync_wrapper_works() -> None:
    provider = FakeLLMProvider([json.dumps({"passed": True, "reason": "ok"})])

    assert provider.structured_output(user("judge"), Verdict).passed is True


def test_extract_json_handles_surrounding_prose() -> None:
    assert _extract_json('Sure! {"a": 1} Hope that helps.') == '{"a": 1}'


def test_extract_json_rejects_text_without_an_object() -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        _extract_json("there is no object here")


# --------------------------------------------------------------------------- #
# OpenAI-compatible provider
# --------------------------------------------------------------------------- #


def openai_transport(
    *, status: int = 200, body: dict[str, object] | None = None, raise_timeout: bool = False
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if raise_timeout:
            raise httpx.ReadTimeout("timed out", request=request)
        if body is not None:
            return httpx.Response(status, json=body, request=request)
        return httpx.Response(status, request=request)

    return httpx.MockTransport(handler)


def openai_ok_body(content: str = "hello") -> dict[str, object]:
    return {
        "model": "gpt-test",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }


async def test_openai_provider_parses_a_successful_response() -> None:
    client = httpx.AsyncClient(transport=openai_transport(body=openai_ok_body("hi there")))
    provider = OpenAICompatibleProvider(api_key="k", model="gpt-test", client=client)

    response = await provider.ainvoke(user("hello"))

    assert response.content == "hi there"
    assert response.model == "gpt-test"
    assert response.usage.total_tokens == 12
    await client.aclose()


async def test_openai_provider_sends_the_expected_request() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=openai_ok_body(), request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(
        api_key="secret-key", model="gpt-test", base_url="https://example.test/v1", client=client
    )

    await provider.ainvoke(user("hello"), temperature=0.25, max_tokens=99)

    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["auth"] == "Bearer secret-key"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "gpt-test"
    assert body["temperature"] == 0.25
    assert body["max_tokens"] == 99
    await client.aclose()


async def test_openai_provider_maps_429_to_rate_limit() -> None:
    client = httpx.AsyncClient(transport=openai_transport(status=429))
    provider = OpenAICompatibleProvider(api_key="k", model="m", client=client)

    with pytest.raises(LLMRateLimitError):
        await provider.ainvoke(user("hello"))

    await client.aclose()


async def test_openai_provider_maps_server_errors() -> None:
    client = httpx.AsyncClient(transport=openai_transport(status=500))
    provider = OpenAICompatibleProvider(api_key="k", model="m", client=client)

    with pytest.raises(LLMError):
        await provider.ainvoke(user("hello"))

    await client.aclose()


async def test_openai_provider_maps_timeouts() -> None:
    client = httpx.AsyncClient(transport=openai_transport(raise_timeout=True))
    provider = OpenAICompatibleProvider(api_key="k", model="m", client=client)

    with pytest.raises(LLMTimeoutError):
        await provider.ainvoke(user("hello"))

    await client.aclose()


async def test_openai_provider_rejects_an_unexpected_payload() -> None:
    client = httpx.AsyncClient(transport=openai_transport(body={"unexpected": True}))
    provider = OpenAICompatibleProvider(api_key="k", model="m", client=client)

    with pytest.raises(LLMError, match="unexpected payload"):
        await provider.ainvoke(user("hello"))

    await client.aclose()


async def test_openai_provider_does_not_leak_the_key_in_errors() -> None:
    client = httpx.AsyncClient(transport=openai_transport(status=401))
    provider = OpenAICompatibleProvider(api_key="super-secret-key", model="m", client=client)

    with pytest.raises(LLMError) as exc_info:
        await provider.ainvoke(user("hello"))

    assert "super-secret-key" not in str(exc_info.value)
    await client.aclose()


# --------------------------------------------------------------------------- #
# Anthropic provider
# --------------------------------------------------------------------------- #


async def test_anthropic_provider_parses_a_successful_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "claude-test",
                "content": [{"type": "text", "text": "the answer"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="k", model="claude-test", client=client)

    response = await provider.ainvoke(user("hello"))

    assert response.content == "the answer"
    assert response.usage.total_tokens == 7
    assert response.finish_reason == "end_turn"
    await client.aclose()


async def test_anthropic_provider_lifts_system_prompts_out_of_the_conversation() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "ok"}], "usage": {}},
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="k", model="claude-test", client=client)

    await provider.ainvoke(
        [Message(role=Role.SYSTEM, content="be brief"), Message(role=Role.USER, content="hello")]
    )

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["system"] == "be brief"
    assert all(turn["role"] != "system" for turn in body["messages"])
    await client.aclose()


async def test_anthropic_provider_sends_the_api_version_header() -> None:
    captured: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["version"] = request.headers.get("anthropic-version")
        return httpx.Response(200, json={"content": [], "usage": {}}, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="k", model="claude-test", client=client)

    await provider.ainvoke(user("hello"))

    assert captured["version"] == "2023-06-01"
    await client.aclose()


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def build_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_factory_builds_an_openai_provider() -> None:
    settings = build_settings(llm_provider="openai", llm_api_key="key")

    assert isinstance(build_llm_provider(settings), OpenAICompatibleProvider)


def test_factory_builds_an_anthropic_provider() -> None:
    settings = build_settings(llm_provider="anthropic", llm_api_key="key")

    assert isinstance(build_llm_provider(settings), AnthropicProvider)


def test_factory_builds_a_local_provider_when_a_base_url_is_given() -> None:
    settings = build_settings(llm_provider="local", llm_base_url="http://localhost:11434/v1")

    assert isinstance(build_llm_provider(settings), OpenAICompatibleProvider)


def test_factory_requires_a_base_url_for_local() -> None:
    settings = build_settings(llm_provider="local")

    with pytest.raises(ConfigurationError, match="LLM_BASE_URL"):
        build_llm_provider(settings)


def test_factory_requires_a_key_for_hosted_providers() -> None:
    settings = build_settings(llm_provider="openai")

    with pytest.raises(ConfigurationError, match="LLM_API_KEY"):
        build_llm_provider(settings)
