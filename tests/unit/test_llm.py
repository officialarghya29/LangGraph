"""Tests for the LLM provider abstraction."""

from __future__ import annotations

import json
from collections.abc import Callable

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
    LLMProvider,
    Message,
    OpenAICompatibleProvider,
    Role,
    _extract_json,
    build_llm_provider,
    is_reasoning_model,
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


def test_the_factory_gives_each_provider_a_model_its_vendor_serves() -> None:
    """The default model follows the provider.

    A single default was an OpenAI name, so ``LLM_PROVIDER=anthropic`` with the
    model left unset sent a GPT model to the Messages API. This asserts the
    resolved name, not that a provider object was built — the object was always
    built; it was the name inside it that was wrong.
    """
    openai = build_settings(llm_provider="openai", llm_api_key="key")
    anthropic = build_settings(llm_provider="anthropic", llm_api_key="key")

    assert build_llm_provider(openai).default_model == "gpt-5.6-terra"
    assert build_llm_provider(anthropic).default_model == "claude-sonnet-5-5"
    assert "gpt" not in build_llm_provider(anthropic).default_model


def test_the_factory_passes_the_sampling_settings_to_the_provider() -> None:
    """``LLM_TEMPERATURE`` and the output ceiling have to reach the provider.

    Both are read by the provider and neither is read anywhere else, so a factory
    that dropped them would leave two settings silently inert.
    """
    settings = build_settings(
        llm_provider="openai", llm_api_key="key", llm_temperature=0.7, llm_max_output_tokens=1234
    )

    provider = build_llm_provider(settings)

    assert provider.default_temperature == 0.7
    assert provider.default_max_tokens == 1234


# --------------------------------------------------------------------------- #
# Sampling parameters the target model will accept
# --------------------------------------------------------------------------- #


def anthropic_ok_body(content: str = "hello") -> dict[str, object]:
    """Return a well-formed Anthropic Messages payload."""
    return {
        "model": "claude-test",
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 4, "output_tokens": 6},
    }


async def _capture(
    handler_body: dict[str, object],
    build: Callable[[httpx.AsyncClient], LLMProvider],
    *,
    call_temperature: float | None = None,
    call_max_tokens: int | None = None,
) -> dict[str, object]:
    """Send one call and return the JSON body that went out.

    Args:
        handler_body: What the scripted endpoint replies with.
        build: Builds the provider under test around the scripted client.
        call_temperature: Temperature for the call, or ``None`` to omit it.
        call_max_tokens: Ceiling for the call, or ``None`` to omit it.

    Returns:
        The parsed request body.
    """
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=handler_body, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await build(client).ainvoke(
            user("hello"), temperature=call_temperature, max_tokens=call_max_tokens
        )
    finally:
        await client.aclose()

    body = captured["body"]
    assert isinstance(body, dict)
    return body


async def openai_body(
    model: str,
    *,
    call_temperature: float | None = None,
    call_max_tokens: int | None = None,
    provider_temperature: float = 0.0,
    provider_max_tokens: int | None = None,
) -> dict[str, object]:
    """Return the body an OpenAI-compatible call to ``model`` sends."""
    return await _capture(
        openai_ok_body(),
        lambda client: OpenAICompatibleProvider(
            api_key="k",
            model=model,
            client=client,
            default_temperature=provider_temperature,
            default_max_tokens=provider_max_tokens,
        ),
        call_temperature=call_temperature,
        call_max_tokens=call_max_tokens,
    )


async def anthropic_body(
    *,
    call_temperature: float | None = None,
    provider_temperature: float = 0.0,
    provider_max_tokens: int = AnthropicProvider.DEFAULT_MAX_TOKENS,
) -> dict[str, object]:
    """Return the body an Anthropic call sends."""
    return await _capture(
        anthropic_ok_body(),
        lambda client: AnthropicProvider(
            api_key="k",
            model="claude-sonnet-5-5",
            client=client,
            max_tokens=provider_max_tokens,
            default_temperature=provider_temperature,
        ),
        call_temperature=call_temperature,
    )


@pytest.mark.parametrize(
    ("model", "reasoning"),
    [
        ("gpt-5.6-terra", True),
        ("gpt-5.6-luna", True),
        ("gpt-5.6-sol", True),
        ("o3-mini", True),
        ("gpt-4o-mini", False),
        ("gpt-4.1", False),
        ("gpt-oss:20b", False),
        ("llama3.1:8b", False),
        ("mistral-nemo", False),
    ],
)
def test_only_models_that_reject_sampling_parameters_are_recognised(
    model: str, reasoning: bool
) -> None:
    """The predicate has to be right in both directions.

    A false positive strips a temperature the server would have honoured; a false
    negative sends one it rejects, and the whole request fails. ``gpt-oss:20b`` is
    the interesting case: an OpenAI model name that is served locally and accepts
    the ordinary parameters.
    """
    assert is_reasoning_model(model) is reasoning


async def test_a_reasoning_model_is_never_sent_a_temperature() -> None:
    """It is a hard 400, not an ignored field.

    "Unsupported value: 'temperature' does not support 0.0 with this model" — and
    the shipped default model is one of these, so getting this wrong makes every
    call in the system fail.
    """
    body = await openai_body("gpt-5.6-terra", call_temperature=0.4)

    assert "temperature" not in body
    assert body["model"] == "gpt-5.6-terra"


async def test_a_reasoning_model_is_given_the_ceiling_under_the_right_name() -> None:
    """``max_tokens`` is rejected as an unsupported parameter on these models."""
    body = await openai_body("gpt-5.6-terra", call_max_tokens=512)

    assert body["max_completion_tokens"] == 512
    assert "max_tokens" not in body


async def test_an_ordinary_model_keeps_both_parameters() -> None:
    """The other half of the branch: names that do take the sampling knobs."""
    body = await openai_body("gpt-4o-mini", call_temperature=0.4, call_max_tokens=512)

    assert body["temperature"] == 0.4
    assert body["max_tokens"] == 512
    assert "max_completion_tokens" not in body


async def test_the_provider_temperature_applies_when_a_call_states_none() -> None:
    """``LLM_TEMPERATURE`` reaches the wire without every call site restating it."""
    body = await openai_body("gpt-4o-mini", provider_temperature=0.7)

    assert body["temperature"] == 0.7


async def test_an_explicit_temperature_beats_the_provider_default() -> None:
    """The setting is a default, not an override of the call site."""
    body = await openai_body("gpt-4o-mini", call_temperature=0.1, provider_temperature=0.7)

    assert body["temperature"] == 0.1


async def test_the_provider_ceiling_applies_when_a_call_states_none() -> None:
    """``LLM_MAX_OUTPUT_TOKENS`` has to reach the wire as well."""
    body = await openai_body("gpt-4o-mini", provider_max_tokens=2048)

    assert body["max_tokens"] == 2048


async def test_anthropic_gets_the_provider_temperature_too() -> None:
    """Both vendors, one setting, so the configuration does not depend on the API."""
    body = await anthropic_body(provider_temperature=0.3, provider_max_tokens=2048)

    assert body["temperature"] == 0.3
    assert body["max_tokens"] == 2048


async def test_structured_output_forwards_the_sampling_parameters() -> None:
    """A structured call is still a model call.

    The correction loop is exactly where a caller wants to pin the temperature,
    and this path used to drop both parameters on the floor.
    """
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200, json=openai_ok_body('{"passed": true, "reason": "fine"}'), request=request
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(api_key="k", model="gpt-4o-mini", client=client)

    result = await provider.astructured_output(
        user("grade this"), Verdict, temperature=0.9, max_tokens=77
    )

    assert result.passed is True
    assert seen[0]["temperature"] == 0.9
    assert seen[0]["max_tokens"] == 77
    await client.aclose()
