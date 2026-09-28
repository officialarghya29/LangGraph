"""Provider-independent access to language models.

Agents and graph nodes depend on :class:`LLMProvider`, never on a vendor SDK.
Adding a vendor means adding one class here; no agent has to change.

Structured output is implemented once, on the base class, by instructing the
model to emit JSON and validating the result against a Pydantic schema. That
keeps structured generation behaving identically across providers instead of
relying on vendor-specific tool-calling formats.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Coroutine, Sequence
from enum import StrEnum
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.exceptions import (
    ConfigurationError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    StructuredOutputError,
)

__all__ = [
    "AnthropicProvider",
    "FakeLLMProvider",
    "LLMProvider",
    "LLMResponse",
    "Message",
    "OpenAICompatibleProvider",
    "Role",
    "TokenUsage",
    "build_llm_provider",
]

T = TypeVar("T", bound=BaseModel)


class Role(StrEnum):
    """Message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """A single chat message."""

    role: Role
    content: str


class TokenUsage(BaseModel):
    """Token accounting for one model call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """A model completion plus its accounting metadata."""

    content: str
    model: str
    usage: TokenUsage = TokenUsage()
    finish_reason: str | None = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _run_sync[R](coro: Coroutine[Any, Any, R]) -> R:
    """Run a coroutine to completion from synchronous code.

    Raises:
        LLMError: If called from inside a running event loop, where blocking
            would deadlock.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise LLMError("synchronous provider calls are not allowed inside an event loop")


def _strip_code_fence(text: str) -> str:
    """Remove a surrounding Markdown code fence, if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json(text: str) -> str:
    """Extract the outermost JSON object from a model response."""
    cleaned = _strip_code_fence(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in response")
    return cleaned[start : end + 1]


def _schema_instruction(schema: type[BaseModel]) -> str:
    """Build the instruction that constrains a model to a JSON schema."""
    return (
        "You must reply with a single JSON object and nothing else. "
        "No prose, no explanation, no Markdown fences.\n"
        "The JSON must validate against this schema:\n"
        f"{json.dumps(schema.model_json_schema(), indent=2)}"
    )


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #


class LLMProvider(ABC):
    """Abstract language-model provider.

    Subclasses implement :meth:`ainvoke` only. Structured output, retries on
    invalid JSON, and the synchronous wrappers are provided here so every
    provider behaves the same way.
    """

    #: Stable identifier used in logs and metadata.
    name: str = "provider"

    def __init__(self, *, default_model: str) -> None:
        self.default_model = default_model

    @abstractmethod
    async def ainvoke(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send messages to the model and return its completion."""

    def invoke(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Synchronous convenience wrapper around :meth:`ainvoke`."""
        return _run_sync(
            self.ainvoke(messages, model=model, temperature=temperature, max_tokens=max_tokens)
        )

    async def astructured_output(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        model: str | None = None,
        max_attempts: int = 3,
    ) -> T:
        """Request output conforming to ``schema``.

        Invalid responses are fed back to the model with the validation error so
        it can correct itself. Exhausting ``max_attempts`` raises
        :class:`StructuredOutputError` rather than returning partial data.

        Args:
            messages: The conversation to send.
            schema: Pydantic model the response must validate against.
            model: Optional model override.
            max_attempts: Total attempts including the first. Must be at least 1.

        Returns:
            A validated instance of ``schema``.

        Raises:
            StructuredOutputError: If no attempt produced valid output.
            ValueError: If ``max_attempts`` is less than 1.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        conversation = [
            Message(role=Role.SYSTEM, content=_schema_instruction(schema)),
            *messages,
        ]
        last_error: Exception | None = None

        for _ in range(max_attempts):
            response = await self.ainvoke(conversation, model=model)
            try:
                return schema.model_validate_json(_extract_json(response.content))
            except (ValidationError, ValueError) as exc:
                last_error = exc
                conversation = [
                    *conversation,
                    Message(role=Role.ASSISTANT, content=response.content),
                    Message(
                        role=Role.USER,
                        content=(
                            "That reply did not satisfy the schema. "
                            f"Problem: {exc}. Reply with only the corrected JSON object."
                        ),
                    ),
                ]

        raise StructuredOutputError(
            f"model did not return valid {schema.__name__} output",
            detail=str(last_error),
        )

    def structured_output(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        model: str | None = None,
        max_attempts: int = 3,
    ) -> T:
        """Synchronous convenience wrapper around :meth:`astructured_output`."""
        return _run_sync(
            self.astructured_output(messages, schema, model=model, max_attempts=max_attempts)
        )


# --------------------------------------------------------------------------- #
# OpenAI-compatible provider
# --------------------------------------------------------------------------- #


class OpenAICompatibleProvider(LLMProvider):
    """Provider for OpenAI and any OpenAI-compatible endpoint.

    The same class serves hosted OpenAI, and self-hosted runtimes such as Ollama
    or vLLM, by varying ``base_url``.
    """

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(default_model=model)
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = client

    def _payload(
        self,
        messages: Sequence[Message],
        model: str | None,
        temperature: float,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    async def ainvoke(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Call ``/chat/completions`` and normalise the response."""
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = self._payload(messages, model, temperature, max_tokens)

        client = self._client or httpx.AsyncClient(timeout=self._timeout)
        try:
            response = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("provider did not respond in time", detail=url) from exc
        except httpx.HTTPError as exc:
            raise LLMError("provider request failed", detail=str(exc)) from exc
        finally:
            if self._client is None:
                await client.aclose()

        if response.status_code == 429:
            raise LLMRateLimitError("provider rate limited the request", detail=url)
        if response.status_code >= 400:
            raise LLMError(
                "provider returned an error",
                detail=f"HTTP {response.status_code} from {url}",
            )

        try:
            body = response.json()
            choice = body["choices"][0]
            content = choice["message"]["content"]
            usage = body.get("usage") or {}
            finish = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("provider returned an unexpected payload", detail=str(exc)) from exc

        return LLMResponse(
            content=content or "",
            model=body.get("model", model or self.default_model),
            usage=TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
            finish_reason=finish,
        )


# --------------------------------------------------------------------------- #
# Anthropic provider
# --------------------------------------------------------------------------- #


class AnthropicProvider(LLMProvider):
    """Provider for the Anthropic Messages API."""

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.anthropic.com/v1",
        timeout: float = 60.0,
        max_tokens: int = 4096,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(default_model=model)
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._default_max_tokens = max_tokens
        self._client = client

    async def ainvoke(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Call ``/messages`` and normalise the response.

        Anthropic takes the system prompt as a top-level parameter, so system
        messages are lifted out of the conversation.
        """
        system_parts = [m.content for m in messages if m.role is Role.SYSTEM]
        turns = [
            {"role": m.role.value, "content": m.content}
            for m in messages
            if m.role is not Role.SYSTEM
        ]
        # The Messages API requires the conversation to start with a user turn.
        if not turns or turns[0]["role"] != Role.USER.value:
            turns.insert(0, {"role": Role.USER.value, "content": "(no user message)"})

        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "max_tokens": max_tokens or self._default_max_tokens,
            "temperature": temperature,
            "messages": turns,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}/messages"

        client = self._client or httpx.AsyncClient(timeout=self._timeout)
        try:
            response = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("provider did not respond in time", detail=url) from exc
        except httpx.HTTPError as exc:
            raise LLMError("provider request failed", detail=str(exc)) from exc
        finally:
            if self._client is None:
                await client.aclose()

        if response.status_code == 429:
            raise LLMRateLimitError("provider rate limited the request", detail=url)
        if response.status_code >= 400:
            raise LLMError(
                "provider returned an error",
                detail=f"HTTP {response.status_code} from {url}",
            )

        try:
            body = response.json()
            blocks = body.get("content") or []
            text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
            usage = body.get("usage") or {}
        except (AttributeError, TypeError, ValueError) as exc:
            raise LLMError("provider returned an unexpected payload", detail=str(exc)) from exc

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        return LLMResponse(
            content=text,
            model=body.get("model", model or self.default_model),
            usage=TokenUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            finish_reason=body.get("stop_reason"),
        )


# --------------------------------------------------------------------------- #
# Deterministic fake
# --------------------------------------------------------------------------- #


class FakeLLMProvider(LLMProvider):
    """Deterministic provider for tests and offline graph runs.

    Performs no network I/O. Responses are drawn from a queue, or produced by
    ``responder``, or returned as ``default``. Every call is recorded so tests
    can assert on what the model was asked.
    """

    name = "fake"

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        *,
        default: str = "ok",
        responder: Callable[[Sequence[Message]], str] | None = None,
        model: str = "fake-model",
    ) -> None:
        super().__init__(default_model=model)
        self._queue: deque[str] = deque(responses or ())
        self._default = default
        self._responder = responder
        self._calls: list[tuple[Message, ...]] = []

    @property
    def calls(self) -> tuple[tuple[Message, ...], ...]:
        """Every conversation this provider has been asked about."""
        return tuple(self._calls)

    @property
    def call_count(self) -> int:
        """How many completions have been requested."""
        return len(self._calls)

    @property
    def pending(self) -> int:
        """How many scripted responses remain unconsumed."""
        return len(self._queue)

    @property
    def responder(self) -> Callable[[Sequence[Message]], str] | None:
        """The callable that produces responses once the queue is empty.

        Exposed as a property so a caller can swap the behaviour of a provider
        that is already wired into a compiled graph. Replacing the provider
        object would not work: the graph holds a reference to this one.
        """
        return self._responder

    @responder.setter
    def responder(self, value: Callable[[Sequence[Message]], str] | None) -> None:
        """Replace the fallback responder."""
        self._responder = value

    def queue_response(self, content: str) -> None:
        """Append a scripted response."""
        self._queue.append(content)

    def last_prompt(self) -> str:
        """Return the concatenated content of the most recent call."""
        if not self._calls:
            return ""
        return "\n".join(m.content for m in self._calls[-1])

    async def ainvoke(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Return the next scripted response without touching the network."""
        self._calls.append(tuple(messages))

        if self._queue:
            content = self._queue.popleft()
        elif self._responder is not None:
            content = self._responder(messages)
        else:
            content = self._default

        # Deterministic token approximation so budget logic is testable offline.
        prompt_chars = sum(len(m.content) for m in messages)
        prompt_tokens = prompt_chars // 4
        completion_tokens = len(content) // 4

        return LLMResponse(
            content=content,
            model=model or self.default_model,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            finish_reason="stop",
        )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Construct the provider named by ``settings.llm_provider``.

    Args:
        settings: Validated application settings.

    Returns:
        A configured provider.

    Raises:
        ConfigurationError: If a required credential or base URL is missing.
    """
    provider = settings.llm_provider
    model = settings.llm_model

    if provider == "local":
        if not settings.llm_base_url:
            raise ConfigurationError("LLM_BASE_URL is required when LLM_PROVIDER=local")
        return OpenAICompatibleProvider(
            api_key=settings.llm_api_key.get_secret_value()
            if settings.llm_api_key
            else "not-required",
            model=model,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
        )

    if settings.llm_api_key is None:
        raise ConfigurationError(f"LLM_API_KEY is required when LLM_PROVIDER={provider}")

    secret = settings.llm_api_key.get_secret_value()

    if provider == "openai":
        return OpenAICompatibleProvider(
            api_key=secret,
            model=model,
            base_url=settings.llm_base_url or "https://api.openai.com/v1",
            timeout=settings.llm_timeout_seconds,
        )

    return AnthropicProvider(
        api_key=secret,
        model=model,
        timeout=settings.llm_timeout_seconds,
    )
