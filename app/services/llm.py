"""Provider-independent access to language models.

Agents and graph nodes depend on :class:`LLMProvider`, never on a vendor SDK.
Adding a vendor means adding one class here; no agent has to change.

Structured output is implemented once, on the base class, by instructing the
model to emit JSON and validating the result against a Pydantic schema. That
keeps structured generation behaving identically across providers instead of
relying on vendor-specific tool-calling formats.

Sampling parameters are a property of the *model*, not of this client, and the
current generation of reasoning models rejects them outright — see
:data:`_REASONING_PREFIXES`. The payload builders therefore decide per model which
fields are legal rather than sending a fixed shape and hoping.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Coroutine, Mapping, Sequence
from datetime import UTC
from email.utils import parsedate_to_datetime
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
    "is_reasoning_model",
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


#: Statuses where the provider is telling the client to come back later, and where
#: a ``Retry-After`` header is meaningful.
#:
#: This mirrors the retry set the official OpenAI client ships with. Three of the
#: members are worth naming: 429 and 503 are explicit back-pressure; 500 and 502
#: and 504 are the gateway and origin failures that are transient far more often
#: than they are permanent; and 501 is deliberately absent, because "not
#: implemented" is a property of the request and will be just as unimplemented on
#: the next attempt. Treating a 5xx as permanent would fail a run that a second
#: attempt would have completed.
_THROTTLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})

#: Model families that do their reasoning internally and reject the sampling
#: knobs as a consequence.
#:
#: OpenAI's reasoning models do not ignore ``temperature``; they fail the whole
#: request with HTTP 400, "Unsupported value: 'temperature' does not support 0.0
#: with this model". They also reject ``max_tokens`` — "Unsupported parameter:
#: 'max_tokens' is not supported with this model. Use 'max_completion_tokens'
#: instead" — and take the ceiling as ``max_completion_tokens``. Sending either
#: field unconditionally, as this module did, means every call to such a model
#: fails, and it fails in a way that reads like a request bug rather than a
#: configuration one. Match is by family prefix because these names carry a
#: version and often a tier suffix (``gpt-5.6-terra``, ``o3-mini``).
_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5", "gpt-6")

#: Prefixes excluded from the match above despite looking like it.
#:
#: ``gpt-oss`` is OpenAI's open-weights family and it is served by the runtimes
#: ``LLM_PROVIDER=local`` targets. It accepts the ordinary sampling parameters, so
#: treating it as a reasoning model would silently drop the temperature an operator
#: configured.
_NOT_REASONING_PREFIXES = ("gpt-oss",)


def is_reasoning_model(name: str) -> bool:
    """Return whether ``name`` is a model that rejects sampling parameters.

    Args:
        name: A model identifier, as passed by the operator or a call site.

    Returns:
        True when the model takes ``max_completion_tokens`` and must not be sent
        ``temperature``. Unknown names return False, because the alternative is to
        guess wrong for a self-hosted model this module has never heard of and
        strip a parameter its server would have accepted.
    """
    lowered = name.strip().lower()
    if lowered.startswith(_NOT_REASONING_PREFIXES):
        return False
    return lowered.startswith(_REASONING_PREFIXES)


def _retry_after(headers: Mapping[str, str]) -> float | None:
    """Return the delay the provider asked for, in seconds.

    Both permitted forms are understood. A number of seconds is taken as given.
    An HTTP date is interpreted against the response's *own* ``Date`` header
    rather than this machine's clock: the header expresses a relative wait, and
    subtracting two different clocks reintroduces exactly the skew the value is
    meant to remove. Without a ``Date`` header there is nothing to measure a date
    against, so it is ignored and the backoff curve applies — which at least
    fails in a known direction.

    Args:
        headers: The response headers.

    Returns:
        A positive number of seconds, or ``None`` when the header is absent,
        malformed, already in the past, or in a form this does not interpret.
    """
    raw = headers.get("retry-after")
    if raw is None:
        return None

    text = raw.strip()
    try:
        seconds = float(text)
    except (TypeError, ValueError):
        return _retry_after_date(text, headers.get("date"))
    return seconds if seconds > 0 else None


def _retry_after_date(value: str, served_at: str | None) -> float | None:
    """Return a wait expressed as an HTTP date, measured from its ``Date`` header.

    Args:
        value: The ``Retry-After`` value, in HTTP-date form.
        served_at: The response's own ``Date`` header, or ``None``.

    Returns:
        Whole seconds until that instant, or ``None`` if either timestamp is
        unusable or the instant has already passed.
    """
    if not served_at:
        return None
    try:
        due = parsedate_to_datetime(value)
        now = parsedate_to_datetime(served_at)
    except (TypeError, ValueError):
        return None

    # HTTP dates always carry a zone, but a server that omits one should not
    # raise on subtraction between an aware and a naive datetime.
    if due.tzinfo is None:
        due = due.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    seconds = (due - now).total_seconds()
    return seconds if seconds > 0 else None


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

    #: Sampling defaults, restated as class attributes so that a provider which
    #: sets only ``default_model`` — every test double, and the decorators reading
    #: these off an inner provider — still has a well-defined value rather than an
    #: ``AttributeError`` at construction time.
    default_temperature: float = 0.0
    default_max_tokens: int | None = None

    def __init__(
        self,
        *,
        default_model: str,
        default_temperature: float = 0.0,
        default_max_tokens: int | None = None,
    ) -> None:
        """Configure the provider.

        Args:
            default_model: Model used when a call does not name one.
            default_temperature: Sampling temperature used when a call does not
                state one. It lives here rather than as a literal default on
                :meth:`ainvoke` so that ``LLM_TEMPERATURE`` reaches every call site
                without each of them having to read the settings.
            default_max_tokens: Output ceiling used when a call does not state one.
                ``None`` means "whatever the vendor defaults to".
        """
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    @abstractmethod
    async def ainvoke(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send messages to the model and return its completion.

        Args:
            messages: The conversation to send.
            model: Optional model override.
            temperature: Optional sampling temperature. ``None`` uses the
                provider's configured default, which is what lets a call site that
                has no opinion inherit one instead of restating ``0.0``.
            max_tokens: Optional ceiling on the completion.
        """

    def _sampling(
        self, temperature: float | None, max_tokens: int | None
    ) -> tuple[float, int | None]:
        """Resolve per-call sampling parameters against the provider defaults.

        Resolution happens here, at the innermost provider, and not in the
        decorators: a decorator that resolved ``None`` into a number would decide
        on the caller's behalf and the setting would be read twice with the outer
        layer winning.

        Args:
            temperature: The call's temperature, or ``None``.
            max_tokens: The call's ceiling, or ``None``.

        Returns:
            The temperature and ceiling to send.
        """
        resolved = self.default_temperature if temperature is None else temperature
        return resolved, (self.default_max_tokens if max_tokens is None else max_tokens)

    def invoke(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
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
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_attempts: int = 3,
    ) -> T:
        """Request output conforming to ``schema``.

        Invalid responses are fed back to the model with the validation error so
        it can correct itself. Exhausting ``max_attempts`` raises
        :class:`StructuredOutputError` rather than returning partial data.

        Sampling parameters are accepted here as well as on :meth:`ainvoke`,
        because a structured call is still a model call. They used to be dropped
        on this path — the correction loop is where a caller most wants to pin the
        temperature, and it had no way to.

        Args:
            messages: The conversation to send.
            schema: Pydantic model the response must validate against.
            model: Optional model override.
            temperature: Optional sampling temperature.
            max_tokens: Optional ceiling on the completion.
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
            response = await self.ainvoke(
                conversation, model=model, temperature=temperature, max_tokens=max_tokens
            )
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
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_attempts: int = 3,
    ) -> T:
        """Synchronous convenience wrapper around :meth:`astructured_output`."""
        return _run_sync(
            self.astructured_output(
                messages,
                schema,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_attempts=max_attempts,
            )
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
        default_temperature: float = 0.0,
        default_max_tokens: int | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            default_model=model,
            default_temperature=default_temperature,
            default_max_tokens=default_max_tokens,
        )
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
        """Build a request body the target model will actually accept.

        Which sampling fields are legal is a property of the model, not of this
        endpoint, and the two families disagree: reasoning models reject
        ``temperature`` and ``max_tokens`` with a 400, everything else requires
        them under exactly those names. The field names are therefore chosen from
        the resolved model rather than fixed.

        Args:
            messages: The conversation to send.
            model: Model override, or ``None`` for the provider default.
            temperature: The already-resolved sampling temperature.
            max_tokens: The already-resolved output ceiling, if any.

        Returns:
            The JSON body to post.
        """
        name = model or self.default_model
        payload: dict[str, Any] = {
            "model": name,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
        }
        if is_reasoning_model(name):
            # No temperature at all. Reasoning models spend the ceiling on their
            # own reasoning before emitting a token of answer, so a small value
            # truncates the reply rather than shortening it, and the vendor's own
            # default is the only well-informed choice for a name this module
            # cannot price.
            if max_tokens is not None:
                payload["max_completion_tokens"] = max_tokens
        else:
            payload["temperature"] = temperature
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
        return payload

    async def ainvoke(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Call ``/chat/completions`` and normalise the response."""
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        temperature, max_tokens = self._sampling(temperature, max_tokens)
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

        if response.status_code in _THROTTLE_STATUSES:
            raise LLMRateLimitError(
                "provider rate limited the request",
                detail=f"HTTP {response.status_code} from {url}",
                retry_after=_retry_after(response.headers),
            )
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

    #: Ceiling used when neither the call nor ``LLM_MAX_OUTPUT_TOKENS`` states one.
    #: Unlike the OpenAI endpoint, the Messages API has no server-side default:
    #: ``max_tokens`` is a required field, so one of these three has to supply it.
    DEFAULT_MAX_TOKENS = 4096

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.anthropic.com/v1",
        timeout: float = 60.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        default_temperature: float = 0.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            default_model=model,
            default_temperature=default_temperature,
            default_max_tokens=max_tokens,
        )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = client

    async def ainvoke(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
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

        temperature, max_tokens = self._sampling(temperature, max_tokens)
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "max_tokens": max_tokens or self.DEFAULT_MAX_TOKENS,
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

        if response.status_code in _THROTTLE_STATUSES:
            raise LLMRateLimitError(
                "provider rate limited the request",
                detail=f"HTTP {response.status_code} from {url}",
                retry_after=_retry_after(response.headers),
            )
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
        default_temperature: float = 0.0,
        default_max_tokens: int | None = None,
    ) -> None:
        super().__init__(
            default_model=model,
            default_temperature=default_temperature,
            default_max_tokens=default_max_tokens,
        )
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
        temperature: float | None = None,
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
    # Resolved, not read raw: an unset LLM_MODEL must become the model the chosen
    # provider actually serves, rather than the empty name a previous version
    # passed on or another vendor's model name.
    model = settings.llm_model_name
    temperature = settings.llm_temperature
    max_tokens = settings.llm_max_output_tokens

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
            default_temperature=temperature,
            default_max_tokens=max_tokens,
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
            default_temperature=temperature,
            default_max_tokens=max_tokens,
        )

    return AnthropicProvider(
        api_key=secret,
        model=model,
        timeout=settings.llm_timeout_seconds,
        max_tokens=max_tokens or AnthropicProvider.DEFAULT_MAX_TOKENS,
        default_temperature=temperature,
    )
