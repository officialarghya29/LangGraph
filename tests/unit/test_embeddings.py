"""Tests for the embedding provider abstraction."""

from __future__ import annotations

import json
import math

import httpx
import pytest
from app.core.config import Settings
from app.core.exceptions import ConfigurationError, EmbeddingError
from app.services.embeddings import (
    LocalHashingEmbeddingProvider,
    OpenAIEmbeddingProvider,
    build_embedding_provider,
)


def cosine(left: list[float], right: list[float]) -> float:
    """Return the cosine similarity of two vectors."""
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


# --------------------------------------------------------------------------- #
# Local provider
# --------------------------------------------------------------------------- #


async def test_local_provider_returns_the_requested_width() -> None:
    provider = LocalHashingEmbeddingProvider(dimensions=64)

    assert len(await provider.embed("hello world")) == 64


async def test_local_provider_is_deterministic_across_instances() -> None:
    """Results must not depend on the process hash seed."""
    first = await LocalHashingEmbeddingProvider(dimensions=32).embed("stable text")
    second = await LocalHashingEmbeddingProvider(dimensions=32).embed("stable text")

    assert first == second


async def test_local_provider_produces_unit_vectors() -> None:
    vector = await LocalHashingEmbeddingProvider(dimensions=128).embed("some words here")

    assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0)


async def test_local_provider_distinguishes_different_text() -> None:
    provider = LocalHashingEmbeddingProvider(dimensions=256)

    assert await provider.embed("apples") != await provider.embed("oranges")


async def test_local_provider_is_case_insensitive() -> None:
    provider = LocalHashingEmbeddingProvider(dimensions=128)

    assert await provider.embed("Hello World") == await provider.embed("hello world")


async def test_local_provider_scores_overlap_higher_than_unrelated_text() -> None:
    """Lexical overlap is what this provider measures, and it should detect it."""
    provider = LocalHashingEmbeddingProvider(dimensions=512)

    reference = await provider.embed("the database connection pool is exhausted")
    related = await provider.embed("database connection pool exhausted")
    unrelated = await provider.embed("a recipe for sourdough bread")

    assert cosine(reference, related) > cosine(reference, unrelated)


async def test_local_provider_handles_empty_text() -> None:
    vector = await LocalHashingEmbeddingProvider(dimensions=16).embed("")

    assert vector == [0.0] * 16


async def test_local_provider_handles_punctuation_only_text() -> None:
    vector = await LocalHashingEmbeddingProvider(dimensions=16).embed("!!! ... ???")

    assert vector == [0.0] * 16


async def test_local_provider_batches() -> None:
    provider = LocalHashingEmbeddingProvider(dimensions=32)

    vectors = await provider.embed_batch(["one", "two", "three"])

    assert len(vectors) == 3
    assert all(len(vector) == 32 for vector in vectors)


async def test_local_provider_batch_matches_single_embedding() -> None:
    provider = LocalHashingEmbeddingProvider(dimensions=64)

    batched = await provider.embed_batch(["alpha", "beta"])
    singles = [await provider.embed("alpha"), await provider.embed("beta")]

    assert batched == singles


def test_provider_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        LocalHashingEmbeddingProvider(dimensions=0)


# --------------------------------------------------------------------------- #
# OpenAI provider
# --------------------------------------------------------------------------- #


def embedding_body(vectors: list[list[float]], *, shuffle: bool = False) -> dict[str, object]:
    data = [{"index": i, "embedding": v} for i, v in enumerate(vectors)]
    if shuffle:
        data.reverse()
    return {"data": data}


async def test_openai_provider_parses_embeddings() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=embedding_body([[0.1, 0.2], [0.3, 0.4]]), request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIEmbeddingProvider(api_key="k", model="m", dimensions=2, client=client)

    vectors = await provider.embed_batch(["a", "b"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    await client.aclose()


async def test_openai_provider_orders_by_index() -> None:
    """Providers may return batch results out of order."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=embedding_body([[1.0, 1.1], [2.0, 2.1]], shuffle=True), request=request
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIEmbeddingProvider(api_key="k", model="m", dimensions=2, client=client)

    vectors = await provider.embed_batch(["a", "b"])

    assert vectors == [[1.0, 1.1], [2.0, 2.1]]
    await client.aclose()


async def test_openai_provider_validates_the_width() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=embedding_body([[0.1, 0.2, 0.3]]), request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIEmbeddingProvider(api_key="k", model="m", dimensions=2, client=client)

    with pytest.raises(EmbeddingError, match="dimensions"):
        await provider.embed("a")

    await client.aclose()


async def test_openai_provider_reports_http_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIEmbeddingProvider(api_key="k", model="m", dimensions=2, client=client)

    with pytest.raises(EmbeddingError):
        await provider.embed("a")

    await client.aclose()


async def test_openai_provider_maps_timeouts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIEmbeddingProvider(api_key="k", model="m", dimensions=2, client=client)

    with pytest.raises(EmbeddingError, match="timed out"):
        await provider.embed("a")

    await client.aclose()


async def test_openai_provider_sends_model_and_input() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=embedding_body([[0.5, 0.5]]), request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIEmbeddingProvider(api_key="k", model="embed-test", dimensions=2, client=client)

    await provider.embed_batch(["hello"])

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "embed-test"
    assert body["input"] == ["hello"]
    await client.aclose()


async def test_openai_provider_skips_the_request_for_an_empty_batch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request should be made for an empty batch")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIEmbeddingProvider(api_key="k", model="m", dimensions=2, client=client)

    assert await provider.embed_batch([]) == []
    await client.aclose()


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def build_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_factory_defaults_to_local() -> None:
    settings = build_settings()

    assert isinstance(build_embedding_provider(settings), LocalHashingEmbeddingProvider)


def test_factory_honours_the_configured_dimensions() -> None:
    settings = build_settings(embedding_dimensions=512)

    assert build_embedding_provider(settings).dimensions == 512


def test_factory_builds_the_openai_provider() -> None:
    settings = build_settings(embedding_provider="openai", embedding_api_key="key")

    assert isinstance(build_embedding_provider(settings), OpenAIEmbeddingProvider)


def test_factory_falls_back_to_the_llm_key() -> None:
    settings = build_settings(embedding_provider="openai", llm_api_key="shared-key")

    assert isinstance(build_embedding_provider(settings), OpenAIEmbeddingProvider)


def test_factory_requires_a_key_for_openai_embeddings() -> None:
    settings = build_settings(embedding_provider="openai")

    with pytest.raises(ConfigurationError, match="EMBEDDING_API_KEY"):
        build_embedding_provider(settings)
