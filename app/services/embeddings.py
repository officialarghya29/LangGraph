"""Embedding provider abstraction.

Kept separate from the LLM abstraction because the two are frequently backed by
different vendors, and because embeddings need to be deterministic in tests
without any network access.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

import httpx

from app.core.config import Settings
from app.core.exceptions import ConfigurationError, EmbeddingError

__all__ = [
    "EmbeddingProvider",
    "LocalHashingEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "build_embedding_provider",
]

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _l2_normalise(vector: list[float]) -> list[float]:
    """Scale a vector to unit length. A zero vector is returned unchanged."""
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return vector
    return [component / norm for component in vector]


class EmbeddingProvider(ABC):
    """Abstract text embedding provider."""

    #: Stable identifier used in logs and metadata.
    name: str = "embeddings"

    def __init__(self, *, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single string."""

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed several strings.

        The default implementation embeds sequentially. Providers with a batch
        endpoint should override this.
        """
        return [await self.embed(text) for text in texts]

    def _validate(self, vector: list[float]) -> list[float]:
        """Check a provider's output is the expected width and is finite."""
        if len(vector) != self.dimensions:
            raise EmbeddingError(
                "provider returned the wrong number of dimensions",
                detail=f"expected {self.dimensions}, got {len(vector)}",
            )
        if not all(math.isfinite(component) for component in vector):
            raise EmbeddingError("provider returned non-finite values")
        return vector


class LocalHashingEmbeddingProvider(EmbeddingProvider):
    """Deterministic, dependency-free embeddings computed locally.

    This is the hashing trick: tokens are hashed into a fixed number of buckets
    and weighted by term frequency. It captures lexical overlap, which is enough
    for exact and near-exact retrieval, and it is fully deterministic.

    It is explicitly *not* a semantic embedding. Text that means the same thing
    in different words will not score as similar. Use a real embedding model when
    semantic recall matters.

    Hashing uses BLAKE2b rather than the builtin :func:`hash`, whose string
    hashing is salted per process and would produce embeddings that differ
    between runs.
    """

    name = "local-hashing"

    async def embed(self, text: str) -> list[float]:
        """Embed text as a normalised term-frequency hash vector."""
        vector = [0.0] * self.dimensions

        for token in _TOKEN_PATTERN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            bucket = value % self.dimensions
            # Use one bit to pick a sign, which reduces collision bias.
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[bucket] += sign

        return self._validate(_l2_normalise(vector))


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embeddings from an OpenAI-compatible ``/embeddings`` endpoint."""

    name = "openai-embeddings"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimensions: int,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(dimensions=dimensions)
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = client

    async def embed(self, text: str) -> list[float]:
        """Embed one string."""
        vectors = await self.embed_batch([text])
        return vectors[0]

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch in a single request."""
        if not texts:
            return []

        payload = {"model": self._model, "input": list(texts)}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}/embeddings"

        client = self._client or httpx.AsyncClient(timeout=self._timeout)
        try:
            response = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise EmbeddingError("embedding provider timed out", detail=url) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingError("embedding request failed", detail=str(exc)) from exc
        finally:
            if self._client is None:
                await client.aclose()

        if response.status_code >= 400:
            raise EmbeddingError(
                "embedding provider returned an error",
                detail=f"HTTP {response.status_code} from {url}",
            )

        try:
            body = response.json()
            ordered = sorted(body["data"], key=lambda item: item["index"])
            vectors = [[float(value) for value in item["embedding"]] for item in ordered]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError("embedding provider returned an unexpected payload") from exc

        return [self._validate(vector) for vector in vectors]


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Construct the embedding provider named by ``settings.embedding_provider``.

    Args:
        settings: Validated application settings.

    Returns:
        A configured embedding provider.

    Raises:
        ConfigurationError: If a required credential is missing.
    """
    if settings.embedding_provider == "local":
        return LocalHashingEmbeddingProvider(dimensions=settings.embedding_dimensions)

    key = settings.embedding_api_key or settings.llm_api_key
    if key is None:
        raise ConfigurationError(
            "EMBEDDING_API_KEY or LLM_API_KEY is required when EMBEDDING_PROVIDER=openai"
        )

    return OpenAIEmbeddingProvider(
        api_key=key.get_secret_value(),
        # Resolved, not read raw: an unset EMBEDDING_MODEL must become the model
        # this provider actually serves rather than an empty name.
        model=settings.embedding_model_name,
        dimensions=settings.embedding_dimensions,
    )
