"""Typed application configuration.

Every setting is declared once, validated at startup, and read through
:func:`get_settings`. Application code never touches ``os.environ`` directly, so
there is a single place to audit what the process consumes and a single place
where defaults and bounds are enforced.

Validation is deliberately strict. A misconfigured production deployment fails
at startup rather than at the first request.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.exceptions import ConfigurationError

__all__ = ["AppEnv", "LogLevel", "Settings", "get_settings", "reset_settings_cache"]

AppEnv = Literal["development", "staging", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LLMProviderName = Literal["openai", "anthropic", "local"]
EmbeddingProviderName = Literal["local", "openai"]


class Settings(BaseSettings):
    """Validated application settings, loaded from the environment and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application ------------------------------------------------------- #
    app_name: str = "langgraph-multi-agent"
    app_env: AppEnv = "development"
    debug: bool = False
    log_level: LogLevel = "INFO"

    # --- LLM --------------------------------------------------------------- #
    llm_provider: LLMProviderName = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    llm_max_retries: int = Field(default=2, ge=0, le=10)

    # --- Embeddings -------------------------------------------------------- #
    embedding_provider: EmbeddingProviderName = "local"
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: SecretStr | None = None
    embedding_dimensions: int = Field(default=256, gt=0, le=8192)

    # --- PostgreSQL -------------------------------------------------------- #
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/langgraph"
    database_pool_size: int = Field(default=10, ge=1, le=100)
    database_max_overflow: int = Field(default=5, ge=0, le=100)
    #: Log every statement. Off outside debugging: a logged statement can carry a
    #: prompt or a credential value.
    database_echo: bool = False

    # --- Database tool ----------------------------------------------------- #
    #: Where the ``database`` tool points. Unset by default, and the tool is then
    #: not registered at all rather than aimed at the application's own tables.
    #: Point this at a read-only replica or a reporting database.
    database_tool_url: str | None = None
    # Read-only by default. Writes and destructive statements each need an
    # explicit opt-in, and destructive ones also need human approval.
    database_allow_writes: bool = False
    database_allow_destructive: bool = False
    database_statement_timeout_ms: int = Field(default=5000, ge=100, le=60_000)
    database_max_rows: int = Field(default=500, ge=1, le=10_000)

    # --- Memory ------------------------------------------------------------ #
    #: Whether runs read from and write to the memory tiers. On by default: the
    #: point of a memory manager is that it is used, and it degrades to a no-op
    #: rather than failing when its store is unavailable.
    memory_enabled: bool = True
    #: Minimum estimated importance for a *durable* memory to be written at all.
    #: Volatile tiers are never thresholded: a short-term note that is not worth
    #: keeping loses nothing by not being kept, whereas a long-term store that
    #: accepts everything becomes unusable.
    memory_importance_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    #: How many candidate rows a retrieval may consider before ranking. Exact
    #: scoring in Python is correct but linear, so the read is bounded explicitly.
    memory_scan_limit: int = Field(default=500, ge=1, le=10_000)
    #: Memories below this blended score are not injected into a prompt. Ranking
    #: a whole store and returning the best of a bad set would feed the model
    #: irrelevant text and invite it to use it.
    memory_min_score: float = Field(default=0.15, ge=0.0, le=1.0)
    #: Volatile memories older than this are structurally unsound to keep.
    memory_volatile_ttl_seconds: int = Field(default=86_400, ge=60)

    # --- Checkpointing ----------------------------------------------------- #
    # "postgres" is durable: a run survives a restart and can be resumed by
    # another worker. "memory" is volatile and exists for tests. The default is
    # durable, because a silent downgrade to a volatile backend would remove the
    # resume guarantee without anyone noticing.
    checkpoint_backend: Literal["postgres", "memory"] = "postgres"
    checkpoint_pool_size: int = Field(default=10, ge=1, le=100)
    checkpoint_open_timeout_seconds: float = Field(default=10.0, gt=0, le=120)

    # --- Redis ------------------------------------------------------------- #
    redis_url: str = "redis://localhost:6379/0"
    #: Namespace prefix for every cache key, so one Redis instance can host
    #: several environments without them reading each other's keys.
    redis_key_prefix: str = "langgraph"
    cache_default_ttl_seconds: int = Field(default=300, ge=1)

    # --- Execution limits -------------------------------------------------- #
    # Every loop in the system is bounded by one of these.
    max_agent_iterations: int = Field(default=10, ge=1, le=100)
    max_tool_calls: int = Field(default=25, ge=1, le=500)
    max_execution_time: int = Field(default=300, ge=1, le=3600)
    max_parallel_tasks: int = Field(default=4, ge=1, le=32)
    max_retries: int = Field(default=3, ge=0, le=10)
    max_token_budget: int = Field(default=100_000, ge=1)

    # --- Security ---------------------------------------------------------- #
    auth_enabled: bool = True
    jwt_secret: SecretStr | None = None
    jwt_algorithm: str = "HS256"
    jwt_audience: str | None = None
    jwt_issuer: str | None = None
    #: Development convenience only: trust the caller's ``X-User-Id`` header as
    #: identity when authentication is disabled. Refused when ``APP_ENV`` is
    #: ``production``, because it lets any caller claim any identity.
    trust_identity_header: bool = True
    rate_limit_requests: int = Field(default=60, ge=1)
    rate_limit_window_seconds: int = Field(default=60, ge=1)
    #: Fail closed when the rate limiter cannot reach its store. Off by default,
    #: since a cache outage taking the whole API down is usually the worse
    #: outcome — but a security-sensitive deployment should turn it on.
    rate_limit_fail_closed: bool = False

    # --- API documentation ------------------------------------------------ #
    # ``None`` means "decide from the environment": served outside production,
    # withheld inside it. The OpenAPI schema is an inventory of every route,
    # every field, and every error shape; that is a convenience while building
    # against the API and a map of the attack surface once it is deployed.
    # Setting this explicitly wins either way, so a deployment that deliberately
    # fronts its docs with an authenticating gateway can still expose them.
    api_docs_enabled: bool | None = None

    # --- Operator console -------------------------------------------------- #
    # Off by default, and deliberately so. The page it serves renders task text
    # from the database and carries approve, reject, and cancel buttons: anyone
    # who can reach it can drive the system. "Off unless asked for" is the only
    # defensible default for a control surface, so local development opts in and
    # a deployment has to make the same decision knowingly.
    #
    # Enabling it does not widen the API. The page authenticates exactly like any
    # other client, so with ``auth_enabled`` set it is inert without a token.
    dashboard_enabled: bool = False

    # --- Filesystem tool --------------------------------------------------- #
    # Comma-separated in the environment; exposed as resolved paths.
    filesystem_allowed_roots: str = "./workspace"
    filesystem_max_file_bytes: int = Field(default=1_048_576, ge=1)

    # --- Code execution ---------------------------------------------------- #
    # Off by default. Arbitrary execution is only safe behind real isolation.
    python_execution_enabled: bool = False

    # --- GitHub tool ------------------------------------------------------- #
    github_token: SecretStr | None = None
    github_tool_allow_writes: bool = False
    github_api_url: str = "https://api.github.com"

    # --- Web search tool --------------------------------------------------- #
    # Provider-agnostic: point SEARCH_API_URL at any endpoint that accepts
    # {"query": ..., "count": ...} and returns {"results": [...]}.
    search_api_url: str | None = None
    search_api_key: SecretStr | None = None
    search_max_results: int = Field(default=5, ge=1, le=25)
    search_timeout_seconds: float = Field(default=15.0, gt=0, le=120)

    # --- Network egress ---------------------------------------------------- #
    # Outbound tools refuse to reach private, loopback, and link-local
    # addresses unless this is explicitly enabled.
    allow_private_network_egress: bool = False

    # --- Observability ----------------------------------------------------- #
    otel_exporter_otlp_endpoint: str | None = None
    otel_service_name: str = "langgraph-multi-agent"
    #: Serve ``GET /metrics`` in the Prometheus text format. On by default, and
    #: deliberately not authenticated: the endpoint exposes operation names,
    #: statuses, and timings, never a prompt, a task id, or a credential. If it
    #: ever would, the fix is to stop recording that label, not to hide the page.
    metrics_enabled: bool = True
    #: How many completed spans the tracer keeps for introspection.
    trace_history_size: int = Field(default=200, ge=0, le=10_000)

    # ------------------------------------------------------------------ #
    # Derived values
    # ------------------------------------------------------------------ #

    @property
    def is_production(self) -> bool:
        """Return whether the application is running in production."""
        return self.app_env == "production"

    @property
    def serve_api_docs(self) -> bool:
        """Return whether ``/docs`` and ``/openapi.json`` are served.

        Returns:
            The explicit setting when one was given, otherwise ``True`` outside
            production and ``False`` inside it.
        """
        if self.api_docs_enabled is not None:
            return self.api_docs_enabled
        return self.app_env != "production"

    @property
    def allowed_roots(self) -> tuple[Path, ...]:
        """Resolved filesystem roots the filesystem tool may access."""
        roots = [part.strip() for part in self.filesystem_allowed_roots.split(",") if part.strip()]
        return tuple(Path(root).expanduser().resolve() for root in roots)

    def secret_values(self) -> tuple[str, ...]:
        """Return every configured secret, for redaction in logs and output."""
        candidates = (
            self.llm_api_key,
            self.embedding_api_key,
            self.jwt_secret,
            self.github_token,
            self.search_api_key,
        )
        return tuple(s.get_secret_value() for s in candidates if s is not None)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    @field_validator(
        "llm_api_key",
        "embedding_api_key",
        "jwt_secret",
        "github_token",
        "search_api_key",
        mode="after",
    )
    @classmethod
    def _blank_secret_means_absent(cls, value: SecretStr | None) -> SecretStr | None:
        """Treat an empty or whitespace-only secret as not set.

        ``LLM_API_KEY=`` in a ``.env`` file is the commonest way to configure
        nothing, and pydantic reads it as the empty string rather than as absent.
        That difference is not cosmetic: a provider built with an empty key sends
        ``Authorization: Bearer `` and fails at the transport layer with an error
        about an illegal header value, which reads like a bug in the client
        rather than a missing credential. Worse, the production validator's
        ``is None`` checks — including the one guarding ``JWT_SECRET`` — pass for
        an empty string, so a deployment could start believing it had a signing
        key.

        Surrounding whitespace is stripped as well: a key pasted with a trailing
        newline would produce a header the server rejects for reasons nobody
        enjoys diagnosing.

        Args:
            value: The parsed secret, if any.

        Returns:
            The stripped secret, or ``None`` when there was nothing in it.
        """
        if value is None:
            return None
        text = value.get_secret_value().strip()
        return SecretStr(text) if text else None

    @field_validator(
        "llm_base_url",
        "database_tool_url",
        "search_api_url",
        "otel_exporter_otlp_endpoint",
        mode="after",
    )
    @classmethod
    def _blank_url_means_unset(cls, value: str | None) -> str | None:
        """Treat an empty or whitespace-only URL as not set.

        The same failure as a blank secret, one layer on: ``DATABASE_TOOL_URL=``
        would register the database tool against an empty target, and the tool's
        own "is a target configured?" check is an ``is not None`` test.

        Args:
            value: The parsed URL, if any.

        Returns:
            The stripped URL, or ``None`` when there was nothing in it.
        """
        if value is None:
            return None
        text = value.strip()
        return text or None

    @model_validator(mode="after")
    def _enforce_production_safety(self) -> Settings:
        """Refuse configurations that are unsafe to run in production."""
        if not self.is_production:
            return self

        if self.debug:
            raise ConfigurationError("DEBUG must be false when APP_ENV=production")

        if not self.auth_enabled:
            raise ConfigurationError("AUTH_ENABLED must be true when APP_ENV=production")

        if self.jwt_secret is None:
            raise ConfigurationError("JWT_SECRET is required when APP_ENV=production")

        if self.trust_identity_header:
            # The header is self-asserted. Trusting it in production would let any
            # caller act as any user, which defeats every ownership check built
            # on top of it.
            raise ConfigurationError("TRUST_IDENTITY_HEADER must be false when APP_ENV=production")

        if self.python_execution_enabled and not self.execution_sandbox_available:
            raise ConfigurationError(
                "PYTHON_EXECUTION_ENABLED requires a sandbox, and none is configured"
            )

        return self

    @property
    def execution_sandbox_available(self) -> bool:
        """Return whether an isolation boundary for arbitrary code exists.

        Always false today. Arbitrary execution is only safe behind a real
        boundary — a container, a jailed worker with its own kernel namespace —
        and this project has not been given one. The property exists so the
        answer is a single explicit fact rather than a scattered assumption, and
        so enabling execution without a sandbox fails loudly instead of
        pretending.
        """
        return False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so the environment is read once. Call :func:`reset_settings_cache` in
    tests that need to change the environment.
    """
    return Settings()


def reset_settings_cache() -> None:
    """Clear the cached settings instance. Intended for tests."""
    get_settings.cache_clear()
