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

from pydantic import Field, SecretStr, model_validator
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

    # --- Database tool ----------------------------------------------------- #
    # Read-only by default. Writes and destructive statements each need an
    # explicit opt-in, and destructive ones also need human approval.
    database_allow_writes: bool = False
    database_allow_destructive: bool = False
    database_statement_timeout_ms: int = Field(default=5000, ge=100, le=60_000)
    database_max_rows: int = Field(default=500, ge=1, le=10_000)

    # --- Redis ------------------------------------------------------------- #
    redis_url: str = "redis://localhost:6379/0"

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
    rate_limit_requests: int = Field(default=60, ge=1)
    rate_limit_window_seconds: int = Field(default=60, ge=1)

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

    # ------------------------------------------------------------------ #
    # Derived values
    # ------------------------------------------------------------------ #

    @property
    def is_production(self) -> bool:
        """Return whether the application is running in production."""
        return self.app_env == "production"

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

        return self


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
