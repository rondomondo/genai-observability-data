"""Runtime configuration for the god package.

Loaded from environment variables or a .env file.
All fields have sensible defaults so the package works without any API keys
when mock_mode=True.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class HttpClientSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    http_connect_timeout: float = Field(default=5.0, alias="HTTP_CONNECT_TIMEOUT")
    http_read_timeout: float = Field(default=30.0, alias="HTTP_READ_TIMEOUT")
    http_write_timeout: float = Field(default=10.0, alias="HTTP_WRITE_TIMEOUT")
    http_pool_timeout: float = Field(default=5.0, alias="HTTP_POOL_TIMEOUT")
    http_max_connections: int = Field(default=20, alias="HTTP_MAX_CONNECTIONS")
    http_max_keepalive: int = Field(default=10, alias="HTTP_MAX_KEEPALIVE")
    http_user_agent: str = Field(default="god/0.1", alias="HTTP_USER_AGENT")
    http_retry_attempts: int = Field(default=3, alias="HTTP_RETRY_ATTEMPTS")
    http_retry_backoff_min: float = Field(default=0.5, alias="HTTP_RETRY_BACKOFF_MIN")
    http_retry_backoff_max: float = Field(default=8.0, alias="HTTP_RETRY_BACKOFF_MAX")


_http_settings: HttpClientSettings | None = None


def get_http_settings() -> HttpClientSettings:
    """Return the singleton HttpClientSettings instance."""
    global _http_settings
    if _http_settings is None:
        _http_settings = HttpClientSettings()
    return _http_settings


def reset_http_settings() -> None:
    """Reset cached HTTP settings (for tests)."""
    global _http_settings
    _http_settings = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Model used by the scenario-analysis agent
    agent_model: str = Field(default="gemini-2.5-flash", alias="GOD_AGENT_MODEL")
    # Fallback model tried when the primary returns a 503/overload error
    agent_fallback_model: str | None = Field(default="gemini-2.0-flash", alias="GOD_AGENT_FALLBACK_MODEL")

    # When True the pydantic-ai agent is replaced with a FunctionModel (no API key needed)
    mock_mode: bool = Field(default=True, alias="GOD_MOCK_MODE")

    # Optional Gemini key -- used when the active model is a Google/Gemini model
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")

    # Generation defaults (all overridable via CLI args)
    default_traces: int = Field(default=100, alias="GOD_DEFAULT_TRACES")
    default_rate: float = Field(default=10.0, alias="GOD_DEFAULT_RATE")
    default_batch_size: int = Field(default=50, alias="GOD_DEFAULT_BATCH_SIZE")
    default_seed: int | None = Field(default=None, alias="GOD_DEFAULT_SEED")
    default_scenario: str | None = Field(default=None, alias="GOD_DEFAULT_SCENARIO")

    # Output defaults
    default_format: list[str] = Field(default=["otlp_traces", "otlp_logs"], alias="GOD_DEFAULT_FORMAT")
    default_otlp: str | None = Field(default=None, alias="GOD_DEFAULT_OTLP")

    # vecstore-api sink
    vecstore_url: str | None = Field(default=None, alias="GOD_VECSTORE_URL")
    vecstore_api_key: str = Field(default="abcdef", alias="GOD_VECSTORE_API_KEY")
    vecstore_summarise_traces: bool = Field(default=True, alias="GOD_VECSTORE_SUMMARISE_TRACES")
    vecstore_summarise_logs: bool = Field(default=False, alias="GOD_VECSTORE_SUMMARISE_LOGS")
    vecstore_async_mode: bool = Field(default=True, alias="GOD_VECSTORE_ASYNC_MODE")


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset cached settings (for tests)."""
    global _settings
    _settings = None
