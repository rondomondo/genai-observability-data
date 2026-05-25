"""Runtime configuration for the god package.

Loaded from environment variables or a .env file.
All fields have sensible defaults so the package works without any API keys
when mock_mode=True.
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Model used by the scenario-analysis agent
    agent_model: str = Field(default="claude-haiku-4-5-20251001", alias="GOD_AGENT_MODEL")

    # When True the pydantic-ai agent is replaced with a FunctionModel (no API key needed)
    mock_mode: bool = Field(default=False, alias="GOD_MOCK_MODE")

    # Optional Anthropic key — only needed when mock_mode=False
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    # Generation defaults (all overridable via CLI args)
    default_traces: int = Field(default=100, alias="GOD_DEFAULT_TRACES")
    default_rate: float = Field(default=10.0, alias="GOD_DEFAULT_RATE")
    default_batch_size: int = Field(default=50, alias="GOD_DEFAULT_BATCH_SIZE")


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
