from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
import os
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_SENSITIVE_FIELDS = {"api_key", "gemini_api_key", "google_access_token"}
_MASK = "***"
_PROFILES_ACTIVE = Path(__file__).parent.parent.parent / "profiles" / ".active"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    # Auth
    api_key: str = Field(default="abcdef")

    # Google Cloud auth (google-cloud backends)
    google_access_token: str = Field(default=os.environ.get('GOOGLE_ACCESS_TOKEN',""))
    google_cloud_project: str = Field(default="")
    google_cloud_location: str = Field(default="")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379")
    redis_keyspace_db: int = Field(default=0)
    redis_keyspace_events: str = Field(default="Ex")

    # Embedding
    gemini_api_key: str = Field(default="")
    embed_backend: Literal["google-cloud", "gemini", "sentence-transformers", "sidecar", "mock"] = Field(default="mock")
    embed_model: str = Field(default="gemini-embedding-001")
    embed_sidecar_url: str = Field(default="http://localhost:8001")
    embed_dim: int = Field(default=256, ge=64, le=3072)
    embed_workers: int = Field(default=4, ge=1, le=16)
    embed_concurrency: int = Field(default=2, ge=1, le=16)
    embed_batch_size: int = Field(default=32, ge=1, le=128)
    embed_rate_limit_rpm: int = Field(default=90, ge=1, le=10000)

    # Vector set keys
    vset_cache: str = Field(default="rag:cache")
    vset_logs: str = Field(default="logs:embeddings")
    vset_traces: str = Field(default="traces:embeddings")

    # Semantic cache
    cache_hit_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    cache_ttl_seconds: int = Field(default=3600, ge=60)

    # Summariser
    summariser_backend: Literal["google-cloud", "anthropic", "gemini", "mock" ] = Field(default="mock")
    summariser_model: str = Field(default="claude-haiku-4-5-20251001")
    summariser_mock: bool = Field(default=False)

    @model_validator(mode="after")
    def _parse_combined_model_strings(self) -> "Settings":
        """Split backend:model combined env var format into separate fields.

        Allows .env to use e.g. EMBED_MODEL=google-cloud:gemini-embedding-001 and
        SUMMARISER_MODEL=google-cloud:gemini-2.5-flash-lite, matching the GOD_AGENT_MODEL pattern.
        """
        if ":" in self.embed_model:
            backend, _, model = self.embed_model.partition(":")
            self.embed_backend = backend  # type: ignore[assignment]
            self.embed_model = model
        if ":" in self.summariser_model:
            backend, _, model = self.summariser_model.partition(":")
            self.summariser_backend = backend  # type: ignore[assignment]
            self.summariser_model = model
        return self
    summariser_concurrency: int = Field(default=2, ge=1, le=16)
    summariser_llm_threshold_duration_ms: int = Field(default=5000)
    summariser_rate_limit_rpm: int = Field(default=50, ge=1, le=10000)

    # OTEL
    otel_exporter_otlp_endpoint: str = Field(default="http://localhost:4317")
    otel_service_name: str = Field(default="vecstore-api")
    otel_enabled: bool = Field(default=True)

    # Logging
    log_level: str = Field(default="DEBUG")
    log_format: Literal["json", "console", "logfmt"] = Field(default="console")

    # Async batch jobs
    job_ttl_seconds: int = Field(default=300)

    # Ingress rate limits (slowapi / limits syntax, e.g. "60/minute", "10/second").
    # Applied per remote IP. Set to "" to disable a specific endpoint's limit.
    rate_limit_ingest: str = Field(default="120/minute")
    rate_limit_embed: str = Field(default="60/minute")
    rate_limit_search: str = Field(default="60/minute")
    rate_limit_cache: str = Field(default="60/minute")
    rate_limit_summarise: str = Field(default="30/minute")

    def as_loggable(self) -> dict[str, Any]:
        """Return all settings with sensitive fields masked for safe console/log output.

        Also includes the active profile name from profiles/.active if present.
        """
        result: dict[str, Any] = {}
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            if field_name in _SENSITIVE_FIELDS:
                raw = str(value)
                result[field_name] = _MASK if not raw else f"{raw[:4]}{_MASK}" if len(raw) > 4 else _MASK
            else:
                result[field_name] = value
        try:
            result["active_profile"] = _PROFILES_ACTIVE.read_text(encoding="utf-8").strip()
        except OSError:
            result["active_profile"] = None
        return result


#@lru_cache
def get_settings() -> Settings:
    return Settings()
