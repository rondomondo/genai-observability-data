from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Auth
    api_key: str = Field(default="abcdef")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379")
    redis_keyspace_db: int = Field(default=0)
    redis_keyspace_events: str = Field(default="Ex")

    # Embedding
    gemini_api_key: str = Field(default="")
    embed_backend: Literal["gemini", "sentence-transformers", "sidecar"] = Field(default="gemini")
    embed_model: str = Field(default="gemini-embedding-001")
    embed_sidecar_url: str = Field(default="http://embedder-sidecar:8001")
    embed_dim: int = Field(default=256, ge=64, le=3072)
    embed_workers: int = Field(default=4, ge=1, le=16)
    embed_concurrency: int = Field(default=2, ge=1, le=16)
    embed_batch_size: int = Field(default=32, ge=1, le=128)

    # Vector set keys
    vset_cache: str = Field(default="rag:cache")
    vset_logs: str = Field(default="logs:embeddings")
    vset_traces: str = Field(default="traces:embeddings")

    # Semantic cache
    cache_hit_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    cache_ttl_seconds: int = Field(default=3600, ge=60)

    # Summariser
    summariser_model: str = Field(default="claude-sonnet-4-6")
    summariser_concurrency: int = Field(default=2, ge=1, le=16)
    summariser_llm_threshold_duration_ms: int = Field(default=5000)

    # OTEL
    otel_exporter_otlp_endpoint: str = Field(default="http://otel-collector:4317")
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


#@lru_cache
def get_settings() -> Settings:
    return Settings()
