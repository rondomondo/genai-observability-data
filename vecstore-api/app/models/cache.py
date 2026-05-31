from typing import Any

from pydantic import BaseModel, Field


class CacheStoreRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    sources: list[str] = Field(default_factory=list)
    model: str | None = None
    ttl_seconds: int | None = Field(default=None, ge=60)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CacheStoreResponse(BaseModel):
    element_id: str
    key: str
    cardinality: int
    ttl_seconds: int
    embed_ms: int


class CacheLookupRequest(BaseModel):
    question: str = Field(..., min_length=1)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int = Field(default=1, ge=1, le=10)


class CacheHit(BaseModel):
    element_id: str
    score: float
    question: str
    answer: str
    sources: list[str]
    model: str | None
    cached_at: int
    metadata: dict[str, Any]


class CacheLookupResponse(BaseModel):
    hit: bool
    score: float | None
    threshold: float
    result: CacheHit | None
    embed_ms: int


class CacheStatusResponse(BaseModel):
    key: str
    cardinality: int
    ttl_watcher_running: bool
    pending_expirations: int
