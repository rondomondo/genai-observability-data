from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    keys: list[str] | None = Field(
        default=None,
        description="Vector set keys to search. Defaults to all (cache, logs, traces).",
    )
    top_k: int = Field(default=5, ge=1, le=50)
    filter_expr: str | None = Field(
        default=None,
        description='Redis filter expression e.g. \'.service == "payments"\'',
    )
    include_vectors: bool = False


class SearchHit(BaseModel):
    element_id: str
    vset_key: str
    score: float
    attrs: dict[str, Any]
    vector: list[float] | None = None


class SearchResponse(BaseModel):
    query: str
    keys_searched: list[str]
    total_hits: int
    embed_ms: int
    results: list[SearchHit]
