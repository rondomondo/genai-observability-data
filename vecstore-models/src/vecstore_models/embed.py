from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=32_000)
    dim: int | None = Field(default=None, ge=64, le=3072, description="Override default embedding dim")


class EmbedResponse(BaseModel):
    text: str
    dim: int
    norm: float
    embed_ms: int
    vector: list[float] | None = Field(default=None, description="Only returned when include_vector=true")


class EmbedBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=128)
    dim: int | None = None
    include_vectors: bool = False


class EmbedBatchResponse(BaseModel):
    count: int
    embed_ms: int
    results: list[EmbedResponse]
