from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    is_query: bool = Field(default=False)


class EmbedResponse(BaseModel):
    vectors: list[list[float]]
    model: str
    dim: int
