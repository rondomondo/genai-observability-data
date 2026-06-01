from fastapi import APIRouter, Depends, Request

from app.dependencies import get_embedder
from app.models.embed import EmbedBatchRequest, EmbedBatchResponse, EmbedRequest, EmbedResponse
from app.rate_limits import limit_for
from app.services.embedder import Embedder
from app.services.vectorstore import VectorStore

router = APIRouter(prefix="/embed", tags=["embed"])


@router.post("", response_model=EmbedResponse, summary="Embed a single text")
@limit_for("rate_limit_embed")
async def embed_single(
    request: Request,
    body: EmbedRequest,
    embedder: Embedder = Depends(get_embedder),
) -> EmbedResponse:
    vector, embed_ms = await embedder.embed(body.text, body.dim)
    return EmbedResponse(
        text=body.text,
        dim=len(vector),
        norm=round(VectorStore.norm(vector), 6),
        embed_ms=embed_ms,
        vector=vector if False else None,  # not exposed by default
    )


@router.post("/batch", response_model=EmbedBatchResponse, summary="Embed multiple texts")
@limit_for("rate_limit_embed")
async def embed_batch(
    request: Request,
    body: EmbedBatchRequest,
    embedder: Embedder = Depends(get_embedder),
) -> EmbedBatchResponse:
    vectors, total_ms = await embedder.embed_batch(body.texts, body.dim)
    results = [
        EmbedResponse(
            text=text,
            dim=len(vec),
            norm=round(VectorStore.norm(vec), 6),
            embed_ms=0,  # individual timing not tracked in batch
            vector=vec if body.include_vectors else None,
        )
        for text, vec in zip(body.texts, vectors)
    ]
    return EmbedBatchResponse(count=len(results), embed_ms=total_ms, results=results)
