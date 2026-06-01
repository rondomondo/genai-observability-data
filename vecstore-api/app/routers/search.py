import asyncio

from fastapi import APIRouter, Depends, Request

from app.config import get_settings
from app.dependencies import get_embedder, get_vectorstore
from app.models.search import SearchHit, SearchRequest, SearchResponse
from app.rate_limits import limit_for
from app.services.embedder import Embedder
from app.services.vectorstore import VectorStore

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse, summary="Semantic search across vector sets")
@limit_for("rate_limit_search")
async def search(
    request: Request,
    body: SearchRequest,
    embedder: Embedder = Depends(get_embedder),
    vs: VectorStore = Depends(get_vectorstore),
) -> SearchResponse:
    settings = get_settings()

    default_keys = [settings.vset_cache, settings.vset_logs, settings.vset_traces]
    keys_to_search = body.keys if body.keys else default_keys

    vector, embed_ms = await embedder.embed(body.query)

    loop = asyncio.get_running_loop()
    all_hits: list[SearchHit] = []

    async def _search_key(key: str) -> list[SearchHit]:
        results = await loop.run_in_executor(
            None,
            lambda: vs.search(key, vector, top_k=body.top_k, filter_expr=body.filter_expr),
        )
        return [
            SearchHit(
                element_id=r["element"],
                vset_key=key,
                score=r["score"],
                attrs=r["attrs"],
                vector=None,  # vectors not returned by vsim
            )
            for r in results
        ]

    results_per_key = await asyncio.gather(*[_search_key(k) for k in keys_to_search])

    for hits in results_per_key:
        all_hits.extend(hits)

    # Re-rank across keys by score descending
    all_hits.sort(key=lambda h: h.score, reverse=True)
    all_hits = all_hits[: body.top_k]

    return SearchResponse(
        query=body.query,
        keys_searched=keys_to_search,
        total_hits=len(all_hits),
        embed_ms=embed_ms,
        results=all_hits,
    )
