"""Embedder sidecar — thin FastAPI wrapper around a local SentenceTransformer model.

Endpoints:
    POST /embed   -- embed a batch of texts
    GET  /health  -- liveness + model info
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.embedder import LocalEmbedder
from app.models import EmbedRequest, EmbedResponse

log = logging.getLogger(__name__)

_embedder: LocalEmbedder | None = None
_executor: ThreadPoolExecutor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _embedder, _executor
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    log.info(
        "config embed_model=%s embed_dim=%d host=%s port=%d log_level=%s",
        settings.embed_model,
        settings.embed_dim,
        settings.host,
        settings.port,
        settings.log_level,
    )
    # Load model in a thread so the event loop is not blocked during startup
    _executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="embed")
    loop = asyncio.get_running_loop()
    _embedder = await loop.run_in_executor(_executor, lambda: LocalEmbedder(settings))
    log.info("embedder_ready model=%s dim=%d", _embedder.model_name, _embedder.dim)
    yield
    if _executor:
        _executor.shutdown(wait=True)


app = FastAPI(title="embedder-sidecar", lifespan=lifespan)


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest) -> EmbedResponse:
    """Embed a batch of texts.

    Args:
        request: texts and optional is_query flag.

    Returns:
        Vectors, model name, and embedding dimension.

    Raises:
        HTTPException: 503 if the model is not yet ready.
    """
    if _embedder is None or _executor is None:
        raise HTTPException(status_code=503, detail="model not ready")
    loop = asyncio.get_running_loop()
    vectors = await loop.run_in_executor(
        _executor,
        lambda: _embedder.embed(request.texts, request.is_query),  # type: ignore[union-attr]
    )
    return EmbedResponse(vectors=vectors, model=_embedder.model_name, dim=_embedder.dim)


@app.get("/health")
async def health() -> JSONResponse:
    if _embedder is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return JSONResponse(content={"status": "ok", "model": _embedder.model_name, "dim": _embedder.dim})
