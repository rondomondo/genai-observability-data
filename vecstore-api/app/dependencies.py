"""FastAPI dependencies.

All service singletons live on app.state and are accessed via these
dependency functions. This keeps routers decoupled from startup logic.
"""

from fastapi import Request

from app.services.cache import CacheService
from app.services.embedder import Embedder
from app.services.jobs import JobStore
from app.services.summariser import Summariser
from app.services.ttl_watcher import TTLWatcher
from app.services.vectorstore import VectorStore


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_vectorstore(request: Request) -> VectorStore:
    return request.app.state.vectorstore


def get_cache_service(request: Request) -> CacheService:
    return request.app.state.cache_service


def get_summariser(request: Request) -> Summariser:
    return request.app.state.summariser


def get_ttl_watcher(request: Request) -> TTLWatcher:
    return request.app.state.ttl_watcher


def get_job_store(request: Request) -> JobStore:
    return request.app.state.job_store
