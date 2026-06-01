"""Shared Pydantic v2 models for vecstore-api request/response contracts.

Both vecstore-api (server) and god (client) depend on this package so that
model changes only need to happen in one place.
"""

from vecstore_models.cache import (
    CacheHit,
    CacheLookupRequest,
    CacheLookupResponse,
    CacheStoreRequest,
    CacheStoreResponse,
    CacheStatusResponse,
)
from vecstore_models.common import ErrorResponse, JobReference, OKResponse
from vecstore_models.embed import (
    EmbedBatchRequest,
    EmbedBatchResponse,
    EmbedRequest,
    EmbedResponse,
)
from vecstore_models.ingest import (
    BatchIngestResponse,
    IngestResponse,
    IngestResult,
    LogBatchIngestRequest,
    LogEntry,
    LogIngestRequest,
    Span,
    TraceBatchIngestRequest,
    TraceEntry,
    TraceIngestRequest,
)
from vecstore_models.search import SearchHit, SearchRequest, SearchResponse
from vecstore_models.status import JobStatus, StatusResponse, VSetInfo
from vecstore_models.summarise import (
    SummariseLogRequest,
    SummariseResponse,
    SummariseTraceRequest,
    TraceSummary,
)

__all__ = [
    # cache
    "CacheHit",
    "CacheLookupRequest",
    "CacheLookupResponse",
    "CacheStoreRequest",
    "CacheStoreResponse",
    "CacheStatusResponse",
    # common
    "ErrorResponse",
    "JobReference",
    "OKResponse",
    # embed
    "EmbedBatchRequest",
    "EmbedBatchResponse",
    "EmbedRequest",
    "EmbedResponse",
    # ingest
    "BatchIngestResponse",
    "IngestResponse",
    "IngestResult",
    "LogBatchIngestRequest",
    "LogEntry",
    "LogIngestRequest",
    "Span",
    "TraceBatchIngestRequest",
    "TraceEntry",
    "TraceIngestRequest",
    # search
    "SearchHit",
    "SearchRequest",
    "SearchResponse",
    # status
    "JobStatus",
    "StatusResponse",
    "VSetInfo",
    # summarise
    "SummariseLogRequest",
    "SummariseResponse",
    "SummariseTraceRequest",
    "TraceSummary",
]
