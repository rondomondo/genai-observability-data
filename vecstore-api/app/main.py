"""vecstore-api — FastAPI application entry point.

Lifespan manages startup/shutdown of all services:
  - Redis connection
  - Embedder (worker pool)
  - VectorStore
  - CacheService
  - Summariser
  - TTLWatcher (background thread + asyncio bridge)
  - JobStore

OTEL instrumentation is configured before the app is created so
all FastAPI routes and outbound HTTP calls are traced automatically.
"""

from contextlib import asynccontextmanager

import redis as redis_lib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.rate_limits import limiter
from app.logging_config import configure_logging, get_logger
from app.middleware.auth import APIKeyMiddleware
from app.middleware.correlation import CorrelationMiddleware
from app.routers import cache, embed, ingest, search, status, summarise
from app.services.cache import CacheService
from app.services.embedder import Embedder
from app.services.jobs import JobStore
from app.services.summariser import Summariser
from app.services.ttl_watcher import TTLWatcher
from app.services.vectorstore import VectorStore

settings = get_settings()

configure_logging(level=settings.log_level, fmt=settings.log_format)
log = get_logger(__name__)


# ── OTEL setup ────────────────────────────────────────────────────────────────


def _setup_otel(app: FastAPI) -> None:
    if not settings.otel_enabled:
        log.info("otel_disabled")
        return
    try:
        import logging

        from opentelemetry import trace
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk._logs import LoggingHandler

        resource = Resource.create({"service.name": settings.otel_service_name})

        # Traces
        trace_provider = TracerProvider(resource=resource)
        span_exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint, insecure=True)
        trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(trace_provider)

        # Logs — bridge stdlib logging -> OTLP -> OTel Collector -> Loki
        log_provider = LoggerProvider(resource=resource)
        log_exporter = OTLPLogExporter(endpoint=settings.otel_exporter_otlp_endpoint, insecure=True)
        log_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
        set_logger_provider(log_provider)

        otel_handler = LoggingHandler(level=logging.DEBUG, logger_provider=log_provider)
        logging.getLogger().addHandler(otel_handler)

        FastAPIInstrumentor.instrument_app(app)
        log.info("otel_configured", endpoint=settings.otel_exporter_otlp_endpoint)
    except Exception as exc:
        log.warning("otel_setup_failed", error=str(exc))


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup_begin", service="vecstore-api")

    # Redis sync client (used by VectorStore, CacheService)
    r = redis_lib.Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        r.ping()
        log.info("redis_connected", url=settings.redis_url)
    except redis_lib.RedisError as exc:
        log.error("redis_connection_failed", error=str(exc))
        raise

    vs = VectorStore(r)
    embedder = Embedder(settings)
    await embedder.start()

    cache_svc = CacheService(settings, embedder, vs, r)
    summariser = Summariser(settings)
    ttl_watcher = TTLWatcher(settings.redis_url, vs, db=settings.redis_keyspace_db)
    await ttl_watcher.start()

    job_store = JobStore(ttl_seconds=settings.job_ttl_seconds)

    # Attach to app state for dependency injection
    app.state.redis = r
    app.state.vectorstore = vs
    app.state.embedder = embedder
    app.state.cache_service = cache_svc
    app.state.summariser = summariser
    app.state.ttl_watcher = ttl_watcher
    app.state.job_store = job_store

    log.info("startup_complete")
    yield

    # Shutdown
    log.info("shutdown_begin")
    await embedder.stop()
    await ttl_watcher.stop()
    r.close()
    log.info("shutdown_complete")


# ── App factory ───────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(
        title="vecstore-api",
        description="Semantic vector store — embeddings, RAG cache, log/trace ingestion",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Middleware (order matters — outermost first)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten for non-local
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(APIKeyMiddleware, api_key=settings.api_key)
    app.add_middleware(CorrelationMiddleware)

    # Routers
    app.include_router(embed.router)
    app.include_router(cache.router)
    app.include_router(ingest.router)
    app.include_router(search.router)
    app.include_router(summarise.router)
    app.include_router(status.router)

    # Health (no auth)
    @app.get("/health", tags=["health"], include_in_schema=False)
    async def health() -> JSONResponse:
        return JSONResponse({"ok": True})

    _setup_otel(app)
    return app


app = create_app()
