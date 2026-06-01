# vecstore-api

Semantic vector store API — embeddings, RAG cache, log/trace ingestion, async batch ops.

Built on FastAPI + Redis Vector Sets (HNSW) + Google GenAI embeddings + pydantic-ai.

---

## Quick start

```bash
cp .env.example .env
# edit .env — set GEMINI_API_KEY and API_KEY at minimum

# Add to your existing compose stack
docker compose up -d vecstore-api

# Or run locally
uv run uvicorn app.main:app --reload
```

Docs at http://localhost:8000/docs

---

## Authentication

All endpoints (except `/health`, `/docs`) require:
```
X-API-Key: <your API_KEY>
```

---

## Endpoints

### Embed
| Method | Path | Description |
|--------|------|-------------|
| POST | `/embed` | Embed a single text |
| POST | `/embed/batch` | Embed multiple texts concurrently |

### Cache (RAG semantic cache)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/cache/query` | Lookup — returns hit/miss + cached answer |
| POST | `/cache` | Store a Q&A pair with TTL |
| DELETE | `/cache/{element_id}` | Remove a cache entry |
| GET | `/cache/status` | Cardinality, TTL watcher state |

### Ingest
| Method | Path | Description |
|--------|------|-------------|
| POST | `/ingest/log` | Ingest a single log entry |
| POST | `/ingest/log/batch` | Batch log ingest (sync or async with `async_mode=true`) |
| POST | `/ingest/trace` | Ingest a single trace |
| POST | `/ingest/trace/batch` | Batch trace ingest (sync or async) |

### Search
| Method | Path | Description |
|--------|------|-------------|
| POST | `/search` | Semantic search across cache + logs + traces |

### Summarise
| Method | Path | Description |
|--------|------|-------------|
| POST | `/summarise/trace` | Summarise a trace (template or LLM, returns structured TraceSummary) |
| POST | `/summarise/log` | Summarise a log entry |

### Status
| Method | Path | Description |
|--------|------|-------------|
| GET | `/status` | Service health, queue depth, vset cardinalities |
| GET | `/status/job/{job_id}` | Async batch job status |

---

## Examples

Set these in your shell before running the examples below:

```bash
BASE=http://localhost:8000
KEY=dev-secret-change-me
```

### Health check (no auth)

```bash
curl -s $BASE/status/health | jq .
# {"status": "ok"}
```

### Service status

```bash
curl -s $BASE/status -H "X-API-Key: $KEY" | jq .
# {
#   "service": "vecstore-api",
#   "redis_connected": true,
#   "embed_queue_depth": 0,
#   "ttl_watcher_running": true,
#   "vsets": [{"key": "rag:cache", "cardinality": 3}, ...],
#   "active_jobs": 0
# }
```

### RAG cache round-trip

```bash
# Store a Q&A pair
curl -s -X POST $BASE/cache \
  -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is HNSW?",
    "answer": "Hierarchical Navigable Small World -- a graph-based ANN algorithm.",
    "sources": ["https://en.wikipedia.org/wiki/HNSW"],
    "model": "gemini-2.0-flash"
  }' | jq .

# Lookup by semantic similarity (different phrasing, same meaning)
curl -s -X POST $BASE/cache/query \
  -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "explain the HNSW algorithm", "threshold": 0.85}' | jq .
# hit: true if the stored question is close enough

# Cache status
curl -s $BASE/cache/status -H "X-API-Key: $KEY" | jq .
```

### Embed text

```bash
curl -s -X POST $BASE/embed \
  -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "connection pool exhausted in payment service"}' | jq '{dim: (.vector | length), embed_ms}'
```

### Ingest a log entry

```bash
curl -s -X POST $BASE/ingest/log \
  -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "entry": {
      "message": "Redis connection pool exhausted after 30s",
      "service": "checkout-svc",
      "severity": "ERROR",
      "trace_id": "abc123"
    },
    "summarise": false
  }' | jq .
```

### Ingest a trace

```bash
curl -s -X POST $BASE/ingest/trace \
  -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "entry": {
      "trace_id": "abc123",
      "root_operation": "POST /checkout",
      "root_service": "checkout-svc",
      "duration_ms": 4200,
      "status": "error",
      "spans": [
        {
          "name": "db.query",
          "span_id": "s1",
          "service": "postgres",
          "start_time_ms": 0,
          "duration_ms": 3800,
          "status": "error"
        }
      ]
    },
    "summarise": true
  }' | jq .
```

### Semantic search

```bash
curl -s -X POST $BASE/search \
  -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "database connection errors in checkout",
    "top_k": 5
  }' | jq '.results[] | {element_id, score, type: .attrs.type}'
```

### Async batch ingest

```bash
# Submit batch (returns immediately with a job_id)
JOB=$(curl -s -X POST $BASE/ingest/log/batch \
  -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "entries": [
      {"message": "payment timeout after 5s", "service": "payments", "severity": "ERROR"},
      {"message": "checkout completed in 120ms", "service": "checkout", "severity": "INFO"},
      {"message": "auth token refreshed", "service": "auth", "severity": "DEBUG"}
    ],
    "summarise": false,
    "async_mode": true
  }' | jq -r .job_id)

echo "Job: $JOB"

# Poll until done
curl -s $BASE/status/job/$JOB -H "X-API-Key: $KEY" | jq '{status, processed, total}'
```

### Summarise a trace directly

```bash
curl -s -X POST $BASE/summarise/trace \
  -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "trace_id": "abc123",
    "root_operation": "POST /checkout",
    "root_service": "checkout-svc",
    "duration_ms": 4200,
    "status": "error",
    "force_llm": false
  }' | jq .summary

---

## Rate limiting

All write and search endpoints are rate-limited per client IP using
[slowapi](https://github.com/laurentS/slowapi). Limits are configurable via environment
variables without code changes. The defaults are permissive for local development; tighten
them before exposing the API to untrusted networks.

| Endpoint group | Variable | Default |
|----------------|----------|---------|
| `/ingest/*` | `RATE_LIMIT_INGEST` | `120/minute` |
| `/embed`, `/embed/batch` | `RATE_LIMIT_EMBED` | `60/minute` |
| `/search` | `RATE_LIMIT_SEARCH` | `60/minute` |
| `/cache/query`, `/cache` | `RATE_LIMIT_CACHE` | `60/minute` |
| `/summarise/*` | `RATE_LIMIT_SUMMARISE` | `30/minute` |
| `/health`, `/status/*` | (unlimited) | -- |

Clients that exceed their limit receive `HTTP 429` with a `Retry-After` header.

See [../RATE_LIMITING.md](../RATE_LIMITING.md) for full details including proxy/multi-replica
caveats and how to switch to a Redis-backed distributed store.

---

## Architecture notes

- **Embedder**: asyncio queue + thread pool executor — concurrent embed requests without unlimited thread spawning
- **TTL watcher**: daemon thread subscribes to Redis keyspace expiry events, bridges to asyncio queue for cleanup — vector set element removed when shadow TTL key expires
- **Summariser**: pydantic-ai agent for error/anomaly traces (structured `TraceSummary` output), fast template for normal traces and logs
- **Async jobs**: in-memory job store, fine for single-process local dev

---

## Environment variables

See `.env.example` for the full list. Key ones:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Required |
| `REDIS_URL` | `redis://redis:6379` | Redis connection |
| `API_KEY` | `dev-secret-change-me` | Auth header value |
| `EMBED_DIM` | `768` | Embedding dimensionality |
| `CACHE_HIT_THRESHOLD` | `0.92` | Cosine similarity threshold for cache hit |
| `CACHE_TTL_SECONDS` | `3600` | Default cache entry TTL |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://otel-collector:4317` | OTLP endpoint |
