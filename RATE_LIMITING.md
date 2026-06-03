# Rate Limiting

This document covers the full rate-limiting strategy across both the `god` data generator
(outbound LLM/embedding calls) and `vecstore-api` (inbound HTTP traffic). The two concerns
are symmetric -- one is an HTTP client talking to external quota-bounded APIs, the other is
an HTTP server protecting downstream services from runaway clients -- and the implementations
mirror each other intentionally.

---

## Contents

- [Concepts](#concepts)
- [Egress: outbound LLM calls (god package)](#egress-outbound-llm-calls-god-package)
  - [Two-layer architecture](#two-layer-architecture)
  - [Token-bucket configuration](#token-bucket-configuration)
  - [Concurrency configuration](#concurrency-configuration)
  - [429 feedback loop](#429-feedback-loop)
  - [Retry and backoff interaction](#retry-and-backoff-interaction)
  - [Egress configuration reference](#egress-configuration-reference)
  - [Egress: known limitations](#egress-known-limitations)
- [Ingress: vecstore-api HTTP rate limiting](#ingress-vecstore-api-http-rate-limiting)
  - [Endpoint limits](#endpoint-limits)
  - [Response on limit exceeded](#response-on-limit-exceeded)
  - [Ingress configuration reference](#ingress-configuration-reference)
  - [Ingress: known limitations](#ingress-known-limitations)
- [Choosing the right values](#choosing-the-right-values)
  - [Egress: matching upstream quotas](#egress-matching-upstream-quotas)
  - [Ingress: protecting vecstore-api](#ingress-protecting-vecstore-api)
- [Running multiple instances](#running-multiple-instances)
- [End-to-end flow](#end-to-end-flow)
- [Troubleshooting](#troubleshooting)

---

## Concepts

Two distinct rate-limiting concepts appear throughout this codebase. It helps to keep them
separate:

**Concurrency limit** -- how many requests are in flight simultaneously. Enforced by
`asyncio.Semaphore`. This caps resource usage (open connections, memory, thread slots) but
does not control *throughput* over time. A concurrency of 4 with 10ms requests allows 400
req/s; the same concurrency with 1s requests allows 4 req/s.

**Throughput limit (rate limit)** -- how many requests per unit of time, regardless of how
long each one takes. Enforced by a token bucket. This is the proactive control that prevents
upstream quota exhaustion before it happens.

Both controls are active simultaneously, per named operation. A request must pass the token
bucket first, then acquire the concurrency semaphore. Either can be the bottleneck depending
on load and latency.

---

## Egress: outbound LLM calls (god package)

Relevant files:
- [src/god/http_client.py](src/god/http_client.py) -- `HttpClient` class
- [src/god/config.py](src/god/config.py) -- `HttpClientSettings`
- [src/god/vecstore_client.py](src/god/vecstore_client.py) -- sets operation names

### Two-layer architecture

```
Request arrives at HttpClient.request(method, path, rate_name="ingest")
    |
    +-- [1] Token bucket (pyrate-limiter)
    |       Wait until a token is available for "ingest"
    |       Capacity: HTTP_RPS_INGEST tokens/second (default: unlimited)
    |       If 0 or unset: skip this layer entirely
    |
    +-- [2] Concurrency semaphore (asyncio.Semaphore)
    |       Wait until a slot is free for "ingest"
    |       Capacity: HTTP_RATE_INGEST concurrent requests (default: 8)
    |
    +-- [3] httpx outbound request
    |       Retried on 429/5xx via tenacity (exponential backoff)
    |
    +-- [4] On 429: drain one extra token from the bucket
                Shrinks burst headroom proportionally to upstream pressure
```

The two layers serve different failure modes:

| Layer | What it prevents | Failure mode it addresses |
|-------|-----------------|--------------------------|
| Token bucket | Too many requests per second | Gemini/Anthropic quota windows (RPM/RPD) |
| Semaphore | Too many simultaneous connections | Thread exhaustion, connection pool overflow |
| Retry + backoff | Transient upstream errors | Transient 5xx, network blips |
| 429 drain | Burst-then-thrash under sustained pressure | Quota exhaustion leading to cascading retries |

### Token-bucket configuration

The token bucket is disabled by default (all `HTTP_RPS_*` values default to `0`). This is
a safe default -- the existing semaphore-only behaviour is preserved until you explicitly
opt in. Set the RPS values to match your upstream provider quotas.

The bucket refills at a constant `rps` tokens/second. When a request arrives it attempts
to acquire one token immediately. If none are available it waits with exponential backoff
(starting at 10ms, capping at 1s) without blocking the event loop. This polling approach
means the wait is approximate -- actual throughput will be slightly below the configured
limit, not exactly at it. For quota compliance purposes this is safe: you will not exceed
the limit.

Named operation buckets are created lazily and are independent. Setting `HTTP_RPS_INGEST=8`
does not affect the search or cache buckets.

**Example: match Gemini's free tier (60 RPM = 1 RPS)**

```env
HTTP_RPS_INGEST=1.0
HTTP_RPS_SEARCH=1.0
HTTP_RPS_CACHE=1.0
HTTP_RPS_ADMIN=0.5
```

**Example: Gemini paid tier (1000 RPM ~= 16 RPS with headroom)**

```env
HTTP_RPS_INGEST=12.0
HTTP_RPS_SEARCH=8.0
HTTP_RPS_CACHE=8.0
HTTP_RPS_ADMIN=2.0
```

**Example: pass per-operation buckets programmatically**

```python
from god.http_client import HttpClient

client = HttpClient(
    base_url="https://generativelanguage.googleapis.com",
    rates={"embed": 4, "chat": 2},
    rps={"embed": 10.0, "chat": 3.0},
)
```

The `rps` dict overrides all settings-based values. Operations not listed in `rps` fall back
to `HTTP_RPS_DEFAULT` (default `0`, i.e. no limit). Pass `rps={}` to disable the bucket
entirely regardless of settings.

### Concurrency configuration

The semaphore is always active. It caps the number of requests that are physically in-flight
(holding a connection, waiting for a response) regardless of throughput. The settings-based
names map as follows:

| `rate_name` | Settings field | Default |
|-------------|---------------|---------|
| `ingest` | `HTTP_RATE_INGEST` | 8 |
| `search` | `HTTP_RATE_SEARCH` | 4 |
| `cache` | `HTTP_RATE_CACHE` | 4 |
| `admin` | `HTTP_RATE_ADMIN` | 2 |
| anything else | `HTTP_DEFAULT_CONCURRENCY` | 4 |

The `rate_name` is passed explicitly at the call site:

```python
resp = await client.request("POST", "/embed", rate_name="ingest", json=payload)
```

### 429 feedback loop

When an upstream API returns 429 (Too Many Requests), two things happen:

1. **Tenacity retries** with exponential backoff (reactive). The request will be re-attempted
   after `HTTP_RETRY_BACKOFF_MIN` to `HTTP_RETRY_BACKOFF_MAX` seconds.

2. **Bucket drain** (proactive). `_drain_token_on_429` calls `try_acquire` on the named
   bucket, consuming an extra token. Since the bucket refills at a fixed rate, draining an
   extra token means the next real request waits proportionally longer. Under sustained 429
   pressure this progressively slows the client down even before each individual retry fires.

The drain is best-effort -- if `try_acquire` raises internally (e.g. the bucket is already
empty) the exception is swallowed and the retry continues normally. The drain never blocks
and never prevents the retry from proceeding.

```
Time -->    [req 1] [req 2] [429] [drain] [backoff] [req 3 delayed by bucket]
                                       ^                   ^
                                  extra token consumed    next token arrives later
```

This mechanism is most effective when 429s are intermittent. If upstream consistently rate
limits you, lower the `HTTP_RPS_*` values -- the drain provides soft pressure, not hard
quota management.

### Retry and backoff interaction

The token bucket is acquired *inside* each retry attempt, not once per `request()` call.
This means each retry attempt is individually rate-limited:

```
HttpClient.request("POST", "/embed", rate_name="ingest")
  attempt 1:
    acquire_token("ingest")   <- wait for bucket
    acquire semaphore
    -> 503                    <- retryable, tenacity fires backoff
  attempt 2:
    acquire_token("ingest")   <- wait for bucket again
    acquire semaphore
    -> 200 OK
```

This is intentional: retries should not bypass the rate limit. Without this, a burst of
retries after a transient 5xx could itself cause a 429.

Tenacity's backoff applies *between* attempts. The token bucket applies *at the start of*
each attempt. Both add latency -- on 503s you will see `backoff_delay + bucket_wait` per
retry. Keep `HTTP_RETRY_BACKOFF_MAX` and `HTTP_RETRY_ATTEMPTS` proportionate to your upstream
quota window to avoid retrying into the same window that caused the original failure.

### Egress configuration reference

All variables are read from environment or `.env` (via `HttpClientSettings`).

**Throughput limits (new)**

| Variable | Default | Description |
|----------|---------|-------------|
| `HTTP_RPS_INGEST` | `0.0` | Max ingest requests/second (0 = unlimited) |
| `HTTP_RPS_SEARCH` | `0.0` | Max search requests/second |
| `HTTP_RPS_CACHE` | `0.0` | Max cache requests/second |
| `HTTP_RPS_ADMIN` | `0.0` | Max admin requests/second |
| `HTTP_RPS_DEFAULT` | `0.0` | Fallback for operations not named above |

**Concurrency limits**

| Variable | Default | Description |
|----------|---------|-------------|
| `HTTP_RATE_INGEST` | `8` | Max simultaneous ingest requests |
| `HTTP_RATE_SEARCH` | `4` | Max simultaneous search requests |
| `HTTP_RATE_CACHE` | `4` | Max simultaneous cache requests |
| `HTTP_RATE_ADMIN` | `2` | Max simultaneous admin requests |
| `HTTP_DEFAULT_CONCURRENCY` | `4` | Fallback for unnamed operations |

**Retry / backoff**

| Variable | Default | Description |
|----------|---------|-------------|
| `HTTP_RETRY_ATTEMPTS` | `3` | Total attempts per `request()` call (includes first) |
| `HTTP_RETRY_BACKOFF_MIN` | `0.5` | Minimum backoff between retries (seconds) |
| `HTTP_RETRY_BACKOFF_MAX` | `8.0` | Maximum backoff between retries (seconds) |

**Connection pool**

| Variable | Default | Description |
|----------|---------|-------------|
| `HTTP_MAX_CONNECTIONS` | `20` | httpx max open connections |
| `HTTP_MAX_KEEPALIVE` | `10` | httpx max keepalive connections |
| `HTTP_CONNECT_TIMEOUT` | `5.0` | Connection timeout (seconds) |
| `HTTP_READ_TIMEOUT` | `30.0` | Read timeout (seconds) |
| `HTTP_WRITE_TIMEOUT` | `10.0` | Write timeout (seconds) |
| `HTTP_POOL_TIMEOUT` | `5.0` | Connection pool acquire timeout (seconds) |

### Egress: known limitations

**Single-process only.** Buckets and semaphores are in-process. If you run two `god`
processes simultaneously against the same upstream API (e.g. `make gen-otlp` in two
terminals), each has its own bucket. The effective throughput hitting Gemini/Anthropic will
be the sum of both. Divide `HTTP_RPS_*` by the number of concurrent processes, or use a
shared Redis-backed limiter (the Redis instance in vecstore-api would work; this is not
currently wired up).

**Approximate throughput.** The bucket uses `try_acquire` with 10ms->1s exponential polling,
not a blocking `acquire`. This means actual throughput will be slightly below the configured
value, which is safe. It also means at very low RPS (< 0.5) the polling granularity becomes
noticeable -- a 0.1 RPS limit may behave as ~0.08 in practice.

**No dynamic adaptation.** The configured RPS is static. The 429 drain applies soft pressure
but does not permanently lower the rate. If you need to automatically respond to sustained
upstream throttling, consider implementing a simple token refill dampener: on each 429, halve
the refill rate for the affected operation until requests succeed again.

**Retry retryable codes only.** The following are retried: `429`, `500`, `502`, `503`, `504`,
`httpx.ConnectError`, `httpx.ReadTimeout`, `httpx.WriteTimeout`, `httpx.RemoteProtocolError`.
Non-retryable 4xx errors (`400`, `401`, `403`, `404`, `422`) are raised immediately.

---

## Ingress: vecstore-api HTTP rate limiting

Relevant files:
- [vecstore-api/app/rate_limits.py](vecstore-api/app/rate_limits.py) -- shared `Limiter` and `limit_for` helper
- [vecstore-api/app/main.py](vecstore-api/app/main.py) -- exception handler wired up
- [vecstore-api/app/config.py](vecstore-api/app/config.py) -- `Settings` fields
- Router files: `app/routers/ingest.py`, `search.py`, `embed.py`, `cache.py`, `summarise.py`

### Endpoint limits

Rate limits are applied per remote IP address. Each endpoint group has an independent limit.
The defaults are intentionally permissive; tighten them in `.env` before exposing to
untrusted networks.

| Endpoint group | Settings field | Default | Notes |
|----------------|---------------|---------|-------|
| `/ingest/*` | `RATE_LIMIT_INGEST` | `120/minute` | All four ingest endpoints share this limit |
| `/embed`, `/embed/batch` | `RATE_LIMIT_EMBED` | `60/minute` | Direct embed calls |
| `/search` | `RATE_LIMIT_SEARCH` | `60/minute` | Semantic search |
| `/cache/query`, `/cache` (store) | `RATE_LIMIT_CACHE` | `60/minute` | Cache reads and writes |
| `/summarise/trace`, `/summarise/log` | `RATE_LIMIT_SUMMARISE` | `30/minute` | LLM calls per request, hence lower |
| `/health`, `/status/*` | (unlimited) | -- | Intentionally not capped |

The limits use `limits`-library syntax: `"N/period"` where period is `second`, `minute`,
`hour`, or `day`. Examples: `"10/second"`, `"500/minute"`, `"1000/hour"`.

The limit strings are read from settings at request time (not at import time), so they can
be overridden per-test without restarting the application:

```python
# In a test, monkeypatch the settings before the request:
monkeypatch.setattr("app.config.get_settings", lambda: Settings(rate_limit_ingest="1000/minute"))
```

### Response on limit exceeded

When a client exceeds its limit, slowapi returns:

```
HTTP 429 Too Many Requests
Retry-After: <seconds until next token>
Content-Type: application/json

{"error": "Rate limit exceeded: 120 per 1 minute"}
```

The `Retry-After` header is always present and accurate. Well-behaved clients should honour
it rather than hammering with retries. The `god` package's `HttpClient` treats 429 as
retryable and will back off automatically (see [429 feedback loop](#429-feedback-loop)).

### Egress: vecstore-api to Gemini

vecstore-api's `Embedder` applies its own outbound token bucket when `EMBED_BACKEND=gemini`.
This is independent of the god client limits and prevents the API server itself from exhausting
the Gemini quota regardless of how many inbound clients are calling it.

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_RATE_LIMIT_RPM` | `90` | Max Gemini embed requests per minute (Gemini backend only) |

The bucket uses `pyrate-limiter` with the same polling approach as the god client. On a 429
from Gemini, an extra token is drained and tenacity retries with exponential backoff
(1s min, 60s max, 3 attempts). Set to 90 for the free tier (100 RPM limit) or raise it
proportionally for paid tiers.

### Ingress configuration reference

| Variable | Default | Description |
|----------|---------|-------------|
| `RATE_LIMIT_INGEST` | `120/minute` | All `/ingest/*` endpoints |
| `RATE_LIMIT_EMBED` | `60/minute` | `/embed` and `/embed/batch` |
| `RATE_LIMIT_SEARCH` | `60/minute` | `/search` |
| `RATE_LIMIT_CACHE` | `60/minute` | `/cache/query` and `/cache` (store) |
| `RATE_LIMIT_SUMMARISE` | `30/minute` | `/summarise/trace` and `/summarise/log` |

### Ingress: known limitations

**IP-based keying breaks behind a reverse proxy.** `slowapi` keys by `request.client.host`,
which is the immediate TCP peer. Behind nginx, traefik, or any other proxy, all requests
arrive from `127.0.0.1` and share a single bucket -- effectively disabling per-client
limiting. To fix this, replace `get_remote_address` with a custom key function that reads
`X-Forwarded-For` or `X-Real-IP`:

```python
# vecstore-api/app/rate_limits.py
from fastapi import Request

def _get_real_ip(request: Request) -> str:
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

limiter = Limiter(key_func=_get_real_ip)
```

Only enable this if you trust the proxy to set the header correctly. A client that can
reach the app directly could spoof `X-Forwarded-For` to bypass limits.

**In-memory buckets are not shared across workers.** The default uvicorn configuration
runs a single process. If you scale to multiple workers (`--workers 4`) or run behind
a multi-replica deployment, each worker has its own bucket and the effective limit per
client is `RATE_LIMIT_* * num_workers`. To enforce true per-client limits across replicas,
configure slowapi with a Redis backend:

```python
from limits.storage import RedisStorage
from slowapi import Limiter

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=settings.redis_url,   # reuse the existing Redis instance
)
```

The Redis instance that vecstore-api already connects to is suitable. This is not currently
wired up.

**`/cache/delete` and read-only endpoints are not rate-limited.** `DELETE /cache/{id}` and
`GET /cache/status` are not currently decorated. This is intentional for the local dev
context, but consider adding limits before exposing publicly. The `limit_for` helper makes
this a two-line change:

```python
@router.delete("/{element_id}", ...)
@limit_for("rate_limit_cache")
async def cache_delete(request: Request, element_id: str, ...):
```

---

## Choosing the right values

### Egress: matching upstream quotas

Start by looking up your actual quota tier in the Gemini or Anthropic console.

**Gemini free tier** (as of 2025): 15 RPM for most models, 2 RPM for Gemini 2.5.

```env
HTTP_RPS_INGEST=0.2     # 12 RPM with headroom
HTTP_RPS_SEARCH=0.2
HTTP_RPS_CACHE=0.2
HTTP_RPS_DEFAULT=0.1
```

**Gemini paid tier** (Tier 1, Gemini 2.5 Flash): 1000 RPM.

```env
HTTP_RPS_INGEST=14.0    # 840 RPM, 16% headroom
HTTP_RPS_SEARCH=8.0
HTTP_RPS_CACHE=8.0
HTTP_RPS_ADMIN=2.0
```

**Rule of thumb:** set RPS to 80-85% of your quota limit. The remaining headroom absorbs
burstiness from the polling approximation and prevents the 429 drain from causing
unnecessary retries.

If you see 429s despite staying under quota, lower the values further. Provider quotas are
often enforced at sub-minute windows (e.g. 100 RPM may mean "no more than 2 in any 1.2s
window") and a simple per-second bucket does not capture that granularity. The `limits`
library supports sliding window strategies; configuring that is out of scope for this
document.

### Ingress: protecting vecstore-api

The ingress limits exist to protect the downstream services (embedding, Redis, the LLM
summariser) from client bursts, not primarily to enforce fair usage. Set them relative to
what the downstream can sustain:

- `RATE_LIMIT_INGEST` should not exceed `embed_workers * embed_concurrency * 60` per minute
  (the throughput the embedder can actually process). With defaults (`workers=4`,
  `concurrency=2`) that is roughly `480/minute`; `120/minute` is conservative.
- `RATE_LIMIT_SUMMARISE` should be lower than `RATE_LIMIT_INGEST` because each summarise
  call hits the LLM, which is the most expensive downstream call. `30/minute` with
  `summariser_concurrency=2` is proportionate.
- `/health` and `/status` are intentionally uncapped -- they must remain available for
  liveness/readiness probes under all load conditions.

---

## Running multiple instances

**Multiple god processes (egress)**

If you run two `god` CLI invocations simultaneously, each has its own in-process buckets.
The upstream provider sees the aggregate. Divide by the number of concurrent processes:

```bash
# Two processes sharing a 10 RPS quota:
HTTP_RPS_INGEST=4.0   # leaves 20% headroom per process, handles one process dying
```

Alternatively, stagger launches: `god --traces 500` then `god --traces 500 --seed 43`.

**Multiple vecstore-api replicas (ingress)**

With multiple replicas each has its own in-memory buckets. Divide `RATE_LIMIT_*` by replica
count, or switch to the Redis backend (see [Ingress: known limitations](#ingress-known-limitations)).

For local single-process development the defaults are fine as-is.

---

## End-to-end flow

A request from `god` to `vecstore-api` and on to Gemini passes through both rate-limiting
layers:

```
god (HttpClient)
    |
    | HTTP_RPS_INGEST token bucket
    | HTTP_RATE_INGEST semaphore
    |
    v
vecstore-api (FastAPI)
    |
    | RATE_LIMIT_INGEST per-IP bucket (slowapi)
    |
    v
Embedder
    |
    | EMBED_RATE_LIMIT_RPM token bucket (pyrate-limiter, Gemini only)
    | embed_concurrency semaphore
    | tenacity retry + backoff on 429
    |
    v
Google Gemini API
    |
    | (quota enforced upstream)
```

The `god` client limits are tuned to stay within Gemini's quota; the `vecstore-api` limits
are tuned to protect the embedder from overload. The two sets of limits operate independently
and should be sized independently.

If the god client fires faster than the vecstore-api ingest limit allows, the `vecstore-api`
will return 429s, which the `god` client will retry with backoff and drain -- correctly
slowing itself down. This feedback loop works even when the two components are not configured
consistently.

---

## Troubleshooting

**Seeing 429s from Gemini/Anthropic despite setting RPS limits**

- Confirm the `.env` is being read: `python -c "from god.config import get_http_settings; s = get_http_settings(); print(s.http_rps_ingest)"`. If it returns `0.0` your `.env` path is wrong.
- Remember RPS limits are per process. Two concurrent processes double the effective rate.
- Provider quotas may be enforced per minute rather than per second. A 60 RPM limit at
  `1.0 RPS` setting can still trigger a quota if the minute window fills early in the minute.
  Set `HTTP_RETRY_BACKOFF_MAX` to at least `60` so retries wait out the window.
- Check for multiple operation names routing to the same upstream bucket. If `search` and
  `embed` both call Gemini but have separate `rate_name` values, their buckets are independent
  but share the same upstream quota.

**Seeing 429s from vecstore-api**

- The response includes `Retry-After`. The `god` client respects this via tenacity; if you
  are calling the API directly, honour the header.
- If all requests come from the same IP (typical in local dev), limits are shared across all
  callers. Raise `RATE_LIMIT_*` if you are the sole legitimate user.
- If the limit feels too tight for batch operations, use `async_mode=true` on the ingest
  endpoints. The async endpoint counts as one HTTP request regardless of batch size, and the
  processing happens in the background outside the rate limit window.

**Token bucket not taking effect**

All `HTTP_RPS_*` defaults are `0.0` (disabled). Verify the value is non-zero:

```bash
python -c "
from god.config import HttpClientSettings
s = HttpClientSettings()
print('RPS ingest:', s.http_rps_ingest)
print('RPS search:', s.http_rps_search)
"
```

If running via `make`, ensure `.env` is in the directory where `make` is invoked
(`src/god/` by default).

**Very low RPS limits feel slower than expected**

Below `~0.5 RPS` the polling backoff (10ms -> 1s exponential) becomes significant relative
to the token interval. At `0.1 RPS` (one request every 10 seconds) a request that arrives
just after a token was consumed will poll for up to 1s before finding a token, then wait
the remaining ~9s at 1s intervals. This is correct behaviour but means actual throughput is
approximately `0.09 RPS` rather than `0.1`. Not a bug; size limits accordingly.

**Bucket not resetting after a 429**

The drain does not permanently reduce the refill rate -- it only consumes an extra token from
the current burst allowance. The bucket refills at the same `rps` rate regardless. If you
are seeing sustained 429s you need to lower `HTTP_RPS_*`, not wait for the bucket to recover.
