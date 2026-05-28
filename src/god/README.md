# god - GenAI Observability Data Generator

Python package (`genai-observability-data`) that generates realistic distributed traces and structured
logs for GenAI-powered systems, following the [OTel GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
(semconv 1.41.0, experimental).

The package ships a CLI (`god`), a library API, and an optional pydantic-ai analysis agent.

---

## Contents

- [Package API](#package-api)
- [CLI quick reference](#cli-quick-reference)
- [Configuration](#configuration)
- [Key modules](#key-modules)
- [Data models](#data-models)
- [Scenarios](#scenarios)
- [Sinks](#sinks)
- [Analysis agent](#analysis-agent)
- [Mock mode](#mock-mode)
- [Running tests](#running-tests)
- [Best practices](#best-practices)
- [Gotchas](#gotchas)

---

## Package API

Everything public is re-exported from `god.__init__`:

```python
from god import (
    # Data models
    Span, SpanEvent, Trace, LogEntry,
    LogLevel, SpanKind, SpanStatus, FinishReason,
    GenAIOperation, GenAIProvider, ServiceType,
    # Service topology
    ServiceDef, ServiceRegistry, LatencyProfile,
    # Scenarios
    Scenario, ScenarioEngine, ScenarioState, SCENARIO_PRESETS,
    # Generation
    generate_trace, generate_dataset,
    # Sinks
    Sink, StdoutSink, FileSink, OTLPHttpSink, MultiSink,
    # Config
    get_settings, reset_settings,
)
```

### Minimal usage - generate and stream to OTLP

```python
import asyncio
import numpy as np
from god import ServiceRegistry, ScenarioEngine, generate_dataset, OTLPHttpSink

registry = ServiceRegistry()
engine = ScenarioEngine.from_preset("rag_meltdown", registry)
sink = OTLPHttpSink(endpoint="http://localhost:4318")

for trace, logs in generate_dataset(
    registry=registry,
    n_traces=500,
    traces_per_second=10.0,
    rng_seed=42,
    engine=engine,
):
    sink.flush([trace], logs)

sink.close()
```

### Analyse a batch with the pydantic-ai agent

```python
import asyncio
from god import ServiceRegistry, generate_dataset
from god.agent import analyse_batch
from god.deps import GodDeps
from god.mock.activate import activate_mock_mode

activate_mock_mode()   # remove this line when you have an ANTHROPIC_API_KEY

registry = ServiceRegistry()
traces, all_logs = [], []
for trace, logs in generate_dataset(registry=registry, n_traces=100):
    traces.append(trace)
    all_logs.extend(logs)

result = asyncio.run(analyse_batch(traces, all_logs, GodDeps()))
print(result.summary)
print(result.anomaly_signals)
```

---

## CLI quick reference

The package installs a `god` entry point. When using `uv`, invoke via `uv run python -m god`.

```bash
# stdout logfmt, no API key needed
uv run python -m god -n 50 --mock

# Write all formats to ./output/
uv run python -m god -n 1000 --seed 42 --output ./output \
  --format otlp_traces otlp_logs logfmt loki

# Stream a scenario to a local OTel collector
uv run python -m god -n 500 --scenario cascading_failure --rate 5 \
  --otlp http://localhost:4318

# Run the analysis agent (mock mode)
uv run python -m god -n 200 --scenario llm_degradation --mock --analyse

# List scenarios
uv run python -m god --list-scenarios
```

See the root [README.md](../../README.md) for the full CLI reference.

---

## Configuration

Settings are loaded by `config.py` using pydantic-settings. The `.env` file is resolved relative to
the **working directory** at the time `get_settings()` is first called, not relative to this file.

```bash
# Create .env in src/god/ (or set env vars directly)
cat > .env <<'EOF'
ANTHROPIC_API_KEY=sk-ant-...      # required for --analyse without --mock
GOD_MOCK_MODE=false
GOD_AGENT_MODEL=claude-haiku-4-5-20251001
GOD_DEFAULT_TRACES=100
GOD_DEFAULT_RATE=10.0
GOD_DEFAULT_BATCH_SIZE=50
EOF
```

`get_settings()` is a singleton. Call `reset_settings()` between tests to force a fresh load.

---

## Key modules

```
god/
  config.py        Settings singleton (pydantic-settings)
  models.py        Span, Trace, LogEntry + OTLP/Loki/logfmt serialisation
  services.py      ServiceRegistry, ServiceDef, LatencyProfile, fault overrides
  scenarios.py     Scenario state machine, ScenarioEngine, 8 built-in presets
  orchestrator.py  generate_trace + generate_dataset (the main generation loop)
  agent.py         pydantic-ai analysis agent + AnomalySignal, ScenarioAnalysis
  deps.py          GodDeps dependency injection container
  sinks.py         StdoutSink, FileSink, OTLPHttpSink, MultiSink
  generators/
    genai.py       GenAI span builders (inference, embedding, retrieval, tool, MCP)
    infra.py       Infra span builders (HTTP, gRPC, DB, queue) + log generators
  mock/
    activate.py    activate_mock_mode() / deactivate_mock_mode()
    models.py      Deterministic FunctionModel for the analysis agent
```

---

## Data models

### Span

```python
@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: SpanKind
    service_name: str
    start_time_unix_nano: int
    end_time_unix_nano: int
    status: SpanStatus
    status_message: str
    attributes: dict[str, Any]
    events: list[SpanEvent]
    resource_attributes: dict[str, Any]

    @property
    def duration_ms(self) -> float: ...

    def to_otlp_dict(self) -> dict: ...
```

### Trace

```python
@dataclass
class Trace:
    trace_id: str
    spans: list[Span]

    def root_span(self) -> Span: ...          # first span with no parent
    def to_otlp_dict(self) -> dict: ...       # {"resourceSpans": [...]}
```

### LogEntry

```python
@dataclass
class LogEntry:
    trace_id: str
    span_id: str
    timestamp_unix_nano: int
    severity: LogLevel
    body: str
    service_name: str
    attributes: dict[str, Any]
    resource_attributes: dict[str, Any]

    def to_otlp_dict(self) -> dict: ...
    def to_loki_dict(self) -> dict: ...
    def to_logfmt(self) -> str: ...
```

### OTLP attribute type mapping

| Python type | OTLP field |
|-------------|-----------|
| `bool` | `boolValue` |
| `int` | `intValue` (serialised as string per spec) |
| `float` | `doubleValue` |
| `list` | `arrayValue` |
| `str` | `stringValue` |

---

## Scenarios

### Built-in presets

| Name | Faults | Services degraded |
|------|--------|------------------|
| `db_incident` | DB pool exhaustion (40% err, 8x lat) | postgres-payments |
| `llm_degradation` | LLM latency spike + token truncations | llm-anthropic |
| `rag_meltdown` | Vector DB slow + LLM spike + provider failover | vector-db, llm-anthropic |
| `payment_outage` | 60% payment errors + DB degradation | payments-service, postgres-payments |
| `cascading_failure` | Auth DB -> auth outage -> payments -> LLM | 4 services, chained |
| `quiet` | No faults | - |

### Scenario state machine

```python
engine = ScenarioEngine.from_preset("cascading_failure", registry)

# Engine is ticked internally by generate_dataset once per virtual second.
# You can also drive it manually:
activated = engine.tick(elapsed_seconds)
active_names = engine.active_scenario_names()
active_tags = engine.active_tags()   # e.g. {"genai", "latency", "auth"}
```

### Custom scenario

```python
from god.scenarios import Scenario, ScenarioEngine, SCENARIO_PRESETS
from god.services import ServiceRegistry

def my_embed_overload(registry: ServiceRegistry) -> list[Scenario]:
    def apply(r: ServiceRegistry) -> None:
        r.propagate_fault("embed-service", error_rate=0.3, latency_multiplier=4.0)

    def recover(r: ServiceRegistry) -> None:
        svc = r.get("embed-service")
        svc._error_rate_override = None
        svc._latency_multiplier = 1.0

    return [Scenario(
        name="embed_overload",
        description="Embedding service saturation",
        start_offset_s=10,
        end_offset_s=120,
        apply=apply,
        recover=recover,
        tags={"genai", "embed", "latency"},
    )]

SCENARIO_PRESETS["embed_overload"] = my_embed_overload
```

---

## Sinks

### StdoutSink

```python
StdoutSink(include_traces=False)
```

Writes logs as logfmt to stdout. If `include_traces=True`, also prints a one-line summary per trace.

### FileSink

```python
FileSink(
    output_dir=Path("./output"),
    formats=["otlp_traces", "otlp_logs", "logfmt"],  # also: "loki"
    compress=False,    # append .gz and use gzip.open in append mode
    append=True,       # accumulate across multiple flush() calls
)
```

Files written:
- `traces.otlp.jsonl` (or `.jsonl.gz`)
- `logs.otlp.jsonl`
- `logs.logfmt`
- `logs.loki.jsonl`

### OTLPHttpSink

```python
OTLPHttpSink(
    endpoint="http://localhost:4318",   # no trailing slash, no /v1/traces
    headers={"Authorization": "Bearer ..."},
    timeout_s=10.0,
)
```

POSTs to `{endpoint}/v1/traces` and `{endpoint}/v1/logs`. Uses stdlib `urllib.request` (synchronous).
Accepts HTTP 200 or 204. On `URLError`, prints to stderr and continues (does not raise).

### MultiSink

```python
sink = MultiSink([FileSink(...), OTLPHttpSink(...)])
sink.flush(traces, logs)
sink.close()
```

### Flushing

All sinks implement the `Sink` protocol:

```python
class Sink(Protocol):
    def flush(self, traces: list[Trace], logs: list[LogEntry]) -> None: ...
    def close(self) -> None: ...
```

Call `flush()` periodically (every `batch_size` traces) and `close()` when done.

---

## Analysis agent

```python
from god.agent import analyse_batch, ScenarioAnalysis, AnomalySignal
from god.deps import GodDeps

analysis: ScenarioAnalysis = await analyse_batch(
    traces=traces,
    logs=logs,
    deps=GodDeps(),
    model_name="claude-haiku-4-5-20251001",   # optional override
)
```

`analyse_batch` computes a summary dict (error rate, p99 latency, per-service error counts, token
truncations, tool failures) and passes it to `scenario_agent` as JSON in the prompt.

### ScenarioAnalysis fields

| Field | Type | Description |
|-------|------|-------------|
| `active_scenarios` | `list[str]` | Named scenarios the agent identified as active |
| `anomaly_signals` | `list[AnomalySignal]` | Per-service anomaly signals |
| `error_rate` | `float` | Overall error fraction (0.0-1.0) |
| `p99_latency_ms` | `float` | p99 span duration across the batch |
| `recommended_action` | `str` | Agent-generated action recommendation |
| `summary` | `str` | Prose summary of observations |

### AnomalySignal

| Field | Values |
|-------|--------|
| `service` | service name string |
| `signal_type` | `error_spike` / `latency_spike` / `token_truncation` / `tool_failure` |
| `severity` | `low` / `medium` / `high` / `critical` |

---

## Mock mode

Mock mode replaces the pydantic-ai `scenario_agent` with a deterministic `FunctionModel` that
implements the analysis logic locally with no API calls.

```python
from god.mock.activate import activate_mock_mode, deactivate_mock_mode

activate_mock_mode()     # idempotent
# ... run analyses ...
deactivate_mock_mode()   # safe to call even if not active
```

Mock severity thresholds:

| Per-service error rate | Severity |
|------------------------|----------|
| > 50% | critical |
| > 20% | high |
| > 5% | medium |
| <= 5% | low |

Additional signals: latency spike if p99 > 5000ms; token truncation if any truncations; tool failure
if any tool spans in ERROR state.

CLI equivalent: `--mock` flag or `GOD_MOCK_MODE=true`.

---

## Running tests

```bash
# All tests (from src/god/)
uv run pytest

# With coverage
uv run pytest --cov=. --cov-report=term-missing

# Specific file
uv run pytest tests/test_generators.py -v

# Specific test
uv run pytest tests/test_scenarios.py::test_cascading_failure_has_four_scenarios -v
```

### pytest configuration (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"        # all async tests run automatically
testpaths = ["tests"]
pythonpath = [".."]          # imports as: from god.models import Span
addopts = "-v --tb=short"
```

### Test fixtures and patterns

**Mock mode in tests:** `test_agent.py` has an `autouse` fixture that calls `activate_mock_mode()` /
`deactivate_mock_mode()` around each test. No real API calls are made, even in CI.

**Determinism:** Pass `rng_seed=42` to `generate_dataset` for reproducible output. Without a seed,
each run is non-deterministic.

**Settings isolation:** Call `reset_settings()` after modifying settings in a test to clear the
singleton:

```python
import pytest
from god.config import get_settings, reset_settings

@pytest.fixture
def custom_settings(monkeypatch):
    monkeypatch.setenv("GOD_MOCK_MODE", "true")
    reset_settings()
    yield get_settings()
    reset_settings()
```

**No API guard needed:** The test suite never calls a real model. `activate_mock_mode()` is sufficient.
If you add a test that could accidentally reach the Anthropic API, add an assertion on
`GOD_MOCK_MODE=true` in the fixture.

### Test coverage by module

| Module | Test file | Key assertions |
|--------|-----------|---------------|
| `models.py` | `test_models.py` | OTLP JSON structure, int-as-string, severity numbers |
| `services.py` | `test_services.py` | Latency lognormal params, fault overrides, bleed |
| `scenarios.py` | `test_scenarios.py` | State transitions, preset counts, unknown preset error |
| `generators/` | `test_generators.py` | Attribute presence, TTFT event, RAG trace shape |
| `orchestrator.py` | `test_orchestrator.py` | Span count, parent relationships, log correlation |
| `sinks.py` | `test_sinks.py` | File content, gzip, OTLP HTTP mock, multi-sink |
| `agent.py` | `test_agent.py` | ScenarioAnalysis structure, severity accuracy |

---

## Best practices

**Always use `rng_seed` for fixture generation.** Deterministic fixtures are easier to debug and
diff. Use `seed=42` (or any constant) in `make gen-fixtures`.

**Run `generate_dataset` as a generator, not eagerly.** It yields `(Trace, list[LogEntry])` pairs.
Collecting everything into a list before flushing loads the full dataset into memory. For large
runs, flush every `batch_size` traces instead:

```python
buffer_t, buffer_l = [], []
for trace, logs in generate_dataset(registry, n_traces=10_000):
    buffer_t.append(trace)
    buffer_l.extend(logs)
    if len(buffer_t) >= batch_size:
        sink.flush(buffer_t, buffer_l)
        buffer_t, buffer_l = [], []
if buffer_t:
    sink.flush(buffer_t, buffer_l)
```

**Use `MultiSink` when writing to multiple destinations.** It handles errors in one sink
independently.

**Prefer `--mock` during development.** All scenario signals, anomaly detection, and severity
thresholds work correctly in mock mode. Only switch to a real model when evaluating the agent's
reasoning quality specifically.

**Pass `engine` to `generate_dataset` for scenario-correlated data.** Without an engine, all
generated data is baseline (no faults). The engine ticks automatically inside the generator loop.

**Set `capture_content=True` only when you need message bodies.** The attribute values are
stringified Python lists, which increases output size substantially.

---

## Gotchas

1. **`.env` resolves from the working directory.** If you `cd src/god && uv run python -m god`,
   the `.env` must be in `src/god/`. If you run from the repo root, it must be in the root. Check
   by calling `get_settings()` and printing `settings.model_dump()` to verify what loaded.

2. **`gen-otlp` make target does NOT set mock mode.** It is the only generation target that will
   attempt real LLM calls when `--analyse` is passed. All other targets (`gen`, `gen-files`,
   `gen-fixtures`) set `GOD_MOCK_MODE=true`.

3. **Token budget breach sets error rate to zero, not a positive value.** The scenario calls
   `_error_rate_override = 0.0` to suppress baseline errors. The anomaly shows up only as
   `finish_reason=length` on spans - not in span status. Write dashboards accordingly.

4. **`gen_ai.server.time_to_first_token` is in seconds; `gen_ai.server.ttft_ms` is in
   milliseconds.** Both are emitted on every successful inference span. Do not sum them or use
   them interchangeably in PromQL/LogQL.

5. **`OTLPHttpSink` is synchronous.** It blocks the thread while POSTing. Use a small `--batch-size`
   (e.g. 10) when streaming to a remote endpoint, or wrap calls in a thread pool if needed.

6. **`int` attributes are strings in OTLP JSON.** `{"intValue": "1204"}` not `{"intValue": 1204}`.
   The OTel spec requires this. Some parsers handle both; do not assume either.

7. **Fault propagation is one hop only.** `propagate_fault` bleeds 30% error rate and 50% latency
   increase to immediate callers in the dependency graph. It is not transitive. Recovery does not
   call `propagate_fault` in reverse - it clears overrides directly on specific services.
