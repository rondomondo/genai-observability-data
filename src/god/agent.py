"""Pydantic-AI scenario analysis agent.

The agent analyses a batch of generated traces and produces a structured
summary: active scenarios, anomaly signals, and recommended next actions.

In mock mode (GOD_MOCK_MODE=true) the agent uses a FunctionModel -- no
network calls, no API key required.
"""

from __future__ import annotations

import asyncio
import json
import warnings
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model, parse_model_id
from pyrate_limiter import Duration, Limiter, Rate

from god.deps import GodDeps
from god.models import LogEntry, SpanStatus, Trace

# Output model returned by the agent


class AnomalySignal(BaseModel):
    service: str
    signal_type: str = Field(description="error_spike | latency_spike | token_truncation | tool_failure")
    severity: str = Field(description="low | medium | high | critical")
    detail: str


class ScenarioAnalysis(BaseModel):
    """Structured analysis of a batch of traces."""

    active_scenarios: list[str] = Field(description="Scenario names currently injected")
    anomaly_signals: list[AnomalySignal] = Field(description="Detected anomaly signals")
    error_rate: float = Field(description="Fraction of spans in ERROR state")
    p99_latency_ms: float = Field(description="Approximate p99 span duration in ms")
    recommended_action: str = Field(description="Short recommended operator action")
    summary: str = Field(description="1-2 sentence human-readable summary")


# Agent definition — model is set at runtime to support per-call overrides

scenario_agent: Agent[GodDeps, ScenarioAnalysis] = Agent(
    model=None,  # type: ignore[arg-type]
    deps_type=GodDeps,
    output_type=ScenarioAnalysis,
    instructions="""You are an observability analyst for a GenAI application stack.

Given a JSON summary of distributed trace spans and logs, identify:
1. Active fault injection scenarios (db_degradation, llm_latency_spike, etc.)
2. Anomaly signals: error spikes, latency regressions, LLM token truncations, tool failures
3. The overall error rate and p99 latency across spans
4. A short recommended operator action

Be concise. Focus on signals that indicate real production problems.""",
)


def _summarise_batch(traces: list[Trace], logs: list[LogEntry]) -> dict[str, Any]:
    """Build a compact JSON-serialisable summary of a trace batch for the agent."""
    total_spans = sum(len(t.spans) for t in traces)
    error_spans = sum(1 for t in traces for s in t.spans if s.status == SpanStatus.ERROR)
    durations = [s.duration_ms for t in traces for s in t.spans if s.duration_ms > 0]
    durations.sort()
    p99_idx = max(0, int(len(durations) * 0.99) - 1)
    p99_ms = durations[p99_idx] if durations else 0.0

    # Aggregate errors by service
    service_errors: dict[str, int] = {}
    service_total: dict[str, int] = {}
    for t in traces:
        for s in t.spans:
            service_total[s.service_name] = service_total.get(s.service_name, 0) + 1
            if s.status == SpanStatus.ERROR:
                service_errors[s.service_name] = service_errors.get(s.service_name, 0) + 1

    # GenAI-specific signals
    token_truncations = sum(
        1 for t in traces for s in t.spans if "length" in str(s.attributes.get("gen_ai.response.finish_reasons", []))
    )
    tool_failures = sum(
        1
        for t in traces
        for s in t.spans
        if s.attributes.get("gen_ai.operation.name") == "execute_tool" and s.status == SpanStatus.ERROR
    )

    return {
        "traces": len(traces),
        "total_spans": total_spans,
        "error_spans": error_spans,
        "error_rate": round(error_spans / max(total_spans, 1), 4),
        "p99_latency_ms": round(p99_ms, 1),
        "service_error_counts": service_errors,
        "service_span_counts": service_total,
        "genai_token_truncations": token_truncations,
        "genai_tool_failures": tool_failures,
        "log_count": len(logs),
    }


_GOOGLE_CLOUD_PROVIDERS = {"google-cloud", "google-cloud-global", "google-gla"}


def _normalise_model_id(raw: str) -> str:
    """Return a canonical 'provider:model' string from any supported model identifier.

    Accepts legacy bare names ('claude-sonnet-4-6', 'gemini-2.5-flash') and
    already-prefixed names ('anthropic:claude-sonnet-4-6', 'google-cloud:gemini-2.5-flash').
    """
    if ":" in raw:
        return raw
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        provider, model = parse_model_id(raw)
    if provider:
        return f"{provider}:{model}"
    return raw


def _build_google_cloud_model(model_name: str) -> Model:
    """Build a GoogleModel backed by GoogleCloudProvider using Application Default Credentials.

    The provider discovers project and location from GOOGLE_CLOUD_PROJECT /
    GOOGLE_CLOUD_LOCATION env vars (or gcloud ADC metadata), so no API key is needed.
    """
    import google.auth
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google_cloud import GoogleCloudProvider

    credentials, project = google.auth.default()
    provider = GoogleCloudProvider(credentials=credentials, project=project)
    return GoogleModel(model_name, provider=provider)


def _resolve_model(primary: str, fallback: str | None) -> Model | str:
    """Build a pydantic-ai model object from primary and optional fallback model strings.

    Falls back only when both models share the same provider -- cross-provider fallback
    is not supported by FallbackModel since providers use incompatible auth and APIs.

    google-cloud: prefix routes through GoogleCloudProvider (ADC / bearer token).
    All other prefixes are passed to pydantic-ai's infer_model unchanged.
    """
    from pydantic_ai.models import infer_model

    norm_primary = _normalise_model_id(primary)
    primary_provider, _, primary_model = norm_primary.partition(":")

    def _make(norm: str) -> Model:
        provider, _, model = norm.partition(":")
        if provider in _GOOGLE_CLOUD_PROVIDERS:
            return _build_google_cloud_model(model)
        return infer_model(norm)

    if not fallback:
        if primary_provider in _GOOGLE_CLOUD_PROVIDERS:
            return _build_google_cloud_model(primary_model)
        return norm_primary

    norm_fallback = _normalise_model_id(fallback)
    fallback_provider = norm_fallback.split(":")[0]

    if primary_provider != fallback_provider:
        if primary_provider in _GOOGLE_CLOUD_PROVIDERS:
            return _build_google_cloud_model(primary_model)
        return norm_primary

    from pydantic_ai.models.fallback import FallbackModel

    return FallbackModel(_make(norm_primary), _make(norm_fallback))


# Agent RPM default matches Gemini 2.5 Flash-Lite free-tier quota; override with GOD_AGENT_RPM.
_AGENT_DEFAULT_RPM = 15

_agent_sem: asyncio.Semaphore | None = None
_agent_limiter: Limiter | None = None


def _get_agent_gates(settings: Any) -> tuple[asyncio.Semaphore, Limiter]:
    global _agent_sem, _agent_limiter
    if _agent_sem is None:
        _agent_sem = asyncio.Semaphore(getattr(settings, "agent_concurrency", 2))
    if _agent_limiter is None:
        rpm = getattr(settings, "agent_rpm", _AGENT_DEFAULT_RPM)
        _agent_limiter = Limiter(Rate(rpm, Duration.MINUTE))
    return _agent_sem, _agent_limiter


async def _acquire_agent_token(limiter: Limiter) -> None:
    loop = asyncio.get_running_loop()
    delay = 0.05
    while True:
        acquired = await loop.run_in_executor(None, lambda: limiter.try_acquire("agent"))
        if acquired:
            return
        await asyncio.sleep(delay)
        delay = min(delay * 2, 2.0)


async def analyse_batch(
    traces: list[Trace],
    logs: list[LogEntry],
    deps: GodDeps,
    model_name: str | None = None,
) -> ScenarioAnalysis:
    """Run the scenario analysis agent on a batch of traces and logs.

    Args:
        traces: Batch of generated traces
        logs: Correlated log entries for the batch
        deps: Injected runtime dependencies
        model_name: Override the model (useful for evals)

    Returns:
        A structured ScenarioAnalysis with anomaly signals and recommendations
    """

    settings = deps.settings
    summary = _summarise_batch(traces, logs)
    if deps.scenario_preset:
        summary["active_scenarios_hint"] = [deps.scenario_preset]
    elif deps.engine:
        summary["active_scenarios_hint"] = deps.engine.activated_scenario_names()

    prompt = f"Analyse this trace batch:\n\n{json.dumps(summary, indent=2)}"

    from god.mock.activate import is_active as _mock_is_active

    if _mock_is_active():
        result = await scenario_agent.run(prompt, deps=deps)
    else:
        primary = model_name or settings.agent_model
        fallback = None if model_name else settings.agent_fallback_model
        effective_model: Model | str = _resolve_model(primary, fallback)
        sem, limiter = _get_agent_gates(settings)
        await _acquire_agent_token(limiter)
        async with sem:
            result = await scenario_agent.run(prompt, deps=deps, model=effective_model)
    return result.output
