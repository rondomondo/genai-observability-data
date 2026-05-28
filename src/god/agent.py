"""Pydantic-AI scenario analysis agent.

The agent analyses a batch of generated traces and produces a structured
summary: active scenarios, anomaly signals, and recommended next actions.

In mock mode (GOD_MOCK_MODE=true or no ANTHROPIC_API_KEY) the agent uses
a FunctionModel — no network calls, no API key required.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent

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
    active = deps.engine.active_scenario_names() if deps.engine else []
    summary["active_scenarios_hint"] = active

    prompt = f"Analyse this trace batch:\n\n{json.dumps(summary, indent=2)}"

    effective_model = model_name or settings.agent_model

    result = await scenario_agent.run(
        prompt,
        deps=deps,
        model=effective_model,
    )
    return result.output
