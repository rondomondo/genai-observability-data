"""Mock pydantic-ai models for the scenario analysis agent.

Uses FunctionModel — fully deterministic, no network calls, no API key.
Produces realistic-looking ScenarioAnalysis output based on the trace summary
passed in the user message.
"""

from __future__ import annotations

import json
import re

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


def _last_user_text(messages: list[ModelMessage]) -> str:
    for msg in reversed(messages):
        for part in getattr(msg, "parts", []):
            if hasattr(part, "content") and isinstance(part.content, str):
                return part.content
    return ""


def _contains(text: str, *terms: str) -> bool:
    lower = text.lower()
    return any(t.lower() in lower for t in terms)


def _analysis_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    text = _last_user_text(messages)

    # Try to parse the JSON summary embedded in the prompt
    summary: dict = {}
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        try:
            summary = json.loads(match.group())
        except json.JSONDecodeError:
            pass

    error_rate: float = summary.get("error_rate", 0.0)
    p99: float = summary.get("p99_latency_ms", 0.0)
    active_hints: list[str] = summary.get("active_scenarios_hint", [])
    service_errors: dict[str, int] = summary.get("service_error_counts", {})
    truncations: int = summary.get("genai_token_truncations", 0)
    tool_failures: int = summary.get("genai_tool_failures", 0)

    # Build anomaly signals
    signals = []
    for svc, count in service_errors.items():
        total = summary.get("service_span_counts", {}).get(svc, 1)
        svc_err_rate = count / max(total, 1)
        if svc_err_rate > 0.5:
            severity = "critical"
        elif svc_err_rate > 0.2:
            severity = "high"
        elif svc_err_rate > 0.05:
            severity = "medium"
        else:
            severity = "low"
        signals.append(
            {
                "service": svc,
                "signal_type": "error_spike",
                "severity": severity,
                "detail": f"{svc_err_rate:.0%} error rate ({count}/{total} spans)",
            }
        )

    if p99 > 5000:
        signals.append(
            {
                "service": "llm-anthropic",
                "signal_type": "latency_spike",
                "severity": "high",
                "detail": f"p99 latency {p99:.0f}ms exceeds 5s threshold",
            }
        )
    if truncations > 0:
        signals.append(
            {
                "service": "llm-anthropic",
                "signal_type": "token_truncation",
                "severity": "medium",
                "detail": f"{truncations} responses truncated at token limit",
            }
        )
    if tool_failures > 0:
        signals.append(
            {
                "service": "llm-anthropic",
                "signal_type": "tool_failure",
                "severity": "medium",
                "detail": f"{tool_failures} tool calls failed",
            }
        )

    # Recommended action
    if error_rate > 0.3:
        action = "Investigate high error rate — check service logs and dependency health"
    elif p99 > 8000:
        action = "Investigate LLM latency — check GPU utilisation and model queue depth"
    elif truncations > 2:
        action = "Increase max_tokens in LLM configuration to reduce response truncations"
    else:
        action = "No immediate action required — monitor for trend continuation"

    # Summary text
    if active_hints:
        scenario_str = ", ".join(active_hints)
        summary_text = (
            f"Active fault scenarios: {scenario_str}. Overall error rate {error_rate:.1%}, p99 latency {p99:.0f}ms."
        )
    else:
        summary_text = f"No active fault scenarios. System nominal — error rate {error_rate:.1%}, p99 {p99:.0f}ms."

    result = {
        "active_scenarios": active_hints,
        "anomaly_signals": signals,
        "error_rate": error_rate,
        "p99_latency_ms": p99,
        "recommended_action": action,
        "summary": summary_text,
    }
    return ModelResponse(parts=[TextPart(json.dumps(result))])


def make_analysis_model() -> FunctionModel:
    """Return a deterministic mock model for the scenario analysis agent."""
    return FunctionModel(_analysis_fn)
