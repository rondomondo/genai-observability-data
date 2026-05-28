"""god — GenAI Observability Data generator.

Generate realistic distributed traces and logs with correlated failures
and OTel GenAI semantic conventions (semconv 1.41.0, experimental).

Examples
--------
# 500 traces to stdout (logfmt), no API key needed:
python -m god --traces 500 --mock

# 1000 traces with a cascading failure scenario, OTLP JSON files:
python -m god --traces 1000 --scenario cascading_failure \\
    --format otlp_traces otlp_logs --output ./data

# Deterministic dataset for test fixtures:
python -m god --traces 200 --seed 42 --format logfmt --output ./fixtures

# Stream to an OTLP collector:
python -m god --traces 10000 --rate 50 --otlp http://localhost:4318

# Run the scenario analysis agent on a batch (mock mode):
python -m god --traces 50 --scenario rag_meltdown --analyse --mock

# List available scenarios:
python -m god --list-scenarios
"""

from __future__ import annotations

import sys

from god.cli import main

if __name__ == "__main__":
    sys.exit(main())
