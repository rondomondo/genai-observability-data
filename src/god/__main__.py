#!/usr/bin/env python3
"""
god — GenAI Observability Data generator.

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

import argparse
import asyncio
import sys
import time
from pathlib import Path

from god import __version__
from god.config import get_settings
from god.orchestrator import generate_dataset
from god.scenarios import SCENARIO_PRESETS, ScenarioEngine
from god.services import ServiceRegistry
from god.sinks import FileSink, MultiSink, OTLPHttpSink, Sink, StdoutSink


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None)

    Returns:
        Parsed namespace
    """
    p = argparse.ArgumentParser(
        prog="god",
        description=(
            "Generate synthetic distributed traces and logs with GenAI instrumentation.\n"
            "All generation is purely local — no network calls unless --otlp or --analyse is used."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--version", "-V",
        action="version",
        version=f"god {__version__}",
    )

    # -- Generation controls ------------------------------------------
    gen = p.add_argument_group("Generation")
    gen.add_argument(
        "--traces", "-n",
        type=int, default=None,
        metavar="N",
        help="Number of traces to generate (default: 100 or GOD_DEFAULT_TRACES)",
    )
    gen.add_argument(
        "--rate",
        type=float, default=None,
        metavar="N",
        help="Simulated arrival rate in traces/second (default: 10.0)",
    )
    gen.add_argument(
        "--scenario",
        choices=list(SCENARIO_PRESETS.keys()),
        default=None,
        metavar="NAME",
        help=f"Fault injection scenario preset. Choices: {', '.join(SCENARIO_PRESETS)}",
    )
    gen.add_argument(
        "--seed",
        type=int, default=None,
        metavar="N",
        help="RNG seed for deterministic output (default: random)",
    )
    gen.add_argument(
        "--capture-content",
        action="store_true",
        default=False,
        help="Include gen_ai.input/output.messages in spans (large, opt-in)",
    )

    # -- Output controls -----------------------------------------------
    out = p.add_argument_group("Output")
    out.add_argument(
        "--format",
        nargs="+",
        choices=["otlp_traces", "otlp_logs", "logfmt", "loki"],
        default=["logfmt"],
        metavar="FMT",
        help="Output format(s) for file sink (default: logfmt)",
    )
    out.add_argument(
        "--output", "-o",
        type=str, default=None,
        metavar="DIR",
        help="Output directory for file sink. Omit to write to stdout.",
    )
    out.add_argument(
        "--otlp",
        type=str, default=None,
        metavar="URL",
        help="OTLP HTTP endpoint, e.g. http://localhost:4318",
    )
    out.add_argument(
        "--otlp-header",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra HTTP headers for OTLP sink (repeatable)",
    )
    out.add_argument(
        "--compress",
        action="store_true",
        default=False,
        help="Gzip-compress file output",
    )
    out.add_argument(
        "--batch-size",
        type=int, default=None,
        metavar="N",
        help="Flush to sinks every N traces (default: 50)",
    )

    # -- Agent controls ------------------------------------------------
    agent = p.add_argument_group("Agent")
    agent.add_argument(
        "--analyse",
        action="store_true",
        default=False,
        help="Run the pydantic-ai scenario analysis agent on the generated batch",
    )
    agent.add_argument(
        "--agent-model",
        type=str, default=None,
        metavar="MODEL",
        help="Override the agent model (e.g. claude-sonnet-4-20250514)",
    )
    agent.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Use mock mode (FunctionModel — no API key or network calls needed)",
    )

    # -- Meta ----------------------------------------------------------
    meta = p.add_argument_group("Meta")
    meta.add_argument(
        "--list-scenarios",
        action="store_true",
        default=False,
        help="List available scenario presets and exit",
    )
    meta.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="Suppress progress output to stderr",
    )

    return p.parse_args(argv)


def build_sinks(args: argparse.Namespace) -> Sink:
    """Construct the sink chain from parsed arguments.

    Args:
        args: Parsed CLI namespace

    Returns:
        A Sink (possibly MultiSink) ready to receive flush() calls
    """
    sinks: list[Sink] = []

    if args.output:
        sinks.append(FileSink(
            output_dir=args.output,
            formats=args.format,
            compress=args.compress,
        ))
    elif not args.otlp:
        sinks.append(StdoutSink(include_traces=False))

    if args.otlp:
        headers: dict[str, str] = {}
        for h in args.otlp_header:
            if "=" in h:
                k, v = h.split("=", 1)
                headers[k.strip()] = v.strip()
        sinks.append(OTLPHttpSink(endpoint=args.otlp, headers=headers))

    if not sinks:
        sinks.append(StdoutSink())

    return MultiSink(sinks) if len(sinks) > 1 else sinks[0]


def _list_scenarios() -> None:
    print("Available scenario presets:\n")
    for name, factory in SCENARIO_PRESETS.items():
        scenarios = factory()
        if scenarios:
            tags: set[str] = set()
            for s in scenarios:
                tags.update(s.tags)
            print(f"  {name:<25} {len(scenarios)} scenario(s)  tags: {', '.join(sorted(tags))}")
            for s in scenarios:
                print(f"    - {s.name}: {s.description}")
        else:
            print(f"  {name:<25} (no fault injection)")


async def _run_analysis(
    traces: list,
    logs: list,
    deps,
    model_override: str | None,
    quiet: bool,
) -> None:
    from god.agent import analyse_batch
    if not quiet:
        print("\n[analyse] Running scenario analysis agent...", file=sys.stderr)
    result = await analyse_batch(traces, logs, deps, model_name=model_override)
    print("\n--- Scenario Analysis ---")
    print(f"Active scenarios : {result.active_scenarios or ['none']}")
    print(f"Error rate       : {result.error_rate:.1%}")
    print(f"p99 latency      : {result.p99_latency_ms:.0f}ms")
    if result.anomaly_signals:
        print("Anomaly signals  :")
        for sig in result.anomaly_signals:
            print(f"  [{sig.severity.upper()}] {sig.service} — {sig.signal_type}: {sig.detail}")
    else:
        print("Anomaly signals  : none")
    print(f"Recommendation   : {result.recommended_action}")
    print(f"Summary          : {result.summary}")
    print("---")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to sys.argv[1:])

    Returns:
        Exit code (0 on success, 1 on failure)
    """
    args = parse_args(argv)

    if args.list_scenarios:
        _list_scenarios()
        return 0

    settings = get_settings()

    # Apply mock mode from flag or env
    use_mock = args.mock or settings.mock_mode
    if use_mock:
        from god.mock.activate import activate_mock_mode
        activate_mock_mode()

    # Resolve defaults (CLI > env > hardcoded)
    n_traces = args.traces or settings.default_traces
    rate = args.rate or settings.default_rate
    batch_size = args.batch_size or settings.default_batch_size

    registry = ServiceRegistry()
    engine: ScenarioEngine | None = None

    if args.scenario:
        engine = ScenarioEngine.from_preset(args.scenario, registry)
        if not args.quiet:
            print(
                f"[scenario] '{args.scenario}' loaded with "
                f"{len(engine.scenarios)} scenario(s)",
                file=sys.stderr,
            )

    if args.analyse:
        from god.deps import GodDeps
        deps = GodDeps(settings=settings, registry=registry, engine=engine)

    sink = build_sinks(args)

    base_epoch_ms = time.time() * 1000
    trace_buf: list = []
    log_buf: list = []
    all_traces_for_analysis: list = []
    all_logs_for_analysis: list = []
    generated = 0
    t0 = time.monotonic()

    if not args.quiet:
        print(
            f"[generate] {n_traces} traces | rate={rate}/s | "
            f"scenario={args.scenario or 'none'} | seed={args.seed} | "
            f"mock={use_mock}",
            file=sys.stderr,
        )

    try:
        for trace, logs in generate_dataset(
            registry=registry,
            n_traces=n_traces,
            base_epoch_ms=base_epoch_ms,
            traces_per_second=rate,
            rng_seed=args.seed,
            engine=engine,
            capture_content=args.capture_content,
        ):
            trace_buf.append(trace)
            log_buf.extend(logs)
            if args.analyse:
                all_traces_for_analysis.append(trace)
                all_logs_for_analysis.extend(logs)
            generated += 1

            if generated % batch_size == 0:
                sink.flush(trace_buf, log_buf)
                trace_buf.clear()
                log_buf.clear()

                if not args.quiet:
                    elapsed = time.monotonic() - t0
                    print(
                        f"[generate] {generated}/{n_traces} traces "
                        f"({elapsed:.1f}s)",
                        file=sys.stderr,
                        end="\r",
                    )

        if trace_buf or log_buf:
            sink.flush(trace_buf, log_buf)

    finally:
        sink.close()

    elapsed = time.monotonic() - t0
    if not args.quiet:
        print(
            f"\n[done] {generated} traces in {elapsed:.2f}s "
            f"({generated/elapsed:.0f} traces/s)",
            file=sys.stderr,
        )

    if args.analyse:
        asyncio.run(_run_analysis(
            all_traces_for_analysis,
            all_logs_for_analysis,
            deps,
            args.agent_model,
            args.quiet,
        ))

    return 0


if __name__ == "__main__":
    sys.exit(main())
