#!/usr/bin/env python3
"""
synthetic_logs — CLI entry point.

Generate realistic distributed traces and logs with correlated failures
and OTel GenAI semantic conventions (semconv 1.41.0, experimental).

Examples
--------
# 500 traces to stdout (logfmt):
python -m god --traces 500

# 1000 traces with a cascading failure scenario, OTLP JSON files:
python -m god --traces 1000 --scenario cascading_failure \\
    --format otlp_traces otlp_logs --output ./data

# Deterministic dataset for test fixtures:
python -m god --traces 200 --seed 42 --format logfmt --output ./fixtures

# Stream to an OTLP collector:
python -m god --traces 10000 --rate 50 --otlp http://localhost:4318

# List available scenarios:
python -m god --list-scenarios
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from synthetic_logs.orchestrator import generate_dataset
from synthetic_logs.scenarios import SCENARIO_PRESETS, ScenarioEngine
from synthetic_logs.services import ServiceRegistry
from synthetic_logs.sinks import FileSink, MultiSink, OTLPHttpSink, Sink, StdoutSink


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m synthetic_logs",
        description="Generate synthetic distributed traces and logs with GenAI instrumentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--traces", "-n",
        type=int, default=100,
        help="Number of traces to generate (default: 100)",
    )
    p.add_argument(
        "--rate",
        type=float, default=10.0,
        help="Simulated arrival rate in traces/second (default: 10.0)",
    )
    p.add_argument(
        "--scenario",
        choices=list(SCENARIO_PRESETS.keys()),
        default=None,
        help="Fault injection scenario preset (default: none)",
    )
    p.add_argument(
        "--seed",
        type=int, default=None,
        help="RNG seed for deterministic output (default: random)",
    )
    p.add_argument(
        "--format",
        nargs="+",
        choices=["otlp_traces", "otlp_logs", "logfmt", "loki"],
        default=["logfmt"],
        help="Output format(s) for file sink (default: logfmt)",
    )
    p.add_argument(
        "--output", "-o",
        type=str, default=None,
        help="Output directory for file sink. If omitted, writes to stdout.",
    )
    p.add_argument(
        "--otlp",
        type=str, default=None,
        metavar="ENDPOINT",
        help="OTLP HTTP endpoint, e.g. http://localhost:4318",
    )
    p.add_argument(
        "--otlp-header",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra HTTP headers for OTLP sink (repeatable)",
    )
    p.add_argument(
        "--capture-content",
        action="store_true",
        default=False,
        help="Include gen_ai.input/output.messages in spans (large, opt-in)",
    )
    p.add_argument(
        "--compress",
        action="store_true",
        default=False,
        help="Gzip-compress file output",
    )
    p.add_argument(
        "--batch-size",
        type=int, default=50,
        help="Flush to sinks every N traces (default: 50)",
    )
    p.add_argument(
        "--list-scenarios",
        action="store_true",
        default=False,
        help="List available scenario presets and exit",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="Suppress progress output to stderr",
    )

    return p.parse_args(argv)


def build_sinks(args: argparse.Namespace) -> Sink:
    sinks: list[Sink] = []

    if args.output:
        sinks.append(FileSink(
            output_dir=args.output,
            formats=args.format,
            compress=args.compress,
        ))
    elif not args.otlp:
        # Default: stdout logfmt
        sinks.append(StdoutSink(include_traces=False))

    if args.otlp:
        headers = {}
        for h in args.otlp_header:
            if "=" in h:
                k, v = h.split("=", 1)
                headers[k.strip()] = v.strip()
        sinks.append(OTLPHttpSink(endpoint=args.otlp, headers=headers))

    if not sinks:
        sinks.append(StdoutSink())

    return MultiSink(sinks) if len(sinks) > 1 else sinks[0]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_scenarios:
        print("Available scenario presets:\n")
        for name, factory in SCENARIO_PRESETS.items():
            scenarios = factory()
            if scenarios:
                tags = set()
                for s in scenarios:
                    tags.update(s.tags)
                print(f"  {name:<25} {len(scenarios)} scenario(s)  tags: {', '.join(sorted(tags))}")
                for s in scenarios:
                    print(f"    - {s.name}: {s.description}")
            else:
                print(f"  {name:<25} (no fault injection)")
        return 0

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

    sink = build_sinks(args)

    # Use epoch-aligned base time for realistic timestamps
    base_epoch_ms = time.time() * 1000

    trace_buf = []
    log_buf = []
    generated = 0
    t0 = time.monotonic()

    if not args.quiet:
        print(
            f"[generate] {args.traces} traces | rate={args.rate}/s | "
            f"scenario={args.scenario or 'none'} | seed={args.seed}",
            file=sys.stderr,
        )

    try:
        for trace, logs in generate_dataset(
            registry=registry,
            n_traces=args.traces,
            base_epoch_ms=base_epoch_ms,
            traces_per_second=args.rate,
            rng_seed=args.seed,
            engine=engine,
            capture_content=args.capture_content,
        ):
            trace_buf.append(trace)
            log_buf.extend(logs)
            generated += 1

            if generated % args.batch_size == 0:
                sink.flush(trace_buf, log_buf)
                trace_buf.clear()
                log_buf.clear()

                if not args.quiet:
                    elapsed = time.monotonic() - t0
                    print(
                        f"[generate] {generated}/{args.traces} traces "
                        f"({elapsed:.1f}s)",
                        file=sys.stderr,
                        end="\r",
                    )

        # Flush remainder
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
