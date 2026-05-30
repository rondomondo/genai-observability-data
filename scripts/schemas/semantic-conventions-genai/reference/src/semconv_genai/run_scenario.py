#!/usr/bin/env python3
"""Run one or more GenAI OTel reference implementation scenarios against Weaver.

This is the CLI entry point. The heavy lifting lives in:

  - :mod:`semconv_genai.uv_env` -- per-library uv env management
  - :mod:`semconv_genai.pipeline` -- mock server, Weaver live-check, pipeline

The CLI orchestrates:

  1. Starting a mock LLM server (if not already running) that simulates
     provider APIs (OpenAI, Anthropic, etc.) on localhost.
  2. Starting a Weaver live-check instance that receives OTel telemetry
     via gRPC and validates it against the semantic conventions registry.
  3. Running the scenario, which uses a real SDK to call the mock server
     and emit telemetry (spans, metrics, logs) to Weaver.
  4. Stopping Weaver, collecting results, and updating the committed per-scenario
     data file used by the checked-in status report.

Usage:
    run-scenario <library> [weaver-args...]
    run-scenario --all [--keep-going] [weaver-args...]
    run-scenario --print-ci-matrix
    python -m semconv_genai.run_scenario <library> [weaver-args...]

Requires:
    - Python 3.12+ (for mock server)
    - Network access on first run if the pinned Weaver release is not already
      installed locally or available on PATH
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys

from semconv_genai.pipeline import (
    MOCK_SERVER_PORT,
    RunScenarioError,
    run_one_library,
    start_mock_server,
    stop_process,
)
from semconv_genai.uv_env import (
    UvNotInstalledError,
    build_reference_scenario_matrix,
    list_reference_libraries,
)

logger = logging.getLogger(__name__)


def _print_available_scenarios() -> None:
    """Print the current list of runnable libraries to stderr."""
    print("Available libraries:", file=sys.stderr)
    for library in list_reference_libraries():
        print(f"  {library}", file=sys.stderr)


def _resolve_library(requested: str) -> str:
    """Resolve CLI input to a Python reference library slug."""
    libraries = set(list_reference_libraries())
    if requested in libraries:
        return requested
    raise RunScenarioError(f"Unknown library: {requested}", show_available_scenarios=True)


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        prog="run-scenario",
        description="Run reference implementation scenarios against Weaver live-check.",
    )
    parser.add_argument("library", nargs="?", help="Library slug")
    parser.add_argument("--all", action="store_true", help="Run all reference scenarios")
    parser.add_argument("--keep-going", action="store_true", help="Continue after failures when using --all")
    parser.add_argument(
        "--print-ci-matrix",
        action="store_true",
        help="Print CI matrix JSON for runnable reference scenarios",
    )
    args, extra_weaver_args = parser.parse_known_args(argv)

    if args.print_ci_matrix:
        if args.all or args.keep_going or args.library:
            parser.error("--print-ci-matrix cannot be combined with scenario execution arguments")
        if extra_weaver_args:
            parser.error("weaver arguments cannot be provided with --print-ci-matrix")
        return args, []

    if args.all and args.library:
        parser.error("library cannot be provided together with --all")
    if not args.all and not args.library:
        parser.error("library is required unless --all is provided")

    return args, extra_weaver_args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    args, extra_weaver_args = _parse_args(sys.argv[1:] if argv is None else argv)

    try:
        libraries = list_reference_libraries()
        if not libraries:
            logger.error("No reference scenarios found.")
            return 1

        if args.print_ci_matrix:
            print(json.dumps(build_reference_scenario_matrix(), separators=(",", ":")))
            return 0

        if args.all:
            selected_libraries = libraries
            logger.info("Running all reference scenarios (%s)", len(selected_libraries))
        else:
            assert args.library is not None
            selected_libraries = [_resolve_library(args.library)]
            logger.info("Scenario: %s", selected_libraries[0])

        failures: list[tuple[str, int]] = []
        mock_url = f"http://127.0.0.1:{MOCK_SERVER_PORT}"
        mock_proc: subprocess.Popen | None = None
        mock_checked = False

        try:
            from semconv_genai import MODEL_ROOT

            registry = str(MODEL_ROOT)

            for index, library in enumerate(selected_libraries, start=1):
                if args.all:
                    logger.info("[%s/%s] %s", index, len(selected_libraries), library)
                try:
                    if not mock_checked:
                        mock_proc = start_mock_server(mock_url)
                        mock_checked = True
                    run_one_library(library, extra_weaver_args, registry, mock_url)
                except RunScenarioError as e:
                    logger.error("%s: %s", library, e)
                    failures.append((library, e.exit_code))
                    if not args.all or not args.keep_going:
                        return e.exit_code
        finally:
            stop_process(mock_proc, "mock server")
    except RunScenarioError as e:
        logger.error("%s", e)
        if e.show_available_scenarios:
            _print_available_scenarios()
        return e.exit_code
    except UvNotInstalledError as e:
        logger.error("%s", e)
        return 1

    if failures:
        logger.error("Failed reference scenarios:")
        for library, exit_code in failures:
            logger.error("  %s (exit %s)", library, exit_code)
        return failures[0][1] or 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
