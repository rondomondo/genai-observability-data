"""Tests for the root ./god-cli script.

The script lives at the repo root and resolves src/god via sys.path injection.
These tests invoke it via importlib (same process) to avoid subprocess overhead,
dispatching through the wrapper's _dispatch() which delegates to god.cli.main().
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure the repo root is on sys.path so we can import the god-cli script directly.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "god-cli"

# Import the root CLI as a module by loading it via importlib.
# spec_from_file_location needs an explicit loader for extension-less scripts.
import importlib.machinery  # noqa: E402
import importlib.util  # noqa: E402

_loader = importlib.machinery.SourceFileLoader("root_cli", str(_SCRIPT))
_spec = importlib.util.spec_from_loader("root_cli", _loader)
assert _spec is not None
_root_cli = importlib.util.module_from_spec(_spec)
_loader.exec_module(_root_cli)


def _run(argv: list[str]) -> tuple[int, str, str]:
    """Run the root CLI with the given argv, returning (exit_code, stdout, stderr)."""
    import god.cli as _cli

    stdout_buf = StringIO()
    stderr_buf = StringIO()
    from rich.console import Console

    out_console = Console(file=stdout_buf, theme=_cli.THEME, highlight=False, force_terminal=False)
    err_console = Console(file=stderr_buf, theme=_cli.THEME, highlight=False, force_terminal=False)

    with (
        patch.object(_cli, "console", err_console),
        patch.object(_cli, "out", out_console),
    ):
        code = _root_cli._dispatch(argv)

    return code, stdout_buf.getvalue(), stderr_buf.getvalue()


def test_version_exits_zero() -> None:
    code, stdout, _ = _run(["--version"])
    assert code == 0
    assert "god" in stdout
    assert "0." in stdout


def test_help_exits_zero() -> None:
    code, _, stderr = _run(["--help"])
    assert code == 0
    assert "--traces" in stderr
    assert "--scenario" in stderr
    assert "--otlp" in stderr


def test_list_scenarios_exits_zero() -> None:
    code, _, stderr = _run(["--list-scenarios"])
    assert code == 0
    assert "cascading_failure" in stderr
    assert "quiet" in stderr


def test_generate_stdout_quiet(capsys: pytest.CaptureFixture[str]) -> None:
    """--quiet --format logfmt: logfmt lines on real stdout, nothing on Rich stderr."""
    from god.config import get_settings, reset_settings

    reset_settings()
    get_settings().default_otlp = None
    try:
        code, _rich_stdout, stderr = _run(["--traces", "10", "--quiet", "--format", "logfmt"])
    finally:
        reset_settings()
    assert code == 0
    assert stderr == ""
    real = capsys.readouterr()
    assert "level=" in real.out
    assert "service=" in real.out


def test_generate_to_file_mock(tmp_path: Path) -> None:
    code, _, _ = _run(
        [
            "--traces",
            "15",
            "--quiet",
            "--output",
            str(tmp_path),
            "--format",
            "logfmt",
        ]
    )
    assert code == 0
    logfmt = tmp_path / "logs.logfmt"
    assert logfmt.exists()
    lines = logfmt.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 0
    assert all("service=" in line for line in lines)


def test_generate_with_scenario_mock() -> None:
    code, stdout, stderr = _run(["--traces", "10", "--quiet", "--scenario", "db_incident"])
    assert code == 0
    assert stderr == ""


def test_generate_with_seed_is_deterministic() -> None:
    def _run_stdout(seed: int) -> str:
        _, stdout, _ = _run(["--traces", "5", "--quiet", "--seed", str(seed)])
        return stdout

    a = _run_stdout(99)
    b = _run_stdout(99)
    assert a == b


def test_generate_stderr_progress_not_quiet(capsys: pytest.CaptureFixture[str]) -> None:
    """Without --quiet, progress/config go to Rich stderr; logfmt data to real stdout."""
    from god.config import get_settings, reset_settings

    reset_settings()
    get_settings().default_otlp = None
    try:
        code, _rich_stdout, stderr = _run(["--traces", "5", "--format", "logfmt"])
    finally:
        reset_settings()
    assert code == 0
    real = capsys.readouterr()
    assert "level=" in real.out
    # Header panel rendered to the patched stderr console
    assert "god" in stderr


def test_generate_otlp_traces_format(tmp_path: Path) -> None:
    code, _, _ = _run(
        [
            "--traces",
            "5",
            "--quiet",
            "--output",
            str(tmp_path),
            "--format",
            "otlp_traces",
        ]
    )
    assert code == 0
    import json

    otlp = tmp_path / "traces.otlp.jsonl"
    assert otlp.exists()
    first_line = otlp.read_text(encoding="utf-8").splitlines()[0]
    obj = json.loads(first_line)
    assert "resourceSpans" in obj
