"""uv-based Python test environment management for reference implementations."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from semconv_genai import (
    REFERENCE_ROOT,
    SCENARIOS_DIR,
    reference_data_file,
    reference_project_dir,
    reference_scenario_file,
)

logger = logging.getLogger(__name__)


class UvNotInstalledError(RuntimeError):
    """Raised when uv is required but not installed."""


def _uv_cmd() -> str:
    """Return the platform-specific uv executable name or raise with guidance."""
    uv = shutil.which("uv.exe" if sys.platform == "win32" else "uv")
    if uv:
        return uv

    raise UvNotInstalledError(
        "uv is required to install Python test dependencies. "
        "Install it and retry: https://docs.astral.sh/uv/getting-started/installation/"
    )


def _python_executable_for_env(env_dir: Path) -> Path:
    if sys.platform == "win32":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _scrub_python_runtime_env(env: Mapping[str, str]) -> dict[str, str]:
    """Remove Python runtime variables that can poison a different interpreter."""
    scrubbed_env = dict(env)
    for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__"):
        scrubbed_env.pop(name, None)
    return scrubbed_env


def _uv_subprocess_env() -> dict[str, str]:
    """Return a clean env for uv subprocesses."""
    return _scrub_python_runtime_env(os.environ)


def ensure_python_scenario_env(library: str) -> Path:
    """Sync the per-library uv project and return its interpreter."""
    project_dir = reference_project_dir(library)
    logger.info("Syncing Python env for %s in %s", library, project_dir)
    subprocess.run(
        [_uv_cmd(), "sync", "--frozen"],
        cwd=project_dir,
        check=True,
        env=_uv_subprocess_env(),
    )
    return _python_executable_for_env(project_dir / ".venv")


def run_reference_scenario(library: str, env: dict[str, str]) -> int:
    scenario_file = reference_scenario_file(library)
    if not scenario_file.is_file():
        raise FileNotFoundError(f"Reference scenario file not found: {scenario_file}")
    python_executable = _python_executable_for_env(reference_project_dir(library) / ".venv")
    if not python_executable.is_file():
        python_executable = ensure_python_scenario_env(library)
    proc = subprocess.run([str(python_executable), str(scenario_file)], env=_scrub_python_runtime_env(env))
    return proc.returncode


def list_reference_libraries() -> list[str]:
    libraries: list[str] = []
    for scenario_dir in sorted(path for path in SCENARIOS_DIR.iterdir() if path.is_dir()):
        library = scenario_dir.name
        if (scenario_dir / "scenario.py").is_file() and (scenario_dir / "pyproject.toml").is_file():
            libraries.append(library)
    return libraries


def build_reference_scenario_matrix() -> dict[str, list[dict[str, str]]]:
    return {
        "scenario": [
            {
                "lib": library,
                "data": reference_data_file(library).relative_to(REFERENCE_ROOT).as_posix(),
            }
            for library in list_reference_libraries()
        ]
    }
