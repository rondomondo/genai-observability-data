"""Local config UI for genai-observability-data.

Serves a browser UI to edit profiles and apply them to .env files.
Run from repo root:
    uv run --directory vecstore-api python ../config_ui.py
Or if fastapi/uvicorn/pyyaml are already on PATH:
    python config_ui.py
"""

# https://platform.claude.com/docs/en/api/rate-limits


import copy
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

log = logging.getLogger("config_ui")

REPO_ROOT = Path(os.environ["REPO_ROOT"]) if "REPO_ROOT" in os.environ else Path(__file__).parent.resolve()
PROFILES_DIR = REPO_ROOT / "profiles"
ROOT_ENV = REPO_ROOT / ".env"
VECSTORE_ENV = REPO_ROOT / "vecstore-api" / ".env"
GOD_ENV = REPO_ROOT / "src" / "god" / ".env"
BACKUP_DIR = REPO_ROOT / "profiles" / ".backups"

EMBED_BACKENDS = ["mock", "sidecar", "gemini", "sentence-transformers", "google-cloud"]
EMBED_MODELS_BY_BACKEND: dict[str, list[str]] = {
    "google-cloud": ["gemini-embedding-001", "gemini-embedding-2-preview"],
    "sidecar": ["Qwen/Qwen3-Embedding-0.6B", "sentence-transformers/all-MiniLM-L6-v2"],
    "sentence-transformers": ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
    "mock": ["mock-model"],
}
EMBED_DIMS_BY_BACKEND: dict[str, int] = {
    "google-cloud": 768,
    "gemini": 768,
    "sidecar": 1024,
    "sentence-transformers": 384,
    "mock": 256,
}
AGENT_BACKENDS = ["mock", "google-cloud", "gemini", "anthropic"]
AGENT_MODELS_BY_BACKEND: dict[str, list[str]] = {
    "mock": ["mock-model"],
    "google-cloud": ["gemini-2.5-flash-lite", "gemini-2.5-flash"],
    "gemini": ["gemini-2.5-flash-lite", "gemini-2.5-flash"],
    "anthropic": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-7"],
}
AGENT_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
    "claude-haiku-4-5-20251001",
]
SUMMARISER_BACKENDS = ["mock", "anthropic", "gemini", "google-cloud"]
SUMMARISER_MODELS_BY_BACKEND: dict[str, list[str]] = {
    "mock": ["mock-model"],
    "anthropic": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-7"],
    "google-cloud": ["gemini-2.5-flash-lite", "gemini-2.5-flash"],
}
SUMMARISER_MODELS = [
    "claude-sonnet-4-6",
    "claude-opus-4-7",
    "claude-haiku-4-5-20251001",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]
LOG_FORMATS = ["console", "json", "logfmt"]

app = FastAPI(title="Config UI", docs_url=None, redoc_url=None)


# Profile I/O

def _load_profile(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_profile(path: Path, data: dict[str, Any]) -> None:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _list_profiles() -> list[dict[str, str]]:
    if not PROFILES_DIR.exists():
        return []
    profiles = []
    for p in sorted(PROFILES_DIR.glob("*.yaml")):
        data = _load_profile(p)
        profiles.append({
            "file": p.name,
            "name": data.get("name", p.stem),
            "description": data.get("description", ""),
            "editor_enabled": bool(data.get("editor_enabled", True)),
        })
    return profiles


def _active_profile_name() -> str | None:
    marker = PROFILES_DIR / ".active"
    if marker.exists():
        return marker.read_text(encoding="utf-8").strip()
    return None


def _set_active_profile(name: str) -> None:
    (PROFILES_DIR / ".active").write_text(name, encoding="utf-8")


# .env rendering

def _read_env_file(path: Path) -> dict[str, str]:
    """Parse key=value lines from an .env file, preserving comments as-is."""
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            result[key.strip()] = val.strip()
    return result


def _backup_env(path: Path) -> None:
    if not path.exists():
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"{path.name}.{ts}.bak"
    shutil.copy2(path, dest)


def _write_env_preserving_secrets(path: Path, new_vals: dict[str, str]) -> None:
    """Merge new_vals into an existing .env, preserving lines we don't own (secrets, comments)."""
    _backup_env(path)

    lines: list[str] = []
    written: set[str] = set()

    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                lines.append(line)
                continue
            if "=" in stripped:
                key = stripped.partition("=")[0].strip()
                if key in new_vals:
                    lines.append(f"{key}={new_vals[key]}")
                    written.add(key)
                    continue
            lines.append(line)

    # Append any keys not already in the file
    for key, val in new_vals.items():
        if key not in written:
            lines.append(f"{key}={val}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _credentials_to_env(profile: dict[str, Any]) -> dict[str, str]:
    """Extract provider credential env vars from the top-level credentials key."""
    c = profile.get("credentials", {})
    result: dict[str, str] = {}
    for key in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "EMBED_SIDECAR_URL"):
        val = c.get(key)
        if val:
            result[key] = str(val)
    return result


def _profile_to_root_env(profile: dict[str, Any]) -> dict[str, str]:
    g = profile.get("god", {})
    env = {
        "GOD_AGENT_MODEL": f"{g.get('agent_backend', 'google-cloud')}:{g.get('agent_model', 'gemini-2.5-flash-lite')}",
        "GOD_AGENT_FALLBACK_MODEL": str(g.get("agent_fallback_model", "")),
        "GOD_MOCK_MODE": str(g.get("mock_mode", False)).lower(),
        "GOD_DEFAULT_TRACES": str(g.get("default_traces", 100)),
        "GOD_DEFAULT_RATE": str(g.get("default_rate", 10.0)),
        "GOD_DEFAULT_BATCH_SIZE": str(g.get("default_batch_size", 50)),
        "GOD_VECSTORE_SUMMARISE_TRACES": str(g.get("vecstore_summarise_traces", True)).lower(),
        "GOD_VECSTORE_SUMMARISE_LOGS": str(g.get("vecstore_summarise_logs", False)).lower(),
        "GOD_VECSTORE_ASYNC_MODE": str(g.get("vecstore_async_mode", True)).lower(),
        "HTTP_RATE_INGEST": str(g.get("http_rate_ingest", 8)),
        "HTTP_RATE_SEARCH": str(g.get("http_rate_search", 4)),
        "HTTP_RATE_CACHE": str(g.get("http_rate_cache", 4)),
        "HTTP_RATE_ADMIN": str(g.get("http_rate_admin", 2)),
        "HTTP_RPS_INGEST": str(g.get("http_rps_ingest", 0.0)),
        "HTTP_RPS_SEARCH": str(g.get("http_rps_search", 0.0)),
        "HTTP_RPS_CACHE": str(g.get("http_rps_cache", 0.0)),
        "HTTP_RPS_ADMIN": str(g.get("http_rps_admin", 0.0)),
        "HTTP_RETRY_ATTEMPTS": str(g.get("http_retry_attempts", 3)),
        "HTTP_RETRY_BACKOFF_MIN": str(g.get("http_retry_backoff_min", 0.5)),
        "HTTP_RETRY_BACKOFF_MAX": str(g.get("http_retry_backoff_max", 8.0)),
    }
    creds = _credentials_to_env(profile)
    for key in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"):
        if key in creds:
            env[key] = creds[key]
    return env


def _profile_to_god_env(profile: dict[str, Any]) -> dict[str, str]:
    """Build env vars for src/god/.env from a profile."""
    g = profile.get("god", {})
    backend = g.get("agent_backend", "google-cloud")
    model = g.get("agent_model", "gemini-2.5-flash-lite")
    env = {
        "GOD_AGENT_MODEL": f"{backend}:{model}",
        "GOD_AGENT_BACKEND": backend,
        "GOD_AGENT_FALLBACK_MODEL": str(g.get("agent_fallback_model", "")),
        "GOD_MOCK_MODE": str(g.get("mock_mode", False)).lower(),
        "GOD_DEFAULT_TRACES": str(g.get("default_traces", 100)),
        "GOD_DEFAULT_RATE": str(g.get("default_rate", 10.0)),
        "GOD_DEFAULT_BATCH_SIZE": str(g.get("default_batch_size", 50)),
        "GOD_VECSTORE_SUMMARISE_TRACES": str(g.get("vecstore_summarise_traces", True)).lower(),
        "GOD_VECSTORE_SUMMARISE_LOGS": str(g.get("vecstore_summarise_logs", False)).lower(),
        "GOD_VECSTORE_ASYNC_MODE": str(g.get("vecstore_async_mode", True)).lower(),
        "LOG_LEVEL": str(g.get("log_level", "INFO")),
        "LOG_FORMAT": str(g.get("log_format", "console")),
    }
    creds = _credentials_to_env(profile)
    for key in ("GEMINI_API_KEY", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"):
        if key in creds:
            env[key] = creds[key]
    return env


def _profile_to_vecstore_env(profile: dict[str, Any]) -> dict[str, str]:
    v = profile.get("vecstore", {})
    env = {
        "EMBED_MODEL": f"{v.get('embed_backend', 'mock')}:{v.get('embed_model', 'gemini-embedding-001')}",
        "EMBED_DIM": str(v.get("embed_dim", 768)),
        "EMBED_WORKERS": str(v.get("embed_workers", 4)),
        "EMBED_CONCURRENCY": str(v.get("embed_concurrency", 2)),
        "EMBED_BATCH_SIZE": str(v.get("embed_batch_size", 32)),
        "EMBED_RATE_LIMIT_RPM": str(v.get("embed_rate_limit_rpm", 90)),
        "SUMMARISER_MODEL": f"{v.get('summariser_backend', 'mock')}:{v.get('summariser_model', 'gemini-2.5-flash-lite')}",
        "SUMMARISER_MOCK": str(v.get("summariser_backend", "mock") == "mock").lower(),
        "SUMMARISER_CONCURRENCY": str(v.get("summariser_concurrency", 2)),
        "SUMMARISER_RATE_LIMIT_RPM": str(v.get("summariser_rate_limit_rpm", 50)),
        "CACHE_HIT_THRESHOLD": str(v.get("cache_hit_threshold", 0.92)),
        "CACHE_TTL_SECONDS": str(v.get("cache_ttl_seconds", 3600)),
        "RATE_LIMIT_INGEST": str(v.get("rate_limit_ingest", "120/minute")),
        "RATE_LIMIT_EMBED": str(v.get("rate_limit_embed", "60/minute")),
        "RATE_LIMIT_SEARCH": str(v.get("rate_limit_search", "60/minute")),
        "RATE_LIMIT_CACHE": str(v.get("rate_limit_cache", "60/minute")),
        "RATE_LIMIT_SUMMARISE": str(v.get("rate_limit_summarise", "30/minute")),
        "LOG_LEVEL": str(v.get("log_level", "INFO")),
        "LOG_FORMAT": str(v.get("log_format", "console")),
    }
    creds = _credentials_to_env(profile)
    for key in ("GEMINI_API_KEY", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "EMBED_SIDECAR_URL"):
        if key in creds:
            env[key] = creds[key]
    return env


# API routes

@app.get("/api/profiles")
async def list_profiles() -> JSONResponse:
    return JSONResponse({"profiles": _list_profiles(), "active": _active_profile_name()})


@app.get("/api/profiles/{filename}")
async def get_profile(filename: str) -> JSONResponse:
    path = PROFILES_DIR / filename
    if not path.exists() or not filename.endswith(".yaml"):
        raise HTTPException(status_code=404, detail="Profile not found")
    return JSONResponse(_load_profile(path))


@app.post("/api/profiles/{filename}")
async def save_profile(filename: str, request: Request) -> JSONResponse:
    if not filename.endswith(".yaml"):
        raise HTTPException(status_code=400, detail="Filename must end in .yaml")
    body = await request.json()
    path = PROFILES_DIR / filename
    _save_profile(path, body)
    return JSONResponse({"ok": True, "file": filename})


@app.post("/api/profiles/{filename}/apply")
async def apply_profile(filename: str) -> JSONResponse:
    path = PROFILES_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Profile not found")
    profile = _load_profile(path)
    root_vals = _profile_to_root_env(profile)
    vecstore_vals = _profile_to_vecstore_env(profile)
    god_vals = _profile_to_god_env(profile)
    _write_env_preserving_secrets(ROOT_ENV, root_vals)
    _write_env_preserving_secrets(VECSTORE_ENV, vecstore_vals)
    _write_env_preserving_secrets(GOD_ENV, god_vals)
    _set_active_profile(filename)
    return JSONResponse({
        "ok": True, "applied": filename,
        "root_env": str(ROOT_ENV), "vecstore_env": str(VECSTORE_ENV), "god_env": str(GOD_ENV),
    })


@app.delete("/api/profiles/{filename}")
async def delete_profile(filename: str) -> JSONResponse:
    path = PROFILES_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Profile not found")
    active = _active_profile_name()
    if active == filename:
        raise HTTPException(status_code=400, detail="Cannot delete the active profile")
    path.unlink()
    return JSONResponse({"ok": True})


@app.get("/api/meta/vecstore-key")
async def vecstore_key() -> JSONResponse:
    """Read API_KEY from vecstore-api/.env for use by the topology UI."""
    key = _read_env_file(VECSTORE_ENV).get("API_KEY", "")
    return JSONResponse({"api_key": key})


@app.get("/api/meta/options")
async def meta_options() -> JSONResponse:
    return JSONResponse({
        "embed_backends": EMBED_BACKENDS,
        "embed_models_by_backend": EMBED_MODELS_BY_BACKEND,
        "embed_dims_by_backend": EMBED_DIMS_BY_BACKEND,
        "agent_backends": AGENT_BACKENDS,
        "agent_models_by_backend": AGENT_MODELS_BY_BACKEND,
        "agent_models": AGENT_MODELS,
        "summariser_backends": SUMMARISER_BACKENDS,
        "summariser_models_by_backend": SUMMARISER_MODELS_BY_BACKEND,
        "summariser_models": SUMMARISER_MODELS,
        "log_levels": LOG_LEVELS,
        "log_formats": LOG_FORMATS,
    })


# HTML UI

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Config UI</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  /* Theme tokens - dark (default) */
  :root {
    --bg:         #0f1117;
    --bg-panel:   #1a1d27;
    --bg-card:    #252a3a;
    --border:     #2d3748;
    --border-hover: #4a5568;
    --text:       #e2e8f0;
    --text-muted: #a0aec0;
    --text-dim:   #718096;
    --text-hint:  #4a5568;
    --accent:     #63b3ed;
    --focus:      #4299e1;
    --green-bg:   #276749;
    --green-text: #9ae6b4;
    --green-dim:  #68d391;
    --active-bg:  #1c2e25;
    --active-bdr: #48bb78;
    --ok-bg:      #1c2e25;
    --ok-text:    #68d391;
    --err-bg:     #2d1515;
    --err-text:   #fc8181;
    --info-bg:    #1a2636;
    --info-text:  #63b3ed;
    --slider-off: #2d3748;
    --slider-dot: #718096;
    --theme-icon: "☀";
  }

  /* Light theme */
  body.light {
    --bg:         #f0f2f7;
    --bg-panel:   #ffffff;
    --bg-card:    #f7f9fc;
    --border:     #cbd5e0;
    --border-hover: #a0aec0;
    --text:       #1a202c;
    --text-muted: #4a5568;
    --text-dim:   #718096;
    --text-hint:  #a0aec0;
    --accent:     #2b6cb0;
    --focus:      #3182ce;
    --green-bg:   #c6f6d5;
    --green-text: #276749;
    --green-dim:  #276749;
    --active-bg:  #e6ffed;
    --active-bdr: #38a169;
    --ok-bg:      #e6ffed;
    --ok-text:    #276749;
    --err-bg:     #fff5f5;
    --err-text:   #c53030;
    --info-bg:    #ebf8ff;
    --info-text:  #2b6cb0;
    --slider-off: #cbd5e0;
    --slider-dot: #a0aec0;
    --theme-icon: "☽";
  }

  body {
    font-family: system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    transition: background .2s, color .2s;
  }
  a { color: var(--accent); }

  /* Centred 60% wrapper */
  .page-wrap {
    max-width: 1400px;
    width: 60%;
    min-width: 320px;
    margin: 0 auto;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  .btn-icon {
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text-muted);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 13px;
    cursor: pointer;
    transition: border-color .15s;
    white-space: nowrap;
  }
  .btn-icon:hover { border-color: var(--border-hover); color: var(--text); }

  /* Help modal */
  .modal-backdrop {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,.6);
    z-index: 100;
    align-items: center;
    justify-content: center;
  }
  .modal-backdrop.open { display: flex; }
  .modal {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    box-shadow: 0 8px 40px rgba(0,0,0,.5);
    width: min(780px, 92vw);
    max-height: 82vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .modal-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 20px 14px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .modal-header h2 { font-size: 15px; font-weight: 700; margin: 0; }
  .modal-close {
    background: transparent;
    border: none;
    color: var(--text-muted);
    font-size: 20px;
    cursor: pointer;
    line-height: 1;
    padding: 0 4px;
  }
  .modal-close:hover { color: var(--text); }
  .modal-body {
    padding: 20px;
    overflow-y: auto;
    font-size: 13px;
    line-height: 1.65;
    color: var(--text-muted);
  }
  .modal-body h3 { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; color: var(--accent); margin: 18px 0 8px; }
  .modal-body h3:first-child { margin-top: 0; }
  .modal-body p { margin-bottom: 10px; }
  .modal-body code { background: var(--bg-card); border: 1px solid var(--border); border-radius: 3px; padding: 1px 5px; font-size: 12px; color: var(--text); font-family: monospace; }
  .modal-body pre { background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 12px 14px; overflow-x: auto; margin-bottom: 12px; }
  .modal-body pre code { background: none; border: none; padding: 0; }
  .modal-body table { width: 100%; border-collapse: collapse; margin-bottom: 12px; font-size: 12px; }
  .modal-body th { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--border); color: var(--text-dim); font-weight: 600; }
  .modal-body td { padding: 5px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
  .modal-body td:first-child { font-family: monospace; font-size: 11px; color: var(--text); white-space: nowrap; }

  .layout {
    display: grid;
    grid-template-columns: 220px 1fr;
    flex: 1;
    border: 0px solid var(--border);
    border-radius: 10px;
    overflow: visible;
    margin-bottom: 16px;
    align-items: start;
  }

  /* Sidebar */
  .sidebar {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    box-shadow: inset 0 1px 3px rgba(0,0,0,.35), inset 0 0 0 1px rgba(0,0,0,.15);
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    align-self: start;
    margin: 8px;
  }
  body.light .sidebar { box-shadow: inset 0 1px 3px rgba(0,0,0,.08), inset 0 0 0 1px rgba(0,0,0,.06); }
  .sidebar h1 { font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; color: var(--text-dim); padding-bottom: 8px; border-bottom: 1px solid var(--border); margin-bottom: 4px; }
  #profile-list { display: flex; flex-direction: column; gap: 6px; }
  .profile-details { background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; transition: border-color .15s; }
  .profile-details:hover { border-color: var(--border-hover); }
  .profile-details.active  { border-color: var(--active-bdr); background: var(--active-bg); }
  .profile-details.editing { border-color: var(--focus); background: var(--info-bg); }
  .profile-details summary { list-style: none; display: flex; align-items: center; gap: 8px; padding: 8px 12px; cursor: default; user-select: none; }
  .profile-details summary::-webkit-details-marker { display: none; }
  .profile-details .pname { font-size: 13px; font-weight: 600; flex: 1; }
  .profile-summary-cb { cursor: pointer; flex-shrink: 0; }
  .profile-body { padding: 4px 12px 10px; cursor: pointer; }
  .profile-body .pdesc { font-size: 11px; color: var(--text-dim); line-height: 1.4; }
  .profile-body .pbadge { display: inline-block; margin-top: 4px; font-size: 10px; padding: 1px 6px; border-radius: 10px; background: var(--green-bg); color: var(--green-text); }
  .profile-body .pbadge-editing { display: inline-block; margin-top: 4px; font-size: 10px; padding: 1px 6px; border-radius: 10px; background: var(--info-bg); color: var(--info-text); border: 1px solid var(--focus); }
  .btn-new { background: #2b6cb0; color: #fff; border: none; border-radius: 6px; padding: 8px 12px; font-size: 12px; cursor: pointer; width: 100%; margin-top: 8px; }
  .btn-new:hover { background: #3182ce; }

  /* Main */
  .main { padding: 24px 28px; overflow-y: auto; background: var(--bg-panel); border: 1px solid var(--border); border-radius: 10px; box-shadow: inset 0 1px 3px rgba(0,0,0,.35), inset 0 0 0 1px rgba(0,0,0,.15); margin: 8px 8px 8px 0; user-select: none; }
  body.light .main { box-shadow: inset 0 1px 3px rgba(0,0,0,.08), inset 0 0 0 1px rgba(0,0,0,.06); }
  .title-bar { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 4px; gap: 12px; }
  .title-bar h2 { font-size: 18px; font-weight: 700; margin: 0; flex: 1; }
  .title-bar-btns { display: flex; gap: 6px; flex-shrink: 0; padding-top: 2px; }
  .main .subtitle { font-size: 13px; color: var(--text-dim); margin-top: 4px; margin-bottom: 12px; }

  /* Sections */
  .section { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 20px; overflow: hidden; }
  .section summary { list-style: none; display: flex; align-items: center; gap: 8px; padding: 14px 20px; cursor: pointer; user-select: none; }
  .section summary::-webkit-details-marker { display: none; }
  .section summary h3 { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; color: var(--accent); margin: 0; flex: 1; }
  .section summary::after { content: '▾'; font-size: 18px; color: var(--accent); transition: transform .2s; }
  .section:not([open]) summary::after { transform: rotate(-90deg); }
  .section-body { padding: 0 20px 20px; }
  .main input, .main select, .main textarea { user-select: text; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }

  /* Fields */
  .field { display: flex; flex-direction: column; gap: 4px; }
  .field label { font-size: 11px; color: var(--text-muted); font-weight: 500; }
  .field input, .field select { background: var(--bg-card); border: 1px solid var(--border); border-radius: 4px; color: var(--text); padding: 6px 8px; font-size: 13px; width: 100%; }
  .field input:focus, .field select:focus { outline: none; border-color: var(--focus); }
  .field input[type=range] { padding: 0; cursor: pointer; }
  .range-row { display: flex; align-items: center; gap: 8px; }
  .range-val { font-size: 12px; color: var(--text); min-width: 36px; text-align: right; }
  .field .hint { font-size: 10px; color: var(--text-hint); margin-top: 1px; }

  /* Tooltips on labels with data-tip */
  .field label[data-tip] { display: inline-flex; align-items: center; gap: 4px; cursor: default; position: relative; width: fit-content; }
  .field label[data-tip]::after {
    content: attr(data-tip);
    position: absolute;
    bottom: calc(100% + 5px);
    left: 0;
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 11px;
    font-weight: 400;
    line-height: 1.5;
    padding: 5px 9px;
    border-radius: 5px;
    white-space: pre-wrap;
    width: max-content;
    max-width: 240px;
    box-shadow: 0 3px 10px rgba(0,0,0,.25);
    pointer-events: none;
    opacity: 0;
    transform: translateY(3px);
    transition: opacity .15s, transform .15s;
    z-index: 10;
  }
  .field label[data-tip]:hover::after { opacity: 1; transform: translateY(0); }
  .field label[data-tip] .tip-dot {
    display: inline-block;
    width: 12px; height: 12px;
    border-radius: 50%;
    border: 1px solid var(--text-hint);
    color: var(--text-hint);
    font-size: 9px;
    line-height: 11px;
    text-align: center;
    flex-shrink: 0;
  }
  .field label[data-tip]:hover .tip-dot { border-color: var(--text-muted); color: var(--text-muted); }

  /* Toggle (boolean switch) */
  .toggle-row { display: flex; align-items: center; justify-content: space-between; }
  .toggle-row label { font-size: 12px; color: var(--text-muted); }
  .toggle { position: relative; display: inline-block; width: 36px; height: 20px; }
  .toggle input { opacity: 0; width: 0; height: 0; }
  .slider { position: absolute; cursor: pointer; inset: 0; background: var(--slider-off); border-radius: 20px; transition: .2s; }
  .slider::before { position: absolute; content: ""; height: 14px; width: 14px; left: 3px; bottom: 3px; background: var(--slider-dot); border-radius: 50%; transition: .2s; }
  input:checked + .slider { background: var(--green-bg); }
  input:checked + .slider::before { transform: translateX(16px); background: var(--green-dim); }

  /* Action bar */
  .actions { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 20px; }
  .btn { border: none; border-radius: 6px; padding: 8px 16px; font-size: 13px; font-weight: 600; cursor: pointer; transition: opacity .15s; }
  .btn:disabled { opacity: .4; cursor: default; }
  .btn-save { background: #2b6cb0; color: #fff; }
  .btn-save:hover:not(:disabled) { background: #3182ce; }
  .btn-apply { background: var(--green-bg); color: var(--green-text); }
  .btn-apply:hover:not(:disabled) { background: #2f855a; }
  .btn-delete { background: transparent; border: 1px solid #c53030; color: #fc8181; }
  .btn-delete:hover:not(:disabled) { background: #742a2a22; }
  .btn-clone { background: #553c9a; color: #d6bcfa; }
  .btn-clone:hover:not(:disabled) { background: #6b46c1; }
  .status { font-size: 12px; padding: 6px 12px; border-radius: 4px; flex: 1; }
  .status.ok  { background: var(--ok-bg);   color: var(--ok-text); }
  .status.err { background: var(--err-bg);  color: var(--err-text); }
  .status.info{ background: var(--info-bg); color: var(--info-text); }

  /* Name/desc edit row */
  .meta-row { display: grid; grid-template-columns: 180px 1fr; gap: 14px; margin-bottom: 20px; }

  /* Empty state */
  .empty { text-align: center; padding: 60px 0; color: var(--text-hint); }
  .empty .big { font-size: 40px; margin-bottom: 12px; }

  /* Topology import */
  .topo-drop {
    border: 2px dashed var(--border);
    border-radius: 8px;
    padding: 28px 20px;
    text-align: center;
    color: var(--text-dim);
    font-size: 13px;
    cursor: pointer;
    transition: border-color .15s, background .15s;
    margin-bottom: 14px;
  }
  .topo-drop.drag-over { border-color: var(--focus); background: var(--info-bg); color: var(--info-text); }
  .topo-drop input[type=file] { display: none; }
  .topo-preview { background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 14px; margin-bottom: 14px; font-size: 12px; }
  .topo-preview h4 { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--accent); margin-bottom: 10px; }
  .topo-svc-table { width: 100%; border-collapse: collapse; font-size: 11px; }
  .topo-svc-table th { text-align: left; padding: 3px 6px; border-bottom: 1px solid var(--border); color: var(--text-dim); font-weight: 600; }
  .topo-svc-table td { padding: 3px 6px; border-bottom: 1px solid var(--border); color: var(--text-muted); font-family: monospace; }
  .topo-list { display: flex; flex-direction: column; gap: 6px; margin-bottom: 10px; }
  .topo-item { background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; display: flex; align-items: center; gap: 8px; }
  .topo-item .tname { font-size: 12px; font-weight: 600; flex: 1; }
  .topo-item .tmeta { font-size: 10px; color: var(--text-dim); }
  .topo-item .btn-del-topo { background: transparent; border: 1px solid #c53030; color: #fc8181; border-radius: 4px; padding: 2px 8px; font-size: 11px; cursor: pointer; }

  /* Responsive - tablet */
  @media (max-width: 900px) {
    .page-wrap { width: 90%; }
  }

  /* Responsive - mobile */
  @media (max-width: 640px) {
    .page-wrap { width: 100%; padding: 0 8px; }
    .layout { grid-template-columns: 1fr; border-radius: 6px; }
    .sidebar { border-radius: 8px; max-height: 260px; overflow-y: auto; }
    .main { padding: 16px; }
    .grid2, .grid3 { grid-template-columns: 1fr; }
    .meta-row { grid-template-columns: 1fr; }
    .actions { flex-direction: column; align-items: stretch; }
    .actions .btn, .actions .status { width: 100%; text-align: center; }
  }
</style>
</head>
<body>

<div class="modal-backdrop" id="help-modal" onclick="if(event.target===this)closeHelp()">
  <div class="modal" role="dialog" aria-modal="true" aria-labelledby="help-title">
    <div class="modal-header">
      <h2 id="help-title">Config UI - Help &amp; Rate Limiting Guide</h2>
      <button class="modal-close" onclick="closeHelp()" title="Close">&times;</button>
    </div>
    <div class="modal-body">
      <h3>Profiles</h3>
      <p>A profile is a named YAML file in <code>profiles/</code>. Edit values here and click <strong>Save</strong> to write the YAML, then <strong>Apply to .env</strong> to merge the values into <code>.env</code> and <code>vecstore-api/.env</code>. Secrets already in those files are preserved. Restart services after applying.</p>

      <h3>Two rate-limiting layers</h3>
      <p>Rate limiting spans two boundaries in this stack:</p>
      <table>
        <tr><th>Layer</th><th>What it controls</th><th>Where configured</th></tr>
        <tr><td>Egress (god client)</td><td>Outbound calls to Gemini / Anthropic</td><td>HTTP_RPS_* / HTTP_RATE_* in root .env</td></tr>
        <tr><td>Ingress (vecstore-api)</td><td>Inbound HTTP traffic from god / clients</td><td>RATE_LIMIT_* in vecstore-api/.env</td></tr>
      </table>

      <h3>Egress: token bucket + semaphore</h3>
      <p>Each named operation (<code>ingest</code>, <code>search</code>, <code>cache</code>, <code>admin</code>) has two independent controls that both apply:</p>
      <table>
        <tr><th>Setting</th><th>Default</th><th>What it does</th></tr>
        <tr><td>HTTP_RPS_INGEST</td><td>0.0</td><td>Token-bucket cap in req/s (0 = unlimited)</td></tr>
        <tr><td>HTTP_RATE_INGEST</td><td>8</td><td>Max simultaneous in-flight requests</td></tr>
        <tr><td>HTTP_RETRY_ATTEMPTS</td><td>3</td><td>Total attempts including first</td></tr>
        <tr><td>HTTP_RETRY_BACKOFF_MIN</td><td>0.5 s</td><td>Min backoff between retries</td></tr>
        <tr><td>HTTP_RETRY_BACKOFF_MAX</td><td>8.0 s</td><td>Max backoff between retries</td></tr>
      </table>
      <p>On a 429 from upstream, an extra token is drained from the bucket (soft feedback) and tenacity retries with exponential backoff. Each retry re-acquires the token - retries do not bypass the rate limit.</p>

      <h3>Egress: matching your quota tier</h3>
      <p><strong>Rule of thumb:</strong> set RPS to 80-85% of your quota limit. Provider windows are often sub-minute, so headroom prevents cascading retries.</p>
      <table>
        <tr><th>Tier</th><th>HTTP_RPS_INGEST</th><th>HTTP_RPS_SEARCH</th></tr>
        <tr><td>Gemini free (~15 RPM)</td><td>0.2</td><td>0.2</td></tr>
        <tr><td>Gemini paid Tier 1 (1000 RPM)</td><td>14.0</td><td>8.0</td></tr>
      </table>

      <h3>Ingress: vecstore-api per-IP limits</h3>
      <p>Uses slowapi / limits syntax: <code>"120/minute"</code>, <code>"10/second"</code>. Limits are per remote IP. Defaults are permissive for local dev.</p>
      <table>
        <tr><th>Setting</th><th>Default</th><th>Endpoint group</th></tr>
        <tr><td>RATE_LIMIT_INGEST</td><td>120/minute</td><td>/ingest/*</td></tr>
        <tr><td>RATE_LIMIT_EMBED</td><td>60/minute</td><td>/embed, /embed/batch</td></tr>
        <tr><td>RATE_LIMIT_SEARCH</td><td>60/minute</td><td>/search</td></tr>
        <tr><td>RATE_LIMIT_CACHE</td><td>60/minute</td><td>/cache/query, /cache</td></tr>
        <tr><td>RATE_LIMIT_SUMMARISE</td><td>30/minute</td><td>/summarise/trace, /summarise/log</td></tr>
      </table>
      <p><strong>Note:</strong> buckets are in-process. Multiple replicas or processes multiply the effective limit. /health and /status are always unlimited.</p>

      <h3>Embedding (vecstore-api to Gemini)</h3>
      <p><code>EMBED_RATE_LIMIT_RPM</code> (default 90) caps outbound Gemini embedding calls independently of the god client limits. Set to ~90% of your Gemini embed quota RPM.</p>

      <h3>Troubleshooting</h3>
      <p><strong>429s from Gemini despite RPS limits:</strong> confirm .env is being read, remember limits are per-process (two processes = double the rate), and consider that provider windows may be sub-minute - raise HTTP_RETRY_BACKOFF_MAX to 60+ so retries wait out the window.</p>
      <p><strong>429s from vecstore-api:</strong> check the Retry-After header. If all traffic comes from one IP in local dev, raise RATE_LIMIT_* freely - you are the sole client.</p>
      <p><strong>Token bucket has no effect:</strong> all HTTP_RPS_* default to 0.0 (disabled). Set a non-zero value to activate.</p>
    </div>
  </div>
</div>

<div class="modal-backdrop" id="topo-modal" onclick="if(event.target===this)closeTopoImport()">
  <div class="modal" role="dialog" aria-modal="true" aria-labelledby="topo-modal-title">
    <div class="modal-header">
      <h2 id="topo-modal-title">Import topology from OTel trace file</h2>
      <button class="modal-close" onclick="closeTopoImport()" title="Close">&times;</button>
    </div>
    <div class="modal-body" id="topo-modal-body">
      <div class="topo-drop" id="topo-drop-zone"
           onclick="document.getElementById('topo-file-input').click()"
           ondragover="topoOnDragOver(event)" ondragleave="topoOnDragLeave(event)" ondrop="topoOnDrop(event)">
        <input type="file" id="topo-file-input" accept=".json,.jsonl" onchange="topoOnFileSelect(event)">
        <div style="font-size:28px;margin-bottom:8px">&#8659;</div>
        Drop an OTel JSON trace file here, or click to browse
        <div style="font-size:11px;margin-top:6px;color:var(--text-hint)">Supports OTLP JSON and JSONL (resourceSpans format)</div>
      </div>
      <div id="topo-preview-area"></div>
    </div>
  </div>
</div>

<div class="page-wrap">
  <div class="layout">
    <aside class="sidebar">
      <h1>Profiles</h1>
      <div id="profile-list"></div>
      <button class="btn-new" onclick="newProfile()">+ New profile</button>
      <div style="margin-top:16px;padding-top:12px;border-top:1px solid var(--border)">
        <h1 style="margin-bottom:8px">Topologies</h1>
        <div id="topo-list"><p style="font-size:12px;color:#4a5568">Loading...</p></div>
        <button class="btn-new" style="margin-top:8px" onclick="openTopoImport()">+ Import topology</button>
      </div>
    </aside>
    <main class="main" id="main" ondblclick="toggleAllSections(event)" ontouchend="toggleAllSections(event)">
      <div class="empty"><div class="big">&#9881;</div><p>Select a profile to edit</p></div>
    </main>
  </div>
</div>

<script>
const API = "";
let state = { profiles: [], active: null, options: {}, current: null, dirty: false };
let _vecstoreKey = '';

function _topoHeaders(extra) {
  const h = { 'X-API-Key': _vecstoreKey, ...extra };
  return h;
}

async function init() {
  const [pl, opts, keyData] = await Promise.all([
    fetch(`${API}/api/profiles`).then(r => r.json()),
    fetch(`${API}/api/meta/options`).then(r => r.json()),
    fetch(`${API}/api/meta/vecstore-key`).then(r => r.json()).catch(() => ({})),
  ]);
  state.profiles = pl.profiles;
  state.active = pl.active;
  state.options = opts;
  _vecstoreKey = keyData.api_key || '';
  renderSidebar();
  _loadTopologies();
  if (state.active) {
    const match = state.profiles.find(p => p.file === state.active);
    if (match) loadProfile(match.file);
  }
}

function renderSidebar() {
  const el = document.getElementById("profile-list");
  if (!state.profiles.length) { el.innerHTML = '<p style="font-size:12px;color:#4a5568">No profiles yet</p>'; return; }
  const sorted = [...state.profiles].sort((a, b) => {
    const ae = a.editor_enabled !== false;
    const be = b.editor_enabled !== false;
    return ae === be ? 0 : ae ? -1 : 1;
  });
  el.innerHTML = sorted.map(p => {
    const isActive  = p.file === state.active;
    const isEditing = p.file === state.currentFile && !isActive;
    const cls       = isActive ? 'active' : isEditing ? 'editing' : '';
    const enabled   = p.editor_enabled !== false;
    const openAttr  = enabled ? ' open' : '';
    const cbChecked = enabled ? ' checked' : '';
    return `
    <details class="profile-details ${cls}"${openAttr}>
      <summary>
        <input class="profile-summary-cb" type="checkbox"${cbChecked}
               onclick="event.stopPropagation();toggleEditorEnabled('${p.file}',this.checked)"
               title="Show this profile in editor">
        <span class="pname">${esc(p.name)}</span>
      </summary>
      <div class="profile-body" onclick="loadProfile('${p.file}')">
        <div class="pdesc">${esc(p.description)}</div>
        ${isActive ? '<span class="pbadge">active</span>' : isEditing ? '<span class="pbadge-editing">editing</span>' : ''}
      </div>
    </details>`;
  }).join('');
}

async function toggleEditorEnabled(filename, enabled) {
  const data = await fetch(`${API}/api/profiles/${filename}`).then(r => r.json());
  data.editor_enabled = enabled;
  await fetch(`${API}/api/profiles/${filename}`, {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(data)
  });
  const pl = await fetch(`${API}/api/profiles`).then(r => r.json());
  state.profiles = pl.profiles;
  state.active   = pl.active;
  renderSidebar();
}

async function loadProfile(filename) {
  state.current = await fetch(`${API}/api/profiles/${filename}`).then(r => r.json());
  state.currentFile = filename;
  state.dirty = false;
  renderSidebar();
  renderEditor();
}

function renderEditor() {
  const p = state.current;
  const o = state.options;
  const g = Object.assign({ agent_backend: 'google-cloud', log_level: 'INFO', log_format: 'console' }, p.god || {});
  const _v = p.vecstore || {};
  // migrate legacy summariser_mock bool to summariser_backend string
  const v = Object.assign({
    summariser_concurrency: 2,
    summariser_rate_limit_rpm: 50,
  }, _v, {
    summariser_backend: _v.summariser_backend || (_v.summariser_mock ? 'mock' : 'anthropic'),
  });
  const c = p.credentials || {};
  const isActive = state.currentFile === state.active;

  document.getElementById("main").innerHTML = `
    <h2 id="ed-name-display">${esc(p.name || state.currentFile)}</h2>
    <p class="subtitle" id="ed-desc-display">${esc(p.description || "")}</p>

    <div class="actions">
      <button class="btn btn-save" onclick="saveProfile()">Save</button>
      <button class="btn btn-apply" onclick="applyProfile()">Apply to .env</button>
      <button class="btn btn-clone" onclick="cloneProfile()">Clone</button>
      ${!isActive ? `<button class="btn btn-delete" onclick="deleteProfile()">Delete</button>` : ''}
      <span class="status info" id="status-bar">Editing: ${esc(state.currentFile)}</span>
      <button class="btn-icon" onclick="openHelp()" title="Help &amp; rate limiting guide">? Help</button>
      <button class="btn-icon" id="theme-btn" onclick="toggleTheme()" title="Toggle light/dark mode">&#9789; Dark</button>
    </div>

    <div class="meta-row">
      <div class="field"><label>Profile name</label><input id="f-name" value="${esc(p.name || '')}" oninput="patch('name',this.value)"></div>
      <div class="field"><label>Description</label><input id="f-desc" value="${esc(p.description || '')}" oninput="patch('description',this.value)"></div>
    </div>

    ${sec('god-agent', 'God agent', `
      <div class="grid3">
        <div class="field">
          <label data-tip="LLM provider for the god agent. mock = no API key needed; google-cloud = Vertex AI; gemini = Gemini API; anthropic = Claude models.">Backend<span class="tip-dot">i</span></label>
          <select id="f-agent-backend" onchange="onAgentBackendChange(this.value)">
            ${o.agent_backends.map(b => `<option ${b===g.agent_backend?'selected':''}>${b}</option>`).join('')}
          </select>
        </div>
        <div class="field">
          <label>Agent model</label>
          <select id="f-agent-model" onchange="patch('g.agent_model',this.value)">
            ${(o.agent_models_by_backend[g.agent_backend] || o.agent_models).map(m => `<option ${m===g.agent_model?'selected':''}>${esc(m)}</option>`).join('')}
          </select>
        </div>
        <div class="field">
          <label>Fallback model</label>
          <select id="f-agent-fallback-model" onchange="patch('g.agent_fallback_model',this.value)">
            ${(o.agent_models_by_backend[g.agent_backend] || o.agent_models).map(m => `<option ${m===g.agent_model?'selected':''}>${esc(m)}</option>`).join('')}
          </select>
        </div>
      </div>
      <div style="margin-top:14px">
        ${tog('Mock mode (no API key)', 'g.mock_mode', g.mock_mode)}
      </div>
      <div class="grid3" style="margin-top:14px">
        ${num('Default traces', 'g.default_traces', g.default_traces, 1, 100000, 1, 'Total OTel traces to generate per run when not overridden on the CLI.')}
        ${num('Default rate (req/s)', 'g.default_rate', g.default_rate, 0.1, 100, 0.1, 'Target ingest rate in traces/second. Actual throughput is also bounded by the egress concurrency and RPS caps below.')}
        ${num('Batch size', 'g.default_batch_size', g.default_batch_size, 1, 500, 1, 'Number of spans bundled into each ingest POST. Larger batches are more efficient but hit the ingest rate limit faster.')}
      </div>
      <div style="margin-top:14px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px">
        ${tog('Summarise traces', 'g.vecstore_summarise_traces', g.vecstore_summarise_traces)}
        ${tog('Summarise logs', 'g.vecstore_summarise_logs', g.vecstore_summarise_logs)}
        ${tog('Async mode', 'g.vecstore_async_mode', g.vecstore_async_mode)}
      </div>
    `)}

    ${sec('egress-rate', 'Egress rate limiting (god HTTP client)', `
      <p style="font-size:11px;color:#718096;margin-bottom:12px">Concurrency = max in-flight requests. RPS = token-bucket cap (0 = unlimited).</p>
      <div class="grid2">
        ${rng('Ingest concurrency', 'g.http_rate_ingest', g.http_rate_ingest, 1, 20, 1, false, 'Max simultaneous ingest requests held open at once (asyncio.Semaphore).')}
        ${rng('Ingest RPS cap', 'g.http_rps_ingest', g.http_rps_ingest, 0, 20, 0.5, true, 'Token-bucket throughput cap for ingest. 0 = unlimited. Set to ~85% of your upstream RPM / 60.')}
        ${rng('Search concurrency', 'g.http_rate_search', g.http_rate_search, 1, 20, 1, false, 'Max simultaneous search requests in flight.')}
        ${rng('Search RPS cap', 'g.http_rps_search', g.http_rps_search, 0, 20, 0.5, true, 'Token-bucket cap for search. 0 = unlimited.')}
        ${rng('Cache concurrency', 'g.http_rate_cache', g.http_rate_cache, 1, 20, 1, false, 'Max simultaneous cache requests in flight.')}
        ${rng('Cache RPS cap', 'g.http_rps_cache', g.http_rps_cache, 0, 20, 0.5, true, 'Token-bucket cap for cache. 0 = unlimited.')}
        ${rng('Admin concurrency', 'g.http_rate_admin', g.http_rate_admin, 1, 10, 1, false, 'Max simultaneous admin/status requests in flight.')}
        ${rng('Admin RPS cap', 'g.http_rps_admin', g.http_rps_admin, 0, 10, 0.5, true, 'Token-bucket cap for admin. 0 = unlimited.')}
      </div>
      <div class="grid3" style="margin-top:14px">
        ${rng('Retry attempts', 'g.http_retry_attempts', g.http_retry_attempts, 0, 10, 1, false, 'Total attempts per request including the first. Retries on 429, 5xx, and transient network errors.')}
        ${num('Backoff min (s)', 'g.http_retry_backoff_min', g.http_retry_backoff_min, 0.1, 10, 0.1, 'Minimum wait between retries. Actual wait grows exponentially up to the max.')}
        ${num('Backoff max (s)', 'g.http_retry_backoff_max', g.http_retry_backoff_max, 1, 60, 0.5, 'Maximum wait between retries. Set to >=60 if upstream quotas reset on a per-minute window.')}
      </div>
    `)}

    ${sec('embedding', 'Embedding', `
      <div class="grid2">
        <div class="field">
          <label data-tip="Where embeddings are computed. mock = no API key needed; gemini = Gemini API; sidecar = local inference server; sentence-transformers = local Python library.">Backend<span class="tip-dot">i</span></label>
          <select id="f-embed-backend" onchange="onBackendChange(this.value)">
            ${o.embed_backends.map(b => `<option ${b===v.embed_backend?'selected':''}>${b}</option>`).join('')}
          </select>
        </div>
        <div class="field">
          <label>Model</label>
          <select id="f-embed-model" onchange="patch('vecstore.embed_model',this.value)">
            ${(o.embed_models_by_backend[v.embed_backend] || [v.embed_model]).map(m => `<option ${m===v.embed_model?'selected':''}>${esc(m)}</option>`).join('')}
          </select>
        </div>
        ${rng('Embedding dimension', 'v.embed_dim', v.embed_dim, 64, 3072, 64, false, 'Vector size produced by the model. Must match what the model actually outputs - changing this requires re-indexing.')}
        ${rng('Workers', 'v.embed_workers', v.embed_workers, 1, 16, 1, false, 'Number of parallel embedding worker coroutines. Increase for higher throughput if the backend supports it.')}
        ${rng('Concurrency', 'v.embed_concurrency', v.embed_concurrency, 1, 16, 1, false, 'Max simultaneous in-flight requests per worker to the embedding backend.')}
        ${rng('Batch size', 'v.embed_batch_size', v.embed_batch_size, 1, 128, 1, false, 'Texts sent per embedding API call. Larger = fewer round trips but higher per-call latency.')}
        ${rng('Rate limit RPM (Gemini)', 'v.embed_rate_limit_rpm', v.embed_rate_limit_rpm, 1, 300, 1, false, 'Outbound token-bucket cap for Gemini embedding calls. Only active when backend=gemini. Set to ~90% of your Gemini embed quota RPM.')}
      </div>
    `)}

    ${sec('summariser', 'Summariser', `
      <div class="grid2">
        <div class="field">
          <label data-tip="LLM provider for summarisation. mock = no API key needed; anthropic = Claude models; gemini = Gemini models.">Backend<span class="tip-dot">i</span></label>
          <select id="f-summariser-backend" onchange="onSummariserBackendChange(this.value)">
            ${o.summariser_backends.map(b => `<option ${b===v.summariser_backend?'selected':''}>${b}</option>`).join('')}
          </select>
        </div>
        <div class="field">
          <label>Model</label>
          <select id="f-summariser-model" onchange="patch('vecstore.summariser_model',this.value)">
            ${(o.summariser_models_by_backend[v.summariser_backend] || [v.summariser_model]).map(m => `<option ${m===v.summariser_model?'selected':''}>${esc(m)}</option>`).join('')}
          </select>
        </div>
        ${rng('Concurrency', 'v.summariser_concurrency', v.summariser_concurrency, 1, 16, 1, false, 'Max simultaneous summarisation calls. Each call hits the LLM - keep proportionate to your LLM rate limit.')}
        ${rng('Rate limit RPM', 'v.summariser_rate_limit_rpm', v.summariser_rate_limit_rpm, 1, 2000, 1, false, 'Outbound token-bucket cap for LLM summarisation calls. Set to ~90% of your model quota RPM.')}
      </div>
    `)}

    ${sec('semantic-cache', 'Semantic cache', `
      <div class="grid3">
        <div class="field">
          <label data-tip="Cosine similarity threshold for a cache hit. 1.0 = exact match only; 0.9 = near-duplicate; lower values return more results but reduce precision.">Hit threshold (${v.cache_hit_threshold})<span class="tip-dot">i</span></label>
          <div class="range-row">
            <input type="range" min="0.5" max="1.0" step="0.01" value="${v.cache_hit_threshold}"
              oninput="patch('vecstore.cache_hit_threshold',parseFloat(this.value));this.previousElementSibling&&(this.closest('.field').querySelector('label').textContent='Hit threshold ('+this.value+')')">
            <span class="range-val">${v.cache_hit_threshold}</span>
          </div>
          <span class="hint">Higher = stricter match required</span>
        </div>
        ${num('TTL (seconds)', 'v.cache_ttl_seconds', v.cache_ttl_seconds, 60, 86400, 1, 'How long a cached result is considered valid. After TTL expires the entry is evicted on next read.')}
      </div>
    `)}

    ${sec('ingress-rate', 'Ingress rate limits (vecstore-api per-IP)', `
      <p style="font-size:11px;color:#718096;margin-bottom:12px">slowapi / limits syntax: "120/minute", "10/second". Leave blank to disable.</p>
      <div class="grid2">
        ${txt('Ingest', 'v.rate_limit_ingest', v.rate_limit_ingest, 'Per-IP limit on /ingest/* endpoints. Should not exceed embed_workers x embed_concurrency x 60/min.')}
        ${txt('Embed', 'v.rate_limit_embed', v.rate_limit_embed, 'Per-IP limit on /embed and /embed/batch.')}
        ${txt('Search', 'v.rate_limit_search', v.rate_limit_search, 'Per-IP limit on /search.')}
        ${txt('Cache', 'v.rate_limit_cache', v.rate_limit_cache, 'Per-IP limit on /cache/query and /cache (store).')}
        ${txt('Summarise', 'v.rate_limit_summarise', v.rate_limit_summarise, 'Per-IP limit on /summarise/*. Each call triggers an LLM request - keep lower than ingest.')}
      </div>
    `)}

    ${sec('logging', 'Logging', `
      <div style="margin-bottom:12px;display:flex;align-items:center;gap:8px">
        <label style="font-size:12px;color:var(--text-muted);display:flex;align-items:center;gap:6px;cursor:pointer">
          <input type="checkbox" id="log-apply-all"${_logApplyAll ? ' checked' : ''} onchange="onLogApplyAll(this.checked)">
          Apply to all components
        </label>
      </div>
      <div class="grid2">
        <div>
          <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--text-dim);margin-bottom:8px">Vecstore</div>
          <div class="grid2">
            ${selFn('Log level', 'v.log_level', v.log_level, o.log_levels, 'onLogFieldChange', 'Minimum severity for vecstore-api logs. DEBUG is very verbose; use INFO or WARNING in production.')}
            ${selFn('Log format', 'v.log_format', v.log_format, o.log_formats, 'onLogFieldChange', 'console = human-readable; json = structured for log aggregators; logfmt = key=value pairs.')}
          </div>
        </div>
        <div>
          <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--text-dim);margin-bottom:8px">God agent</div>
          <div class="grid2">
            ${selFn('Log level', 'g.log_level', g.log_level, o.log_levels, 'onLogFieldChange', 'Minimum severity for god agent logs.')}
            ${selFn('Log format', 'g.log_format', g.log_format, o.log_formats, 'onLogFieldChange', 'Output format for god agent logs.')}
          </div>
        </div>
      </div>
    `)}

    ${_credentialsSection(v, g, c)}
  `;
  _applyStoredTheme();
}

function _credentialsSection(v, g, c) {
  const eb = v.embed_backend || 'mock';
  const sb = v.summariser_backend || 'mock';
  const ab = g.agent_backend || 'mock';
  const needsGoogle  = eb === 'google-cloud' || sb === 'google-cloud' || ab === 'google-cloud';
  const needsGemini  = eb === 'gemini' || sb === 'gemini' || ab === 'gemini';
  const needsAnthropic = sb === 'anthropic' || ab === 'anthropic';
  const needsSidecar = eb === 'sidecar';
  if (!needsGoogle && !needsGemini && !needsAnthropic && !needsSidecar) return '';
  const rows = [];
  if (needsAnthropic)
    rows.push(credTxt('Anthropic API key', 'credentials.ANTHROPIC_API_KEY', c.ANTHROPIC_API_KEY, 'sk-ant-...', 'Used by the summariser or agent when backend=anthropic. Written to root .env as ANTHROPIC_API_KEY.'));
  if (needsGemini)
    rows.push(credTxt('Gemini API key', 'credentials.GEMINI_API_KEY', c.GEMINI_API_KEY, 'AIza...', 'Used when embed/summariser/agent backend=gemini. Written to both .env files as GEMINI_API_KEY.'));
  if (needsGoogle) {
    rows.push(credTxt('Google Cloud project', 'credentials.GOOGLE_CLOUD_PROJECT', c.GOOGLE_CLOUD_PROJECT, 'my-project-123', 'GCP project ID for Vertex AI / google-cloud backends. Written to both .env files as GOOGLE_CLOUD_PROJECT.'));
    rows.push(credTxt('Google Cloud location', 'credentials.GOOGLE_CLOUD_LOCATION', c.GOOGLE_CLOUD_LOCATION, 'us-central1', 'Vertex AI region. Written to both .env files as GOOGLE_CLOUD_LOCATION.'));
  }
  if (needsSidecar)
    rows.push(credTxt('Embed sidecar URL', 'credentials.EMBED_SIDECAR_URL', c.EMBED_SIDECAR_URL, 'http://localhost:8001', 'Base URL of the local embedding inference server when embed_backend=sidecar. Written to vecstore .env as EMBED_SIDECAR_URL.'));
  const grid = `<div class="grid2">${rows.join('')}</div>`;
  return sec('credentials', 'Provider credentials', grid);
}

// Section helper
const SECTION_KEY = 'cfg-sections-open';
function _secOpen(id) {
  try { const s = JSON.parse(localStorage.getItem(SECTION_KEY)||'{}'); return s[id] !== false; } catch { return true; }
}
function _secToggle(el) {
  try {
    const s = JSON.parse(localStorage.getItem(SECTION_KEY)||'{}');
    s[el.id] = el.open;
    localStorage.setItem(SECTION_KEY, JSON.stringify(s));
  } catch {}
}
let _tapTimer = null;
function toggleAllSections(ev) {
  if (ev.type === 'touchend') {
    if (_tapTimer) { clearTimeout(_tapTimer); _tapTimer = null; } else { _tapTimer = setTimeout(() => { _tapTimer = null; }, 350); return; }
  }
  ev.preventDefault();
  ev.stopPropagation();
  const sections = document.querySelectorAll('#main details.section');
  if (!sections.length) return;
  const anyOpen = Array.from(sections).some(d => d.open);
  sections.forEach(d => { d.open = !anyOpen; _secToggle(d); });
}
function sec(id, title, body) {
  const open = _secOpen(id) ? ' open' : '';
  return `<details class="section" id="${id}"${open} ontoggle="_secToggle(this)">
    <summary><h3>${title}</h3></summary>
    <div class="section-body">${body}</div>
  </details>`;
}

// Field helpers
function esc(s) { return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

function _lbl(label, tip) {
  if (!tip) return label;
  return `<label data-tip="${esc(tip)}">${label}<span class="tip-dot">i</span></label>`;
}

function sel(label, key, val, opts, tip='') {
  return `<div class="field">${_lbl(label,tip)}<select onchange="patch('${key}',this.value)">${(opts||[]).map(o=>`<option ${o==val?'selected':''}>${esc(o)}</option>`).join('')}</select></div>`;
}
function selFn(label, key, val, opts, fnName, tip='') {
  return `<div class="field">${_lbl(label,tip)}<select onchange="${fnName}('${key}',this.value)">${(opts||[]).map(o=>`<option ${o==val?'selected':''}>${esc(o)}</option>`).join('')}</select></div>`;
}
function num(label, key, val, min, max, step=1, tip='') {
  return `<div class="field">${_lbl(label,tip)}<input type="number" value="${val}" min="${min}" max="${max}" step="${step}" oninput="patch('${key}',+this.value)"></div>`;
}
function txt(label, key, val, tip='') {
  return `<div class="field">${_lbl(label,tip)}<input type="text" value="${esc(val)}" oninput="patch('${key}',this.value)"></div>`;
}
function credTxt(label, key, val, placeholder='', tip='') {
  return `<div class="field">${_lbl(label,tip)}<input type="password" autocomplete="off" value="${esc(val||'')}" placeholder="${esc(placeholder)}" oninput="patch('${key}',this.value)"></div>`;
}
function rng(label, key, val, min, max, step=1, float=false, tip='') {
  const id = 'rng_' + key.replace(/\./g,'_');
  return `<div class="field">${_lbl(label,tip)}
    <div class="range-row">
      <input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"
        oninput="patch('${key}',${float}?parseFloat(this.value):parseInt(this.value));document.getElementById('v_${id}').textContent=this.value">
      <span class="range-val" id="v_${id}">${val}</span>
    </div></div>`;
}
function tog(label, key, val) {
  const checked = val ? 'checked' : '';
  return `<div class="toggle-row"><label>${label}</label>
    <label class="toggle"><input type="checkbox" ${checked} onchange="patch('${key}',this.checked)"><span class="slider"></span></label></div>`;
}

// Patch a nested key like "god.mock_mode" into state.current
function patch(key, val) {
  const parts = key.split('.');
  let obj = state.current;
  // map short prefix to actual profile section
  const ALIASES = {g: 'god', v: 'vecstore'};
  const section = ALIASES[parts[0]] || parts[0];
  if (parts.length >= 2) {
    if (!obj[section]) obj[section] = {};
    obj[section][parts[1]] = val;
  } else {
    obj[section] = val;
  }
  state.dirty = true;
}

function onBackendChange(backend) {
  const opts = state.options;
  patch('v.embed_backend', backend);
  const models = opts.embed_models_by_backend[backend] || [];
  const dim = opts.embed_dims_by_backend[backend] || 768;
  if (models.length) patch('v.embed_model', models[0]);
  patch('v.embed_dim', dim);
  renderEditor();
}

function onSummariserBackendChange(backend) {
  const opts = state.options;
  patch('v.summariser_backend', backend);
  const models = opts.summariser_models_by_backend[backend] || [];
  if (models.length) patch('v.summariser_model', models[0]);
  renderEditor();
}

function onAgentBackendChange(backend) {
  const opts = state.options;
  patch('g.agent_backend', backend);
  const models = opts.agent_models_by_backend[backend] || [];
  if (models.length) patch('g.agent_model', models[0]);
  renderEditor();
}

let _logApplyAll = false;

function onLogApplyAll(enabled) {
  _logApplyAll = enabled;
  if (!enabled) return;
  const level  = (state.current.vecstore || {}).log_level  || 'INFO';
  const format = (state.current.vecstore || {}).log_format || 'console';
  patch('g.log_level',  level);
  patch('g.log_format', format);
  renderEditor();
  const cb = document.getElementById('log-apply-all');
  if (cb) cb.checked = true;
}

function onLogFieldChange(key, val) {
  patch(key, val);
  if (!_logApplyAll) return;
  const subKey = key.split('.').pop();
  patch('v.' + subKey, val);
  patch('g.' + subKey, val);
  renderEditor();
  const cb = document.getElementById('log-apply-all');
  if (cb) cb.checked = true;
}

async function saveProfile() {
  setStatus('Saving...', 'info');
  const resp = await fetch(`${API}/api/profiles/${state.currentFile}`, {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(state.current)
  });
  if (resp.ok) {
    state.dirty = false;
    setStatus('Saved.', 'ok');
    // refresh sidebar in case name/desc changed
    const pl = await fetch(`${API}/api/profiles`).then(r=>r.json());
    state.profiles = pl.profiles;
    renderSidebar();
  } else {
    setStatus('Save failed.', 'err');
  }
}

async function applyProfile() {
  await saveProfile();
  setStatus('Applying...', 'info');
  const resp = await fetch(`${API}/api/profiles/${state.currentFile}/apply`, { method: 'POST' });
  const data = await resp.json();
  if (resp.ok) {
    state.active = state.currentFile;
    setStatus(`Applied to .env. Restart services to pick up changes.`, 'ok');
    const pl = await fetch(`${API}/api/profiles`).then(r=>r.json());
    state.profiles = pl.profiles;
    renderSidebar();
  } else {
    setStatus(`Error: ${data.detail}`, 'err');
  }
}

async function deleteProfile() {
  if (!confirm(`Delete profile "${state.currentFile}"?`)) return;
  const resp = await fetch(`${API}/api/profiles/${state.currentFile}`, { method: 'DELETE' });
  if (resp.ok) {
    state.current = null;
    state.currentFile = null;
    const pl = await fetch(`${API}/api/profiles`).then(r=>r.json());
    state.profiles = pl.profiles;
    state.active = pl.active;
    renderSidebar();
    document.getElementById("main").innerHTML = '<div class="empty"><div class="big">&#9881;</div><p>Profile deleted</p></div>';
  }
}

async function cloneProfile() {
  const base = state.current.name || state.currentFile.replace('.yaml','');
  const newName = prompt('New profile name:', base + '-copy');
  if (!newName) return;
  const newFile = newName.replace(/[^a-z0-9_-]/gi,'_').toLowerCase() + '.yaml';
  const clone = JSON.parse(JSON.stringify(state.current));
  clone.name = newName;
  const resp = await fetch(`${API}/api/profiles/${newFile}`, {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(clone)
  });
  if (resp.ok) {
    const pl = await fetch(`${API}/api/profiles`).then(r=>r.json());
    state.profiles = pl.profiles;
    renderSidebar();
    loadProfile(newFile);
  }
}

function newProfile() {
  const newName = prompt('Profile name:');
  if (!newName) return;
  const newFile = newName.replace(/[^a-z0-9_-]/gi,'_').toLowerCase() + '.yaml';
  state.current = {
    name: newName, description: '', editor_enabled: true,
    god: { agent_backend:'mock', agent_model:'gemini-2.5-flash-lite', agent_fallback_model:'', mock_mode:true,
           default_traces:100, default_rate:10.0, default_batch_size:50,
           vecstore_summarise_traces:true, vecstore_summarise_logs:false, vecstore_async_mode:true,
           http_rate_ingest:8, http_rate_search:4, http_rate_cache:4, http_rate_admin:2,
           http_rps_ingest:0.0, http_rps_search:0.0, http_rps_cache:0.0, http_rps_admin:0.0,
           http_retry_attempts:3, http_retry_backoff_min:0.5, http_retry_backoff_max:8.0,
           log_level:'INFO', log_format:'console' },
    vecstore: { embed_backend:'mock', embed_model:'gemini-embedding-001', embed_dim:768,
                embed_workers:4, embed_concurrency:2, embed_batch_size:32, embed_rate_limit_rpm:90,
                summariser_backend:'mock', summariser_model:'mock-model', summariser_concurrency:2, summariser_rate_limit_rpm:50,
                cache_hit_threshold:0.92, cache_ttl_seconds:3600,
                rate_limit_ingest:'120/minute', rate_limit_embed:'60/minute',
                rate_limit_search:'60/minute', rate_limit_cache:'60/minute', rate_limit_summarise:'30/minute',
                log_level:'INFO', log_format:'console' }
  };
  state.currentFile = newFile;
  state.dirty = true;
  renderEditor();
  setStatus('New profile - fill in details then Save', 'info');
}

function setStatus(msg, cls) {
  const el = document.getElementById('status-bar');
  if (!el) return;
  el.textContent = msg;
  el.className = `status ${cls}`;
}

function _themeLabel(isLight) { return isLight ? '\u263d Dark' : '\u2600 Light'; }

function toggleTheme() {
  const isLight = document.body.classList.toggle('light');
  localStorage.setItem('cfg-ui-theme', isLight ? 'light' : 'dark');
  const btn = document.getElementById('theme-btn');
  if (btn) btn.textContent = _themeLabel(isLight);
}

function _applyStoredTheme() {
  const stored = localStorage.getItem('cfg-ui-theme');
  const useDark = stored === 'dark';
  if (!useDark) document.body.classList.add('light');
  const btn = document.getElementById('theme-btn');
  if (btn) btn.textContent = _themeLabel(!useDark);
}

function openHelp() {
  document.getElementById('help-modal').classList.add('open');
}
function closeHelp() {
  document.getElementById('help-modal').classList.remove('open');
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') { closeHelp(); closeTopoImport(); } });

// Topology import

const VECSTORE_API = 'http://localhost:8000';
let _topoExtracted = null;

async function _loadTopologies() {
  const el = document.getElementById('topo-list');
  if (!el) return;
  try {
    const data = await fetch(`${VECSTORE_API}/topologies`, { headers: _topoHeaders() }).then(r => r.json());
    const topos = data.topologies || [];
    if (!topos.length) {
      el.innerHTML = '<p style="font-size:12px;color:#4a5568">None saved yet</p>';
      return;
    }
    el.innerHTML = '<div class="topo-list">' + topos.map(t => `
      <div class="topo-item">
        <div>
          <div class="tname">${esc(t.name)}</div>
          <div class="tmeta">${t.service_count} services</div>
        </div>
        <button class="btn-del-topo" onclick="deleteTopo('${esc(t.file)}')">x</button>
      </div>`).join('') + '</div>';
  } catch {
    el.innerHTML = '<p style="font-size:11px;color:var(--text-hint)">vecstore-api not reachable</p>';
  }
}

async function deleteTopo(filename) {
  if (!confirm(`Delete topology "${filename}"?`)) return;
  await fetch(`${VECSTORE_API}/topologies/${filename}`, { method: 'DELETE', headers: _topoHeaders() });
  _loadTopologies();
}

function openTopoImport() {
  _topoExtracted = null;
  document.getElementById('topo-preview-area').innerHTML = '';
  document.getElementById('topo-file-input').value = '';
  document.getElementById('topo-modal').classList.add('open');
}
function closeTopoImport() {
  document.getElementById('topo-modal').classList.remove('open');
}

function topoOnDragOver(e) { e.preventDefault(); document.getElementById('topo-drop-zone').classList.add('drag-over'); }
function topoOnDragLeave()  { document.getElementById('topo-drop-zone').classList.remove('drag-over'); }
function topoOnDrop(e) {
  e.preventDefault();
  topoOnDragLeave();
  const file = e.dataTransfer.files[0];
  if (file) _topoReadFile(file);
}
function topoOnFileSelect(e) {
  const file = e.target.files[0];
  if (file) _topoReadFile(file);
}

function _parseJsonOrJsonl(text) {
  const trimmed = text.trim();
  if (trimmed.startsWith('[') || trimmed.startsWith('{')) {
    try {
      return JSON.parse(trimmed);
    } catch {}
  }
  // JSONL - merge all resourceSpans arrays into one envelope
  const allResourceSpans = [];
  for (const line of trimmed.split('\n')) {
    const l = line.trim();
    if (!l) continue;
    const obj = JSON.parse(l);
    const rs = obj.resourceSpans || obj.resource_spans || [];
    allResourceSpans.push(...rs);
  }
  return { resourceSpans: allResourceSpans };
}

function _topoReadFile(file) {
  const reader = new FileReader();
  reader.onload = ev => {
    try {
      const raw = _parseJsonOrJsonl(ev.target.result);
      _topoExtracted = _parseOtelTrace(raw, file.name);
      _renderTopoPreview(_topoExtracted, file.name);
    } catch(err) {
      document.getElementById('topo-preview-area').innerHTML =
        `<div class="status err">Failed to parse file: ${esc(String(err))}</div>`;
    }
  };
  reader.readAsText(file);
}

function _parseOtelTrace(raw, filename) {
  const resourceSpans = raw.resourceSpans || raw.resource_spans || [];
  if (!resourceSpans.length) throw new Error('No resourceSpans found - is this an OTLP JSON trace file?');

  // Collect per-service span durations and error counts
  const svcs = {};

  for (const rs of resourceSpans) {
    const attrs = _otelAttrsToObj(rs.resource?.attributes || []);
    const svcName = attrs['service.name'] || 'unknown';

    if (!svcs[svcName]) {
      svcs[svcName] = {
        name: svcName,
        resource_attrs: attrs,
        durations_ms: [],
        error_count: 0,
        span_count: 0,
        parent_services: new Set(),
        span_ids: new Set(),
        parent_span_ids: {},  // spanId -> parentSpanId
      };
    }
    const svc = svcs[svcName];

    for (const ss of rs.scopeSpans || rs.scope_spans || []) {
      for (const span of ss.spans || []) {
        const startNs = BigInt(span.startTimeUnixNano || span.start_time_unix_nano || 0);
        const endNs   = BigInt(span.endTimeUnixNano   || span.end_time_unix_nano   || 0);
        const durMs   = Number(endNs - startNs) / 1_000_000;
        if (durMs > 0) svc.durations_ms.push(durMs);

        const statusCode = span.status?.code ?? span.status?.Code ?? 0;
        if (statusCode === 2) svc.error_count++;
        svc.span_count++;

        if (span.spanId || span.span_id) svc.span_ids.add(span.spanId || span.span_id);
        if (span.parentSpanId || span.parent_span_id) {
          svc.parent_span_ids[span.spanId || span.span_id] = span.parentSpanId || span.parent_span_id;
        }
      }
    }
  }

  // Resolve cross-service parent relationships -> dependencies
  const spanToService = {};
  for (const [svcName, svc] of Object.entries(svcs)) {
    for (const sid of svc.span_ids) spanToService[sid] = svcName;
  }
  for (const [svcName, svc] of Object.entries(svcs)) {
    for (const [, parentId] of Object.entries(svc.parent_span_ids)) {
      const parentSvc = spanToService[parentId];
      if (parentSvc && parentSvc !== svcName) svc.parent_services.add(parentSvc);
    }
  }

  // Build services list - dependencies are services this svc calls (i.e. its children are its deps,
  // but since we track parent_services = who called me, deps of caller = this svc)
  const callerToDeps = {};
  for (const [svcName, svc] of Object.entries(svcs)) {
    for (const caller of svc.parent_services) {
      if (!callerToDeps[caller]) callerToDeps[caller] = new Set();
      callerToDeps[caller].add(svcName);
    }
  }

  const services = Object.values(svcs).map(svc => {
    const durs = svc.durations_ms.sort((a, b) => a - b);
    const p50 = durs.length ? durs[Math.floor(durs.length * 0.5)] : 50;
    const p99 = durs.length ? durs[Math.floor(durs.length * 0.99)] : 500;
    const errorRate = svc.span_count ? svc.error_count / svc.span_count : 0.01;
    const deps = Array.from(callerToDeps[svc.name] || []);

    const ra = svc.resource_attrs;
    return {
      name: svc.name,
      type: _inferServiceType(svc.name, ra),
      version: ra['service.version'] || '1.0.0',
      environment: ra['deployment.environment'] || 'production',
      region: ra['cloud.region'] || 'eu-west-1',
      p50_ms: Math.round(p50),
      p99_ms: Math.round(p99),
      error_rate: Math.round(errorRate * 1000) / 1000,
      dependencies: deps,
      provider: ra['gen_ai.provider.name'] || null,
      model: ra['gen_ai.request.model'] || null,
    };
  });

  return { services, source_file: filename };
}

function _otelAttrsToObj(attrs) {
  const obj = {};
  for (const kv of attrs) {
    const v = kv.value;
    obj[kv.key] = v.stringValue ?? v.intValue ?? v.doubleValue ?? v.boolValue ?? '';
  }
  return obj;
}

function _inferServiceType(name, attrs) {
  const n = name.toLowerCase();
  if (attrs['gen_ai.provider.name']) {
    if (n.includes('embed')) return 'genai_embed';
    if (n.includes('agent')) return 'genai_agent';
    if (n.includes('memory')) return 'genai_memory';
    if (n.includes('workflow')) return 'genai_workflow';
    if (n.includes('tool')) return 'genai_tool_executor';
    if (n.includes('rag') || n.includes('retriev')) return 'genai_rag';
    if (n.includes('vector') || n.includes('vect')) return 'genai_vector_db';
    return 'genai_llm';
  }
  if (n.includes('postgres') || n.includes('mysql') || n.includes('db') || n.includes('database')) return 'database';
  if (n.includes('redis') || n.includes('cache') || n.includes('memcache')) return 'cache';
  if (n.includes('kafka') || n.includes('rabbit') || n.includes('queue') || n.includes('pubsub')) return 'queue';
  if (n.includes('grpc')) return 'grpc_service';
  return 'http_api';
}

function _renderTopoPreview(extracted, filename) {
  const svcs = extracted.services;
  const rows = svcs.map(s => `<tr>
    <td>${esc(s.name)}</td>
    <td>${esc(s.type)}</td>
    <td>${s.p50_ms}</td>
    <td>${s.p99_ms}</td>
    <td>${(s.error_rate * 100).toFixed(1)}%</td>
    <td>${s.dependencies.join(', ') || '-'}</td>
  </tr>`).join('');

  document.getElementById('topo-preview-area').innerHTML = `
    <div class="topo-preview">
      <h4>Extracted: ${svcs.length} services from ${esc(filename)}</h4>
      <div class="field" style="margin-bottom:10px">
        <label>Topology name</label>
        <input id="topo-name" type="text" value="${esc(filename.replace(/\.json$/i,''))}"
               style="background:var(--bg-card);border:1px solid var(--border);border-radius:4px;color:var(--text);padding:5px 8px;font-size:12px;width:100%">
      </div>
      <div class="field" style="margin-bottom:12px">
        <label>Description (optional)</label>
        <input id="topo-desc" type="text" placeholder="e.g. production trace from 2026-06-05 incident"
               style="background:var(--bg-card);border:1px solid var(--border);border-radius:4px;color:var(--text);padding:5px 8px;font-size:12px;width:100%">
      </div>
      <table class="topo-svc-table">
        <thead><tr><th>Service</th><th>Type</th><th>p50 ms</th><th>p99 ms</th><th>Err%</th><th>Dependencies</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <div style="display:flex;gap:10px;align-items:center">
      <button class="btn btn-save" onclick="saveTopology()">Save topology</button>
      <button class="btn btn-delete" onclick="document.getElementById('topo-preview-area').innerHTML=''">Discard</button>
      <span id="topo-save-status" class="status info" style="display:none"></span>
    </div>`;
}

async function saveTopology() {
  if (!_topoExtracted) return;
  const name = document.getElementById('topo-name')?.value?.trim() || 'imported';
  const desc = document.getElementById('topo-desc')?.value?.trim() || '';
  const statusEl = document.getElementById('topo-save-status');
  statusEl.style.display = '';
  statusEl.className = 'status info';
  statusEl.textContent = 'Saving...';

  const payload = {
    name,
    description: desc,
    source_file: _topoExtracted.source_file,
    services: _topoExtracted.services,
  };

  try {
    const resp = await fetch(`${VECSTORE_API}/topologies`, {
      method: 'POST',
      headers: _topoHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(payload),
    });
    if (resp.ok) {
      statusEl.className = 'status ok';
      statusEl.textContent = 'Saved.';
      _loadTopologies();
      setTimeout(closeTopoImport, 1200);
    } else {
      const err = await resp.json().catch(() => ({}));
      statusEl.className = 'status err';
      statusEl.textContent = `Error: ${err.detail || resp.statusText}`;
    }
  } catch(err) {
    statusEl.className = 'status err';
    statusEl.textContent = `Network error: ${err.message}`;
  }
}

_applyStoredTheme();
init();
</script>
</body>
</html>
"""


_FAVICON = REPO_ROOT / "favicon.ico"


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    if not _FAVICON.exists():
        return Response(status_code=204)
    return Response(content=_FAVICON.read_bytes(), media_type="image/x-icon")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(_HTML)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    missing = []
    try:
        import yaml as _  # noqa: F401
    except ImportError:
        missing.append("pyyaml")
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}. Run: pip install {' '.join(missing)}", file=sys.stderr)
        sys.exit(1)
    if not PROFILES_DIR.exists():
        PROFILES_DIR.mkdir(parents=True)
        log.info("Created profiles/ directory")
    log.info("Config UI starting at http://localhost:8080")
    log.info("Repo root: %s", REPO_ROOT)
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")


if __name__ == "__main__":
    main()
