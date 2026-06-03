"""Local config UI for genai-observability-data.

Serves a browser UI to edit profiles and apply them to .env files.
Run from repo root:
    uv run --directory vecstore-api python ../config_ui.py
Or if fastapi/uvicorn/pyyaml are already on PATH:
    python config_ui.py
"""

import copy
import json
import logging
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

REPO_ROOT = Path(__file__).parent.resolve()
PROFILES_DIR = REPO_ROOT / "profiles"
ROOT_ENV = REPO_ROOT / ".env"
VECSTORE_ENV = REPO_ROOT / "vecstore-api" / ".env"
BACKUP_DIR = REPO_ROOT / "profiles" / ".backups"

EMBED_BACKENDS = ["mock", "sidecar", "gemini", "sentence-transformers"]
EMBED_MODELS_BY_BACKEND: dict[str, list[str]] = {
    "gemini": ["gemini-embedding-001", "gemini-embedding-2"],
    "sidecar": ["Qwen/Qwen3-Embedding-0.6B", "sentence-transformers/all-MiniLM-L6-v2"],
    "sentence-transformers": ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
    "mock": ["mock-model"],
}
EMBED_DIMS_BY_BACKEND: dict[str, int] = {
    "gemini": 768,
    "sidecar": 1024,
    "sentence-transformers": 384,
    "mock": 256,
}
AGENT_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
    "claude-haiku-4-5-20251001",
]
SUMMARISER_MODELS = [
    "claude-sonnet-4-6",
    "claude-opus-4-7",
    "claude-haiku-4-5-20251001",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
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
        profiles.append({"file": p.name, "name": data.get("name", p.stem), "description": data.get("description", "")})
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


def _profile_to_root_env(profile: dict[str, Any]) -> dict[str, str]:
    g = profile.get("god", {})
    return {
        "GOD_AGENT_MODEL": str(g.get("agent_model", "gemini-2.5-flash")),
        "GOD_AGENT_FALLBACK_MODEL": str(g.get("agent_fallback_model", "gemini-2.0-flash")),
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


def _profile_to_vecstore_env(profile: dict[str, Any]) -> dict[str, str]:
    v = profile.get("vecstore", {})
    return {
        "EMBED_BACKEND": str(v.get("embed_backend", "mock")),
        "EMBED_MODEL": str(v.get("embed_model", "gemini-embedding-001")),
        "EMBED_DIM": str(v.get("embed_dim", 768)),
        "EMBED_WORKERS": str(v.get("embed_workers", 4)),
        "EMBED_CONCURRENCY": str(v.get("embed_concurrency", 2)),
        "EMBED_BATCH_SIZE": str(v.get("embed_batch_size", 32)),
        "EMBED_RATE_LIMIT_RPM": str(v.get("embed_rate_limit_rpm", 90)),
        "SUMMARISER_MODEL": str(v.get("summariser_model", "claude-sonnet-4-6")),
        "SUMMARISER_CONCURRENCY": str(v.get("summariser_concurrency", 2)),
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
    _write_env_preserving_secrets(ROOT_ENV, root_vals)
    _write_env_preserving_secrets(VECSTORE_ENV, vecstore_vals)
    _set_active_profile(filename)
    return JSONResponse({"ok": True, "applied": filename, "root_env": str(ROOT_ENV), "vecstore_env": str(VECSTORE_ENV)})


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


@app.get("/api/meta/options")
async def meta_options() -> JSONResponse:
    return JSONResponse({
        "embed_backends": EMBED_BACKENDS,
        "embed_models_by_backend": EMBED_MODELS_BY_BACKEND,
        "embed_dims_by_backend": EMBED_DIMS_BY_BACKEND,
        "agent_models": AGENT_MODELS,
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
    border: 1px solid var(--border);
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
  .profile-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; cursor: pointer; transition: border-color .15s; }
  .profile-card:hover { border-color: var(--border-hover); }
  .profile-card.active  { border-color: var(--active-bdr); background: var(--active-bg); }
  .profile-card.editing { border-color: var(--focus); background: var(--info-bg); }
  .profile-card .pname { font-size: 13px; font-weight: 600; }
  .profile-card .pdesc { font-size: 11px; color: var(--text-dim); margin-top: 2px; line-height: 1.4; }
  .profile-card .pbadge { display: inline-block; margin-top: 4px; font-size: 10px; padding: 1px 6px; border-radius: 10px; background: var(--green-bg); color: var(--green-text); }
  .profile-card .pbadge-editing { display: inline-block; margin-top: 4px; font-size: 10px; padding: 1px 6px; border-radius: 10px; background: var(--info-bg); color: var(--info-text); border: 1px solid var(--focus); }
  .btn-new { background: #2b6cb0; color: #fff; border: none; border-radius: 6px; padding: 8px 12px; font-size: 12px; cursor: pointer; width: 100%; margin-top: 8px; }
  .btn-new:hover { background: #3182ce; }

  /* Main */
  .main { padding: 24px 28px; overflow-y: auto; background: var(--bg-panel); border: 1px solid var(--border); border-radius: 10px; box-shadow: inset 0 1px 3px rgba(0,0,0,.35), inset 0 0 0 1px rgba(0,0,0,.15); margin: 8px 8px 8px 0; }
  body.light .main { box-shadow: inset 0 1px 3px rgba(0,0,0,.08), inset 0 0 0 1px rgba(0,0,0,.06); }
  .title-bar { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 4px; gap: 12px; }
  .title-bar h2 { font-size: 18px; font-weight: 700; margin: 0; flex: 1; }
  .title-bar-btns { display: flex; gap: 6px; flex-shrink: 0; padding-top: 2px; }
  .main .subtitle { font-size: 13px; color: var(--text-dim); margin-bottom: 20px; }

  /* Sections */
  .section { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 20px; }
  .section h3 { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; color: var(--accent); margin-bottom: 14px; }
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

<div class="page-wrap">
  <div class="layout">
    <aside class="sidebar">
      <h1>Profiles</h1>
      <div id="profile-list"></div>
      <button class="btn-new" onclick="newProfile()">+ New profile</button>
    </aside>
    <main class="main" id="main">
      <div class="empty"><div class="big">&#9881;</div><p>Select a profile to edit</p></div>
    </main>
  </div>
</div>

<script>
const API = "";
let state = { profiles: [], active: null, options: {}, current: null, dirty: false };

async function init() {
  const [pl, opts] = await Promise.all([
    fetch(`${API}/api/profiles`).then(r => r.json()),
    fetch(`${API}/api/meta/options`).then(r => r.json()),
  ]);
  state.profiles = pl.profiles;
  state.active = pl.active;
  state.options = opts;
  renderSidebar();
  if (state.active) {
    const match = state.profiles.find(p => p.file === state.active);
    if (match) loadProfile(match.file);
  }
}

function renderSidebar() {
  const el = document.getElementById("profile-list");
  if (!state.profiles.length) { el.innerHTML = '<p style="font-size:12px;color:#4a5568">No profiles yet</p>'; return; }
  el.innerHTML = state.profiles.map(p => {
    const isActive  = p.file === state.active;
    const isEditing = p.file === state.currentFile && !isActive;
    const cls = isActive ? 'active' : isEditing ? 'editing' : '';
    return `
    <div class="profile-card ${cls}" onclick="loadProfile('${p.file}')">
      <div class="pname">${esc(p.name)}</div>
      <div class="pdesc">${esc(p.description)}</div>
      ${isActive ? '<span class="pbadge">active</span>' : isEditing ? '<span class="pbadge-editing">editing</span>' : ''}
    </div>`;
  }).join('');
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
  const g = p.god || {};
  const v = p.vecstore || {};
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

    <div class="section">
      <h3>God agent</h3>
      <div class="grid2">
        ${sel('Agent model', 'g.agent_model', g.agent_model, o.agent_models)}
        ${sel('Fallback model', 'g.agent_fallback_model', g.agent_fallback_model, o.agent_models)}
      </div>
      <div style="margin-top:14px">
        ${tog('Mock mode (no API key)', 'g.mock_mode', g.mock_mode)}
      </div>
      <div class="grid3" style="margin-top:14px">
        ${num('Default traces', 'g.default_traces', g.default_traces, 1, 100000)}
        ${num('Default rate (req/s)', 'g.default_rate', g.default_rate, 0.1, 100, 0.1)}
        ${num('Batch size', 'g.default_batch_size', g.default_batch_size, 1, 500)}
      </div>
      <div style="margin-top:14px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px">
        ${tog('Summarise traces', 'g.vecstore_summarise_traces', g.vecstore_summarise_traces)}
        ${tog('Summarise logs', 'g.vecstore_summarise_logs', g.vecstore_summarise_logs)}
        ${tog('Async mode', 'g.vecstore_async_mode', g.vecstore_async_mode)}
      </div>
    </div>

    <div class="section">
      <h3>Egress rate limiting (god HTTP client)</h3>
      <p style="font-size:11px;color:#718096;margin-bottom:12px">Concurrency = max in-flight requests. RPS = token-bucket cap (0 = unlimited).</p>
      <div class="grid2">
        ${rng('Ingest concurrency', 'g.http_rate_ingest', g.http_rate_ingest, 1, 20)}
        ${rng('Ingest RPS cap', 'g.http_rps_ingest', g.http_rps_ingest, 0, 20, 0.5, true)}
        ${rng('Search concurrency', 'g.http_rate_search', g.http_rate_search, 1, 20)}
        ${rng('Search RPS cap', 'g.http_rps_search', g.http_rps_search, 0, 20, 0.5, true)}
        ${rng('Cache concurrency', 'g.http_rate_cache', g.http_rate_cache, 1, 20)}
        ${rng('Cache RPS cap', 'g.http_rps_cache', g.http_rps_cache, 0, 20, 0.5, true)}
        ${rng('Admin concurrency', 'g.http_rate_admin', g.http_rate_admin, 1, 10)}
        ${rng('Admin RPS cap', 'g.http_rps_admin', g.http_rps_admin, 0, 10, 0.5, true)}
      </div>
      <div class="grid3" style="margin-top:14px">
        ${rng('Retry attempts', 'g.http_retry_attempts', g.http_retry_attempts, 0, 10)}
        ${num('Backoff min (s)', 'g.http_retry_backoff_min', g.http_retry_backoff_min, 0.1, 10, 0.1)}
        ${num('Backoff max (s)', 'g.http_retry_backoff_max', g.http_retry_backoff_max, 1, 60, 0.5)}
      </div>
    </div>

    <div class="section">
      <h3>Embedding</h3>
      <div class="grid2">
        <div class="field">
          <label>Backend</label>
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
        ${rng('Embedding dimension', 'v.embed_dim', v.embed_dim, 64, 3072, 64)}
        ${rng('Workers', 'v.embed_workers', v.embed_workers, 1, 16)}
        ${rng('Concurrency', 'v.embed_concurrency', v.embed_concurrency, 1, 16)}
        ${rng('Batch size', 'v.embed_batch_size', v.embed_batch_size, 1, 128)}
        ${rng('Rate limit RPM (Gemini)', 'v.embed_rate_limit_rpm', v.embed_rate_limit_rpm, 1, 300)}
      </div>
    </div>

    <div class="section">
      <h3>Summariser</h3>
      <div class="grid2">
        ${sel('Summariser model', 'v.summariser_model', v.summariser_model, o.summariser_models)}
        ${rng('Summariser concurrency', 'v.summariser_concurrency', v.summariser_concurrency, 1, 16)}
      </div>
    </div>

    <div class="section">
      <h3>Semantic cache</h3>
      <div class="grid3">
        <div class="field">
          <label>Hit threshold (${v.cache_hit_threshold})</label>
          <div class="range-row">
            <input type="range" min="0.5" max="1.0" step="0.01" value="${v.cache_hit_threshold}"
              oninput="patch('vecstore.cache_hit_threshold',parseFloat(this.value));this.previousElementSibling&&(this.closest('.field').querySelector('label').textContent='Hit threshold ('+this.value+')')">
            <span class="range-val">${v.cache_hit_threshold}</span>
          </div>
          <span class="hint">Higher = stricter match required</span>
        </div>
        ${num('TTL (seconds)', 'v.cache_ttl_seconds', v.cache_ttl_seconds, 60, 86400)}
      </div>
    </div>

    <div class="section">
      <h3>Ingress rate limits (vecstore-api per-IP)</h3>
      <p style="font-size:11px;color:#718096;margin-bottom:12px">slowapi / limits syntax: "120/minute", "10/second". Leave blank to disable.</p>
      <div class="grid2">
        ${txt('Ingest', 'v.rate_limit_ingest', v.rate_limit_ingest)}
        ${txt('Embed', 'v.rate_limit_embed', v.rate_limit_embed)}
        ${txt('Search', 'v.rate_limit_search', v.rate_limit_search)}
        ${txt('Cache', 'v.rate_limit_cache', v.rate_limit_cache)}
        ${txt('Summarise', 'v.rate_limit_summarise', v.rate_limit_summarise)}
      </div>
    </div>

    <div class="section">
      <h3>Logging</h3>
      <div class="grid2">
        ${sel('Log level', 'v.log_level', v.log_level, o.log_levels)}
        ${sel('Log format', 'v.log_format', v.log_format, o.log_formats)}
      </div>
    </div>
  `;
  _applyStoredTheme();
}

// Field helpers
function esc(s) { return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

function sel(label, key, val, opts) {
  return `<div class="field"><label>${label}</label><select onchange="patch('${key}',this.value)">${(opts||[]).map(o=>`<option ${o==val?'selected':''}>${esc(o)}</option>`).join('')}</select></div>`;
}
function num(label, key, val, min, max, step=1) {
  return `<div class="field"><label>${label}</label><input type="number" value="${val}" min="${min}" max="${max}" step="${step}" oninput="patch('${key}',+this.value)"></div>`;
}
function txt(label, key, val) {
  return `<div class="field"><label>${label}</label><input type="text" value="${esc(val)}" oninput="patch('${key}',this.value)"></div>`;
}
function rng(label, key, val, min, max, step=1, float=false) {
  const id = 'rng_' + key.replace(/\./g,'_');
  return `<div class="field"><label>${label}</label>
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
  const section = parts[0] === 'g' ? 'god' : parts[0] === 'v' ? 'vecstore' : null;
  if (section) {
    if (!obj[section]) obj[section] = {};
    obj[section][parts[1]] = val;
  } else {
    obj[parts[0]] = val;
  }
  state.dirty = true;
}

function onBackendChange(backend) {
  const opts = state.options;
  patch('vecstore.embed_backend', backend);
  // Set sensible defaults for dimension and model when backend changes
  const models = opts.embed_models_by_backend[backend] || [];
  const dim = opts.embed_dims_by_backend[backend] || 768;
  if (models.length) patch('vecstore.embed_model', models[0]);
  patch('vecstore.embed_dim', dim);
  renderEditor();
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
    name: newName, description: '',
    god: { agent_model:'gemini-2.5-flash', agent_fallback_model:'gemini-2.0-flash', mock_mode:true,
           default_traces:100, default_rate:10.0, default_batch_size:50,
           vecstore_summarise_traces:true, vecstore_summarise_logs:false, vecstore_async_mode:true,
           http_rate_ingest:8, http_rate_search:4, http_rate_cache:4, http_rate_admin:2,
           http_rps_ingest:0.0, http_rps_search:0.0, http_rps_cache:0.0, http_rps_admin:0.0,
           http_retry_attempts:3, http_retry_backoff_min:0.5, http_retry_backoff_max:8.0 },
    vecstore: { embed_backend:'mock', embed_model:'gemini-embedding-001', embed_dim:768,
                embed_workers:4, embed_concurrency:2, embed_batch_size:32, embed_rate_limit_rpm:90,
                summariser_model:'claude-sonnet-4-6', summariser_concurrency:2,
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
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeHelp(); });

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
