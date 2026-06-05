"""Topology import and management endpoints.

Topologies are extracted service graphs derived from real OTel trace files.
They are stored as YAML under src/god/.god/topologies/ and augment the
synthetic DEFAULT_SERVICES when god generates data.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

router = APIRouter(prefix="/topologies", tags=["topologies"])

_REPO_ROOT = (
    Path(os.environ["REPO_ROOT"])
    if "REPO_ROOT" in os.environ
    else Path(__file__).parents[3].resolve()
)
TOPOLOGIES_DIR = _REPO_ROOT / "src" / "god" / ".god" / "topologies"

VALID_SERVICE_TYPES = {
    "http_api", "grpc_service", "database", "cache", "queue",
    "genai_llm", "genai_embed", "genai_vector_db", "genai_rag",
    "genai_agent", "genai_tool_executor", "genai_memory", "genai_workflow",
}


class ServiceSchema(BaseModel):
    name: str
    type: str = "http_api"
    version: str = "1.0.0"
    environment: str = "production"
    region: str = "eu-west-1"
    p50_ms: float = 50.0
    p99_ms: float = 500.0
    timeout_ms: float = 5000.0
    error_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    dependencies: list[str] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None


class TopologyImport(BaseModel):
    name: str
    description: str = ""
    source_file: str = ""
    services: list[ServiceSchema]


def _topology_path(filename: str) -> Path:
    return TOPOLOGIES_DIR / filename


def _load_topology(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_topology(path: Path, data: dict[str, Any]) -> None:
    TOPOLOGIES_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


@router.get("", summary="List saved topologies")
async def list_topologies() -> JSONResponse:
    """Return metadata for all saved topology YAML files."""
    if not TOPOLOGIES_DIR.exists():
        return JSONResponse({"topologies": []})
    result = []
    for p in sorted(TOPOLOGIES_DIR.glob("*.yaml")):
        data = _load_topology(p)
        result.append({
            "file": p.name,
            "name": data.get("name", p.stem),
            "description": data.get("description", ""),
            "source_file": data.get("source_file", ""),
            "imported_at": data.get("imported_at", ""),
            "service_count": len(data.get("services", [])),
        })
    return JSONResponse({"topologies": result})


@router.get("/{filename}", summary="Get a single topology")
async def get_topology(filename: str) -> JSONResponse:
    """Return the full YAML content of a saved topology."""
    if not filename.endswith(".yaml"):
        raise HTTPException(status_code=400, detail="Filename must end in .yaml")
    path = _topology_path(filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Topology not found")
    return JSONResponse(_load_topology(path))


@router.post("", summary="Save an imported topology", status_code=201)
async def save_topology(request: Request) -> JSONResponse:
    """Accept an extracted topology schema and persist it as YAML.

    Args:
        request: JSON body matching TopologyImport schema.

    Returns:
        The filename and path of the saved topology.

    Raises:
        HTTPException: 400 if the payload is invalid or service types are unknown.
    """
    body = await request.json()
    try:
        topology = TopologyImport(**body)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid topology payload: {exc}") from exc

    unknown_types = {s.type for s in topology.services} - VALID_SERVICE_TYPES
    if unknown_types:
        log.warning("import contains unknown service types: %s", unknown_types)

    filename = topology.name.replace(" ", "_").lower()
    filename = "".join(c for c in filename if c.isalnum() or c in "_-") + ".yaml"

    data: dict[str, Any] = {
        "name": topology.name,
        "description": topology.description,
        "source_file": topology.source_file,
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "services": [
            {k: v for k, v in s.model_dump().items() if v is not None}
            for s in topology.services
        ],
    }

    path = _topology_path(filename)
    _save_topology(path, data)
    log.info("topology saved: %s (%d services)", filename, len(topology.services))

    return JSONResponse({"ok": True, "file": filename, "path": str(path)}, status_code=201)


@router.delete("/{filename}", summary="Delete a topology")
async def delete_topology(filename: str) -> JSONResponse:
    """Remove a saved topology YAML file.

    Args:
        filename: The YAML filename to delete.

    Raises:
        HTTPException: 404 if the file does not exist.
    """
    if not filename.endswith(".yaml"):
        raise HTTPException(status_code=400, detail="Filename must end in .yaml")
    path = _topology_path(filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Topology not found")
    path.unlink()
    log.info("topology deleted: %s", filename)
    return JSONResponse({"ok": True})
