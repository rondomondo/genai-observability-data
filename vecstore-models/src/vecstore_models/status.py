from typing import Literal

from pydantic import BaseModel


class VSetInfo(BaseModel):
    key: str
    cardinality: int


class JobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "done", "failed"]
    submitted_at: float
    completed_at: float | None = None
    total: int
    processed: int
    errors: int
    results: list | None = None
    error_detail: str | None = None


class StatusResponse(BaseModel):
    service: str
    version: str
    redis_connected: bool
    embed_workers: int
    embed_queue_depth: int
    ttl_watcher_running: bool
    vsets: list[VSetInfo]
    active_jobs: int
