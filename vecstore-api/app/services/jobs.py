"""Async batch job tracker.

In-memory store of background job state. Jobs are created with a UUID,
run as asyncio Tasks, and cleaned up after job_ttl_seconds.

Not distributed — single process only. Fine for local dev.
For multi-process, back this with Redis hashes instead.
"""

import asyncio
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from app.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class Job:
    job_id: str
    status: str = "pending"  # pending | running | done | failed
    submitted_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    total: int = 0
    processed: int = 0
    errors: int = 0
    results: list[Any] | None = None
    error_detail: str | None = None


class JobStore:
    def __init__(self, ttl_seconds: int = 300) -> None:
        self._jobs: dict[str, Job] = {}
        self._ttl = ttl_seconds

    def create(self, total: int) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(job_id=job_id, total=total)
        self._jobs[job_id] = job
        log.debug("job_created", job_id=job_id, total=total)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def submit(
        self,
        job: Job,
        coro_fn: Callable[[], Coroutine],
    ) -> asyncio.Task:
        """Create and schedule an asyncio Task for the job."""

        async def _run() -> None:
            job.status = "running"
            try:
                await coro_fn()
                job.status = "done"
            except Exception as exc:
                job.status = "failed"
                job.error_detail = str(exc)
                log.error("job_failed", job_id=job.job_id, error=str(exc))
            finally:
                job.completed_at = time.time()
                # Schedule cleanup
                asyncio.create_task(self._cleanup(job.job_id))

        task = asyncio.create_task(_run(), name=f"job-{job.job_id}")
        log.info("job_submitted", job_id=job.job_id, total=job.total)
        return task

    async def _cleanup(self, job_id: str) -> None:
        await asyncio.sleep(self._ttl)
        self._jobs.pop(job_id, None)
        log.debug("job_cleaned_up", job_id=job_id)

    def active_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status in ("pending", "running"))
