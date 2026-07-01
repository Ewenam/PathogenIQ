"""
saas/jobs.py
Durable job model for pipeline runs.

The dashboard tracks jobs in an in-process dict (_JOBS), which is fine for a
single process but loses state on restart and can't be shared by a worker pool.
This provides:

  * Job / JobStatus       — a serializable job record.
  * JobStore              — durable status persistence via saas.storage (so a
                            web process can enqueue and a separate worker can
                            update, and status survives restarts).
  * ThreadJobQueue        — a local executor (ThreadPoolExecutor) that runs a
                            function and writes status transitions to the
                            JobStore. Drop-in for the dashboard's in-process
                            jobs, but durable.
  * production seam       — swap ThreadJobQueue for a Celery/RQ-backed queue
                            with the same submit()/get() surface; the JobStore
                            and Storage layers stay identical.
"""
from __future__ import annotations

import json
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable

from .storage import Storage, get_storage


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    status: str = JobStatus.QUEUED.value
    org_id: str | None = None
    kind: str = "pipeline"
    result: dict | None = None
    error: str | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)


class JobStore:
    """Persists Job records as JSON under jobs/<id>.json in a Storage backend."""

    def __init__(self, storage: Storage | None = None):
        self.storage = storage or get_storage()

    def _key(self, job_id: str) -> str:
        return f"jobs/{job_id}.json"

    def save(self, job: Job) -> None:
        job.updated_at = _now()
        self.storage.put_bytes(self._key(job.id),
                               json.dumps(job.to_dict()).encode())

    def get(self, job_id: str) -> Job | None:
        if not self.storage.exists(self._key(job_id)):
            return None
        return Job(**json.loads(self.storage.get_bytes(self._key(job_id))))

    def create(self, org_id: str | None = None, kind: str = "pipeline") -> Job:
        job = Job(id=uuid.uuid4().hex[:12], org_id=org_id, kind=kind)
        self.save(job)
        return job


class ThreadJobQueue:
    """Local durable job queue: runs `fn(job, *args, **kwargs)` in a thread,
    recording QUEUED→RUNNING→DONE/ERROR transitions to the JobStore."""

    def __init__(self, store: JobStore | None = None, max_workers: int = 2):
        self.store = store or JobStore()
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: dict[str, Future] = {}

    def submit(self, fn: Callable, *args, org_id: str | None = None,
               kind: str = "pipeline", **kwargs) -> str:
        job = self.store.create(org_id=org_id, kind=kind)

        def _run():
            job.status = JobStatus.RUNNING.value
            self.store.save(job)
            try:
                result = fn(job, *args, **kwargs)
                job.status = JobStatus.DONE.value
                job.result = result if isinstance(result, dict) else {"value": result}
            except Exception as exc:  # capture full context for debugging
                job.status = JobStatus.ERROR.value
                job.error = f"{exc}\n{traceback.format_exc()}"
            self.store.save(job)

        self._futures[job.id] = self._pool.submit(_run)
        return job.id

    def get(self, job_id: str) -> Job | None:
        return self.store.get(job_id)

    def wait(self, job_id: str, timeout: float | None = None) -> Job:
        """Block until a submitted job finishes (test/CLI convenience)."""
        fut = self._futures.get(job_id)
        if fut is not None:
            fut.result(timeout=timeout)
        return self.store.get(job_id)

    def shutdown(self):
        self._pool.shutdown(wait=True)
