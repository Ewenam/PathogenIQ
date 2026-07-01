"""Tests for the object-storage abstraction and durable job queue."""
import pytest

from pathogeniq.saas.storage import LocalStorage, get_storage
from pathogeniq.saas.jobs import JobStore, ThreadJobQueue, JobStatus


# ── storage ──────────────────────────────────────────────────────────────────
def test_local_storage_roundtrip(tmp_path):
    s = LocalStorage(tmp_path)
    assert not s.exists("org1/a.json")
    s.put_bytes("org1/a.json", b"hello")
    assert s.exists("org1/a.json")
    assert s.get_bytes("org1/a.json") == b"hello"
    assert s.local_path("org1/a.json").read_bytes() == b"hello"


def test_local_storage_put_file(tmp_path):
    src = tmp_path / "src.txt"
    src.write_bytes(b"payload")
    s = LocalStorage(tmp_path / "store")
    s.put_file("k/x.txt", src)
    assert s.get_bytes("k/x.txt") == b"payload"


def test_local_storage_rejects_escape(tmp_path):
    s = LocalStorage(tmp_path)
    with pytest.raises(ValueError):
        s.put_bytes("../escape.txt", b"x")


def test_get_storage_defaults_local(tmp_path, monkeypatch):
    monkeypatch.setenv("PATHOGENIQ_STORAGE", "local")
    monkeypatch.setenv("PATHOGENIQ_STORAGE_ROOT", str(tmp_path))
    assert isinstance(get_storage(), LocalStorage)


# ── jobs ─────────────────────────────────────────────────────────────────────
def _queue(tmp_path):
    return ThreadJobQueue(JobStore(LocalStorage(tmp_path)), max_workers=2)


def test_job_runs_to_completion_and_persists(tmp_path):
    q = _queue(tmp_path)
    jid = q.submit(lambda job, x: {"doubled": x * 2}, 21, org_id="org1")
    job = q.wait(jid, timeout=5)
    assert job.status == JobStatus.DONE.value
    assert job.result == {"doubled": 42}
    assert job.org_id == "org1"
    # durable: a fresh store reads the same terminal state
    assert JobStore(LocalStorage(tmp_path)).get(jid).status == JobStatus.DONE.value
    q.shutdown()


def test_job_captures_error(tmp_path):
    q = _queue(tmp_path)

    def boom(job):
        raise ValueError("kaboom")

    jid = q.submit(boom)
    job = q.wait(jid, timeout=5)
    assert job.status == JobStatus.ERROR.value
    assert "kaboom" in job.error
    q.shutdown()


def test_unknown_job_is_none(tmp_path):
    assert _queue(tmp_path).get("nope") is None
