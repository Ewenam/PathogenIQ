"""
saas/storage.py
Object-storage abstraction: one interface, a local-filesystem backend for
dev/self-hosted, and an S3 seam for hosted SaaS. Upload payloads, run outputs,
and baselines all go through this so nothing else needs to know where bytes
live.

    store = get_storage()                      # LocalStorage or S3Storage by env
    store.put_bytes("org1/runs/42/report.json", data)
    data = store.get_bytes("org1/runs/42/report.json")

Backends are selected by PATHOGENIQ_STORAGE ("local" | "s3"); local root from
PATHOGENIQ_STORAGE_ROOT, S3 bucket from PATHOGENIQ_S3_BUCKET.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol


class Storage(Protocol):
    def put_bytes(self, key: str, data: bytes) -> str: ...
    def get_bytes(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def put_file(self, key: str, path: str | Path) -> str: ...
    def local_path(self, key: str) -> Path | None: ...  # None if not on local FS


class LocalStorage:
    """Filesystem-backed storage rooted at a directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _p(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if not str(p).startswith(str(self.root.resolve())):
            raise ValueError(f"key escapes storage root: {key}")
        return p

    def put_bytes(self, key: str, data: bytes) -> str:
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return str(p)

    def get_bytes(self, key: str) -> bytes:
        return self._p(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._p(key).exists()

    def put_file(self, key: str, path: str | Path) -> str:
        return self.put_bytes(key, Path(path).read_bytes())

    def local_path(self, key: str) -> Path | None:
        return self._p(key)


class S3Storage:
    """S3-backed storage (hosted SaaS). Lazy-imports boto3 so it's optional."""

    def __init__(self, bucket: str, prefix: str = ""):
        try:
            import boto3  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "S3Storage requires boto3 (pip install boto3), or set "
                "PATHOGENIQ_STORAGE=local."
            ) from e
        import boto3
        self._s3 = boto3.client("s3")
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")

    def _k(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_bytes(self, key: str, data: bytes) -> str:  # pragma: no cover
        self._s3.put_object(Bucket=self.bucket, Key=self._k(key), Body=data)
        return f"s3://{self.bucket}/{self._k(key)}"

    def get_bytes(self, key: str) -> bytes:  # pragma: no cover
        return self._s3.get_object(Bucket=self.bucket, Key=self._k(key))["Body"].read()

    def exists(self, key: str) -> bool:  # pragma: no cover
        from botocore.exceptions import ClientError
        try:
            self._s3.head_object(Bucket=self.bucket, Key=self._k(key))
            return True
        except ClientError:
            return False

    def put_file(self, key: str, path: str | Path) -> str:  # pragma: no cover
        self._s3.upload_file(str(path), self.bucket, self._k(key))
        return f"s3://{self.bucket}/{self._k(key)}"

    def local_path(self, key: str) -> Path | None:  # pragma: no cover
        return None  # objects are remote; callers must get_bytes


def get_storage() -> Storage:
    backend = os.environ.get("PATHOGENIQ_STORAGE", "local").lower()
    if backend == "s3":  # pragma: no cover
        return S3Storage(bucket=os.environ["PATHOGENIQ_S3_BUCKET"],
                         prefix=os.environ.get("PATHOGENIQ_S3_PREFIX", ""))
    root = os.environ.get("PATHOGENIQ_STORAGE_ROOT",
                          str(Path.home() / ".pathogeniq" / "storage"))
    return LocalStorage(root)
