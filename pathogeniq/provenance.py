"""
provenance.py
Per-run audit/evidence trail for PathogenIQ.

Every pipeline run writes a provenance.json alongside report.json, recording:
  - sha256 + size of every ingested input file
  - the classifier format and taxonomic rank used
  - a hash of the active PipelineConfig
  - who/what triggered the run (actor) and which history store it landed in
  - a content hash of the report's `samples` payload, for later tamper checks

`verify_manifest()` re-derives the same hashes from what's on disk today and
compares them against what was recorded at run time — this is the function
behind `pathogeniq verify <report_dir>` and is exercised directly (no
subprocess shelling) by tests/test_provenance.py.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

_FALLBACK_VERSION = "0.1.0"
_CHUNK_SIZE = 65536


def get_version() -> str:
    try:
        from importlib.metadata import version
        return version("pathogeniq")
    except Exception:
        return _FALLBACK_VERSION


def hash_file(path: str | Path) -> tuple[str, int]:
    """Stream-hash a file in fixed-size chunks. Returns (sha256_hex, size_in_bytes)."""
    h = hashlib.sha256()
    n_bytes = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
            n_bytes += len(chunk)
    return h.hexdigest(), n_bytes


def hash_dict(obj) -> str:
    """sha256 over canonical (sorted-key, whitespace-free) JSON."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_report_content_hash(samples: list[dict]) -> str:
    """
    Tamper-evidence hash over a report's `samples` payload only — deliberately
    excludes `generated_at`/timestamps so re-verifying later is reproducible.
    """
    return hash_dict(samples)


def build_manifest(
    input_files: list[str | Path],
    classifier_format: str,
    rank: str,
    config_snapshot: dict,
    actor: str | None,
    store_backend: str,
) -> dict:
    """
    Build a provenance manifest for one pipeline run.

    `input_files` should include every file that was actually ingested
    (both files of a merged paired-end pair, not just the first). The
    returned dict's `report_content_hash` is a placeholder (None) — the
    caller fills it in once the report body has been assembled.
    """
    file_records = []
    for f in input_files:
        f = Path(f)
        sha256, n_bytes = hash_file(f)
        file_records.append({
            "filename": f.name,
            "path": str(f.resolve()),
            "sha256": sha256,
            "bytes": n_bytes,
        })
    file_records.sort(key=lambda r: r["filename"])

    input_hash = hash_dict([{"filename": r["filename"], "sha256": r["sha256"]} for r in file_records])
    config_hash = hash_dict(config_snapshot)

    return {
        "pathogeniq_version": get_version(),
        "generated_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "input_files": file_records,
        "classifier_format": classifier_format,
        "rank": rank,
        "config_snapshot": config_snapshot,
        "config_hash": config_hash,
        "input_hash": input_hash,
        "actor": actor,
        "store_backend": store_backend,
        "report_content_hash": None,
    }


def verify_manifest(report_dir: str | Path) -> dict:
    """
    Re-verify a run's provenance against what's on disk now.

    Returns {"checks": [{"name", "passed", "detail"}, ...], "all_passed": bool}.
    """
    report_dir = Path(report_dir)
    manifest_path = report_dir / "provenance.json"
    report_path = report_dir / "report.json"
    checks: list[dict] = []

    if not manifest_path.exists():
        checks.append({"name": "manifest_exists", "passed": False,
                        "detail": f"{manifest_path} not found"})
        return {"checks": checks, "all_passed": False}
    checks.append({"name": "manifest_exists", "passed": True, "detail": str(manifest_path)})
    manifest = json.loads(manifest_path.read_text())

    if not report_path.exists():
        checks.append({"name": "report_exists", "passed": False,
                        "detail": f"{report_path} not found"})
        return {"checks": checks, "all_passed": False}
    checks.append({"name": "report_exists", "passed": True, "detail": str(report_path)})
    report = json.loads(report_path.read_text())

    recorded_hash = manifest.get("report_content_hash")
    actual_hash = compute_report_content_hash(report.get("samples", []))
    report_match = recorded_hash is not None and actual_hash == recorded_hash
    checks.append({
        "name": "report_content_hash",
        "passed": report_match,
        "detail": "matches recorded hash" if report_match else
                  f"expected {recorded_hash}, got {actual_hash} — report.json may have been altered",
    })

    for record in manifest.get("input_files", []):
        candidate = Path(record.get("path", "")) if record.get("path") else None
        if candidate is None or not candidate.exists():
            candidate = report_dir / record["filename"]
        if not candidate.exists():
            checks.append({
                "name": f"input_file:{record['filename']}",
                "passed": False,
                "detail": f"{candidate} no longer exists on disk",
            })
            continue
        actual_sha, _ = hash_file(candidate)
        match = actual_sha == record["sha256"]
        checks.append({
            "name": f"input_file:{record['filename']}",
            "passed": match,
            "detail": "hash matches" if match else
                      f"expected {record['sha256']}, got {actual_sha} — file may have been altered",
        })

    all_passed = all(c["passed"] for c in checks)
    return {"checks": checks, "all_passed": all_passed}
