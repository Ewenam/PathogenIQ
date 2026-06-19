"""
tests/test_provenance.py
Per-run audit/evidence trail: manifest shape, hash round-trips, tamper
detection, actor attribution, and SQLite migration idempotency.

Fully self-contained via tmp_path and synthetic Kraken2-format fixtures —
does not depend on tests/fixtures/, which is missing sample_a.report/
sample_b.report.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from pathogeniq import provenance


def _write_synthetic_reports(directory: Path, names: list[str], taxa_reads: dict[str, list[int]]):
    """Write minimal Kraken2-format .report files, one per name in `names`."""
    for i, sample_name in enumerate(names):
        lines = []
        for taxon, reads in taxa_reads.items():
            r = reads[i]
            lines.append(f"10.00\t{r}\t{r}\tG\t1\t  {taxon}")
        (directory / f"{sample_name}.report").write_text("\n".join(lines) + "\n")


@pytest.fixture
def pipeline_run(tmp_path):
    """Run the full pipeline once on synthetic Kraken2 reports."""
    from pathogeniq.pipeline.runner import run, PipelineConfig
    from pathogeniq.temporal.store import TimeSeriesStore

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_synthetic_reports(input_dir, ["siteA", "siteB"], {
        "Escherichia": [4000, 3500], "Bacteroides": [2000, 2200],
        "Salmonella": [800, 100], "Vibrio": [50, 0], "Klebsiella": [300, 250],
    })

    output_dir = tmp_path / "output"
    store = TimeSeriesStore(tmp_path / "history.db")
    cfg = PipelineConfig(output_dir=str(output_dir), alert_threshold=0.6)
    run(input_path=input_dir, config=cfg, rank="G", quiet=True, store=store, actor="test-actor")

    return output_dir, input_dir, store


# ── manifest shape ───────────────────────────────────────────────────────────

def test_manifest_file_presence_and_shape(pipeline_run):
    output_dir, _input_dir, _store = pipeline_run
    report_path = output_dir / "report.json"
    manifest_path = output_dir / "provenance.json"
    assert report_path.exists()
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text())
    expected_keys = {
        "pathogeniq_version", "generated_at", "input_files", "classifier_format",
        "rank", "config_snapshot", "config_hash", "input_hash", "actor",
        "store_backend", "report_content_hash",
    }
    assert expected_keys.issubset(manifest.keys())
    assert manifest["classifier_format"] == "kraken2"
    assert manifest["rank"] == "G"
    assert manifest["actor"] == "test-actor"
    assert manifest["store_backend"] == "TimeSeriesStore"
    assert len(manifest["input_files"]) == 2
    assert manifest["report_content_hash"]

    report = json.loads(report_path.read_text())
    assert report["metadata"]["provenance"]["report_content_hash"] == manifest["report_content_hash"]


# ── hash round-trips ─────────────────────────────────────────────────────────

def test_input_hash_round_trip(tmp_path):
    f1 = tmp_path / "a.report"
    f2 = tmp_path / "b.report"
    f1.write_text("10.00\t100\t100\tG\t1\tEscherichia\n")
    f2.write_text("10.00\t200\t200\tG\t1\tBacteroides\n")

    m1 = provenance.build_manifest([f1, f2], "kraken2", "G", {"x": 1}, "tester", "TimeSeriesStore")
    m2 = provenance.build_manifest([f2, f1], "kraken2", "G", {"x": 1}, "tester", "TimeSeriesStore")
    assert m1["input_hash"] == m2["input_hash"], "input_hash must be order-independent"

    f2.write_text("10.00\t999\t999\tG\t1\tBacteroides\n")
    m3 = provenance.build_manifest([f1, f2], "kraken2", "G", {"x": 1}, "tester", "TimeSeriesStore")
    assert m3["input_hash"] != m1["input_hash"], "changing file content must change input_hash"


def test_report_content_hash_round_trip():
    samples = [{"name": "siteA", "score": 0.42}]
    h1 = provenance.compute_report_content_hash(samples)
    h2 = provenance.compute_report_content_hash(samples)
    assert h1 == h2

    tampered = [{"name": "siteA", "score": 0.9999}]
    assert provenance.compute_report_content_hash(tampered) != h1


# ── verify_manifest ──────────────────────────────────────────────────────────

def test_verify_manifest_passes_on_untouched_run(pipeline_run):
    output_dir, _input_dir, _store = pipeline_run
    result = provenance.verify_manifest(output_dir)
    assert result["all_passed"] is True
    assert all(c["passed"] for c in result["checks"])


def test_verify_manifest_detects_tampered_report(pipeline_run):
    output_dir, _input_dir, _store = pipeline_run
    report_path = output_dir / "report.json"
    report = json.loads(report_path.read_text())
    report["samples"][0]["score"] = 0.9999
    report_path.write_text(json.dumps(report, indent=2))

    result = provenance.verify_manifest(output_dir)
    assert result["all_passed"] is False
    failed_names = {c["name"] for c in result["checks"] if not c["passed"]}
    assert "report_content_hash" in failed_names


def test_verify_manifest_detects_deleted_input_file(pipeline_run):
    output_dir, input_dir, _store = pipeline_run
    deleted = next(input_dir.glob("*.report"))
    deleted.unlink()

    result = provenance.verify_manifest(output_dir)
    assert result["all_passed"] is False
    failed = [c for c in result["checks"] if not c["passed"]]
    assert any(c["name"].startswith("input_file:") for c in failed)


# ── actor attribution ────────────────────────────────────────────────────────

def test_actor_round_trips_through_sqlite_store(pipeline_run):
    _output_dir, _input_dir, store = pipeline_run
    row = store.conn.execute(
        "SELECT actor, classifier_format, pathogeniq_version, input_hash, config_hash "
        "FROM runs ORDER BY run_id DESC LIMIT 1"
    ).fetchone()
    assert row["actor"] == "test-actor"
    assert row["classifier_format"] == "kraken2"
    assert row["input_hash"]
    assert row["config_hash"]
    assert row["pathogeniq_version"]


# ── migration idempotency ───────────────────────────────────────────────────

def test_migration_idempotency(tmp_path):
    from pathogeniq.temporal.store import init_db

    db_path = tmp_path / "fresh.db"
    init_db(db_path)
    init_db(db_path)  # second call against an already-migrated DB must be a no-op

    conn = sqlite3.connect(str(db_path))
    cols = [row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()]
    for expected in ("pathogeniq_version", "classifier_format", "input_hash", "config_hash", "actor"):
        assert cols.count(expected) == 1
    conn.close()
