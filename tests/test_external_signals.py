"""
tests/test_external_signals.py
External signal fusion: cross-validates internal risk-score history against
an external (qPCR/ddPCR/case-count) feed via Pearson correlation.

Follows test_provenance.py's pattern: synthetic Kraken2-format .report
fixtures + the real run() pipeline, fully self-contained via tmp_path.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pathogeniq.external.signals import validate_site


def _write_synthetic_reports(directory: Path, names: list[str], taxa_reads: dict[str, list[int]]):
    """Write minimal Kraken2-format .report files, one per name in `names`."""
    for i, sample_name in enumerate(names):
        lines = []
        for taxon, reads in taxa_reads.items():
            r = reads[i]
            lines.append(f"10.00\t{r}\t{r}\tG\t1\t  {taxon}")
        (directory / f"{sample_name}.report").write_text("\n".join(lines) + "\n")


def _run_once(tmp_path, store, run_date, taxa_reads, tag, external_signals=None):
    from pathogeniq.pipeline.runner import run, PipelineConfig

    input_dir = tmp_path / f"input_{tag}"
    input_dir.mkdir()
    _write_synthetic_reports(input_dir, ["siteA", "siteB"], taxa_reads)

    output_dir = tmp_path / f"output_{tag}"
    cfg = PipelineConfig(output_dir=str(output_dir), alert_threshold=0.6, run_date=run_date)
    run(
        input_path=input_dir, config=cfg, rank="G", quiet=True, store=store,
        actor="test-actor", external_signals=external_signals,
    )
    return json.loads((output_dir / "report.json").read_text())


@pytest.fixture
def history_and_scores(tmp_path):
    """Build 3 history runs (no external signals yet) for siteA/siteB and
    return their siteA scores keyed by run_date, plus the shared store."""
    from pathogeniq.temporal.store import TimeSeriesStore

    store = TimeSeriesStore(tmp_path / "history.db")
    dates = ["2026-01-01", "2026-01-08", "2026-01-15"]
    # Only Escherichia (risk_weight 0.65, below the single-pathogen "direct
    # detection" thresholds) is a recognized pathogen here, and it's the only
    # one present — this keeps direct_detection_score at 0 so the composite
    # score (driven by abundance_score) varies smoothly with read count
    # instead of saturating at a step-function plateau.
    taxa_variants = [
        {"Escherichia": [600, 200], "Akkermansia": [3000, 3000], "Faecalibacterium": [3000, 3000],
         "Bifidobacterium": [3000, 3000], "Ruminococcus": [3000, 3000]},
        {"Escherichia": [1200, 220], "Akkermansia": [3000, 3000], "Faecalibacterium": [3000, 3000],
         "Bifidobacterium": [3000, 3000], "Ruminococcus": [3000, 3000]},
        {"Escherichia": [2400, 240], "Akkermansia": [3000, 3000], "Faecalibacterium": [3000, 3000],
         "Bifidobacterium": [3000, 3000], "Ruminococcus": [3000, 3000]},
    ]

    siteA_scores = {}
    for i, (run_date, taxa_reads) in enumerate(zip(dates, taxa_variants)):
        report = _run_once(tmp_path, store, run_date, taxa_reads, tag=f"hist{i}")
        sample = next(s for s in report["samples"] if s["name"] == "siteA")
        siteA_scores[run_date] = sample["score"]

    return tmp_path, store, dates, siteA_scores


# ── full pipeline integration ───────────────────────────────────────────────

def test_matching_site_has_high_correlation(history_and_scores):
    tmp_path, store, dates, siteA_scores = history_and_scores

    external_path = tmp_path / "external.csv"
    with open(external_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["site", "date", "value"])
        for d in dates:
            writer.writerow(["siteA", d, 10 * siteA_scores[d] + 1])

    current_taxa = {"Escherichia": [3600, 230], "Akkermansia": [3000, 3000], "Faecalibacterium": [3000, 3000],
                     "Bifidobacterium": [3000, 3000], "Ruminococcus": [3000, 3000]}
    report = _run_once(tmp_path, store, "2026-01-22", current_taxa, tag="current",
                        external_signals=str(external_path))

    siteA = next(s for s in report["samples"] if s["name"] == "siteA")
    ev = siteA["external_validation"]
    assert ev is not None
    assert ev["n_matched"] == 3
    assert ev["pearson_r"] > 0.99
    assert ev["concordant"] is True


def test_non_matching_site_has_no_external_validation(history_and_scores):
    tmp_path, store, dates, siteA_scores = history_and_scores

    external_path = tmp_path / "external.csv"
    with open(external_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["site", "date", "value"])
        for d in dates:
            writer.writerow(["siteA", d, 10 * siteA_scores[d] + 1])

    current_taxa = {"Escherichia": [3600, 230], "Akkermansia": [3000, 3000], "Faecalibacterium": [3000, 3000],
                     "Bifidobacterium": [3000, 3000], "Ruminococcus": [3000, 3000]}
    report = _run_once(tmp_path, store, "2026-01-22", current_taxa, tag="current2",
                        external_signals=str(external_path))

    siteB = next(s for s in report["samples"] if s["name"] == "siteB")
    assert siteB["external_validation"] is None


# ── direct unit test of validate_site's insufficient-overlap path ──────────

def test_insufficient_overlap_returns_none_correlation():
    import pandas as pd

    external_df = pd.DataFrame({
        "site": ["siteA", "siteA"],
        "date": ["2026-01-01", "2026-01-08"],
        "value": [1.0, 2.0],
        "signal_type": ["external", "external"],
    })
    history = [
        {"run_date": "2026-01-01", "risk_score": 0.2},
        {"run_date": "2026-01-08", "risk_score": 0.3},
    ]
    result = validate_site(
        site="siteA", external_df=external_df, history=history,
        current_score=0.4, current_date="2026-01-15",
    )
    assert result.pearson_r is None
    assert result.n_matched == 2
    assert "Insufficient overlapping dates" in result.summary
