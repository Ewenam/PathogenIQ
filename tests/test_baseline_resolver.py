"""Tests for the tiered baseline resolver, rolling fit, and store matrix accessor."""
import numpy as np
import pandas as pd
import pytest

from pathogeniq.scoring.baseline_resolver import (
    resolve_baseline, history_matrix_from_store, DEFAULT_GLOBAL_PRIOR,
)
from pathogeniq.scoring.baseline_risk import AbundanceBaseline, score_sample_relative


def _rel(n=6, ps=0.4):
    return pd.DataFrame({f"s{i}": {"Pseudomonas": ps, "Vibrio": 0.0, "Bg1": 1 - ps}
                         for i in range(n)})


# ── tier order ───────────────────────────────────────────────────────────────
def test_tier1_controls():
    bl, tier = resolve_baseline(_rel(), control_names=["s0", "s1", "s2"])
    assert tier == "controls" and bl is not None


def test_tier2_rolling_when_enough_history():
    hist = pd.DataFrame({f"h{i}": {"Pseudomonas": 0.4, "Bg1": 0.6} for i in range(10)})
    bl, tier = resolve_baseline(_rel(), history_matrix=hist, min_history=8)
    assert tier == "rolling" and bl is not None


def test_tier2_skipped_when_history_thin():
    hist = pd.DataFrame({f"h{i}": {"Pseudomonas": 0.4, "Bg1": 0.6} for i in range(3)})
    _, tier = resolve_baseline(_rel(), history_matrix=hist, min_history=8)
    assert tier != "rolling"     # falls through to global prior


def test_tier3_global_prior_ships_and_loads():
    assert DEFAULT_GLOBAL_PRIOR.exists()
    bl, tier = resolve_baseline(_rel(), history_matrix=None)
    assert tier == "global_prior" and len(bl.median) > 0


def test_tier4_self_when_nothing_available():
    bl, tier = resolve_baseline(_rel(), history_matrix=None, global_prior_path=None)
    assert tier == "self" and bl is None      # None → caller fits cohort self-baseline


def test_tier_precedence_controls_beats_history():
    hist = pd.DataFrame({f"h{i}": {"Pseudomonas": 0.4, "Bg1": 0.6} for i in range(10)})
    _, tier = resolve_baseline(_rel(), control_names=["s0", "s1"], history_matrix=hist)
    assert tier == "controls"


# ── rolling fit excludes likely-event samples ────────────────────────────────
def test_fit_rolling_excludes_outbreak_samples():
    # 9 clean + 1 heavy outbreak; the outbreak must not inflate the baseline scale
    cols = {f"c{i}": pd.Series({"Pseudomonas": 0.4, "Vibrio": 0.0, "Bg1": 0.6})
            for i in range(9)}
    cols["event"] = pd.Series({"Pseudomonas": 0.2, "Vibrio": 0.6, "Bg1": 0.2})
    hist = pd.DataFrame(cols)
    bl = AbundanceBaseline.fit_rolling(hist, exclude_outlier_frac=0.1)
    # Vibrio was only in the excluded event sample → baseline treats it as absent
    assert bl.median.get("Vibrio", 0.0) == 0.0
    # so a new Vibrio outbreak still scores as elevated
    s = pd.Series({"Vibrio": 0.4, "Pseudomonas": 0.3, "Bg1": 0.3})
    assert score_sample_relative("x", s, bl).breakdown["top_driver"]["genus"] == "Vibrio"


# ── store accessor ────────────────────────────────────────────────────────────
def test_history_matrix_from_store(tmp_path):
    from pathogeniq.temporal.store import TimeSeriesStore
    from pathogeniq.ingestion.reader import SampleSet, Sample
    from pathogeniq.scoring.baseline_risk import score_sample_relative

    store = TimeSeriesStore(tmp_path / "h.db")
    # record two runs with abundances
    for run in range(2):
        mat = pd.DataFrame({"siteA": {"Pseudomonas": 3000, "Bg1": 7000}})
        rel = mat.div(mat.sum(axis=0), axis=1)
        ss = SampleSet(samples=[Sample("siteA")], taxa_matrix=mat, relative_abundance=rel)
        scores = [score_sample_relative("siteA", rel["siteA"], AbundanceBaseline.fit(rel))]
        store.record_run(scores, ss, run_date=f"2026-01-0{run+1}", input_path="x", rank="G")

    mat = history_matrix_from_store(store, sites=["siteA"], window=26)
    assert mat is not None and mat.shape[1] >= 1 and "Pseudomonas" in mat.index


def test_history_matrix_none_without_accessor():
    class _NoAccessor: pass
    assert history_matrix_from_store(_NoAccessor()) is None
