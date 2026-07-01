"""Tests for Farrington aberration detection and negative-control baselines."""
import numpy as np
import pandas as pd

from pathogeniq.temporal.farrington import farrington_detect, farrington_series
from pathogeniq.scoring.baseline_risk import (
    AbundanceBaseline, score_all_relative, score_sample_relative,
)
from pathogeniq.ingestion.reader import SampleSet, Sample


# ── Farrington ───────────────────────────────────────────────────────────────
STABLE = [100, 105, 98, 102, 99, 101, 97, 103, 100, 104]


def test_stable_baseline_no_alarm():
    r = farrington_detect(STABLE, 102)
    assert not r["alarm"] and r["exceedance"] < 0.2


def test_spike_alarms():
    r = farrington_detect(STABLE, 180)
    assert r["alarm"] and r["exceedance"] > 0.5


def test_trend_is_accounted_for():
    trend = [50, 55, 62, 70, 78, 88, 99, 110, 124, 140]
    on_trend = farrington_detect(trend, 155)
    spike = farrington_detect(trend, 320)
    assert not on_trend["alarm"]          # on-trend value is expected
    assert spike["alarm"]                 # a real jump above trend alarms
    assert on_trend["trend_slope"] > 0


def test_past_outbreak_downweighted():
    # a past outbreak (300) must not pull the expected baseline up
    hist = [100, 102, 98, 300, 101, 99, 103, 100, 102, 98]
    r = farrington_detect(hist, 120)
    assert r["expected"] < 140            # expected stays near the true ~100


def test_insufficient_history_fallback():
    r = farrington_detect([100, 102], 300)
    assert r["note"].startswith("insufficient")
    assert r["alarm"]                     # still catches an obvious jump


def test_series_warmup_then_detects():
    vals = STABLE + [190]
    res = farrington_series(vals)
    assert res[0]["note"] == "warmup"
    assert res[-1]["alarm"]               # the trailing spike alarms


def test_continuous_scores_via_scale():
    scores = [0.10, 0.12, 0.09, 0.11, 0.10, 0.13, 0.08, 0.12]
    assert farrington_detect(scores, 0.45, scale=100)["alarm"]
    assert not farrington_detect(scores, 0.11, scale=100)["alarm"]


# ── negative-control / clean-window baseline ─────────────────────────────────
def _matrix():
    rng = np.random.default_rng(0)
    cols = {}
    # 10 clean controls (endemic Pseudomonas only) + 2 test samples
    for i in range(10):
        ps = max(0, rng.normal(0.40, 0.03)); rest = 1 - ps
        cols[f"ctrl{i}"] = pd.Series({"Pseudomonas": ps, "Vibrio": 0.0, "Bg1": rest})
    cols["test_clean"] = pd.Series({"Pseudomonas": 0.41, "Vibrio": 0.0, "Bg1": 0.59})
    cols["test_outbreak"] = pd.Series({"Pseudomonas": 0.30, "Vibrio": 0.35, "Bg1": 0.35})
    mat = pd.concat(cols, axis=1)
    rel = mat.div(mat.sum(axis=0), axis=1)
    return SampleSet(samples=[Sample(n) for n in rel.columns], taxa_matrix=mat,
                     relative_abundance=rel)


def test_control_baseline_flags_only_outbreak():
    ss = _matrix()
    controls = [f"ctrl{i}" for i in range(10)]
    scores = {r.sample_name: r for r in
              score_all_relative(ss, control_names=controls)}
    assert scores["test_clean"].level == "LOW"          # endemic → benign vs controls
    assert scores["test_outbreak"].level in ("HIGH", "CRITICAL")
    assert scores["test_outbreak"].breakdown["top_driver"]["genus"] == "Vibrio"


def test_baseline_save_load_roundtrip(tmp_path):
    ss = _matrix()
    bl = AbundanceBaseline.from_controls(ss.relative_abundance,
                                         [f"ctrl{i}" for i in range(10)])
    bl.save(tmp_path / "bl.json")
    bl2 = AbundanceBaseline.load(tmp_path / "bl.json")
    s = ss.relative_abundance["test_outbreak"]
    r1 = score_sample_relative("x", s, bl)
    r2 = score_sample_relative("x", s, bl2)
    assert abs(r1.score - r2.score) < 1e-9


def test_run_farrington_all_sites_with_store():
    from pathogeniq.temporal.farrington import run_farrington_all_sites

    class _FakeStore:
        def __init__(self, hist): self._hist = hist
        def get_site_history(self, site, last_n=52):
            return [{"risk_score": v} for v in self._hist.get(site, [])]

    class _RS:
        def __init__(self, name, score): self.sample_name, self.score = name, score

    store = _FakeStore({"siteA": [0.10, 0.12, 0.09, 0.11, 0.10, 0.12, 0.08, 0.11]})
    res = run_farrington_all_sites(store, [_RS("siteA", 0.55)], window=16)
    assert res["siteA"]["alarm"]                 # 0.55 vs ~0.10 baseline → aberration
    res2 = run_farrington_all_sites(store, [_RS("siteA", 0.11)], window=16)
    assert not res2["siteA"]["alarm"]            # on-baseline → no alarm
