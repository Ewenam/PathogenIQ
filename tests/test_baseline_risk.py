"""Tests for the baseline-relative risk scorer — the trustworthy replacement
for the absolute-threshold scorer that saturated on endemic wastewater flora."""
import numpy as np
import pandas as pd

from pathogeniq.scoring.baseline_risk import (
    AbundanceBaseline, score_sample_relative, score_all_relative, _squash,
)
from pathogeniq.ingestion.reader import SampleSet, Sample


def _baseline_with_endemic_flora(seed=0, n=20):
    rng = np.random.default_rng(seed)
    cols = []
    for _ in range(n):
        ps = max(0, rng.normal(0.35, 0.05))
        st = max(0, rng.normal(0.05, 0.02))
        rest = 1 - ps - st
        cols.append(pd.Series({"Pseudomonas": ps, "Streptococcus": st,
                               "Bg1": rest * 0.6, "Bg2": rest * 0.4}))
    return AbundanceBaseline.fit(pd.concat(cols, axis=1).clip(lower=0))


def test_squash_monotone_bounded():
    assert _squash(0) == 0.0
    assert 0 < _squash(3) < _squash(30) < 1.0


def test_endemic_flora_at_baseline_scores_low():
    bl = _baseline_with_endemic_flora()
    s = pd.Series({"Pseudomonas": 0.38, "Streptococcus": 0.06, "Bg1": 0.56})
    r = score_sample_relative("x", s / s.sum(), bl)
    assert r.level == "LOW"
    assert r.score < 0.3


def test_true_outbreak_absent_from_baseline_alerts():
    bl = _baseline_with_endemic_flora()
    s = pd.Series({"Vibrio": 0.40, "Pseudomonas": 0.30, "Bg1": 0.30})
    r = score_sample_relative("x", s / s.sum(), bl)
    assert r.level in ("HIGH", "CRITICAL")
    assert r.breakdown["top_driver"]["genus"] == "Vibrio"


def test_tripwire_fires_on_low_abundance_bsl_agent():
    bl = _baseline_with_endemic_flora()
    s = pd.Series({"Yersinia": 0.02, "Pseudomonas": 0.35, "Bg1": 0.63})
    r = score_sample_relative("x", s / s.sum(), bl)
    assert r.level == "CRITICAL"
    assert r.breakdown["tripwire_hits"]
    assert r.score >= 0.85


def test_community_signal_excluded_from_score():
    bl = _baseline_with_endemic_flora()
    s = pd.Series({"Pseudomonas": 0.38, "Bg1": 0.62})
    r = score_sample_relative("x", s / s.sum(), bl)
    assert r.community_signal == 0.0


def test_endemic_dominant_genus_does_not_saturate_cohort():
    # the real-data failure mode: a genus endemic at ~40% across the whole
    # cohort must NOT push every sample HIGH under a cohort self-baseline.
    rng = np.random.default_rng(1)
    cols = {}
    for i in range(30):
        ps = max(0, rng.normal(0.40, 0.03))          # endemic Pseudomonas
        rest = 1 - ps
        cols[f"s{i}"] = pd.Series({"Pseudomonas": ps, "Bg1": rest * 0.7, "Bg2": rest * 0.3})
    mat = pd.concat(cols, axis=1)
    rel = mat.div(mat.sum(0), axis=1)
    ss = SampleSet(samples=[Sample(n) for n in rel.columns],
                   taxa_matrix=mat, relative_abundance=rel)
    scores = score_all_relative(ss)          # cohort self-baseline
    levels = [r.level for r in scores]
    assert levels.count("HIGH") == 0         # nothing saturates
    assert all(r.score < 0.4 for r in scores)


def test_injected_spike_is_flagged_over_endemic_cohort():
    rng = np.random.default_rng(2)
    cols = {}
    for i in range(30):
        ps = max(0, rng.normal(0.40, 0.03))
        ac = 0.30 if i == 0 else max(0, rng.normal(0.01, 0.005))  # one Acinetobacter spike
        rest = max(0.01, 1 - ps - ac)
        cols[f"s{i}"] = pd.Series({"Pseudomonas": ps, "Acinetobacter": ac, "Bg1": rest})
    mat = pd.concat(cols, axis=1)
    rel = mat.div(mat.sum(0), axis=1)
    ss = SampleSet(samples=[Sample(n) for n in rel.columns],
                   taxa_matrix=mat, relative_abundance=rel)
    scores = score_all_relative(ss)
    top = scores[0]                          # highest-scoring sample
    assert top.sample_name == "s0"
    assert top.breakdown["top_driver"]["genus"] == "Acinetobacter"
    assert top.score > scores[-1].score      # spike ranks above normal samples


def test_clr_space_endemic_low_outbreak_ranks_top():
    # CLR (compositional) baseline: endemic flora benign, novel outbreak ranks top
    import pandas as pd
    from pathogeniq.scoring.baseline_risk import AbundanceBaseline
    rng = np.random.default_rng(3)
    cols = {}
    for i in range(15):
        ps = max(0.01, rng.normal(0.40, 0.03)); rest = 1 - ps
        cols[f"c{i}"] = pd.Series({"Pseudomonas": ps, "Vibrio": 0.0, "Bg1": rest})
    cols["clean"] = pd.Series({"Pseudomonas": 0.41, "Vibrio": 0.0, "Bg1": 0.59})
    cols["outbreak"] = pd.Series({"Pseudomonas": 0.30, "Vibrio": 0.35, "Bg1": 0.35})
    mat = pd.concat(cols, axis=1); rel = mat.div(mat.sum(axis=0), axis=1)
    ss = SampleSet(samples=[Sample(n) for n in rel.columns], taxa_matrix=mat,
                   relative_abundance=rel)
    scores = {r.sample_name: r for r in score_all_relative(ss, space="clr")}
    assert scores["clean"].level == "LOW"
    assert scores["outbreak"].score > scores["clean"].score
    assert scores["outbreak"].breakdown["top_driver"]["genus"] == "Vibrio"


def test_clr_baseline_save_load_roundtrip(tmp_path):
    import pandas as pd
    from pathogeniq.scoring.baseline_risk import AbundanceBaseline, score_sample_relative
    rng = np.random.default_rng(4)
    cols = {f"c{i}": pd.Series({"Pseudomonas": max(0.01, rng.normal(0.4, 0.03)),
                                "Bg1": 0.5}) for i in range(12)}
    rel = pd.concat(cols, axis=1)
    bl = AbundanceBaseline.fit(rel, space="clr")
    bl.save(tmp_path / "clr.json")
    bl2 = AbundanceBaseline.load(tmp_path / "clr.json")
    assert bl2.space == "clr"
    s = pd.Series({"Vibrio": 0.3, "Pseudomonas": 0.3, "Bg1": 0.4})
    assert abs(score_sample_relative("x", s, bl).score
               - score_sample_relative("x", s, bl2).score) < 1e-9


def test_corroboration_cannot_alert_alone():
    # endemic sample (no elevation) with maxed novelty must NOT reach the alert
    # band — corroboration is a bounded bonus, not an independent trigger.
    bl = _baseline_with_endemic_flora()
    s = pd.Series({"Pseudomonas": 0.40, "Bg1": 0.60})
    r = score_sample_relative("x", s / s.sum(), bl, novelty_score=1.0)
    assert r.score < 0.6 and r.level in ("LOW", "MODERATE")


def test_strong_outbreak_reaches_high_band():
    # the composite must let a strongly-elevated dangerous genus reach HIGH/CRITICAL
    bl = _baseline_with_endemic_flora()
    s = pd.Series({"Vibrio": 0.45, "Pseudomonas": 0.30, "Bg1": 0.25})
    r = score_sample_relative("x", s / s.sum(), bl)
    assert r.score >= 0.6

def test_zcap_limits_rare_genus_artifact():
    # a genus at ~0 in the baseline (tiny MAD) must not produce an unbounded z
    from pathogeniq.scoring.baseline_risk import Z_CAP, SQUASH_K
    assert SQUASH_K["clr"] == 2.0
    rng = np.random.default_rng(9)
    cols = {}
    for i in range(20):
        # Clostridium present in only 2 samples -> near-zero MAD
        cl = 0.05 if i < 2 else 0.0
        ps = max(0.01, rng.normal(0.4, 0.03)); rest = max(0.01, 1 - ps - cl)
        cols[f"c{i}"] = pd.Series({"Pseudomonas": ps, "Clostridium": cl, "Bg1": rest})
    rel = pd.concat(cols, axis=1); rel = rel.div(rel.sum(axis=0), axis=1)
    bl = AbundanceBaseline.fit(rel)
    r = score_sample_relative("spike", pd.Series({"Clostridium": 0.10, "Pseudomonas": 0.4, "Bg1": 0.5}), bl)
    assert r.detected_pathogens[0]["exceedance_z"] <= Z_CAP
