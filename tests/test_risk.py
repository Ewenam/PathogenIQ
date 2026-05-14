"""Tests for scoring/risk.py — risk scorer and alert thresholds."""
import pytest
import pandas as pd
from pathogeniq.scoring.risk import score_sample, score_all_samples, _score_level


def test_score_level_boundaries():
    assert _score_level(0.0)  == "LOW"
    assert _score_level(0.29) == "LOW"
    assert _score_level(0.30) == "MODERATE"
    assert _score_level(0.59) == "MODERATE"
    assert _score_level(0.60) == "HIGH"
    assert _score_level(0.79) == "HIGH"
    assert _score_level(0.80) == "CRITICAL"
    assert _score_level(1.0)  == "CRITICAL"


def test_clean_sample_scores_low():
    """A sample with no known pathogens should score LOW."""
    taxa = pd.Series({"Caulobacter": 0.5, "Methanobrevibacter": 0.3, "Rhizobium": 0.2})
    result = score_sample("clean_site", taxa)
    assert result.level == "LOW"
    assert result.score < 0.35
    assert result.detected_pathogens == []


def test_high_risk_pathogen_triggers_alert():
    """Yersinia at 20% abundance (BSL-3, risk_weight=0.9) must reach HIGH."""
    taxa = pd.Series({"Yersinia": 0.20, "Caulobacter": 0.80})
    result = score_sample("danger_site", taxa)
    assert result.level in ("HIGH", "CRITICAL")
    assert result.is_alert(threshold=0.6)
    assert any(p["genus"] == "Yersinia" for p in result.detected_pathogens)


def test_moderate_pathogen_at_low_abundance():
    """Salmonella at 2% with benign background should not reach HIGH threshold."""
    # Caulobacter and Methanobrevibacter are not in PATHOGEN_DB — true benign background
    taxa = pd.Series({"Salmonella": 0.02, "Caulobacter": 0.50, "Methanobrevibacter": 0.48})
    result = score_sample("low_contam", taxa)
    assert not result.is_alert(threshold=0.6)


def test_multi_pathogen_load():
    """Multiple pathogens together should elevate score via additive load rule."""
    taxa = pd.Series({
        "Vibrio": 0.15,
        "Salmonella": 0.10,
        "Klebsiella": 0.08,
        "Caulobacter": 0.67,
    })
    result = score_sample("multi_site", taxa)
    assert result.score >= 0.35


def test_score_all_samples(sampleset):
    """score_all_samples should return one RiskScore per sample."""
    results = score_all_samples(sampleset)
    assert len(results) == 2
    names = {r.sample_name for r in results}
    assert "sample_a" in names
    assert "sample_b" in names
    # sample_b has Yersinia + Vibrio + Salmonella — should alert
    b = next(r for r in results if r.sample_name == "sample_b")
    assert b.score > 0.5


def test_amr_annotations_populated(sampleset):
    """After AMR annotation step, detected ESKAPE pathogens should have entries."""
    from pathogeniq.amr.annotator import annotate_amr
    results = score_all_samples(sampleset)
    b = next(r for r in results if r.sample_name == "sample_b")
    amr = annotate_amr(b.detected_pathogens)
    genera = {a.genus for a in amr}
    # sample_b has Klebsiella-level pathogens — check at least one is annotated
    assert len(amr) > 0
    assert any(g in genera for g in ("Salmonella", "Vibrio", "Yersinia",
                                      "Pseudomonas", "Enterococcus", "Staphylococcus"))
