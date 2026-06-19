"""
tests/test_mock_community.py
ZymoBIOMICS-style mock-community ground-truth validation.

Generates replicate samples matching a fully-known reference community
composition (no separate "background", unlike generate_scenario), and runs
them through the real scoring pipeline via the same _run_scenario_pipeline
helper benchmark/runner.py uses.
"""
from __future__ import annotations

import pytest

from pathogeniq.benchmark.synthetic import (
    ZYMOBIOMICS_COMPOSITION,
    generate_mock_community,
)
from pathogeniq.benchmark.runner import _run_scenario_pipeline
from pathogeniq.scoring.risk import PATHOGEN_DB


def test_generate_mock_community_shape():
    dataset = generate_mock_community(n_replicates=5, seed=42)

    assert set(dataset.count_matrix.index) == set(ZYMOBIOMICS_COMPOSITION.keys())
    assert dataset.count_matrix.shape[1] == 5

    for sample_name, fracs in dataset.pathogen_fractions.items():
        assert sum(fracs.values()) == pytest.approx(1.0, abs=1e-3)


def test_mock_community_detects_all_pathogen_db_genera():
    """Mirrors MARTi's "detected all 10 expected species" framing: every
    PATHOGEN_DB-recognized genus in the ZymoBIOMICS standard must be
    detected in at least one replicate — no false negatives."""
    dataset = generate_mock_community(n_replicates=5, seed=42)
    risk_scores = _run_scenario_pipeline(dataset, alert_threshold=0.6, quick=True)

    expected_genera = {g for g in ZYMOBIOMICS_COMPOSITION if g in PATHOGEN_DB}
    assert expected_genera == {
        "Listeria", "Pseudomonas", "Bacillus", "Escherichia",
        "Salmonella", "Enterococcus", "Staphylococcus",
    }

    detected_genera = {
        p["genus"] for rs in risk_scores for p in rs.detected_pathogens
    }
    assert expected_genera <= detected_genera


def test_lactobacillus_never_flagged():
    """Lactobacillus is deliberately absent from PATHOGEN_DB (benign/
    probiotic genus) — sanity check against false positives."""
    assert "Lactobacillus" not in PATHOGEN_DB

    dataset = generate_mock_community(n_replicates=5, seed=42)
    risk_scores = _run_scenario_pipeline(dataset, alert_threshold=0.6, quick=True)

    detected_genera = {
        p["genus"] for rs in risk_scores for p in rs.detected_pathogens
    }
    assert "Lactobacillus" not in detected_genera


def test_bacillus_anthrax_genus_collision_is_documented_behavior():
    """
    Known, expected limitation — NOT a bug: PATHOGEN_DB scores at genus
    granularity (scoring/risk.py's `genus = taxon.split()[0]`), and the
    ZymoBIOMICS standard's harmless Bacillus subtilis component collides
    with the "Bacillus" key, which is mapped to anthrax-tier risk
    (risk_weight=0.90, disease="Anthrax (B.anthracis)"). Kraken2 itself
    can't distinguish B. subtilis from B. anthracis without species-level
    confidence, so this isn't unique to PathogenIQ.

    This test is intentionally an assertion, not a skip: if PATHOGEN_DB or
    score_sample's thresholds ever change this behavior, this test must
    break loudly rather than silently stop covering the finding.
    """
    assert PATHOGEN_DB["Bacillus"]["disease"] == "Anthrax (B.anthracis)"
    assert PATHOGEN_DB["Bacillus"]["risk_weight"] >= 0.85

    dataset = generate_mock_community(n_replicates=5, seed=42)
    risk_scores = _run_scenario_pipeline(dataset, alert_threshold=0.6, quick=True)

    bacillus_hits = [
        p for rs in risk_scores for p in rs.detected_pathogens
        if p["genus"] == "Bacillus"
    ]
    assert bacillus_hits, "expected Bacillus to be detected in the mock community"
    assert all(p["disease"] == "Anthrax (B.anthracis)" for p in bacillus_hits)

    bacillus_sample_scores = [
        rs.score for rs in risk_scores
        if any(p["genus"] == "Bacillus" for p in rs.detected_pathogens)
    ]
    assert all(s >= 0.6 for s in bacillus_sample_scores), (
        "expected the harmless B. subtilis component to score HIGH/CRITICAL "
        "due to genus-level collision with B. anthracis — this is the "
        "documented limitation, not a bug"
    )
