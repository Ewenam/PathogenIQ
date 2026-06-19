"""
tests/test_outbreak_similarity.py
Bray-Curtis sample distances, dendrogram construction, and outbreak-source
cluster identification.

Self-contained: constructs SampleSet/Sample objects directly (this module
only touches sampleset.relative_abundance, so no need to round-trip through
Kraken2 report parsing).
"""
from __future__ import annotations

import pandas as pd
import pytest

from pathogeniq.ingestion.reader import Sample, SampleSet
from pathogeniq.outbreak.similarity import (
    compute_sample_distances,
    build_dendrogram,
    identify_outbreak_clusters,
)


def _make_sampleset(rel_abund: dict[str, dict[str, float]]) -> SampleSet:
    """rel_abund: {sample_name: {taxon: abundance}}"""
    df = pd.DataFrame(rel_abund).fillna(0.0)  # taxa × samples
    samples = [Sample(name=name) for name in rel_abund]
    return SampleSet(samples=samples, taxa_matrix=df, relative_abundance=df)


def test_identical_profiles_have_zero_distance():
    sampleset = _make_sampleset({
        "s1": {"Escherichia": 0.5, "Bacteroides": 0.5},
        "s2": {"Escherichia": 0.5, "Bacteroides": 0.5},
        "s3": {"Escherichia": 0.1, "Bacteroides": 0.9},
    })
    condensed, names = compute_sample_distances(sampleset)
    dendro = build_dendrogram(condensed, names)
    idx1, idx2 = names.index("s1"), names.index("s2")
    assert dendro.distance_matrix[idx1][idx2] == pytest.approx(0.0, abs=1e-9)


def test_fully_disjoint_profiles_have_distance_one():
    sampleset = _make_sampleset({
        "s1": {"Escherichia": 1.0, "Bacteroides": 0.0},
        "s2": {"Escherichia": 0.0, "Bacteroides": 1.0},
        "s3": {"Escherichia": 0.5, "Bacteroides": 0.5},
    })
    condensed, names = compute_sample_distances(sampleset)
    dendro = build_dendrogram(condensed, names)
    idx1, idx2 = names.index("s1"), names.index("s2")
    assert dendro.distance_matrix[idx1][idx2] == pytest.approx(1.0, abs=1e-9)


def test_outbreak_pair_identified_and_surfaced_first():
    sampleset = _make_sampleset({
        "outbreakA": {"Salmonella": 0.7, "Escherichia": 0.3},
        "outbreakB": {"Salmonella": 0.69, "Escherichia": 0.31},
        "bg1": {"Bacteroides": 0.9, "Vibrio": 0.1},
        "bg2": {"Klebsiella": 0.8, "Bacteroides": 0.2},
        "bg3": {"Vibrio": 0.6, "Klebsiella": 0.4},
    })
    condensed, names = compute_sample_distances(sampleset)
    dendro = build_dendrogram(condensed, names)
    assert dendro is not None

    flagged = {"outbreakA", "outbreakB"}
    clusters = identify_outbreak_clusters(dendro, flagged, threshold=0.3)

    assert clusters, "expected at least one cluster"
    top = clusters[0]
    assert top["contains_flagged"] is True
    assert set(top["members"]) >= flagged


def test_two_all_zero_samples_do_not_produce_nan():
    sampleset = _make_sampleset({
        "s1": {"Escherichia": 0.0, "Bacteroides": 0.0},
        "s2": {"Escherichia": 0.0, "Bacteroides": 0.0},
        "s3": {"Escherichia": 1.0, "Bacteroides": 0.0},
    })
    condensed, names = compute_sample_distances(sampleset)
    assert not any(c != c for c in condensed)  # no NaNs
    dendro = build_dendrogram(condensed, names)
    assert dendro is not None


def test_fewer_than_three_samples_returns_none():
    sampleset = _make_sampleset({
        "s1": {"Escherichia": 0.5, "Bacteroides": 0.5},
        "s2": {"Escherichia": 0.3, "Bacteroides": 0.7},
    })
    condensed, names = compute_sample_distances(sampleset)
    dendro = build_dendrogram(condensed, names)
    assert dendro is None
