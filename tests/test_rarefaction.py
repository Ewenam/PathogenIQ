"""
tests/test_rarefaction.py
Pure-math tests for the closed-form rarefaction / species-accumulation curve.
No fixtures needed — every check is closed-form or a round-trip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathogeniq.ingestion.reader import Sample, SampleSet
from pathogeniq.planning.rarefaction import (
    expected_richness,
    rarefaction_curve,
    rarefaction_for_sampleset,
)


def test_expected_richness_at_full_depth_equals_observed():
    counts = np.array([500, 300, 150, 50, 0, 0])
    observed = int((counts > 0).sum())
    N = int(counts.sum())
    assert expected_richness(counts, N) == pytest.approx(observed, abs=1e-6)


def test_expected_richness_at_zero_depth_is_zero():
    counts = np.array([500, 300, 150, 50])
    assert expected_richness(counts, 0) == 0.0


def test_expected_richness_monotonic_non_decreasing():
    counts = np.array([1000, 400, 100, 30, 5, 1])
    N = int(counts.sum())
    depths = sorted(set(int(d) for d in np.geomspace(1, N, num=15)))
    richness = [expected_richness(counts, d) for d in depths]
    assert all(b >= a - 1e-9 for a, b in zip(richness, richness[1:]))


def test_single_taxon_always_richness_one():
    counts = np.array([1000])
    for depth in (1, 10, 500, 1000):
        assert expected_richness(counts, depth) == pytest.approx(1.0, abs=1e-9)


def test_rarefaction_for_sampleset_round_trip():
    df = pd.DataFrame(
        {
            "siteA": [500, 300, 0, 50],
            "siteB": [0, 100, 200, 0],
        },
        index=["Escherichia", "Salmonella", "Listeria", "Vibrio"],
    )
    samples = [Sample(name=c) for c in df.columns]
    sampleset = SampleSet(samples=samples, taxa_matrix=df, relative_abundance=df)

    curves = rarefaction_for_sampleset(sampleset)

    assert set(curves.keys()) == {"siteA", "siteB"}

    curve_a = curves["siteA"]
    assert curve_a.observed_richness == (df["siteA"] > 0).sum()
    assert curve_a.total_reads == int(df["siteA"].sum())
    assert len(curve_a.depths) == len(curve_a.richness)
    assert curve_a.depths[-1] == curve_a.total_reads

    curve_b = curves["siteB"]
    assert curve_b.observed_richness == (df["siteB"] > 0).sum()
