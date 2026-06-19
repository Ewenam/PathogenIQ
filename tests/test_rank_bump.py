"""
tests/test_rank_bump.py
Rank-bump abundance filtering (species->genus only): low-prevalence/
low-read taxa are summed into a synthesized genus row instead of dropped,
mirroring MARTi's LCA "bump up to parent rank" behavior.

Self-contained: constructs SampleSet/Sample objects directly, same pattern
as test_outbreak_similarity.py.
"""
from __future__ import annotations

import pandas as pd
import pytest

from pathogeniq.ingestion.reader import Sample, SampleSet, filter_taxa


def _species_rank_sampleset() -> SampleSet:
    """One high-prevalence species (Escherichia coli) and two low-prevalence
    same-genus species (Escherichia albertii/fergusonii) that fail the
    filter thresholds — all rows are species-rank (bumpable)."""
    df = pd.DataFrame(
        {
            "s1": [1000, 5, 3],
            "s2": [900, 0, 0],
            "s3": [800, 0, 0],
            "s4": [700, 0, 1],
        },
        index=["Escherichia coli", "Escherichia albertii", "Escherichia fergusonii"],
    )
    df.index.name = "taxon"
    samples = [Sample(name=c) for c in df.columns]
    return SampleSet(samples=samples, taxa_matrix=df, relative_abundance=df)


def test_default_drops_low_prevalence_species():
    sampleset = _species_rank_sampleset()
    original_totals = sampleset.taxa_matrix.sum(axis=0)

    filtered = filter_taxa(sampleset, min_prevalence=0.5, min_total_reads=50)

    assert "Escherichia albertii" not in filtered.taxa_matrix.index
    assert "Escherichia fergusonii" not in filtered.taxa_matrix.index
    assert "Escherichia coli" in filtered.taxa_matrix.index

    filtered_totals = filtered.taxa_matrix.sum(axis=0)
    assert (filtered_totals < original_totals).any()


def test_rank_bump_conserves_reads():
    sampleset = _species_rank_sampleset()
    original_totals = sampleset.taxa_matrix.sum(axis=0)

    filtered = filter_taxa(sampleset, min_prevalence=0.5, min_total_reads=50, rank_bump=True)

    assert "Escherichia coli" in filtered.taxa_matrix.index
    assert "Escherichia" in filtered.taxa_matrix.index  # synthesized genus row

    # Synthesized genus row equals the sum of originally-dropped same-genus species
    dropped_species = sampleset.taxa_matrix.loc[["Escherichia albertii", "Escherichia fergusonii"]]
    expected_genus_row = dropped_species.sum(axis=0)
    pd.testing.assert_series_equal(
        filtered.taxa_matrix.loc["Escherichia"], expected_genus_row,
        check_names=False, check_dtype=False,
    )

    # Read-conservation invariant: nothing lost, only merged
    filtered_totals = filtered.taxa_matrix.sum(axis=0)
    pd.testing.assert_series_equal(filtered_totals, original_totals, check_names=False, check_dtype=False)


def test_genus_rank_input_is_noop_regardless_of_rank_bump():
    df = pd.DataFrame(
        {
            "s1": [1000, 5],
            "s2": [900, 0],
        },
        index=["Escherichia", "Vibrio"],
    )
    samples = [Sample(name=c) for c in df.columns]
    sampleset = SampleSet(samples=samples, taxa_matrix=df, relative_abundance=df)

    no_bump = filter_taxa(sampleset, min_prevalence=0.5, min_total_reads=50, rank_bump=False)
    with_bump = filter_taxa(sampleset, min_prevalence=0.5, min_total_reads=50, rank_bump=True)

    pd.testing.assert_frame_equal(no_bump.taxa_matrix, with_bump.taxa_matrix, check_dtype=False)
