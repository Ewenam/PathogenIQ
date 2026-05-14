"""Tests for ingestion/reader.py — Kraken2 report parsing and merging."""
import pytest
import pandas as pd
from pathlib import Path
from pathogeniq.ingestion.reader import (
    load_kraken_report, load_sample_directory, filter_taxa,
)


def test_load_single_report(fixture_dir):
    series = load_kraken_report(fixture_dir / "sample_a.report", rank="G")
    assert isinstance(series, pd.Series)
    assert "Escherichia" in series.index
    assert "Klebsiella" in series.index
    assert series["Escherichia"] == 820
    assert series["Klebsiella"] == 430


def test_load_directory(two_sample_dir):
    ss = load_sample_directory(two_sample_dir, rank="G", pattern="*.report")
    assert len(ss.samples) == 2
    assert ss.taxa_matrix is not None
    assert ss.relative_abundance is not None
    # All relative abundance columns should sum to ~1
    col_sums = ss.relative_abundance.sum(axis=0)
    assert (col_sums - 1.0).abs().max() < 1e-6


def test_sample_names(sampleset):
    names = sampleset.sample_names
    assert "sample_a" in names
    assert "sample_b" in names


def test_filter_taxa_removes_rare(sampleset):
    filtered = filter_taxa(sampleset, min_prevalence=0.0, min_total_reads=1000)
    # Only taxa with >=1000 total reads across both samples should remain
    assert (filtered.taxa_matrix.sum(axis=1) >= 1000).all()


def test_paired_end_merge(tmp_path):
    """_1 / _2 reports for the same sample should be merged into one."""
    import shutil
    from pathlib import Path
    fixtures = Path(__file__).parent / "fixtures"
    shutil.copy(fixtures / "sample_a.report", tmp_path / "site1_1.report")
    shutil.copy(fixtures / "sample_b.report", tmp_path / "site1_2.report")
    ss = load_sample_directory(tmp_path, rank="G", pattern="*.report")
    assert len(ss.samples) == 1
    assert ss.samples[0].name == "site1"
    # Merged counts should be the sum of both individual files
    e_col_1 = load_kraken_report(fixtures / "sample_a.report", rank="G")["Escherichia"]
    e_col_2 = load_kraken_report(fixtures / "sample_b.report", rank="G").get("Escherichia", 0)
    merged_e = ss.taxa_matrix.loc["Escherichia", "site1"]
    assert merged_e == e_col_1 + e_col_2
