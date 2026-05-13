"""
ingestion/reader.py
Reads Kraken2 report files and raw FASTQ samples into a unified SampleSet.
Supports single samples or directories of reports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Sample:
    name: str
    kraken_report: Path | None = None
    fastq_r1: Path | None = None
    fastq_r2: Path | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class SampleSet:
    samples: list[Sample]
    taxa_matrix: pd.DataFrame | None = None   # taxa × samples count matrix
    relative_abundance: pd.DataFrame | None = None

    @property
    def sample_names(self) -> list[str]:
        return [s.name for s in self.samples]


def load_kraken_report(path: Path, rank: str = "G") -> pd.Series:
    """
    Parse a Kraken2 report file into a Series of read counts indexed by taxon name.
    rank: 'G'=genus, 'S'=species, 'F'=family
    """
    counts: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            pct, covered, direct, r, taxid, name = parts[:6]
            r = r.strip()
            if r != rank:
                continue
            taxon = name.strip()
            try:
                counts[taxon] = int(direct)
            except ValueError:
                continue
    return pd.Series(counts, dtype=float, name=path.stem)


def load_sample_directory(
    directory: str | Path,
    rank: str = "G",
    pattern: str = "*.report",
) -> SampleSet:
    """
    Load all Kraken2 reports from a directory.
    Builds a taxa × samples count matrix and computes relative abundance.
    """
    directory = Path(directory)
    report_files = sorted(directory.glob(pattern))
    if not report_files:
        report_files = sorted(directory.glob("*.txt"))
    if not report_files:
        raise FileNotFoundError(f"No Kraken2 reports found in {directory}")

    samples = []
    series_list = []
    for rp in report_files:
        sample = Sample(name=rp.stem, kraken_report=rp)
        samples.append(sample)
        series_list.append(load_kraken_report(rp, rank=rank))

    counts = pd.concat(series_list, axis=1).fillna(0)
    rel = counts.div(counts.sum(axis=0), axis=1).fillna(0)

    return SampleSet(
        samples=samples,
        taxa_matrix=counts,
        relative_abundance=rel,
    )


def load_count_matrix(path: str | Path, transpose: bool = False) -> SampleSet:
    """
    Load a pre-built taxa × samples count matrix (TSV/CSV).
    Use transpose=True if the file is samples × taxa.
    """
    path = Path(path)
    sep = "\t" if path.suffix in (".tsv", ".txt") else ","
    df = pd.read_csv(path, sep=sep, index_col=0)
    if transpose:
        df = df.T
    rel = df.div(df.sum(axis=0), axis=1).fillna(0)
    samples = [Sample(name=col) for col in df.columns]
    return SampleSet(samples=samples, taxa_matrix=df, relative_abundance=rel)


def filter_taxa(
    sampleset: SampleSet,
    min_prevalence: float = 0.1,
    min_total_reads: int = 50,
) -> SampleSet:
    """
    Remove taxa that are too rare to produce reliable co-occurrence signal.
    min_prevalence: fraction of samples in which taxon must appear (reads > 0)
    min_total_reads: minimum summed reads across all samples
    """
    mat = sampleset.taxa_matrix
    n_samples = mat.shape[1]

    prevalence = (mat > 0).sum(axis=1) / n_samples
    total = mat.sum(axis=1)

    keep = (prevalence >= min_prevalence) & (total >= min_total_reads)
    filtered = mat.loc[keep]
    rel = filtered.div(filtered.sum(axis=0), axis=1).fillna(0)

    return SampleSet(
        samples=sampleset.samples,
        taxa_matrix=filtered,
        relative_abundance=rel,
    )
