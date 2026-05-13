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


def _merge_paired_reports(
    samples: list[Sample],
    series_list: list[pd.Series],
) -> tuple[list[Sample], list[pd.Series]]:
    """
    Detect paired-end Kraken2 reports named <base>_1 / <base>_2 and merge
    them by summing raw counts.  Equivalent to running Kraken2 with --paired.

    Samples without a matching counterpart are kept as-is (name unchanged).
    """
    name_map: dict[str, tuple[Sample, pd.Series]] = {
        s.name: (s, ser) for s, ser in zip(samples, series_list)
    }
    processed: set[str] = set()
    out_samples: list[Sample] = []
    out_series: list[pd.Series] = []

    for s, ser in zip(samples, series_list):
        if s.name in processed:
            continue

        m = re.match(r'^(.+)_([12])$', s.name)
        if m:
            base = m.group(1)
            other = base + ('_2' if m.group(2) == '1' else '_1')
            if other in name_map and other not in processed:
                _, other_ser = name_map[other]
                merged = ser.add(other_ser, fill_value=0)
                merged.name = base
                out_series.append(merged)
                out_samples.append(Sample(name=base, kraken_report=s.kraken_report))
                processed.update({s.name, other})
                continue

        # Unpaired — keep as-is
        out_series.append(ser)
        out_samples.append(s)
        processed.add(s.name)

    n_merged = (len(samples) - len(out_samples))
    if n_merged > 0:
        print(f"  Merged {n_merged} paired-end file(s) into "
              f"{len(out_samples) - (len(samples) - 2 * n_merged)} combined sample(s)")

    return out_samples, out_series


def load_sample_directory(
    directory: str | Path,
    rank: str = "G",
    pattern: str = "*.report",
) -> SampleSet:
    """
    Load all Kraken2 reports from a directory.
    Automatically merges paired-end reports (<name>_1 / <name>_2) by summing
    their raw counts — equivalent to Kraken2 --paired processing.
    Builds a taxa × samples count matrix and computes relative abundance.
    """
    directory = Path(directory)
    report_files = sorted(directory.glob(pattern))
    if not report_files:
        report_files = sorted(directory.glob("*.txt"))
    if not report_files:
        raise FileNotFoundError(f"No Kraken2 reports found in {directory}")

    raw_samples: list[Sample] = []
    raw_series: list[pd.Series] = []
    for rp in report_files:
        raw_samples.append(Sample(name=rp.stem, kraken_report=rp))
        raw_series.append(load_kraken_report(rp, rank=rank))

    samples, series_list = _merge_paired_reports(raw_samples, raw_series)

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
