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
    kraken_report_2: Path | None = None  # second file of a merged paired-end pair, if any
    fastq_r1: Path | None = None
    fastq_r2: Path | None = None
    metadata: dict = field(default_factory=dict)
    source_format: str = "kraken2"  # "kraken2" | "bracken" | "metaphlan" | "fastq" | "fasta" | "matrix"
    # Provenance of the abundance for this sample: "uploaded" (user-supplied
    # classifier report) vs "classified" (we ran the classifier on uploaded
    # reads with our pinned DB). Cross-provenance comparisons carry batch-effect
    # risk, so the pipeline/report can surface this.
    abundance_provenance: str = "uploaded"


# Capability tokens a sample can carry. Downstream stages are gated on these.
CAP_ABUNDANCE = "abundance"   # has a taxa-abundance profile (always required)
CAP_READS = "reads"           # has retained sequence reads → read-level novelty


@dataclass
class SampleSet:
    samples: list[Sample]
    taxa_matrix: pd.DataFrame | None = None   # taxa × samples count matrix
    relative_abundance: pd.DataFrame | None = None
    # Optional per-sample retained read subsample (sample_name -> list of read
    # strings). Present only for FASTQ/FASTA uploads; enables read-level novelty.
    reads_by_sample: dict[str, list[str]] | None = None

    @property
    def sample_names(self) -> list[str]:
        return [s.name for s in self.samples]

    def has_reads(self, name: str) -> bool:
        return bool(self.reads_by_sample and self.reads_by_sample.get(name))

    def capabilities(self, name: str) -> set[str]:
        """Per-sample capability set that gates downstream stages."""
        caps = set()
        if (self.taxa_matrix is not None and name in self.taxa_matrix.columns
                and float(self.taxa_matrix[name].sum()) > 0):
            caps.add(CAP_ABUNDANCE)
        if self.has_reads(name):
            caps.add(CAP_READS)
        return caps

    def samples_with_reads(self) -> list[str]:
        return [s.name for s in self.samples if self.has_reads(s.name)]


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
                other_sample, other_ser = name_map[other]
                merged = ser.add(other_ser, fill_value=0)
                merged.name = base
                out_series.append(merged)
                out_samples.append(Sample(
                    name=base,
                    kraken_report=s.kraken_report,
                    kraken_report_2=other_sample.kraken_report,
                    source_format=s.source_format,
                ))
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


# Extra glob patterns tried, in order, only when the caller's `pattern` is
# still the default and matches nothing — lets a directory of Bracken or
# MetaPhlAn output "just work" with no new flags, without risking double-
# counting files that would happen if multiple patterns were OR'd together.
_FALLBACK_PATTERNS = ("*.bracken", "*_profile.tsv", "*.tsv", "*.txt")


def load_sample_directory(
    directory: str | Path,
    rank: str = "G",
    pattern: str = "*.report",
    format: str = "auto",
) -> SampleSet:
    """
    Load all classifier reports from a directory (Kraken2, Bracken, or
    MetaPhlAn4 — auto-detected per file by content unless `format` is set
    explicitly).
    Automatically merges paired-end reports (<name>_1 / <name>_2) by summing
    their raw counts — equivalent to Kraken2 --paired processing.
    Builds a taxa × samples count matrix and computes relative abundance.
    """
    directory = Path(directory)
    report_files = sorted(directory.glob(pattern))
    if not report_files and pattern == "*.report":
        for fallback in _FALLBACK_PATTERNS:
            report_files = sorted(directory.glob(fallback))
            if report_files:
                break
    if not report_files:
        raise FileNotFoundError(f"No classifier reports found in {directory}")

    from .formats import load_report

    raw_samples: list[Sample] = []
    raw_series: list[pd.Series] = []
    for rp in report_files:
        series, resolved_fmt = load_report(rp, rank=rank, fmt=format)
        raw_samples.append(Sample(name=rp.stem, kraken_report=rp, source_format=resolved_fmt))
        raw_series.append(series)

    formats_seen = {s.source_format for s in raw_samples}
    if len(formats_seen) > 1:
        print(f"  Warning: mixing classifier formats in one run ({sorted(formats_seen)}) — "
              "count semantics may not be comparable across samples.")

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
    rank_bump: bool = False,
) -> SampleSet:
    """
    Remove taxa that are too rare to produce reliable co-occurrence signal.
    min_prevalence: fraction of samples in which taxon must appear (reads > 0)
    min_total_reads: minimum summed reads across all samples

    rank_bump: if True, species-rank (or finer) taxa that fail the
        thresholds are not discarded — their reads are summed into a
        synthesized genus-level row (named taxon.split()[0], the same
        convention scoring/risk.py already uses) instead. Mirrors MARTi's
        LCA "bump up to parent rank" behavior, scoped to species->genus
        since PathogenIQ has no general taxonomy table. No-op for
        already-genus-rank input (taxon.split()[0] == taxon — there is no
        parent to bump to, so a failing genus-rank row is dropped exactly
        as it would be with rank_bump=False).
    """
    mat = sampleset.taxa_matrix
    n_samples = mat.shape[1]

    prevalence = (mat > 0).sum(axis=1) / n_samples
    total = mat.sum(axis=1)

    keep_mask = (prevalence >= min_prevalence) & (total >= min_total_reads)

    if not rank_bump:
        filtered = mat.loc[keep_mask]
    else:
        kept = mat.loc[keep_mask]
        dropped = mat.loc[~keep_mask]
        genus_of = dropped.index.to_series().apply(lambda t: t.split()[0])
        bumpable_mask = (genus_of != dropped.index).to_numpy()
        if not bumpable_mask.any():
            filtered = kept
        else:
            bumped = dropped.loc[bumpable_mask].groupby(genus_of[bumpable_mask]).sum()
            filtered = kept.add(bumped, fill_value=0)  # outer-aligns on index; introduces new genus rows automatically

    rel = filtered.div(filtered.sum(axis=0), axis=1).fillna(0)

    return SampleSet(
        samples=sampleset.samples,
        taxa_matrix=filtered,
        relative_abundance=rel,
        reads_by_sample=sampleset.reads_by_sample,  # taxa filtering doesn't touch reads
    )
