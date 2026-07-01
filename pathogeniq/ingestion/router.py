"""
ingestion/router.py
One entry point that accepts heterogeneous uploads and returns a unified
`SampleSet` with per-sample capability flags.

    ingest(path) -> SampleSet

Accepted inputs (auto-detected):
  * directory of classifier reports  (Kraken2 / Bracken / MetaPhlAn)   → abundance
  * count matrix TSV/CSV             (taxa × samples)                   → abundance
  * FASTQ / FASTA files (or a dir of them)                             → abundance + reads
  * a MIX of reports and read files in one directory                   → per-sample tiers

Read files are classified with a pinned Kraken2 DB (see classify.py), which both
normalizes abundance across the whole cohort and retains a read subsample so
read-level novelty can run. Samples built from reports keep their uploaded
abundance (abundance_provenance="uploaded"); samples built from reads are
"classified". Everything shares one taxa × samples matrix, so the co-occurrence
graph / SBM / risk scoring are input-agnostic; read-level novelty simply
activates for the subset of samples that carry reads.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .reader import Sample, SampleSet
from .classify import classify_reads, subsample_reads, kraken2_available, DEFAULT_KRAKEN2_DB

_FASTQ_EXT = (".fastq", ".fq", ".fastq.gz", ".fq.gz")
_FASTA_EXT = (".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz", ".fna.gz")
_REPORT_EXT = (".report", ".txt", ".tsv", ".kreport", ".kraken")
_MATRIX_EXT = (".tsv", ".csv")


def _is_reads(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(_FASTQ_EXT) or n.endswith(_FASTA_EXT)


def _base_and_mate(name: str) -> tuple[str, str | None]:
    """Return (sample_base, mate) for paired-end naming (_1/_2, _R1/_R2)."""
    m = re.match(r"^(.+?)[._](R?)([12])$", name)
    if m:
        return m.group(1), m.group(3)
    return name, None


def _pair_read_files(files: list[Path]) -> dict[str, list[Path]]:
    """Group read files into samples, pairing _1/_2 (or _R1/_R2) mates."""
    groups: dict[str, dict[str, Path]] = {}
    order: list[str] = []
    for f in sorted(files):
        stem = f.name
        for ext in _FASTQ_EXT + _FASTA_EXT:
            if stem.lower().endswith(ext):
                stem = stem[: -len(ext)]
                break
        base, mate = _base_and_mate(stem)
        if base not in groups:
            groups[base] = {}
            order.append(base)
        groups[base][mate or "1"] = f
    return {b: [groups[b][k] for k in sorted(groups[b])] for b in order}


def _assemble(series_list: list[pd.Series], samples: list[Sample],
              reads_by_sample: dict[str, list[str]] | None) -> SampleSet:
    counts = pd.concat(series_list, axis=1).fillna(0)
    rel = counts.div(counts.sum(axis=0), axis=1).fillna(0)
    return SampleSet(samples=samples, taxa_matrix=counts, relative_abundance=rel,
                     reads_by_sample=reads_by_sample or None)


def ingest(
    input_path: str | Path,
    rank: str = "G",
    fmt: str = "auto",
    db_path: str | Path = DEFAULT_KRAKEN2_DB,
    kraken2_bin: str = "kraken2",
    retain_reads: int = 10_000,
    classify: bool = True,
) -> SampleSet:
    """Ingest any supported input into a unified SampleSet (see module docstring)."""
    from .reader import load_sample_directory, load_count_matrix

    input_path = Path(input_path)

    # ── single file ──────────────────────────────────────────────────────────
    if input_path.is_file():
        if _is_reads(input_path):
            return _ingest_read_files([input_path], rank, db_path, kraken2_bin,
                                      retain_reads, classify)
        # report or matrix file
        if input_path.suffix.lower() in _MATRIX_EXT:
            # a count matrix has taxa as the index and many sample columns; a
            # single classifier report is one column. Disambiguate by width.
            probe = pd.read_csv(input_path, sep=None, engine="python", nrows=1)
            if probe.shape[1] >= 3:
                return load_count_matrix(input_path)
        # fall through: treat as a one-file report directory
        return load_sample_directory(input_path.parent, rank=rank, fmt=fmt,
                                     pattern=input_path.name)

    # ── directory: may mix reports and read files ─────────────────────────────
    all_files = [p for p in sorted(input_path.iterdir()) if p.is_file()]
    read_files = [p for p in all_files if _is_reads(p)]
    report_files = [p for p in all_files
                    if not _is_reads(p) and p.suffix.lower() in _REPORT_EXT]

    if read_files and not report_files:
        return _ingest_read_files(read_files, rank, db_path, kraken2_bin,
                                  retain_reads, classify)
    if report_files and not read_files:
        return load_sample_directory(input_path, rank=rank, fmt=fmt)

    if read_files and report_files:
        # mixed cohort: build both, then merge on the taxa axis
        ss_reports = load_sample_directory(input_path, rank=rank, fmt=fmt)
        ss_reads = _ingest_read_files(read_files, rank, db_path, kraken2_bin,
                                      retain_reads, classify)
        counts = pd.concat([ss_reports.taxa_matrix, ss_reads.taxa_matrix],
                           axis=1).fillna(0)
        rel = counts.div(counts.sum(axis=0), axis=1).fillna(0)
        return SampleSet(
            samples=ss_reports.samples + ss_reads.samples,
            taxa_matrix=counts, relative_abundance=rel,
            reads_by_sample=ss_reads.reads_by_sample,
        )

    raise FileNotFoundError(f"No ingestible files found in {input_path}")


def _ingest_read_files(read_files, rank, db_path, kraken2_bin, retain_reads,
                       classify) -> SampleSet:
    grouped = _pair_read_files([Path(f) for f in read_files])
    series_list, samples, reads_by_sample = [], [], {}
    for base, files in grouped.items():
        fmt = "fasta" if any(_f.name.lower().endswith(_FASTA_EXT)
                             for _f in files) else "fastq"
        if classify:
            counts, reads = classify_reads(
                files, sample_name=base, db_path=db_path, rank=rank,
                kraken2_bin=kraken2_bin, retain_reads=retain_reads,
            )
            prov = "classified"
        else:
            # no classification available/desired: retain reads only, no abundance.
            counts = pd.Series(dtype=float, name=base)
            reads = subsample_reads(files[0], n=retain_reads)
            prov = "reads_only"
        series_list.append(counts)
        reads_by_sample[base] = reads
        samples.append(Sample(name=base, fastq_r1=files[0],
                              fastq_r2=files[1] if len(files) > 1 else None,
                              source_format=fmt, abundance_provenance=prov))
    return _assemble(series_list, samples, reads_by_sample)
