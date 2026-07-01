"""
ingestion/classify.py
Turn raw reads (FASTQ) or sequences (FASTA) into a taxa-abundance profile by
running a classifier with a *pinned* database, and retain a small read
subsample so downstream read-level novelty can run.

Why classify server-side instead of trusting an uploaded report?
  A user-uploaded classifier report reflects *their* tool/DB/version/confidence
  settings — unknown and not comparable across samples or tenants. Re-classifying
  the uploaded reads with our pinned DB gives abundance profiles that are
  directly comparable across the whole product (no batch effects) AND yields the
  reads needed for read-level novelty. So `abundance_provenance="classified"`.

No third-party Python deps: reads are parsed and subsampled in pure Python
(reservoir sampling), and classification shells out to the `kraken2` binary.
If `kraken2` is unavailable the caller gets a clear, actionable error — but
read retention / FASTA windowing work without it.
"""
from __future__ import annotations

import gzip
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

DEFAULT_KRAKEN2_DB = Path.home()  # dir containing hash.k2d / opts.k2d / taxo.k2d
DEFAULT_RETAIN_READS = 10_000     # per-sample read subsample kept for novelty


def _open_maybe_gzip(path: Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path)


def iter_fastq(path: Path):
    """Yield sequence strings from a FASTQ (every 4th line), gzip-aware."""
    with _open_maybe_gzip(path) as fh:
        for i, line in enumerate(fh):
            if i % 4 == 1:
                yield line.strip()


def iter_fasta(path: Path):
    """Yield full sequence strings from a FASTA (headers start with '>')."""
    seq: list[str] = []
    with _open_maybe_gzip(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq:
                    yield "".join(seq)
                    seq = []
            else:
                seq.append(line)
    if seq:
        yield "".join(seq)


def reservoir_subsample(iterable, n: int, seed: int = 42) -> list[str]:
    """Uniform size-`n` reservoir sample over a stream (single pass, O(n) memory)."""
    rng = random.Random(seed)
    reservoir: list[str] = []
    for i, item in enumerate(iterable):
        if i < n:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < n:
                reservoir[j] = item
    return reservoir


def window_sequences(seqs, read_len: int = 150, stride: int = 75, max_windows: int | None = None):
    """Chop long sequences (e.g. FASTA contigs/genomes) into read-length windows
    so the read-trained novelty models see inputs of the right size."""
    out: list[str] = []
    for s in seqs:
        if len(s) <= read_len:
            out.append(s)
            continue
        for i in range(0, len(s) - read_len + 1, stride):
            out.append(s[i:i + read_len])
            if max_windows and len(out) >= max_windows:
                return out
    return out


def subsample_reads(
    path: str | Path,
    n: int = DEFAULT_RETAIN_READS,
    seed: int = 42,
    read_len: int = 150,
) -> list[str]:
    """Retain up to `n` reads from a FASTQ/FASTA upload for read-level novelty.
    FASTA sequences longer than `read_len` are windowed into read-length pieces."""
    path = Path(path)
    name = path.name.lower()
    if name.endswith((".fastq", ".fq", ".fastq.gz", ".fq.gz")):
        return reservoir_subsample(iter_fastq(path), n, seed)
    if name.endswith((".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz", ".fna.gz")):
        windowed = window_sequences(iter_fasta(path), read_len=read_len, max_windows=n * 4)
        return reservoir_subsample(iter(windowed), n, seed)
    raise ValueError(f"Unrecognized sequence file type: {path}")


def kraken2_available(kraken2_bin: str = "kraken2") -> bool:
    return shutil.which(kraken2_bin) is not None


def classify_reads(
    seq_paths: list[str | Path],
    sample_name: str,
    db_path: str | Path = DEFAULT_KRAKEN2_DB,
    rank: str = "G",
    kraken2_bin: str = "kraken2",
    confidence: float = 0.05,
    threads: int = 4,
    retain_reads: int = DEFAULT_RETAIN_READS,
    seed: int = 42,
) -> tuple[pd.Series, list[str]]:
    """
    Classify uploaded reads/sequences with a pinned Kraken2 DB → (counts, reads).

    Returns:
      counts: pd.Series of read counts at `rank`, indexed by taxon, named
              `sample_name` (same contract as reader.load_kraken_report).
      reads:  retained read subsample (list[str]) for read-level novelty.

    Raises RuntimeError with an actionable message if kraken2 is unavailable or
    the DB is missing — the caller decides whether to fall back to an uploaded
    report.
    """
    seq_paths = [Path(p) for p in seq_paths]
    reads = subsample_reads(seq_paths[0], n=retain_reads, seed=seed)

    if not kraken2_available(kraken2_bin):
        raise RuntimeError(
            f"'{kraken2_bin}' not found on PATH. Install Kraken2 to classify raw "
            "reads server-side, or upload a precomputed classifier report instead."
        )
    db = Path(db_path)
    if not (db / "hash.k2d").exists():
        raise RuntimeError(
            f"Kraken2 DB not found at {db} (expected hash.k2d/opts.k2d/taxo.k2d)."
        )

    from .reader import load_kraken_report

    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / f"{sample_name}.report"
        paired = len(seq_paths) >= 2
        cmd = [
            kraken2_bin, "--db", str(db),
            "--threads", str(threads),
            "--confidence", str(confidence),
            "--report", str(report),
            "--output", "-",
        ]
        if paired:
            cmd += ["--paired", str(seq_paths[0]), str(seq_paths[1])]
        else:
            cmd += [str(seq_paths[0])]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
        counts = load_kraken_report(report, rank=rank)

    counts.name = sample_name
    return counts, reads
