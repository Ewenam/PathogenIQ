"""
Tests for the polymorphic ingestion router, capability model, read retention,
FASTA windowing, and the sample-level k-mer novelty detector.

Self-contained: no Kraken2, no torch, no external data — synthetic reads are
written to tmp files. The Kraken2-dependent classification path is exercised
only for its graceful-failure behavior.
"""
import random
import pytest

from pathogeniq.ingestion.reader import SampleSet, Sample, CAP_ABUNDANCE, CAP_READS
from pathogeniq.ingestion.router import ingest, _pair_read_files, _is_reads
from pathogeniq.ingestion import classify as C
from pathogeniq.novelty.kmer_novelty import (
    KmerProfiler, KmerNoveltyReference, sample_novelty_scores,
)

import numpy as np
import pandas as pd


def _write_fastq(path, reads):
    with open(path, "w") as fh:
        for i, r in enumerate(reads):
            fh.write(f"@r{i}\n{r}\n+\n{'I' * len(r)}\n")


def _rand_reads(n, length=150, seed=0, alphabet="ACGT"):
    rng = random.Random(seed)
    return ["".join(rng.choice(alphabet) for _ in range(length)) for _ in range(n)]


# ── router: pairing + capabilities ───────────────────────────────────────────
def test_pairing_groups_mates(tmp_path):
    for name in ["s1_1.fastq", "s1_2.fastq", "s2.fastq"]:
        _write_fastq(tmp_path / name, _rand_reads(20))
    grouped = _pair_read_files(list(tmp_path.glob("*.fastq")))
    assert set(grouped) == {"s1", "s2"}
    assert len(grouped["s1"]) == 2 and len(grouped["s2"]) == 1


def test_ingest_reads_only_capabilities(tmp_path):
    _write_fastq(tmp_path / "a_1.fastq", _rand_reads(50, seed=1))
    _write_fastq(tmp_path / "a_2.fastq", _rand_reads(50, seed=2))
    _write_fastq(tmp_path / "b.fastq", _rand_reads(50, seed=3))
    ss = ingest(tmp_path, classify=False, retain_reads=40)
    assert set(ss.sample_names) == {"a", "b"}
    # classify=False → reads retained, no abundance
    for n in ss.sample_names:
        assert ss.capabilities(n) == {CAP_READS}
        assert len(ss.reads_by_sample[n]) == 40
        assert ss.has_reads(n)
    assert ss.samples[0].fastq_r2 is not None  # paired sample kept its mate


def test_capability_requires_nonzero_abundance():
    # empty column must NOT count as abundance capability
    mat = pd.DataFrame({"x": [0.0, 0.0], "y": [1.0, 2.0]}, index=["t1", "t2"])
    ss = SampleSet(samples=[Sample("x"), Sample("y")], taxa_matrix=mat,
                   relative_abundance=mat)
    assert ss.capabilities("x") == set()          # all-zero column
    assert ss.capabilities("y") == {CAP_ABUNDANCE}


# ── read subsampling + FASTA windowing ───────────────────────────────────────
def test_reservoir_subsample_size(tmp_path):
    p = tmp_path / "reads.fastq"
    _write_fastq(p, _rand_reads(1000))
    assert len(C.subsample_reads(p, n=100)) == 100
    assert len(C.subsample_reads(p, n=5000)) == 1000  # capped at available


def test_fasta_windowing():
    seq = "ACGT" * 100  # 400 bp
    windows = C.window_sequences([seq], read_len=150, stride=75)
    assert all(len(w) == 150 for w in windows)
    assert len(windows) == len(range(0, 400 - 150 + 1, 75))


def test_fasta_ingest_windows_long_contigs(tmp_path):
    fa = tmp_path / "contigs.fasta"
    fa.write_text(">c1\n" + "ACGTAC" * 60 + "\n")  # 360 bp
    reads = C.subsample_reads(fa, n=100, read_len=150)
    assert reads and all(len(r) == 150 for r in reads)


# ── classification graceful failure without kraken2 ──────────────────────────
def test_classify_requires_kraken2(tmp_path, monkeypatch):
    p = tmp_path / "x.fastq"
    _write_fastq(p, _rand_reads(30))
    monkeypatch.setattr(C, "kraken2_available", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="kraken2"):
        C.classify_reads([p], "x")


# ── k-mer novelty: explicit-background mode flags a divergent sample ──────────
def test_kmer_novelty_flags_divergent_sample():
    prof = KmerProfiler(k=4)
    # background: GC-poor composition; query: GC-rich → clearly divergent
    bg = [_rand_reads(200, length=120, seed=s, alphabet="ATATATGC") for s in range(12)]
    ref = KmerNoveltyReference.fit(bg, k=4, n_neighbors=5)
    normal = _rand_reads(200, length=120, seed=99, alphabet="ATATATGC")
    novel = _rand_reads(200, length=120, seed=99, alphabet="GCGCGCAT")
    s_normal = ref.score_profiles(prof.sample_profile(normal).reshape(1, -1))[0]
    s_novel = ref.score_profiles(prof.sample_profile(novel).reshape(1, -1))[0]
    assert s_novel > s_normal
    assert s_novel >= 0.5


def test_sample_novelty_scores_selfref_runs():
    samples = {f"s{i}": _rand_reads(100, seed=i) for i in range(6)}
    scores = sample_novelty_scores(samples)
    assert set(scores) == set(samples)
    assert all(0.0 <= v <= 1.0 for v in scores.values())
