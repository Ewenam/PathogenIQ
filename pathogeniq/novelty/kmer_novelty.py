"""
novelty/kmer_novelty.py
Sample-level, model-free read novelty via k-mer composition.

Motivation
----------
Per-read lineage/variant identity is near-impossible (a random 150 bp read
rarely covers a variable site), so read-level representations sit at chance.
But a *sample* is a pool of reads: aggregating the k-mer composition over
thousands of reads amplifies the small frequency shifts at variable sites into
a strong signal. Empirically, mean 6-mer frequency separates SARS-CoV-2
lineages at the sample level (5-fold LR acc ~0.87) while both raw reads and the
current VQ-VAE embeddings sit at chance.

This module turns a sample's reads into a mean k-mer frequency profile and
scores novelty as the sample's distance from a reference (background) set of
sample profiles. It is dependency-light (numpy + scikit-learn only; no torch),
so it can serve as the near-term read-level novelty signal for the pipeline
while the VQ-VAE is retrained.

Output is a per-sample novelty score in [0, 1] suitable for the composite
risk score's novelty term.

Two operating modes (important):
  * EXPLICIT BACKGROUND (recommended): fit the reference on a clean baseline
    (e.g. a site's historical/normal samples), score new samples against it.
    This is the operationally correct WBE pattern and separates well
    (validated AUROC ~0.66-0.90 for held-out novel lineages).
  * SELF-REFERENTIAL (fallback): no baseline supplied, the cohort is its own
    background (leave-one-out). This reliably flags gross composition spread
    but is weak for a *lone* rare outlier whose signal lives in a few k-mers —
    such a sample is too rare to move the reference statistics it is judged
    against. Prefer an explicit baseline whenever one exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np


def _kmer_index(k: int) -> dict[str, int]:
    return {"".join(p): i for i, p in enumerate(product("ACGT", repeat=k))}


@dataclass
class KmerProfiler:
    """Turns reads into L1-normalized k-mer frequency vectors."""
    k: int = 6

    def __post_init__(self):
        self.index = _kmer_index(self.k)
        self.dim = len(self.index)

    def read_profile(self, seq: str) -> np.ndarray:
        v = np.zeros(self.dim)
        seq = seq.upper()
        idx = self.index
        k = self.k
        for i in range(len(seq) - k + 1):
            j = idx.get(seq[i:i + k])
            if j is not None:
                v[j] += 1.0
        return v

    def sample_profile(self, reads: list[str]) -> np.ndarray:
        """Mean k-mer frequency over a sample's reads (the sample representation)."""
        if not reads:
            return np.zeros(self.dim)
        acc = np.zeros(self.dim)
        for r in reads:
            p = self.read_profile(r)
            s = p.sum()
            if s > 0:
                acc += p / s
        return acc / len(reads)


def read_fastq(path: str | Path, limit: int | None = None) -> list[str]:
    seqs: list[str] = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i % 4 == 1:
                seqs.append(line.strip())
                if limit and len(seqs) >= limit:
                    break
    return seqs


def _l2norm(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


@dataclass
class KmerNoveltyReference:
    """k-NN novelty detector over reference SAMPLE k-mer profiles, in a
    per-k-mer *standardized* space.

    Sample k-mer profiles are ~identical in bulk (shared genomic background);
    the discriminative signal (a variant, a community shift) lives in a small
    number of k-mers. Raw cosine/Euclidean distance is therefore dominated by
    the shared bulk and misses the signal. We z-score each k-mer against the
    reference (upweighting dimensions the background holds tight, so a query's
    departure there stands out), drop near-constant dimensions, clip extreme
    z-scores, and take Euclidean k-NN distance in that whitened space. Scores
    are calibrated to [0,1] against the reference's own leave-one-out kNN
    distances.
    """
    ref_z: np.ndarray             # (n_ref, dim) standardized reference profiles
    profiler: KmerProfiler
    mean: np.ndarray
    std: np.ndarray
    keep: np.ndarray              # dims with above-floor reference variance
    n_neighbors: int
    z_clip: float
    cal_med: float
    cal_scale: float

    @staticmethod
    def _knn_dist(Q: np.ndarray, R: np.ndarray, k: int, exclude_self: bool) -> np.ndarray:
        # Euclidean distances query→reference, mean of k nearest
        d = np.sqrt(np.maximum(
            ((Q[:, None, :] - R[None, :, :]) ** 2).sum(-1), 0.0))
        if exclude_self:
            np.fill_diagonal(d, np.inf)
        d.sort(axis=1)
        return d[:, :k].mean(axis=1)

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        Z = (X[:, self.keep] - self.mean[self.keep]) / self.std[self.keep]
        return np.clip(Z, -self.z_clip, self.z_clip)

    @classmethod
    def fit(cls, reference_reads_per_sample: list[list[str]],
            k: int = 6, n_neighbors: int = 5, z_clip: float = 8.0,
            std_floor_quantile: float = 0.5):
        prof = KmerProfiler(k=k)
        X = np.vstack([prof.sample_profile(r) for r in reference_reads_per_sample])
        # Standardize per k-mer against the reference. Mean/std is used (rather
        # than robust median/MAD) because the intended mode is an EXPLICIT clean
        # background (baseline samples) with novel samples scored separately —
        # there the reference isn't contaminated by the anomaly, and mean/std
        # gives the sharpest separation. (Self-referential scoring, where the
        # anomaly is in its own reference, is a weaker fallback — see score_self.)
        mean = X.mean(0)
        std = X.std(0)
        # keep only dimensions the reference actually varies in (above a
        # floor), so shared-background k-mers with ~0 variance don't blow up.
        floor = max(1e-9, float(np.quantile(std[std > 0], std_floor_quantile)) if (std > 0).any() else 1e-9)
        keep = std > floor
        if keep.sum() < 2:
            keep = std > 0
        std_safe = np.where(std > 0, std, 1.0)
        obj = cls(ref_z=None, profiler=prof, mean=mean, std=std_safe, keep=keep,
                  n_neighbors=min(n_neighbors, max(1, X.shape[0] - 1)),
                  z_clip=z_clip, cal_med=0.0, cal_scale=1.0)
        obj.ref_z = obj._standardize(X)
        loo = cls._knn_dist(obj.ref_z, obj.ref_z, obj.n_neighbors, exclude_self=True)
        med = float(np.median(loo))
        mad = float(np.median(np.abs(loo - med))) + 1e-9
        obj.cal_med = med
        obj.cal_scale = 3.0 * 1.4826 * mad
        return obj

    def _calibrate(self, knn: np.ndarray) -> np.ndarray:
        return np.clip((knn - self.cal_med) / (self.cal_scale + 1e-9), 0.0, 1.0)

    def score_profiles(self, profiles: np.ndarray) -> np.ndarray:
        """Score NEW query samples (not in the reference)."""
        Z = self._standardize(profiles)
        knn = self._knn_dist(Z, self.ref_z, self.n_neighbors, exclude_self=False)
        return self._calibrate(knn)

    def score_self(self) -> np.ndarray:
        """Leave-one-out novelty for the reference samples themselves
        (self-referential cohort anomaly). Must exclude each sample's own
        zero-distance self-match — otherwise every sample scores below the
        calibration median and collapses to 0."""
        knn = self._knn_dist(self.ref_z, self.ref_z, self.n_neighbors, exclude_self=True)
        return self._calibrate(knn)

    def score_sample(self, reads: list[str]) -> float:
        """Per-sample novelty in [0, 1] for one sample's reads."""
        p = self.profiler.sample_profile(reads).reshape(1, -1)
        return float(self.score_profiles(p)[0])


def sample_novelty_scores(
    samples: dict[str, list[str]],
    reference: KmerNoveltyReference | None = None,
    k: int = 6,
    n_neighbors: int = 5,
) -> dict[str, float]:
    """Score a set of samples.

    samples: {sample_name: [reads]}. If `reference` is None, the samples
    themselves are used as their own background (self-referential anomaly —
    flags samples whose composition is unusual within the cohort).
    Returns {sample_name: novelty_score in [0,1]}.
    """
    names = list(samples)
    if reference is None:
        # self-referential: samples ARE their own background → leave-one-out scoring
        reference = KmerNoveltyReference.fit(
            [samples[n] for n in names], k=k, n_neighbors=n_neighbors)
        scores = reference.score_self()
    else:
        prof = reference.profiler
        X = np.vstack([prof.sample_profile(samples[n]) for n in names])
        scores = reference.score_profiles(X)
    return {n: round(float(s), 4) for n, s in zip(names, scores)}
