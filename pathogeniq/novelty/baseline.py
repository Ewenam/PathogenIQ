"""
novelty/baseline.py
Persisted baseline (background) reference for read-level novelty.

The k-mer novelty detector is reliable in EXPLICIT-BACKGROUND mode: fit on a
site's normal/historical samples, then score new samples against that frozen
baseline (the self-referential fallback is weak — see kmer_novelty). In a
deployed product the baseline is established once from a clean baseline window
and reused for every subsequent run, so it must be persistable per site/org.

This wraps KmerNoveltyReference with save/load (JSON: reference profiles +
calibration + profiler settings) and a small manager for keying baselines by
(org, site).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .kmer_novelty import KmerNoveltyReference, KmerProfiler


def save_baseline(ref: KmerNoveltyReference, path: str | Path) -> None:
    """Serialize a fitted KmerNoveltyReference to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "version": 1,
        "k": ref.profiler.k,
        "n_neighbors": ref.n_neighbors,
        "z_clip": ref.z_clip,
        "mean": ref.mean.tolist(),
        "std": ref.std.tolist(),
        "keep": ref.keep.astype(bool).tolist(),
        "ref_z": ref.ref_z.tolist(),
        "cal_med": ref.cal_med,
        "cal_scale": ref.cal_scale,
    }
    path.write_text(json.dumps(blob))


def load_baseline(path: str | Path) -> KmerNoveltyReference:
    """Reconstruct a KmerNoveltyReference saved by save_baseline."""
    blob = json.loads(Path(path).read_text())
    return KmerNoveltyReference(
        ref_z=np.asarray(blob["ref_z"], dtype=float),
        profiler=KmerProfiler(k=blob["k"]),
        mean=np.asarray(blob["mean"], dtype=float),
        std=np.asarray(blob["std"], dtype=float),
        keep=np.asarray(blob["keep"], dtype=bool),
        n_neighbors=int(blob["n_neighbors"]),
        z_clip=float(blob["z_clip"]),
        cal_med=float(blob["cal_med"]),
        cal_scale=float(blob["cal_scale"]),
    )


def fit_and_save_baseline(reads_per_sample: list[list[str]], path: str | Path,
                          k: int = 6, n_neighbors: int = 5) -> KmerNoveltyReference:
    ref = KmerNoveltyReference.fit(reads_per_sample, k=k, n_neighbors=n_neighbors)
    save_baseline(ref, path)
    return ref


@dataclass
class BaselineManager:
    """Filesystem-backed baseline store keyed by (org, site).

    root/<org>/<site>.baseline.json — one frozen background reference per site.
    The path layout mirrors the object-storage seam (see saas storage), so the
    same manager can be pointed at S3 later by swapping read/write.
    """
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)

    def _path(self, org: str | None, site: str) -> Path:
        org_dir = self.root / (org or "_selfhosted")
        return org_dir / f"{site}.baseline.json"

    def has(self, site: str, org: str | None = None) -> bool:
        return self._path(org, site).exists()

    def fit(self, site: str, reads_per_sample: list[list[str]],
            org: str | None = None, k: int = 6, n_neighbors: int = 5) -> KmerNoveltyReference:
        return fit_and_save_baseline(reads_per_sample, self._path(org, site),
                                     k=k, n_neighbors=n_neighbors)

    def load(self, site: str, org: str | None = None) -> KmerNoveltyReference:
        return load_baseline(self._path(org, site))

    def score(self, site: str, sample_reads: dict[str, list[str]],
              org: str | None = None) -> dict[str, float]:
        """Score new samples for a site against its frozen baseline."""
        ref = self.load(site, org)
        prof = ref.profiler
        names = list(sample_reads)
        X = np.vstack([prof.sample_profile(sample_reads[n]) for n in names])
        scores = ref.score_profiles(X)
        return {n: round(float(s), 4) for n, s in zip(names, scores)}
