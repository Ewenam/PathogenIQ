"""
outbreak/similarity.py
Sample-to-sample compositional similarity for outbreak-source comparison.

Computes Bray-Curtis dissimilarity between samples' relative-abundance
profiles, builds a hierarchical dendrogram, and flags clusters of
near-identical samples (a signature of a shared contamination source)
that also contain a currently-flagged/alerting sample.

This is a substitute for true sourmash/MASH k-mer distance — which would
require raw FASTQ access that classifier-report ingestion doesn't have.
Bray-Curtis over relative abundance is the documented stand-in, not a gap
silently papered over.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist, squareform


@dataclass
class DendrogramResult:
    linkage_matrix: list[list[float]]
    sample_order: list[str]
    distance_matrix: list[list[float]]


def compute_sample_distances(sampleset) -> tuple[np.ndarray, list[str]]:
    """
    Bray-Curtis dissimilarity between every pair of samples, in condensed
    form (as required by scipy.cluster.hierarchy.linkage).

    Two all-zero sample profiles produce a 0/0 NaN distance, which would
    otherwise poison linkage() — replaced with 0.0 (identical/no signal).
    """
    rel = sampleset.relative_abundance
    sample_names = list(rel.columns)
    condensed = pdist(rel.T.values, metric="braycurtis")
    condensed = np.nan_to_num(condensed, nan=0.0)
    return condensed, sample_names


def build_dendrogram(
    condensed: np.ndarray,
    sample_names: list[str],
    method: str = "average",
) -> DendrogramResult | None:
    """Build a hierarchical dendrogram from a condensed distance vector."""
    if len(sample_names) < 3:
        return None

    from scipy.cluster.hierarchy import linkage

    Z = linkage(condensed, method=method)
    distance_matrix = squareform(condensed)
    return DendrogramResult(
        linkage_matrix=Z.tolist(),
        sample_order=sample_names,
        distance_matrix=distance_matrix.tolist(),
    )


def identify_outbreak_clusters(
    dendro: DendrogramResult,
    flagged_samples: set[str],
    threshold: float = 0.3,
) -> list[dict]:
    """
    Group samples into flat clusters at `threshold` distance and surface
    clusters that contain a currently-flagged/alerting sample first —
    these are the candidate shared-source outbreak groups.
    """
    Z = np.array(dendro.linkage_matrix)
    labels = fcluster(Z, t=threshold, criterion="distance")

    groups: dict[int, list[str]] = {}
    for name, label in zip(dendro.sample_order, labels):
        groups.setdefault(int(label), []).append(name)

    clusters = []
    for label, members in groups.items():
        idxs = [dendro.sample_order.index(m) for m in members]
        if len(idxs) > 1:
            sub_distances = [
                dendro.distance_matrix[i][j]
                for n, i in enumerate(idxs)
                for j in idxs[n + 1:]
            ]
            max_distance = max(sub_distances) if sub_distances else 0.0
        else:
            max_distance = 0.0
        contains_flagged = any(m in flagged_samples for m in members)
        clusters.append({
            "members": members,
            "max_distance": round(max_distance, 4),
            "contains_flagged": contains_flagged,
        })

    clusters.sort(key=lambda c: (-(c["contains_flagged"] and len(c["members"]) > 1), -len(c["members"])))
    return clusters
