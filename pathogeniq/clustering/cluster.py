"""
clustering/cluster.py
Unsupervised hierarchical clustering of samples on CLR-transformed abundances.

Automatically selects the number of clusters (k) by finding the largest gap
in the Ward linkage dendrogram — no pre-labeling required.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score

from ..community.graph import clr_transform


@dataclass
class ClusterResult:
    n_clusters: int
    labels: dict[str, int]            # {sample_name: cluster_id}
    silhouette: float                 # silhouette score; higher = better separation
    cluster_profiles: dict            # {cluster_id: {taxon: mean_relative_abundance}}


def cluster_samples(
    sampleset,
    max_k: int = 6,
) -> ClusterResult:
    """
    Cluster samples by their CLR-transformed abundance profiles.
    Returns a ClusterResult with auto-selected k.
    Falls back to a single cluster when there are fewer than 3 samples.
    """
    mat = sampleset.taxa_matrix
    sample_names = list(mat.columns) if mat is not None else []

    fallback = ClusterResult(
        n_clusters=1,
        labels={s: 0 for s in sample_names},
        silhouette=0.0,
        cluster_profiles={},
    )

    if mat is None or mat.empty or len(sample_names) < 3:
        return fallback

    clr = clr_transform(mat)
    X = clr.values.T  # samples × taxa

    Z = linkage(X, method="ward")

    # Pick k by the biggest gap between consecutive merge distances (top-down view)
    dists = Z[:, 2]
    gaps = np.diff(dists[::-1])   # largest merge distances first
    k_auto = int(np.argmax(gaps)) + 2   # +2: gap at index i means k = i+2
    k_auto = max(2, min(k_auto, max_k, len(sample_names) - 1))

    labels_arr = fcluster(Z, t=k_auto, criterion="maxclust") - 1  # 0-indexed

    sil = 0.0
    if len(set(labels_arr)) > 1 and len(labels_arr) > 2:
        try:
            sil = float(silhouette_score(X, labels_arr))
        except Exception:
            pass

    labels = {name: int(labels_arr[i]) for i, name in enumerate(sample_names)}

    # Mean relative abundance per taxon per cluster
    rel = sampleset.relative_abundance  # taxa × samples
    cluster_profiles: dict = {}
    for c in sorted(set(labels.values())):
        members = [n for n, cl in labels.items() if cl == c]
        if members:
            cluster_profiles[c] = {
                taxon: round(float(rel.loc[taxon, members].mean()), 6)
                for taxon in rel.index
            }

    return ClusterResult(
        n_clusters=k_auto,
        labels=labels,
        silhouette=round(sil, 4),
        cluster_profiles=cluster_profiles,
    )
