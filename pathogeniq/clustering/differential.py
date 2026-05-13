"""
clustering/differential.py
Kruskal-Wallis differential abundance testing across sample clusters.

For each taxon, tests whether relative abundance differs significantly across
the automatically discovered clusters, with Benjamini-Hochberg FDR correction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


@dataclass
class DiffAbundResult:
    taxon: str
    H_stat: float
    p_value: float
    p_adj: float                     # BH-corrected
    significant: bool
    cluster_means: dict              # {cluster_id: mean_relative_abundance}
    fold_change: float               # max_cluster_mean / min_cluster_mean


def differential_abundance(
    sampleset,
    cluster_labels: dict[str, int],
    fdr_alpha: float = 0.05,
    min_prevalence: float = 0.05,
) -> list[DiffAbundResult]:
    """
    Run Kruskal-Wallis per taxon across clusters with BH correction.
    Returns results sorted by adjusted p-value (most significant first).
    Returns [] if fewer than 2 clusters.
    """
    rel = sampleset.relative_abundance  # taxa × samples
    sample_names = list(rel.columns)
    clusters = sorted(set(cluster_labels.values()))

    if len(clusters) < 2:
        return []

    groups: dict[int, list[str]] = {
        c: [s for s in sample_names if cluster_labels.get(s) == c]
        for c in clusters
    }

    candidates: list[DiffAbundResult] = []

    for taxon in rel.index:
        abund = rel.loc[taxon]
        if (abund > 0).mean() < min_prevalence:
            continue

        group_vals = [abund[groups[c]].values for c in clusters if groups[c]]
        if len(group_vals) < 2:
            continue

        try:
            H, p = stats.kruskal(*group_vals)
        except Exception:
            continue

        cmeans = {c: round(float(abund[groups[c]].mean()), 6)
                  for c in clusters if groups[c]}
        vals = list(cmeans.values())
        fc = max(vals) / (min(vals) + 1e-10)

        candidates.append(DiffAbundResult(
            taxon=taxon,
            H_stat=round(float(H), 4),
            p_value=round(float(p), 6),
            p_adj=0.0,
            significant=False,
            cluster_means=cmeans,
            fold_change=round(fc, 4),
        ))

    if not candidates:
        return []

    pvals = [r.p_value for r in candidates]
    reject, padj, _, _ = multipletests(pvals, alpha=fdr_alpha, method="fdr_bh")
    for i, r in enumerate(candidates):
        r.p_adj = round(float(padj[i]), 6)
        r.significant = bool(reject[i])

    candidates.sort(key=lambda r: r.p_adj)
    return candidates
