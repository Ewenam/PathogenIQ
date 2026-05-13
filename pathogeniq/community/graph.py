"""
community/graph.py
Builds a co-occurrence graph from CLR-normalized taxa abundances.

Fixes the degeneracy in the original SBM repo:
  - Applies Benjamini-Hochberg FDR correction to Spearman p-values
  - Only includes edges where corrected p < fdr_alpha
  - This prevents the dense graph (density > 0.25) that caused all block
    probabilities to converge to ~1.0
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def clr_transform(counts: pd.DataFrame) -> pd.DataFrame:
    """
    Centered log-ratio transform for compositional data.
    Adds pseudo-count of 0.5 to handle zeros.
    counts: taxa × samples
    """
    X = counts.values.astype(float) + 0.5
    log_X = np.log(X)
    gm = log_X.mean(axis=0)           # geometric mean per sample
    clr = log_X - gm
    return pd.DataFrame(clr, index=counts.index, columns=counts.columns)


def build_cooccurrence_graph(
    sampleset,
    spearman_threshold: float = 0.45,
    fdr_alpha: float = 0.05,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute pairwise Spearman correlations between taxa (across samples),
    apply BH-FDR correction, and threshold to build adjacency matrix.

    Returns:
        adjacency: (n_taxa × n_taxa) binary symmetric matrix
        taxa_names: list of taxon names corresponding to rows/cols
    """
    clr = clr_transform(sampleset.taxa_matrix)
    taxa = list(clr.index)
    n = len(taxa)
    X = clr.values  # (n_taxa × n_samples)

    # Pairwise Spearman across samples (each row is a taxon's profile)
    rho_matrix = np.zeros((n, n))
    pval_matrix = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            r, p = stats.spearmanr(X[i], X[j])
            rho_matrix[i, j] = rho_matrix[j, i] = r
            pval_matrix[i, j] = pval_matrix[j, i] = p

    # Flatten upper triangle for FDR correction
    upper_idx = np.triu_indices(n, k=1)
    pvals_flat = pval_matrix[upper_idx]
    rhos_flat = rho_matrix[upper_idx]

    reject, pvals_corrected, _, _ = multipletests(pvals_flat, alpha=fdr_alpha, method="fdr_bh")

    # Build adjacency: edge if FDR-significant AND |rho| >= threshold
    adj = np.zeros((n, n), dtype=int)
    for idx, (i, j) in enumerate(zip(*upper_idx)):
        if reject[idx] and abs(rhos_flat[idx]) >= spearman_threshold:
            adj[i, j] = adj[j, i] = 1

    density = adj.sum() / (n * (n - 1))
    n_edges = adj.sum() // 2

    print(f"  Graph: {n} taxa, {n_edges} edges, density={density:.4f}")
    if density > 0.15:
        print(f"  Warning: density={density:.3f} is high — consider raising spearman_threshold or fdr_alpha")

    return adj, taxa, rho_matrix
