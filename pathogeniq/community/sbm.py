"""
community/sbm.py
Stochastic Block Model via variational EM with ICL model selection.
Adapted and fixed from the original repo — now uses the FDR-corrected graph
so block probabilities are no longer degenerate (~1.0).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class SBMResult:
    k: int                          # optimal number of communities
    q: np.ndarray                   # (n × k) soft community assignments
    labels: np.ndarray              # (n,) hard assignments (argmax of q)
    block_p: np.ndarray             # (k × k) block connection probability matrix
    pi: np.ndarray                  # (k,) community mixing proportions
    icl_scores: dict[int, float]    # ICL score per K tried
    taxa_names: list[str]


def _sbm_em(adj: np.ndarray, k: int, n_init: int = 10, max_iter: int = 300) -> tuple:
    """
    Variational EM for SBM with K communities.
    Returns (q, P, pi, log_likelihood) for the best initialization.
    """
    n = adj.shape[0]
    best_ll = -np.inf
    best_q = best_P = best_pi = None

    for _ in range(n_init):
        # Random init: Dirichlet-sampled soft assignments
        q = np.random.dirichlet(np.ones(k), size=n)  # (n × k)
        q = np.clip(q, 1e-10, 1.0)
        q /= q.sum(axis=1, keepdims=True)

        for iteration in range(max_iter):
            q_old = q.copy()

            # M-step: update block probabilities and mixing proportions
            nk = q.sum(axis=0) + 1e-10                          # (k,)
            pi = nk / nk.sum()

            # Expected edges between blocks
            E = q.T @ adj @ q                                    # (k × k)
            N = np.outer(nk, nk) - np.diag(q.T @ q)            # (k × k) expected pairs
            N = np.clip(N, 1e-10, None)
            P = np.clip(E / N, 1e-6, 1 - 1e-6)                 # (k × k)

            # E-step: update responsibilities
            log_q = np.zeros((n, k))
            for g in range(k):
                log_pi_g = np.log(pi[g])
                # log P(adj[i,:] | z_i = g)
                log_lik = np.zeros(n)
                for h in range(k):
                    p_gh = P[g, h]
                    log_p = np.log(p_gh)
                    log_1mp = np.log(1 - p_gh)
                    # expected contribution from block h neighbors
                    log_lik += q[:, h] @ (adj * log_p + (1 - adj) * log_1mp)
                log_q[:, g] = log_pi_g + log_lik

            # Numerically stable softmax
            log_q -= log_q.max(axis=1, keepdims=True)
            q = np.exp(log_q)
            q = np.clip(q, 1e-10, None)
            q /= q.sum(axis=1, keepdims=True)

            if np.max(np.abs(q - q_old)) < 1e-6:
                break

        # Compute log-likelihood
        ll = 0.0
        for g in range(k):
            for h in range(k):
                p_gh = P[g, h]
                ll += (q[:, g].reshape(-1, 1) * q[:, h].reshape(1, -1) *
                       (adj * np.log(p_gh) + (1 - adj) * np.log(1 - p_gh))).sum()

        if ll > best_ll:
            best_ll = ll
            best_q = q.copy()
            best_P = P.copy()
            best_pi = pi.copy()

    return best_q, best_P, best_pi, best_ll


def _icl(adj: np.ndarray, q: np.ndarray, P: np.ndarray, pi: np.ndarray) -> float:
    """
    Integrated Completed Likelihood = log-likelihood + entropy penalty.
    Lower ICL → better model (penalizes unnecessary communities).
    """
    n, k = q.shape
    # Entropy of soft assignments
    entropy = -np.sum(q * np.log(np.clip(q, 1e-10, 1.0)))

    # Log-likelihood
    ll = 0.0
    for g in range(k):
        for h in range(k):
            p_gh = P[g, h]
            ll += (q[:, g].reshape(-1, 1) * q[:, h].reshape(1, -1) *
                   (adj * np.log(p_gh) + (1 - adj) * np.log(1 - p_gh))).sum()

    # Parameter count: k*(k+1)/2 block probs + (k-1) mixing props
    n_params = k * (k + 1) / 2 + (k - 1)
    bic_penalty = 0.5 * n_params * np.log(n * (n - 1) / 2)

    return ll - entropy - bic_penalty


def fit_sbm(
    adj: np.ndarray,
    taxa_names: list[str],
    max_k: int = 12,
    n_init: int = 10,
    max_iter: int = 300,
) -> SBMResult:
    """
    Fit SBM for K=2..max_k, select optimal K by ICL.
    """
    n = adj.shape[0]
    max_k = min(max_k, n - 1)

    icl_scores: dict[int, float] = {}
    best_result = None
    best_icl = -np.inf

    print(f"  Fitting SBM for K=2..{max_k} ({n} taxa)...")
    for k in range(2, max_k + 1):
        q, P, pi, ll = _sbm_em(adj, k=k, n_init=n_init, max_iter=max_iter)
        icl = _icl(adj, q, P, pi)
        icl_scores[k] = icl

        if icl > best_icl:
            best_icl = icl
            best_result = (k, q, P, pi)

    k_opt, q_opt, P_opt, pi_opt = best_result
    labels = np.argmax(q_opt, axis=1)

    print(f"  Optimal K={k_opt} (ICL={best_icl:.2f})")
    _check_degeneracy(P_opt, k_opt)

    return SBMResult(
        k=k_opt,
        q=q_opt,
        labels=labels,
        block_p=P_opt,
        pi=pi_opt,
        icl_scores=icl_scores,
        taxa_names=taxa_names,
    )


def _check_degeneracy(P: np.ndarray, k: int):
    """Warn if block probabilities are near-degenerate (all ~equal)."""
    off_diag = P[~np.eye(k, dtype=bool)]
    diag = np.diag(P)
    if np.std(P) < 0.05:
        print(f"  Warning: Block probability matrix is near-degenerate (std={np.std(P):.4f}).")
        print("  All blocks have similar connectivity — communities may not be meaningful.")
        print("  Consider raising spearman_threshold or fdr_alpha to reduce graph density.")
    else:
        print(f"  Block matrix looks healthy: diag mean={diag.mean():.3f}, off-diag mean={off_diag.mean():.3f}")
