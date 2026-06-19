"""
planning/rarefaction.py
Per-sample rarefaction / species-accumulation curves: taxa detected vs.
sequencing depth, computed from a sample's own raw read counts.

This is the empirical, retrospective counterpart to calculator.py's
prospective/hypothetical "how much should I sequence" planning tool —
rarefaction answers "how much of what I already found would I have found
at lower depth," a sensitivity/QC check on the current run. It is
interpolation up to the sample's actual depth, not extrapolation/prediction
of additional sequencing.

Exact closed-form rarefaction (Hurlbert 1971 / Heck et al. 1975), no Monte
Carlo subsampling:

    E[S(m)] = S_total - sum_i [ C(N - n_i, m) / C(N, m) ]

where N is the sample's total reads, n_i is taxon i's read count, and m is
the subsample depth. Computed in log-space via scipy.special.gammaln for
numerical stability with large N.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln


def expected_richness(counts: np.ndarray, depth: int) -> float:
    """Expected number of distinct taxa observed when subsampling `depth`
    reads without replacement from the full set described by `counts`."""
    arr = np.asarray(counts, dtype=float)
    arr = arr[arr > 0]
    n_taxa = len(arr)
    total = arr.sum()

    if depth <= 0 or total <= 0 or n_taxa == 0:
        return 0.0
    depth = min(depth, int(total))

    log_denom = gammaln(total + 1) - gammaln(depth + 1) - gammaln(total - depth + 1)

    # C(N - n_i, depth) is 0 (taxon guaranteed present) when N - n_i < depth,
    # so only taxa with N - n_i >= depth contribute a nonzero "absent" term.
    eligible = (total - arr) >= depth
    n_i = arr[eligible]
    log_numer = (
        gammaln(total - n_i + 1) - gammaln(depth + 1) - gammaln(total - n_i - depth + 1)
    )
    prob_absent = np.exp(log_numer - log_denom)

    return float(n_taxa - prob_absent.sum())


@dataclass
class RarefactionCurve:
    sample_name: str
    depths: list[int]
    richness: list[float]
    total_reads: int
    observed_richness: int


def rarefaction_curve(sample_name: str, counts, n_points: int = 20) -> RarefactionCurve:
    """Log-spaced rarefaction curve from depth 1 up to the sample's total
    reads N — interpolation within the observed data, not extrapolation."""
    arr = np.asarray(counts, dtype=float)
    arr = arr[arr > 0]
    total = int(arr.sum())
    observed_richness = int(len(arr))

    if total <= 0:
        return RarefactionCurve(
            sample_name=sample_name, depths=[], richness=[],
            total_reads=0, observed_richness=0,
        )

    n_points = max(1, min(n_points, total))
    depths = sorted(set(int(d) for d in np.geomspace(1, total, num=n_points)))
    richness = [expected_richness(arr, d) for d in depths]

    return RarefactionCurve(
        sample_name=sample_name,
        depths=depths,
        richness=richness,
        total_reads=total,
        observed_richness=observed_richness,
    )


def rarefaction_for_sampleset(sampleset, n_points: int = 20) -> dict[str, RarefactionCurve]:
    """Compute a RarefactionCurve per sample column in sampleset.taxa_matrix.
    Expects raw, unfiltered per-sample counts (not the cohort-filtered
    matrix) so genuinely-present-but-cohort-rare taxa aren't undercounted."""
    mat = sampleset.taxa_matrix
    return {
        col: rarefaction_curve(col, mat[col].to_numpy(), n_points=n_points)
        for col in mat.columns
    }
