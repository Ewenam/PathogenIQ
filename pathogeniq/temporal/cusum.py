"""
temporal/cusum.py
CUSUM (Cumulative Sum) changepoint detection for biosurveillance.

CUSUM is the gold standard algorithm used by CDC, ECDC, and WHO for
disease outbreak detection. Unlike a simple threshold, it accumulates
small consistent increases over time — catching slow-rising outbreaks
that never trigger a single-point alert.

Two variants:
  - Upper CUSUM: detects sustained increases (outbreak signal)
  - Two-sided CUSUM: detects both increases and decreases

Reference: Page (1954); Farrington et al. (1996) for epidemiological use.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CUSUMResult:
    site: str
    cusum_upper: float          # current upper CUSUM statistic
    cusum_lower: float          # current lower CUSUM statistic (decrease detection)
    threshold: float            # alert threshold h
    alert: bool                 # True if cusum_upper >= threshold
    alert_decrease: bool        # True if cusum_lower >= threshold (unusual drop)
    cusum_series: list[float]   # full upper CUSUM series for plotting
    n_consecutive_above: int    # weeks CUSUM has been > 0 (persistence indicator)
    signal_strength: str        # "none" | "weak" | "moderate" | "strong"


def run_cusum(
    site: str,
    scores: list[float],
    k: float = 0.5,
    h: float = 4.0,
) -> CUSUMResult:
    """
    Run upper and lower CUSUM on a series of risk scores.

    k: allowance / slack parameter — how much deviation to tolerate
       before accumulating. Typically set to half the expected shift size.
       Default 0.5 = sensitive to shifts > 0.5 standard deviations.

    h: decision interval / threshold. Alert fires when CUSUM >= h.
       Lower h = more sensitive (more false positives).
       Higher h = less sensitive (fewer false positives).
       h=4 is standard in epidemiological surveillance (Farrington).

    Operates on standardized scores: z = (x - mu) / sigma
    where mu and sigma are estimated from the first half of the series
    (treated as baseline period).
    """
    if len(scores) < 3:
        return CUSUMResult(
            site=site, cusum_upper=0.0, cusum_lower=0.0,
            threshold=h, alert=False, alert_decrease=False,
            cusum_series=[], n_consecutive_above=0, signal_strength="none",
        )

    arr = np.array(scores, dtype=float)

    # Estimate baseline from first half of series
    baseline_end = max(2, len(arr) // 2)
    mu = float(np.mean(arr[:baseline_end]))
    sigma = float(np.std(arr[:baseline_end], ddof=1)) or 1.0

    # Standardize
    z = (arr - mu) / sigma

    # Upper CUSUM (detect increases)
    cusum_u = np.zeros(len(z))
    for i in range(1, len(z)):
        cusum_u[i] = max(0.0, cusum_u[i - 1] + z[i] - k)

    # Lower CUSUM (detect decreases — unusual drops can indicate data issues)
    cusum_l = np.zeros(len(z))
    for i in range(1, len(z)):
        cusum_l[i] = max(0.0, cusum_l[i - 1] - z[i] - k)

    current_u = float(cusum_u[-1])
    current_l = float(cusum_l[-1])

    # Consecutive weeks above zero (persistence)
    consec = 0
    for val in reversed(cusum_u):
        if val > 0:
            consec += 1
        else:
            break

    # Signal strength classification
    if current_u >= h * 2:
        strength = "strong"
    elif current_u >= h:
        strength = "moderate"
    elif current_u >= h * 0.5:
        strength = "weak"
    else:
        strength = "none"

    return CUSUMResult(
        site=site,
        cusum_upper=round(current_u, 4),
        cusum_lower=round(current_l, 4),
        threshold=h,
        alert=current_u >= h,
        alert_decrease=current_l >= h,
        cusum_series=[round(v, 4) for v in cusum_u.tolist()],
        n_consecutive_above=consec,
        signal_strength=strength,
    )


def run_cusum_all_sites(
    store,
    current_risk_scores: list,
    window: int = 16,
    k: float = 0.5,
    h: float = 4.0,
) -> dict[str, CUSUMResult]:
    """
    Run CUSUM for all sites, pulling history from the store and appending
    the current score to build the full series.
    """
    results = {}
    for rs in current_risk_scores:
        history = store.get_site_history(rs.sample_name, last_n=window)
        past = [h_["risk_score"] for h_ in history]
        full_series = past + [rs.score]
        results[rs.sample_name] = run_cusum(
            site=rs.sample_name,
            scores=full_series,
            k=k,
            h=h,
        )
    return results


def run_cusum_taxon(
    store,
    site: str,
    taxon: str,
    current_abundance: float,
    window: int = 16,
    k: float = 0.5,
    h: float = 4.0,
) -> CUSUMResult:
    """
    Run CUSUM on a specific taxon's abundance at a site.
    Useful for tracking individual pathogen trends over time.
    """
    history = store.get_taxon_history(site, taxon, last_n=window)
    past = [h_["rel_abund"] for h_ in history]
    full_series = past + [current_abundance]
    return run_cusum(site=f"{site}::{taxon}", scores=full_series, k=k, h=h)
