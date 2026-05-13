"""
temporal/baseline.py
Rolling baseline and z-score anomaly detection.

For each site, computes a rolling mean and std from the last N runs and
flags the current score as anomalous if it exceeds the baseline by
more than `z_threshold` standard deviations.

This is the simplest temporal signal — fast to compute, easy to explain
to public health customers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BaselineResult:
    site: str
    current_score: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    is_anomaly: bool
    n_history: int          # how many historical runs were used
    pct_above_baseline: float   # how many % above the rolling mean


def compute_baseline(
    site: str,
    history: list[dict],
    current_score: float,
    window: int = 8,
    z_threshold: float = 2.0,
) -> BaselineResult:
    """
    Compare current_score against a rolling baseline from history.

    history: list of dicts with 'risk_score' key (oldest → newest),
             NOT including the current run.
    window: number of past runs to use for baseline (default: 8 weeks)
    z_threshold: z-score above which current is flagged as anomalous
    """
    past_scores = [h["risk_score"] for h in history[-window:]]

    if len(past_scores) < 2:
        # Not enough history — return neutral result
        return BaselineResult(
            site=site,
            current_score=current_score,
            baseline_mean=current_score,
            baseline_std=0.0,
            z_score=0.0,
            is_anomaly=False,
            n_history=len(past_scores),
            pct_above_baseline=0.0,
        )

    mean = float(np.mean(past_scores))
    std = float(np.std(past_scores, ddof=1))

    z = (current_score - mean) / (std + 1e-10)
    pct_above = ((current_score - mean) / (mean + 1e-10)) * 100

    return BaselineResult(
        site=site,
        current_score=current_score,
        baseline_mean=round(mean, 4),
        baseline_std=round(std, 4),
        z_score=round(z, 3),
        is_anomaly=z >= z_threshold,
        n_history=len(past_scores),
        pct_above_baseline=round(pct_above, 1),
    )


def compute_baselines_all_sites(
    store,
    current_risk_scores: list,
    window: int = 8,
    z_threshold: float = 2.0,
) -> dict[str, BaselineResult]:
    """
    Compute baseline anomaly check for all sites in the current run.
    Pulls history from the store (excludes the current run — call this
    BEFORE recording the current run, or pass history explicitly).
    """
    results = {}
    for rs in current_risk_scores:
        history = store.get_site_history(rs.sample_name, last_n=window + 1)
        # Exclude current run if already recorded (defensive)
        history = history[:-1] if history and history[-1]["risk_score"] == rs.score else history
        results[rs.sample_name] = compute_baseline(
            site=rs.sample_name,
            history=history,
            current_score=rs.score,
            window=window,
            z_threshold=z_threshold,
        )
    return results
