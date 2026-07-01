"""
scoring/baseline_resolver.py
Tiered baseline resolution — the SaaS-appropriate answer to "what is normal?"

Every new tenant/site starts with zero history, so no single baseline strategy
works day one. This resolves a baseline in priority order, each tier degrading
gracefully so a customer always gets a defensible result:

  1. controls      — customer-designated clean/negative-control samples
                     (CZ ID-style; the reliable regime, available immediately).
  2. rolling site  — the site's own trailing history, robustly fit with likely
                     event periods excluded (reproducible, site-calibrated).
  3. global prior  — a shipped "typical wastewater flora" reference, so a brand
                     new site with no controls and no history still scores
                     against something realistic on day one.
  4. self          — cold-start cohort self-baseline (last resort; returns None
                     so the caller fits it from the batch being scored).

Returns (AbundanceBaseline | None, tier_name). None means "fit a self-baseline".
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .baseline_risk import AbundanceBaseline

# Shipped global prior (fit from the Site-207 series; see scripts/build_global_prior.py)
DEFAULT_GLOBAL_PRIOR = Path(__file__).parent / "data" / "global_prior_baseline.json"


def resolve_baseline(
    rel_abundance: pd.DataFrame,
    *,
    control_names: list[str] | None = None,
    history_matrix: pd.DataFrame | None = None,
    global_prior_path: str | Path | None = DEFAULT_GLOBAL_PRIOR,
    min_history: int = 8,
    space: str = "relative",
    exclude_outlier_frac: float = 0.1,
) -> tuple[AbundanceBaseline | None, str]:
    """Resolve a baseline for scoring `rel_abundance` (taxa × samples). See
    module docstring for the tier order."""
    # 1. customer-designated controls
    if control_names:
        cols = [c for c in control_names if c in rel_abundance.columns]
        if cols:
            return AbundanceBaseline.from_controls(rel_abundance, cols, space=space), "controls"

    # 2. rolling site baseline (enough trailing history)
    if history_matrix is not None and history_matrix.shape[1] >= min_history:
        return (AbundanceBaseline.fit_rolling(history_matrix, space=space,
                                              exclude_outlier_frac=exclude_outlier_frac),
                "rolling")

    # 3. shipped global prior
    if global_prior_path and Path(global_prior_path).exists():
        bl = AbundanceBaseline.load(global_prior_path)
        return bl, "global_prior"

    # 4. cold-start self-baseline (caller fits from the cohort)
    return None, "self"


def history_matrix_from_store(store, sites: list[str] | None = None,
                              window: int = 26) -> pd.DataFrame | None:
    """Best-effort trailing abundance matrix (taxa × past-samples) from a store
    exposing get_abundance_matrix(). Returns None if unavailable/empty."""
    getter = getattr(store, "get_abundance_matrix", None)
    if getter is None:
        return None
    try:
        mat = getter(last_n=window, sites=sites)
    except TypeError:
        mat = getter(window)
    if mat is None or getattr(mat, "empty", True):
        return None
    return mat
