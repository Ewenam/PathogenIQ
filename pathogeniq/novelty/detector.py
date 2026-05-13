"""
novelty/detector.py
Novelty detection — flags samples with unusual taxonomic profiles.

Uses two complementary approaches:
  1. Isolation Forest on CLR-transformed abundance vectors (unsupervised)
  2. Z-score of per-taxon abundance relative to population mean
     (catches sudden spikes of individual taxa)

Returns a novelty score in [0, 1] per sample, where 1.0 = maximally anomalous.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def compute_novelty_scores(
    sampleset,
    contamination: float = 0.1,
    zscore_threshold: float = 2.5,
) -> dict[str, float]:
    """
    Compute novelty score for each sample.

    contamination: expected fraction of outlier samples (Isolation Forest param)
    zscore_threshold: z-score above which a taxon spike contributes to novelty

    Returns dict: {sample_name: novelty_score_0_to_1}
    """
    from ..community.graph import clr_transform

    mat = sampleset.taxa_matrix
    if mat is None or mat.empty:
        return {s.name: 0.0 for s in sampleset.samples}

    clr = clr_transform(mat)              # taxa × samples
    X = clr.values.T                      # samples × taxa

    sample_names = [s.name for s in sampleset.samples]

    scores: dict[str, float] = {}

    # ── Signal 1: Isolation Forest ─────────────────────────────────────────────
    if X.shape[0] >= 4:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        iso = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
        )
        iso.fit(X_scaled)
        # decision_function: negative = more anomalous, positive = more normal
        raw_scores = iso.decision_function(X_scaled)
        # Normalize to [0, 1] with 1 = most anomalous
        iso_scores = 1 - (raw_scores - raw_scores.min()) / (np.ptp(raw_scores) + 1e-10)
    else:
        iso_scores = np.zeros(len(sample_names))

    # ── Signal 2: Per-taxon z-score spike detection ────────────────────────────
    rel = sampleset.relative_abundance.values.T   # samples × taxa
    mean_abund = rel.mean(axis=0)
    std_abund = rel.std(axis=0) + 1e-10
    z = (rel - mean_abund) / std_abund            # (samples × taxa)

    # Max z-score per sample, normalized
    max_z_per_sample = np.abs(z).max(axis=1)
    zscore_signal = np.clip(max_z_per_sample / (zscore_threshold * 3), 0, 1)

    # Combine: weighted average
    for i, name in enumerate(sample_names):
        combined = 0.6 * iso_scores[i] + 0.4 * zscore_signal[i]
        scores[name] = float(np.clip(combined, 0.0, 1.0))

    return scores


def flag_novel_taxa(
    sampleset,
    reference_taxa: set[str] | None = None,
    abundance_threshold: float = 0.001,
) -> dict[str, list[str]]:
    """
    Identify taxa in a sample that are either:
    - Not present in the reference set (truly novel)
    - Spiking well above their historical mean

    Returns dict: {sample_name: [novel_taxon, ...]}
    """
    flagged: dict[str, list[str]] = {}
    rel = sampleset.relative_abundance

    population_mean = rel.mean(axis=1)

    for col in rel.columns:
        novel = []
        for taxon, abund in rel[col].items():
            if abund < abundance_threshold:
                continue
            if reference_taxa and taxon not in reference_taxa:
                novel.append(taxon)
            elif abund > population_mean[taxon] * 5:
                novel.append(taxon)
        if novel:
            flagged[col] = novel

    return flagged
