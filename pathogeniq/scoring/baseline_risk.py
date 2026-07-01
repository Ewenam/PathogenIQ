"""
scoring/baseline_risk.py
Baseline-relative risk scoring — the trustworthy replacement for the absolute
threshold scorer.

Problem it fixes
----------------
The original scorer keys off ABSOLUTE abundance (`risk_weight * abundance`, plus
a `multi-pathogen load >= 0.10 -> 0.62` override). In wastewater, opportunistic
"pathogen" genera are endemic — Pseudomonas alone averages ~39% on Site-207 —
so the absolute rule fires on every normal sample and every sample scores HIGH.
The score has no discriminative power.

Design (grounded in established surveillance methods)
-----------------------------------------------------
Actionability in wastewater = DEPARTURE from the site's normal, weighted by how
dangerous the departing taxon is — not raw abundance. This mirrors:
  * CZ ID / IDseq: per-taxon z-score vs a background (water-control) model
    (Kalantar et al. 2020).
  * CDC NWSS "Wastewater Viral Activity Levels": report each sample as a
    percentile of the site's own history.
  * EARS / Farrington aberration detection: flag counts above a historical
    baseline (the temporal term; CUSUM is already in the pipeline).

Components:
  1. elevation  — max over present pathogen genera of danger×squash(one-sided
                  robust z above the site baseline). Endemic genus at baseline → 0.
  2. tripwire   — narrow set of should-never-be-here agents (BSL-3/4 / hemorrhagic)
                  that alert on mere presence, regardless of baseline.
  3. temporal   — CUSUM/z aberration (passed in; optional).
  4. novelty    — composition anomaly (passed in; optional).
Community signal is intentionally dropped from scoring (non-discriminative
constant on real data; SBM remains an interpretability layer).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .risk import PATHOGEN_DB, RiskScore, _score_level

# ── tripwire: agents whose mere presence at appreciable level is actionable ──
# High-consequence, NON-endemic to wastewater (BSL-3/4, hemorrhagic fevers,
# select agents). Kept deliberately small; these do not occur as background.
TRIPWIRE_GENERA: dict[str, float] = {
    "Yersinia": 0.01, "Francisella": 0.01, "Bacillus": 0.02, "Brucella": 0.01,
    "Burkholderia": 0.02, "Ebolavirus": 0.001, "Marburgvirus": 0.001,
    "Mammarenavirus": 0.005, "Orthonairovirus": 0.005, "Orthopoxvirus": 0.002,
    "Lyssavirus": 0.002, "Phlebovirus": 0.005, "Orthohantavirus": 0.005,
}
_TRIPWIRE_SCORE = 0.85   # presence above threshold → CRITICAL-band


def _squash(z: float, k: float = 3.0) -> float:
    """Saturating map of a one-sided z-score to [0,1): z=k→0.5, z=3k→0.75."""
    z = max(0.0, z)
    return z / (z + k)


@dataclass
class AbundanceBaseline:
    """Per-taxon robust location/scale from a set of baseline samples.

    A taxon's exceedance is a one-sided robust z: (abund - median) / (1.4826·MAD),
    floored at 0. Taxa absent from the baseline are treated as median 0 with a
    small scale, so a newly-appearing taxon reads as strongly elevated.
    """
    median: dict[str, float]
    scale: dict[str, float]
    scale_floor: float = 1e-3
    novel_scale: float = 5e-3   # scale used for taxa unseen in the baseline

    @classmethod
    def fit(cls, rel_abundance: pd.DataFrame, scale_floor: float = 1e-3) -> "AbundanceBaseline":
        """rel_abundance: taxa × samples relative-abundance matrix (the baseline
        window, or the cohort itself for a self-referential cold start)."""
        med, scl = {}, {}
        for taxon, row in rel_abundance.iterrows():
            v = row.to_numpy(dtype=float)
            m = float(np.median(v))
            mad = float(np.median(np.abs(v - m)))
            med[taxon] = m
            scl[taxon] = max(1.4826 * mad, scale_floor)
        return cls(median=med, scale=scl, scale_floor=scale_floor)

    def exceedance(self, taxon: str, abundance: float) -> float:
        if taxon in self.median:
            m, s = self.median[taxon], self.scale[taxon]
        else:
            m, s = 0.0, self.novel_scale
        return max(0.0, (float(abundance) - m) / s)


def _elevation(rel_abund: pd.Series, baseline: AbundanceBaseline) -> tuple[float, list[dict]]:
    """Max danger-weighted exceedance over present pathogen genera."""
    best = 0.0
    detected: list[dict] = []
    for taxon, ab in rel_abund.items():
        if ab <= 0:
            continue
        genus = taxon.split()[0]
        info = PATHOGEN_DB.get(genus)
        if info is None:
            continue
        z = baseline.exceedance(taxon, ab)
        rw = info["risk_weight"]
        contribution = _squash(z) * (0.5 + 0.5 * rw)   # danger-scaled anomaly
        detected.append({
            "taxon": taxon, "genus": genus, "risk_weight": rw,
            "abundance": round(float(ab), 5), "exceedance_z": round(z, 2),
            "contribution": round(contribution, 4), "disease": info["disease"],
        })
        best = max(best, contribution)
    detected.sort(key=lambda d: d["contribution"], reverse=True)
    return best, detected


def _tripwire(rel_abund: pd.Series) -> tuple[float, list[str]]:
    fired = []
    for taxon, ab in rel_abund.items():
        genus = taxon.split()[0]
        thr = TRIPWIRE_GENERA.get(genus)
        if thr is not None and ab >= thr:
            fired.append(f"{taxon} @ {ab*100:.2f}% (tripwire)")
    return (_TRIPWIRE_SCORE if fired else 0.0), fired


def score_sample_relative(
    sample_name: str,
    rel_abund: pd.Series,
    baseline: AbundanceBaseline,
    novelty_score: float = 0.0,
    temporal_z: float = 0.0,
    weights: dict | None = None,
) -> RiskScore:
    """Baseline-relative composite risk score for one sample."""
    w = weights or {"elevation": 0.7, "temporal": 0.15, "novelty": 0.15}
    elevation, detected = _elevation(rel_abund, baseline)
    tripwire, fired = _tripwire(rel_abund)
    temporal = _squash(max(0.0, float(temporal_z)), k=2.0)
    novelty = float(np.clip(novelty_score, 0.0, 1.0))

    composite = (w["elevation"] * elevation
                 + w["temporal"] * temporal
                 + w["novelty"] * novelty)
    score = float(np.clip(max(composite, tripwire), 0.0, 1.0))

    return RiskScore(
        sample_name=sample_name,
        score=score,
        level=_score_level(score),
        detected_pathogens=detected,
        community_signal=0.0,           # intentionally not part of the score
        novelty_signal=novelty,
        breakdown={
            "elevation_score": round(elevation, 4),
            "abundance_score": round(elevation, 4),  # alias for store/report compat
            "temporal_score": round(temporal, 4),
            "novelty_signal": round(novelty, 4),
            "composite_score": round(composite, 4),
            "tripwire_score": round(tripwire, 4),
            "tripwire_hits": fired,
            "top_driver": detected[0] if detected else None,
            "weights": w,
        },
    )


def score_all_relative(
    sampleset,
    baseline: AbundanceBaseline | None = None,
    novelty_scores: dict[str, float] | None = None,
    temporal_z: dict[str, float] | None = None,
    weights: dict | None = None,
) -> list[RiskScore]:
    """Score all samples against a baseline. If `baseline` is None, fit a robust
    self-referential baseline from the cohort (cold-start): flags samples whose
    pathogen composition departs from the site's typical profile. Supply an
    explicit baseline (a clean historical window) for the reliable regime."""
    rel = sampleset.relative_abundance
    if baseline is None:
        baseline = AbundanceBaseline.fit(rel)
    novelty_scores = novelty_scores or {}
    temporal_z = temporal_z or {}

    scores = []
    for sample in sampleset.samples:
        name = sample.name
        s = score_sample_relative(
            name, rel.get(name, pd.Series(dtype=float)), baseline,
            novelty_score=novelty_scores.get(name, 0.0),
            temporal_z=temporal_z.get(name, 0.0),
            weights=weights,
        )
        scores.append(s)
    scores.sort(key=lambda x: x.score, reverse=True)
    return scores
