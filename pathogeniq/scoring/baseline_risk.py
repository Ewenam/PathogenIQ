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


def _clr_transform(rel: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    """Centered-log-ratio transform of a taxa×samples relative-abundance matrix.
    CLR removes the compositional constant-sum constraint, so a taxon's value is
    interpreted relative to the sample's geometric mean rather than in isolation
    (Aitchison; as in ANCOM-BC/ALDEx2). Zeros handled by a small pseudocount."""
    X = rel.to_numpy(dtype=float) + pseudocount
    logX = np.log(X)
    gm = logX.mean(axis=0, keepdims=True)      # per-sample geometric mean (in log)
    return pd.DataFrame(logX - gm, index=rel.index, columns=rel.columns)


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
    space: str = "relative"     # "relative" | "clr" (compositional)
    taxa: list = None           # baseline taxa (needed to CLR-transform samples)
    pseudocount: float = 1e-6

    @classmethod
    def fit(cls, rel_abundance: pd.DataFrame, scale_floor: float = 1e-3,
            space: str = "relative", pseudocount: float = 1e-6) -> "AbundanceBaseline":
        """rel_abundance: taxa × samples relative-abundance matrix (the baseline
        window, or the cohort itself for a self-referential cold start).
        space="clr" fits per-taxon location/scale in centered-log-ratio space
        (compositionally sound); "relative" uses raw relative abundance."""
        mat = _clr_transform(rel_abundance, pseudocount) if space == "clr" else rel_abundance
        med, scl = {}, {}
        for taxon, row in mat.iterrows():
            v = row.to_numpy(dtype=float)
            m = float(np.median(v))
            mad = float(np.median(np.abs(v - m)))
            med[taxon] = m
            scl[taxon] = max(1.4826 * mad, scale_floor)
        return cls(median=med, scale=scl, scale_floor=scale_floor, space=space,
                   taxa=list(rel_abundance.index), pseudocount=pseudocount)

    def sample_values(self, rel_abund: pd.Series) -> dict[str, float]:
        """Per-taxon value in the baseline's space. In CLR space the sample is
        transformed over the baseline taxa plus any taxa it introduces, so the
        geometric-mean reference matches how the baseline was built."""
        if self.space != "clr":
            return {t: float(v) for t, v in rel_abund.items()}
        present = [t for t, v in rel_abund.items() if v > 0]
        taxa = list(dict.fromkeys((self.taxa or []) + present))
        x = np.array([max(float(rel_abund.get(t, 0.0)), 0.0) for t in taxa]) + self.pseudocount
        logx = np.log(x)
        clr = logx - logx.mean()
        return dict(zip(taxa, clr))

    def exceedance(self, taxon: str, value: float) -> float:
        if taxon in self.median:
            m, s = self.median[taxon], self.scale[taxon]
        elif self.space == "clr":
            # unseen taxon: baseline presence ≈ pseudocount → very low CLR; use
            # the lowest baseline location and a typical scale as its reference.
            m = min(self.median.values()) if self.median else 0.0
            s = float(np.median(list(self.scale.values()))) if self.scale else self.novel_scale
        else:
            m, s = 0.0, self.novel_scale
        return max(0.0, (float(value) - m) / s)

    # ── persistence (freeze a clean-window baseline, reuse across runs) ──────
    def to_dict(self) -> dict:
        return {"version": 2, "median": self.median, "scale": self.scale,
                "scale_floor": self.scale_floor, "novel_scale": self.novel_scale,
                "space": self.space, "taxa": self.taxa, "pseudocount": self.pseudocount}

    @classmethod
    def from_dict(cls, d: dict) -> "AbundanceBaseline":
        return cls(median=dict(d["median"]), scale=dict(d["scale"]),
                   scale_floor=d.get("scale_floor", 1e-3),
                   novel_scale=d.get("novel_scale", 5e-3),
                   space=d.get("space", "relative"), taxa=d.get("taxa"),
                   pseudocount=d.get("pseudocount", 1e-6))

    def save(self, path) -> None:
        import json
        from pathlib import Path
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path) -> "AbundanceBaseline":
        import json
        from pathlib import Path
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_controls(cls, rel_abundance: pd.DataFrame, control_names: list[str],
                      scale_floor: float = 1e-3, space: str = "relative") -> "AbundanceBaseline":
        """Fit from a designated set of clean/negative-control sample columns
        (CZ ID-style background model) rather than the whole cohort."""
        cols = [c for c in control_names if c in rel_abundance.columns]
        if not cols:
            raise ValueError("none of the control_names are in the abundance matrix")
        return cls.fit(rel_abundance[cols], scale_floor=scale_floor, space=space)


# Per-space squash constant: CLR z-scores are naturally tighter than raw
# relative z-scores (which explode for rare genera with ~0 MAD), so CLR needs a
# smaller k to reach the alert band. `clr` value calibrated in
# scripts (see calibrate_clr) against a realistic endemic baseline.
SQUASH_K = {"relative": 3.0, "clr": 2.0}
# Cap the exceedance z. Under a cohort self-baseline a genus present in only a
# few samples has ~0 MAD, so any appearance yields an absurd z (e.g. 100+); the
# cap keeps such artifacts from dominating while leaving genuine strong spikes
# (z well past the alert band) unaffected.
Z_CAP = 12.0
# Corroboration (temporal + novelty) is a bounded BONUS on top of the pathogen
# elevation floor — it can lift a borderline elevation but never manufacture an
# alert on its own (endemic samples with incidental novelty stay LOW).
CORROB_WEIGHT = 0.4


def _elevation(rel_abund: pd.Series, baseline: AbundanceBaseline,
               squash_k: float = 3.0) -> tuple[float, list[dict]]:
    """Max danger-weighted exceedance over present pathogen genera."""
    best = 0.0
    detected: list[dict] = []
    values = baseline.sample_values(rel_abund)   # raw or CLR, per baseline.space
    for taxon, ab in rel_abund.items():
        if ab <= 0:
            continue
        genus = taxon.split()[0]
        info = PATHOGEN_DB.get(genus)
        if info is None:
            continue
        z = min(baseline.exceedance(taxon, values.get(taxon, ab)), Z_CAP)
        rw = info["risk_weight"]
        contribution = _squash(z, k=squash_k) * (0.5 + 0.5 * rw)   # danger-scaled anomaly
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
    squash_k: float | None = None,
    corrob_weight: float = CORROB_WEIGHT,
) -> RiskScore:
    """Baseline-relative composite risk score for one sample.

    Composite design: the danger-weighted pathogen ELEVATION sets a floor in
    [0,1]; corroborating temporal (CUSUM/Farrington) + novelty signals add a
    bounded bonus on top of that floor but cannot fire an alert alone. So a
    strongly-elevated dangerous genus can reach HIGH/CRITICAL by itself, while
    an endemic sample with incidental novelty stays LOW.
        base = elevation + (1 - elevation) * corrob_weight * corroboration
        score = max(tripwire, base)
    """
    k = squash_k if squash_k is not None else SQUASH_K.get(baseline.space, 3.0)
    elevation, detected = _elevation(rel_abund, baseline, squash_k=k)
    tripwire, fired = _tripwire(rel_abund)
    temporal = _squash(max(0.0, float(temporal_z)), k=2.0)
    novelty = float(np.clip(novelty_score, 0.0, 1.0))

    corroboration = 0.5 * temporal + 0.5 * novelty
    base = elevation + (1.0 - elevation) * corrob_weight * corroboration
    score = float(np.clip(max(base, tripwire), 0.0, 1.0))

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
            "corroboration": round(corroboration, 4),
            "composite_score": round(base, 4),
            "tripwire_score": round(tripwire, 4),
            "tripwire_hits": fired,
            "top_driver": detected[0] if detected else None,
            "squash_k": k,
        },
    )


def score_all_relative(
    sampleset,
    baseline: AbundanceBaseline | None = None,
    control_names: list[str] | None = None,
    novelty_scores: dict[str, float] | None = None,
    temporal_z: dict[str, float] | None = None,
    squash_k: float | None = None,
    space: str = "relative",
) -> list[RiskScore]:
    """Score all samples against a baseline. Baseline resolution order:
      1. an explicit `baseline` (a frozen clean historical window), else
      2. fit from `control_names` (CZ ID-style negative/clean controls), else
      3. a robust cohort self-baseline (cold start) — flags samples whose
         pathogen composition departs from the site's typical profile.
    space: "relative" (raw) or "clr" (compositional) — used only when fitting a
    baseline here (an explicit `baseline` carries its own space).
    """
    rel = sampleset.relative_abundance
    if baseline is None:
        baseline = (AbundanceBaseline.from_controls(rel, control_names, space=space)
                    if control_names else AbundanceBaseline.fit(rel, space=space))
    novelty_scores = novelty_scores or {}
    temporal_z = temporal_z or {}

    scores = []
    for sample in sampleset.samples:
        name = sample.name
        s = score_sample_relative(
            name, rel.get(name, pd.Series(dtype=float)), baseline,
            novelty_score=novelty_scores.get(name, 0.0),
            temporal_z=temporal_z.get(name, 0.0),
            squash_k=squash_k,
        )
        scores.append(s)
    scores.sort(key=lambda x: x.score, reverse=True)
    return scores
