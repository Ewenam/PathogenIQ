#!/usr/bin/env python3
"""
site207_recalibration.py
=========================
Applies the baseline-relative direct-detection recalibration
(`pathogeniq.scoring.risk.recalibrate_direct_detection`) to the 27-sample
site-207 longitudinal report, orders the results chronologically using
data_site207_accessions.tsv, runs CUSUM on the recalibrated risk-score
series, and writes a consolidated report_recalibrated.json for use by
generate_figures.py and the paper write-up.

Usage:
    python paper/site207_recalibration.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pathogeniq.scoring.risk import RiskScore, recalibrate_direct_detection, _score_level
from pathogeniq.temporal.cusum import run_cusum
REPORT_IN = ROOT / "reports" / "site207_longitudinal" / "report.json"
ACCESSIONS = ROOT / "data_site207_accessions.tsv"
REPORT_OUT = ROOT / "reports" / "site207_longitudinal" / "report_recalibrated.json"


def main():
    # ── Chronological order from accession metadata ─────────────────────────
    order = []
    with open(ACCESSIONS) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            order.append((row["collection_date"], row["run_accession"]))
    order.sort(key=lambda x: x[0])
    date_by_accession = {acc: date for date, acc in order}
    chrono_index = {acc: i for i, (_, acc) in enumerate(order)}

    # ── Load report + reconstruct RiskScore objects ──────────────────────────
    data = json.load(open(REPORT_IN))
    samples = data["samples"]

    scores = [
        RiskScore(
            sample_name=s["name"],
            score=s["score"],
            level=s["level"],
            detected_pathogens=s["detected_pathogens"],
            community_signal=s["community_signal"],
            novelty_signal=s["novelty_signal"],
            breakdown=s["breakdown"],
        )
        for s in samples
    ]

    # ── Recalibrate ────────────────────────────────────────────────────────
    recal = recalibrate_direct_detection(scores)

    # ── Sort chronologically ──────────────────────────────────────────────
    recal.sort(key=lambda s: chrono_index[s.sample_name])

    # ── CUSUM on the recalibrated score series (== composite_score here, ──
    #    since direct_detection_score_recal == 0 for all 27 samples) ───────
    score_series = [s.score for s in recal]
    cusum_result = run_cusum(site="site207", scores=score_series, k=0.5, h=4.0)

    # ── Level distribution before / after ────────────────────────────────
    before_levels = {"LOW": 0, "MODERATE": 0, "HIGH": 0, "CRITICAL": 0}
    after_levels = {"LOW": 0, "MODERATE": 0, "HIGH": 0, "CRITICAL": 0}
    for s, r in zip(scores, recal):
        before_levels[s.level] += 1
        after_levels[r.level] += 1

    composite = np.array([s.breakdown["composite_score"] for s in recal])
    novelty = np.array([s.novelty_signal for s in recal])
    abundance = np.array([s.breakdown["abundance_score"] for s in recal])
    loads = np.array([s.breakdown["load"] for s in recal])

    print("=" * 70)
    print("Level distribution BEFORE recalibration:", before_levels)
    print("Level distribution AFTER  recalibration:", after_levels)
    print()
    print(f"composite_score: min={composite.min():.4f} max={composite.max():.4f} "
          f"mean={composite.mean():.4f} std={composite.std(ddof=1):.4f}")
    print(f"novelty_signal:  min={novelty.min():.4f} max={novelty.max():.4f} "
          f"mean={novelty.mean():.4f} std={novelty.std(ddof=1):.4f}")
    print(f"abundance_score: min={abundance.min():.4f} max={abundance.max():.4f} "
          f"mean={abundance.mean():.4f} std={abundance.std(ddof=1):.4f}")
    print(f"load:            min={loads.min():.4f} max={loads.max():.4f} "
          f"mean={loads.mean():.4f} std={loads.std(ddof=1):.4f}")
    print()
    print("CUSUM (k=0.5, h=4.0) on recalibrated score series, chronological order:")
    print(f"  cusum_upper={cusum_result.cusum_upper}  alert={cusum_result.alert}  "
          f"signal_strength={cusum_result.signal_strength}")
    print(f"  max(C_t) over series = {max(cusum_result.cusum_series):.4f}")
    print(f"  cusum_series = {cusum_result.cusum_series}")
    print()
    print("Chronological per-sample table:")
    print(f"{'wk':>3} {'date':>10} {'accession':>12} {'score':>7} {'level':>9} "
          f"{'load':>7} {'load_z':>7} {'C_t':>7}")
    for i, (s, ct) in enumerate(zip(recal, cusum_result.cusum_series)):
        date = date_by_accession[s.sample_name]
        print(f"{i:>3} {date:>10} {s.sample_name:>12} {s.score:>7.4f} {s.level:>9} "
              f"{s.breakdown['load']:>7.4f} {s.breakdown['load_z']:>7.4f} {ct:>7.4f}")

    # ── Write consolidated output ────────────────────────────────────────
    out = dict(data)  # copy top-level metadata/network/clusters
    out["samples"] = []
    for i, (s, ct) in enumerate(zip(recal, cusum_result.cusum_series)):
        d = {
            "name": s.sample_name,
            "run_index": i,
            "collection_date": date_by_accession[s.sample_name],
            "score": round(s.score, 4),
            "level": s.level,
            "detected_pathogens": s.detected_pathogens,
            "community_signal": s.community_signal,
            "novelty_signal": s.novelty_signal,
            "breakdown": s.breakdown,
            "cusum": ct,
            "cusum_alert": ct >= cusum_result.threshold,
        }
        out["samples"].append(d)

    out["summary"] = {
        "total_samples": len(recal),
        "critical": after_levels["CRITICAL"],
        "high": after_levels["HIGH"],
        "moderate": after_levels["MODERATE"],
        "low": after_levels["LOW"],
    }
    out["temporal"] = {
        "site": "site207",
        "k": 0.5,
        "h": cusum_result.threshold,
        "cusum_upper": cusum_result.cusum_upper,
        "cusum_lower": cusum_result.cusum_lower,
        "alert": cusum_result.alert,
        "alert_decrease": cusum_result.alert_decrease,
        "n_consecutive_above": cusum_result.n_consecutive_above,
        "signal_strength": cusum_result.signal_strength,
        "cusum_series": cusum_result.cusum_series,
    }

    REPORT_OUT.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {REPORT_OUT}")


if __name__ == "__main__":
    main()
