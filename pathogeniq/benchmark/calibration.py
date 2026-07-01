"""
benchmark/calibration.py
Product-grade validation & threshold calibration.

The existing benchmark reports metrics at a *fixed* alert threshold. For a
surveillance product you instead need to (a) characterize the score's ranking
quality independent of any threshold (ROC / PR, AUROC / AUPRC) and (b) *choose*
an operating threshold to hit a stated sensitivity or specificity target — and
report the confusion matrix you'd get there. That operating point + its
error rates is the artifact a public-health or biosecurity buyer (and a
reviewer) actually needs.

This runs each labeled benchmark scenario through the pipeline ONCE to collect
threshold-independent risk scores, pools them, and calibrates.

Usage:
    from pathogeniq.benchmark.calibration import validate
    report = validate(target_sensitivity=0.95)      # dict, JSON-serializable
    print(report["operating_points"]["youden"])
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LabeledScores:
    scores: np.ndarray            # per-sample risk score in [0,1]
    labels: np.ndarray            # 1 = should-alert (true positive class), 0 = should-not
    scenario: list[str]           # scenario name per sample (for stratified metrics)


def collect_labeled_scores(seed: int = 42, quick: bool = False) -> LabeledScores:
    """Run every benchmark scenario once and pool (score, label) for samples
    whose ground-truth alert status is known (expected_alert is not None)."""
    from .synthetic import SCENARIOS, generate_scenario
    from .runner import _run_scenario_pipeline

    scores, labels, scen = [], [], []
    for i, scenario in enumerate(SCENARIOS):
        dataset = generate_scenario(scenario, seed=seed + i)
        risk = {r.sample_name: r.score for r in _run_scenario_pipeline(dataset, 0.6, quick)}
        for name, row in dataset.ground_truth.iterrows():
            expected = row.get("expected_alert")
            if expected is None:
                continue
            scores.append(float(risk.get(name, 0.0)))
            labels.append(1 if bool(expected) else 0)
            scen.append(scenario.name)
    return LabeledScores(np.asarray(scores), np.asarray(labels, dtype=int), scen)


def _confusion_at(scores: np.ndarray, labels: np.ndarray, theta: float) -> dict:
    pred = scores >= theta
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    prec = tp / (tp + fp) if (tp + fp) else None
    f1 = (2 * prec * sens / (prec + sens)) if (prec and sens and (prec + sens)) else 0.0
    return {"threshold": round(float(theta), 4), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": None if sens is None else round(sens, 4),
            "specificity": None if spec is None else round(spec, 4),
            "precision": None if prec is None else round(prec, 4),
            "f1": round(float(f1), 4)}


def calibrate(scores: np.ndarray, labels: np.ndarray,
              target_sensitivity: float = 0.95,
              target_specificity: float = 0.95) -> dict:
    """ROC/PR summary + operating points (Youden-J, target-sensitivity,
    target-specificity). Degenerate single-class input is handled gracefully."""
    from sklearn.metrics import (roc_curve, precision_recall_curve,
                                  roc_auc_score, average_precision_score)

    out: dict = {"n": int(len(scores)), "n_pos": int(labels.sum()),
                 "n_neg": int((labels == 0).sum())}
    if out["n_pos"] == 0 or out["n_neg"] == 0:
        out["note"] = "only one class present — ROC/PR undefined"
        return out

    out["auroc"] = round(float(roc_auc_score(labels, scores)), 4)
    out["auprc"] = round(float(average_precision_score(labels, scores)), 4)

    fpr, tpr, thr = roc_curve(labels, scores)
    # Youden's J = max(tpr - fpr); thr[0] is +inf sentinel from sklearn
    j = tpr - fpr
    j_idx = int(np.argmax(j))
    theta_youden = float(thr[j_idx]) if np.isfinite(thr[j_idx]) else float(scores.max())

    # roc_curve returns thresholds in DESCENDING order, so as the index grows
    # the threshold falls and tpr rises / spec falls (both monotone in index).
    def theta_for_sensitivity(target):
        ok = np.where(tpr >= target)[0]
        # tpr is non-decreasing in index → {tpr>=target} is a suffix; its first
        # index is the HIGHEST (most specific) threshold that still hits target.
        if not len(ok):
            return float(scores.min())
        t = thr[ok[0]]
        return float(t) if np.isfinite(t) else float(scores.max())

    def theta_for_specificity(target):
        spec = 1 - fpr
        ok = np.where(spec >= target)[0]
        # spec is non-increasing in index → {spec>=target} is a prefix; its last
        # index is the LOWEST (most sensitive) threshold that still hits target.
        if not len(ok):
            return float(scores.max())
        t = thr[ok[-1]]
        return float(t) if np.isfinite(t) else float(scores.min())

    out["operating_points"] = {
        "youden": _confusion_at(scores, labels, theta_youden),
        f"target_sensitivity_{target_sensitivity}":
            _confusion_at(scores, labels, theta_for_sensitivity(target_sensitivity)),
        f"target_specificity_{target_specificity}":
            _confusion_at(scores, labels, theta_for_specificity(target_specificity)),
        "default_0.6": _confusion_at(scores, labels, 0.6),
    }
    # compact ROC/PR curves for plotting/reporting
    prec, rec, _ = precision_recall_curve(labels, scores)
    out["roc_curve"] = {"fpr": [round(float(x), 4) for x in fpr],
                        "tpr": [round(float(x), 4) for x in tpr]}
    out["pr_curve"] = {"recall": [round(float(x), 4) for x in rec],
                       "precision": [round(float(x), 4) for x in prec]}
    return out


def validate(seed: int = 42, quick: bool = False,
             target_sensitivity: float = 0.95,
             target_specificity: float = 0.95) -> dict:
    """End-to-end: run scenarios, pool labeled scores, calibrate. Returns a
    JSON-serializable validation report."""
    ls = collect_labeled_scores(seed=seed, quick=quick)
    report = calibrate(ls.scores, ls.labels,
                       target_sensitivity=target_sensitivity,
                       target_specificity=target_specificity)
    report["seed"] = seed
    report["mode"] = "quick" if quick else "full"
    # per-scenario positive-rate context
    report["scenarios"] = sorted(set(ls.scenario))
    return report


def _print_report(report: dict) -> None:
    print(f"Validation on {report['n']} labeled samples "
          f"({report['n_pos']} pos / {report['n_neg']} neg)")
    if "auroc" not in report:
        print("  " + report.get("note", "insufficient data"))
        return
    print(f"  AUROC={report['auroc']}  AUPRC={report['auprc']}")
    print("  Operating points:")
    for name, op in report["operating_points"].items():
        print(f"    {name:28s} θ={op['threshold']:.3f}  "
              f"sens={op['sensitivity']}  spec={op['specificity']}  "
              f"prec={op['precision']}  F1={op['f1']}  "
              f"(TP/FP/TN/FN={op['tp']}/{op['fp']}/{op['tn']}/{op['fn']})")


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--target-sensitivity", type=float, default=0.95)
    ap.add_argument("--target-specificity", type=float, default=0.95)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()
    rep = validate(seed=args.seed, quick=args.quick,
                   target_sensitivity=args.target_sensitivity,
                   target_specificity=args.target_specificity)
    _print_report(rep)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(rep, f, indent=2)
        print(f"\nWrote {args.output}")
