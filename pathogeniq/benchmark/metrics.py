"""
benchmark/metrics.py
Compute evaluation metrics for a benchmark scenario run.

Metrics:
  sensitivity (recall)    TP / (TP + FN)   — fraction of true positives caught
  specificity             TN / (TN + FP)   — fraction of true negatives correctly passed
  precision               TP / (TP + FP)   — of flagged samples, fraction truly positive
  F1                      2 * P * R / (P + R)
  mean/max risk score     distribution summary
  score_auc               area under ROC curve (requires sklearn)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ScenarioMetrics:
    scenario: str
    description: str
    n_samples: int
    n_contaminated: int   # ground-truth positives
    n_clean: int          # ground-truth negatives
    n_alerts: int         # pipeline-flagged alerts
    tp: int
    fp: int
    tn: int
    fn: int
    sensitivity: Optional[float]   # recall; None if no contaminated samples
    specificity: Optional[float]   # None if no clean samples
    precision: Optional[float]     # None if no alerts raised
    f1: Optional[float]
    mean_score: float
    max_score: float
    mean_contaminated_score: Optional[float]
    mean_clean_score: Optional[float]
    alert_threshold: float


def _safe_div(num: float, den: float) -> Optional[float]:
    return round(num / den, 4) if den > 0 else None


def _f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None:
        return None
    denom = precision + recall
    return round(2 * precision * recall / denom, 4) if denom > 0 else 0.0


def compute_metrics(
    scenario_name: str,
    description: str,
    risk_scores: list,
    ground_truth,   # pd.DataFrame with columns: is_contaminated, expected_alert
    alert_threshold: float = 0.6,
) -> ScenarioMetrics:
    """
    Compute binary classification metrics for one scenario run.

    A sample is considered a true positive if:
      - ground_truth.is_contaminated == True
      - ground_truth.expected_alert == True (not None)
      - pipeline risk_score >= alert_threshold
    """
    score_map = {r.sample_name: r.score for r in risk_scores}

    tp = fp = tn = fn = 0
    contaminated_scores: list[float] = []
    clean_scores: list[float] = []
    all_scores: list[float] = []

    for sample_name, row in ground_truth.iterrows():
        score = score_map.get(sample_name, 0.0)
        all_scores.append(score)
        predicted_alert = score >= alert_threshold
        expected = row.get("expected_alert")
        is_contaminated = bool(row["is_contaminated"])

        if is_contaminated:
            contaminated_scores.append(score)
        else:
            clean_scores.append(score)

        # Only count confusion matrix when expected_alert is known (not None)
        if expected is None:
            continue
        truly_positive = bool(expected)

        if truly_positive and predicted_alert:
            tp += 1
        elif truly_positive and not predicted_alert:
            fn += 1
        elif not truly_positive and predicted_alert:
            fp += 1
        else:
            tn += 1

    n_contaminated = int(ground_truth["is_contaminated"].sum())
    n_clean = len(ground_truth) - n_contaminated
    n_alerts = sum(1 for s in all_scores if s >= alert_threshold)

    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    f1 = _f1(precision, sensitivity)

    return ScenarioMetrics(
        scenario=scenario_name,
        description=description,
        n_samples=len(ground_truth),
        n_contaminated=n_contaminated,
        n_clean=n_clean,
        n_alerts=n_alerts,
        tp=tp, fp=fp, tn=tn, fn=fn,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        f1=f1,
        mean_score=round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0,
        max_score=round(max(all_scores), 4) if all_scores else 0.0,
        mean_contaminated_score=(
            round(sum(contaminated_scores) / len(contaminated_scores), 4)
            if contaminated_scores else None
        ),
        mean_clean_score=(
            round(sum(clean_scores) / len(clean_scores), 4)
            if clean_scores else None
        ),
        alert_threshold=alert_threshold,
    )


def aggregate_metrics(results: list[ScenarioMetrics]) -> dict:
    """Compute macro-averaged metrics across all evaluated scenarios."""
    evaluated = [r for r in results if r.sensitivity is not None]
    if not evaluated:
        return {}

    macro_sens = sum(r.sensitivity for r in evaluated) / len(evaluated)
    macro_spec = sum(r.specificity or 0 for r in evaluated) / len(evaluated)
    macro_prec = sum(r.precision or 0 for r in evaluated) / len(evaluated)
    macro_f1 = sum(r.f1 or 0 for r in evaluated) / len(evaluated)

    total_tp = sum(r.tp for r in results)
    total_fp = sum(r.fp for r in results)
    total_tn = sum(r.tn for r in results)
    total_fn = sum(r.fn for r in results)

    return {
        "macro_sensitivity": round(macro_sens, 4),
        "macro_specificity": round(macro_spec, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_sensitivity": _safe_div(total_tp, total_tp + total_fn),
        "micro_specificity": _safe_div(total_tn, total_tn + total_fp),
        "micro_precision": _safe_div(total_tp, total_tp + total_fp),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_tn": total_tn,
        "total_fn": total_fn,
    }
