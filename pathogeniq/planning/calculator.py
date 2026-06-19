"""
planning/calculator.py
Pre-deployment sequencing sensitivity/cost planning calculator.

Given a target pathogen prevalence and a sequencing depth, estimates the
probability of detecting at least `min_reads` reads from that pathogen
(Poisson model) and the associated cost. Also solves the inverse problem:
the depth required to hit a target detection confidence.

Fully decoupled from the analysis pipeline — this is a pre-deployment
budgeting tool, not part of pipeline.runner.run().
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import poisson


@dataclass
class SequencingPlan:
    prevalence: float
    depth: int = 0
    min_reads: int = 1
    cost_per_million_reads: float = 5.0
    sample_prep_cost: float = 50.0


def estimate_sensitivity(plan: SequencingPlan, depth: int | None = None) -> float:
    """Poisson detection probability: P(reads >= min_reads | depth, prevalence)."""
    d = depth if depth is not None else plan.depth
    lam = d * plan.prevalence
    return float(1 - poisson.cdf(plan.min_reads - 1, lam))


def estimate_cost(plan: SequencingPlan, depth: int | None = None) -> float:
    d = depth if depth is not None else plan.depth
    return plan.sample_prep_cost + (d / 1e6) * plan.cost_per_million_reads


def required_depth_for_confidence(plan: SequencingPlan, target_confidence: float) -> int:
    """
    Solve for the sequencing depth that achieves `target_confidence`
    detection probability, given plan.prevalence and plan.min_reads.
    """
    if plan.prevalence <= 0:
        raise ValueError("prevalence must be > 0 to solve for required depth")
    if not (0 < target_confidence < 1):
        raise ValueError("target_confidence must be strictly between 0 and 1")
    if plan.min_reads < 1:
        raise ValueError("min_reads must be >= 1")

    def f(lam: float) -> float:
        return (1 - poisson.cdf(plan.min_reads - 1, lam)) - target_confidence

    lam_root = brentq(f, 1e-9, 1e9)
    return math.ceil(lam_root / plan.prevalence)


def sweep(plan: SequencingPlan, depths: list[int]) -> list[dict]:
    return [
        {
            "depth": d,
            "sensitivity": estimate_sensitivity(plan, d),
            "cost": estimate_cost(plan, d),
        }
        for d in depths
    ]


def plan_report(
    plan: SequencingPlan,
    confidence_targets: tuple[float, ...] = (0.90, 0.95, 0.99),
    n_sweep_points: int = 20,
) -> dict:
    """
    Bundle current-depth sensitivity/cost (if plan.depth > 0), required
    depth/cost per confidence target, and a log-spaced sweep curve.
    """
    report: dict = {
        "plan": {
            "prevalence": plan.prevalence,
            "depth": plan.depth,
            "min_reads": plan.min_reads,
            "cost_per_million_reads": plan.cost_per_million_reads,
            "sample_prep_cost": plan.sample_prep_cost,
        },
        "current": None,
        "targets": {},
        "sweep": [],
    }

    if plan.depth > 0:
        report["current"] = {
            "sensitivity": estimate_sensitivity(plan),
            "cost": estimate_cost(plan),
        }

    required_depths = []
    for target in confidence_targets:
        depth = required_depth_for_confidence(plan, target)
        required_depths.append(depth)
        report["targets"][str(target)] = {
            "depth": depth,
            "cost": estimate_cost(plan, depth),
        }

    max_depth = max(required_depths + ([plan.depth] if plan.depth > 0 else []))
    floor_depth = max(1, max_depth // 1000)
    sweep_depths = sorted(set(
        int(d) for d in np.logspace(math.log10(floor_depth), math.log10(max_depth), n_sweep_points)
    ))
    report["sweep"] = sweep(plan, sweep_depths)

    return report
