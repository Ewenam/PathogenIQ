"""
tests/test_planning_calculator.py
Pure-math tests for the sequencing sensitivity/cost planning calculator.
No fixtures needed — every check is closed-form or a round-trip.
"""
from __future__ import annotations

import math

import pytest

from pathogeniq.planning.calculator import (
    SequencingPlan,
    estimate_sensitivity,
    estimate_cost,
    required_depth_for_confidence,
    plan_report,
)


def test_closed_form_sensitivity():
    plan = SequencingPlan(prevalence=0.001, depth=3000, min_reads=1)
    # lam = 3000 * 0.001 = 3 -> sensitivity = 1 - e^-3
    expected = 1 - math.exp(-3)
    assert estimate_sensitivity(plan) == pytest.approx(expected, abs=1e-6)


def test_estimate_cost():
    plan = SequencingPlan(prevalence=0.001, depth=2_000_000,
                           cost_per_million_reads=5.0, sample_prep_cost=50.0)
    assert estimate_cost(plan) == pytest.approx(50.0 + 2 * 5.0)


def test_required_depth_round_trip():
    plan = SequencingPlan(prevalence=0.0005, min_reads=1)
    depth = required_depth_for_confidence(plan, 0.95)
    achieved = estimate_sensitivity(plan, depth)
    assert achieved >= 0.949


def test_higher_prevalence_needs_lower_depth():
    low_prev = SequencingPlan(prevalence=0.0001)
    high_prev = SequencingPlan(prevalence=0.001)
    depth_low = required_depth_for_confidence(low_prev, 0.95)
    depth_high = required_depth_for_confidence(high_prev, 0.95)
    assert depth_high < depth_low


@pytest.mark.parametrize("kwargs,target", [
    ({"prevalence": 0.0}, 0.95),
    ({"prevalence": 0.001}, 0.0),
    ({"prevalence": 0.001}, 1.0),
    ({"prevalence": 0.001, "min_reads": 0}, 0.95),
])
def test_guard_value_errors(kwargs, target):
    plan = SequencingPlan(**kwargs)
    with pytest.raises(ValueError):
        required_depth_for_confidence(plan, target)


def test_plan_report_shape():
    plan = SequencingPlan(prevalence=0.0001, depth=5_000_000)
    report = plan_report(plan)
    assert set(report.keys()) == {"plan", "current", "targets", "sweep"}
    assert report["current"] is not None
    assert set(report["targets"].keys()) == {"0.9", "0.95", "0.99"}
    for vals in report["targets"].values():
        assert "depth" in vals and "cost" in vals
    assert len(report["sweep"]) > 0
    for point in report["sweep"]:
        assert set(point.keys()) == {"depth", "sensitivity", "cost"}


def test_plan_report_without_depth_has_no_current():
    plan = SequencingPlan(prevalence=0.0001)
    report = plan_report(plan)
    assert report["current"] is None
