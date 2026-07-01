"""
Tests for the validation/threshold-calibration harness. These exercise the
calibration math directly on constructed score/label sets (no pipeline), so
they are fast and deterministic.
"""
import numpy as np
import pytest

from pathogeniq.benchmark.calibration import calibrate, _confusion_at


def test_perfect_separation():
    scores = np.array([0.1, 0.2, 0.15, 0.8, 0.9, 0.85])
    labels = np.array([0, 0, 0, 1, 1, 1])
    rep = calibrate(scores, labels, target_sensitivity=0.95, target_specificity=0.95)
    assert rep["auroc"] == 1.0
    # every operating point should achieve perfect sens & spec
    for op in rep["operating_points"].values():
        assert op["sensitivity"] == 1.0 and op["specificity"] == 1.0
    # the chosen threshold must sit in the separating gap (0.2, 0.8]
    assert 0.2 < rep["operating_points"]["youden"]["threshold"] <= 0.8


def test_target_sensitivity_is_met_and_specific():
    # overlapping distributions → non-trivial trade-off
    rng = np.random.default_rng(0)
    neg = rng.normal(0.35, 0.1, 200).clip(0, 1)
    pos = rng.normal(0.6, 0.1, 200).clip(0, 1)
    scores = np.concatenate([neg, pos])
    labels = np.concatenate([np.zeros(200, int), np.ones(200, int)])
    rep = calibrate(scores, labels, target_sensitivity=0.95, target_specificity=0.90)

    assert 0.5 < rep["auroc"] < 1.0
    op_sens = rep["operating_points"]["target_sensitivity_0.95"]
    assert op_sens["sensitivity"] >= 0.95            # target met
    op_spec = rep["operating_points"]["target_specificity_0.9"]
    assert op_spec["specificity"] >= 0.90            # target met
    # higher sensitivity target ⇒ lower (more permissive) threshold than the
    # specificity-oriented operating point
    assert op_sens["threshold"] <= op_spec["threshold"]


def test_single_class_is_graceful():
    scores = np.array([0.2, 0.3, 0.4])
    labels = np.array([0, 0, 0])
    rep = calibrate(scores, labels)
    assert "auroc" not in rep and "note" in rep


def test_confusion_at_counts():
    scores = np.array([0.1, 0.7, 0.7, 0.2])
    labels = np.array([0, 1, 0, 1])
    c = _confusion_at(scores, labels, 0.5)
    assert (c["tp"], c["fp"], c["tn"], c["fn"]) == (1, 1, 1, 1)
    assert c["sensitivity"] == 0.5 and c["specificity"] == 0.5


def test_hard_scenarios_generate_with_expected_labels():
    from pathogeniq.benchmark.synthetic import HARD_SCENARIOS, generate_scenario
    assert len(HARD_SCENARIOS) >= 3
    for i, sc in enumerate(HARD_SCENARIOS):
        ds = generate_scenario(sc, seed=1 + i)
        assert ds.count_matrix.shape[1] == sc.n_samples
        # ground-truth alert label matches the scenario's expected_alert
        assert (ds.ground_truth["expected_alert"] == sc.expected_alert).all()
