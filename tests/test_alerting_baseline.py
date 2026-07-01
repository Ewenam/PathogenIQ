"""Tests for the alert rule engine and the persisted novelty baseline."""
import random

import numpy as np

from pathogeniq.alerting import (
    AlertRule, evaluate, evaluate_and_dispatch, level_at_least,
)
from pathogeniq.novelty.baseline import (
    fit_and_save_baseline, load_baseline, BaselineManager,
)


def _report():
    return {
        "generated_at": "2026-06-12",
        "samples": [
            {"name": "s_high", "level": "HIGH", "score": 0.72,
             "community_signal": 1.0, "novelty_signal": 0.3,
             "breakdown": {"abundance_score": 0.4}, "temporal": {}},
            {"name": "s_low", "level": "LOW", "score": 0.12,
             "novelty_signal": 0.1, "breakdown": {}, "temporal": {}},
            {"name": "s_cusum", "level": "MODERATE", "score": 0.4,
             "novelty_signal": 0.2, "breakdown": {}, "temporal": {"cusum_alert": True}},
        ],
    }


# ── alerting ─────────────────────────────────────────────────────────────────
def test_level_ordering():
    assert level_at_least("CRITICAL", "HIGH")
    assert not level_at_least("MODERATE", "HIGH")


def test_evaluate_default_rule_fires_high_and_cusum():
    alerts = evaluate(_report(), AlertRule())
    sites = {a.site for a in alerts}
    assert "s_high" in sites          # HIGH level
    assert "s_cusum" in sites         # CUSUM excursion
    assert "s_low" not in sites       # nothing fires
    # most-severe first
    assert alerts[0].level == "HIGH"
    assert any("CUSUM" in r for a in alerts for r in a.reasons if a.site == "s_cusum")


def test_rule_thresholds_and_disable():
    # raise the bar: only CRITICAL, no CUSUM firing → nothing
    strict = AlertRule(min_level="CRITICAL", min_score=0.99, fire_on_cusum=False)
    assert evaluate(_report(), strict) == []
    assert evaluate(_report(), AlertRule(enabled=False)) == []


def test_novelty_floor_fires():
    rule = AlertRule(min_level="CRITICAL", min_score=0.99,
                     fire_on_cusum=False, novelty_min=0.25)
    alerts = evaluate(_report(), rule)
    # s_high has novelty 0.3 ≥ 0.25
    assert {a.site for a in alerts} == {"s_high"}


def test_evaluate_and_dispatch_no_channels_is_noop():
    # no channels configured → dispatch returns {} but still returns the alerts
    from pathogeniq.alerting import AlertConfig
    alerts, results = evaluate_and_dispatch(_report(), AlertRule(), config=AlertConfig())
    assert len(alerts) == 2
    assert results == {}


# ── baseline persistence ─────────────────────────────────────────────────────
def _reads(n, seed):
    rng = random.Random(seed)
    return ["".join(rng.choice("ACGT") for _ in range(120)) for _ in range(n)]


def test_baseline_save_load_roundtrip(tmp_path):
    ref = fit_and_save_baseline([_reads(150, s) for s in range(10)],
                                tmp_path / "b.json", k=4)
    q = np.vstack([ref.profiler.sample_profile(_reads(150, 100))])
    s_before = ref.score_profiles(q)
    s_after = load_baseline(tmp_path / "b.json").score_profiles(q)
    assert np.allclose(s_before, s_after)


def test_baseline_manager_keys_by_org_site(tmp_path):
    mgr = BaselineManager(root=tmp_path)
    assert not mgr.has("siteA", org="org1")
    mgr.fit("siteA", [_reads(150, s) for s in range(10)], org="org1", k=4)
    assert mgr.has("siteA", org="org1")
    assert not mgr.has("siteA", org="org2")     # org isolation
    scores = mgr.score("siteA", {"new": _reads(150, 99)}, org="org1")
    assert "new" in scores and 0.0 <= scores["new"] <= 1.0
