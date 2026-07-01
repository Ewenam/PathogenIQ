"""
alerting/rules.py
Configurable alert RULE ENGINE — decides which samples become alerts.

The dashboard's /api/alerts hard-codes the firing rule (HIGH/CRITICAL or a CUSUM
flag). This makes that policy a per-organization, configurable object and emits
structured Alert records that plug straight into the existing transport layer
(alerting.dispatcher.dispatch_alerts → email/slack), whose senders read `.level`
and `.score`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

_LEVEL_ORDER = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}


def level_at_least(level: str, minimum: str) -> bool:
    return _LEVEL_ORDER.get(level, 0) >= _LEVEL_ORDER.get(minimum, 0)


@dataclass
class AlertRule:
    """A per-organization alerting policy."""
    name: str = "default"
    min_level: str = "HIGH"           # fire if sample level >= this ...
    min_score: float = 0.6            # ...or composite score >= this ...
    novelty_min: float | None = None  # ...or novelty signal >= this (if set)
    fire_on_cusum: bool = True        # a temporal CUSUM excursion alone fires
    cooldown_days: int = 7            # hint for the caller to suppress repeats
    enabled: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "AlertRule":
        return cls(**{k: d[k] for k in cls().__dict__ if k in d})


@dataclass
class Alert:
    site: str
    level: str                        # sample risk level (email/slack read this)
    score: float                      # composite score (email/slack read this)
    reasons: list[str]
    run_date: str = ""
    signals: dict = field(default_factory=dict)
    rule: str = "default"
    fired_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


def _reasons(sample: dict, rule: AlertRule) -> list[str]:
    out: list[str] = []
    level = sample.get("level", "LOW")
    score = float(sample.get("score", 0.0))
    novelty = float(sample.get("novelty_signal",
                    sample.get("breakdown", {}).get("novelty_signal", 0.0)) or 0.0)
    temporal = sample.get("temporal", {}) or {}
    if level_at_least(level, rule.min_level):
        out.append(f"risk level {level} ≥ {rule.min_level}")
    if score >= rule.min_score:
        out.append(f"score {score:.3f} ≥ {rule.min_score:.2f}")
    if rule.novelty_min is not None and novelty >= rule.novelty_min:
        out.append(f"novelty {novelty:.3f} ≥ {rule.novelty_min:.2f}")
    if rule.fire_on_cusum and (temporal.get("cusum_alert")
                               or (temporal.get("z_score") or 0) > 2.0):
        out.append("temporal CUSUM excursion")
    return out


def evaluate(report: dict, rule: AlertRule | None = None) -> list[Alert]:
    """Turn a pipeline report into Alerts (one per firing sample), severe-first."""
    rule = rule or AlertRule()
    if not rule.enabled:
        return []
    alerts: list[Alert] = []
    for s in report.get("samples", []):
        reasons = _reasons(s, rule)
        if not reasons:
            continue
        alerts.append(Alert(
            site=s.get("name", "?"),
            level=s.get("level", "LOW"),
            score=float(s.get("score", 0.0)),
            reasons=reasons,
            run_date=s.get("collection_date", report.get("generated_at", "")),
            signals={
                "abundance": s.get("breakdown", {}).get("abundance_score"),
                "community": s.get("community_signal"),
                "novelty": s.get("novelty_signal"),
                "cusum_alert": (s.get("temporal", {}) or {}).get("cusum_alert"),
            },
            rule=rule.name,
        ))
    alerts.sort(key=lambda a: (_LEVEL_ORDER.get(a.level, 0), a.score), reverse=True)
    return alerts


def evaluate_and_dispatch(report: dict, rule: AlertRule | None = None,
                          config=None, run_date: str = ""):
    """Rule engine → existing transport layer. Returns (alerts, channel_results)."""
    from .dispatcher import dispatch_alerts
    alerts = evaluate(report, rule)
    results = dispatch_alerts(alerts, config=config, run_date=run_date or
                              report.get("generated_at", ""))
    return alerts, results
