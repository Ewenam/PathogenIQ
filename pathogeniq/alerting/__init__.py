"""Alerting: a configurable rule engine (rules.py) feeding pluggable transports
(dispatcher.py → email.py / slack.py)."""
from .rules import AlertRule, Alert, evaluate, evaluate_and_dispatch, level_at_least
from .dispatcher import AlertConfig, dispatch_alerts

__all__ = [
    "AlertRule", "Alert", "evaluate", "evaluate_and_dispatch", "level_at_least",
    "AlertConfig", "dispatch_alerts",
]
