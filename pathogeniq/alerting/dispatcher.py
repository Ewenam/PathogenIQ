"""
alerting/dispatcher.py
Unified alert dispatcher — fires all enabled channels in sequence.

Usage:
    from pathogeniq.alerting.dispatcher import AlertConfig, dispatch_alerts

    cfg = AlertConfig(
        email_enabled=True,
        smtp_user="me@gmail.com",
        smtp_pass="app-password",
        to_addrs=["team@example.com"],
        slack_enabled=True,
        slack_webhook="https://hooks.slack.com/services/...",
    )
    dispatch_alerts(alerts, config=cfg, run_date="2026-05-13")

Environment variable fallbacks (no config needed for basic use):
    PATHOGENIQ_SMTP_USER, PATHOGENIQ_SMTP_PASS, PATHOGENIQ_ALERT_EMAILS
    PATHOGENIQ_SLACK_WEBHOOK
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class AlertConfig:
    # Email
    email_enabled: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_pass: str = ""
    from_addr: str = ""
    to_addrs: list[str] = field(default_factory=list)
    # Slack
    slack_enabled: bool = False
    slack_webhook: str = ""

    @classmethod
    def from_env(cls) -> "AlertConfig":
        """Build config from environment variables."""
        smtp_user = os.environ.get("PATHOGENIQ_SMTP_USER", "")
        to_raw = os.environ.get("PATHOGENIQ_ALERT_EMAILS", "")
        to_addrs = [e.strip() for e in to_raw.split(",") if e.strip()]
        slack_wh = os.environ.get("PATHOGENIQ_SLACK_WEBHOOK", "")
        return cls(
            email_enabled=bool(smtp_user and to_addrs),
            smtp_user=smtp_user,
            smtp_pass=os.environ.get("PATHOGENIQ_SMTP_PASS", ""),
            to_addrs=to_addrs,
            slack_enabled=bool(slack_wh),
            slack_webhook=slack_wh,
        )

    @classmethod
    def from_yaml_section(cls, cfg: dict) -> "AlertConfig":
        """Build config from the 'alerting' section of a YAML config dict."""
        email = cfg.get("email", {})
        slack = cfg.get("slack", {})
        return cls(
            email_enabled=email.get("enabled", False),
            smtp_host=email.get("smtp_host", "smtp.gmail.com"),
            smtp_port=int(email.get("smtp_port", 587)),
            smtp_user=email.get("smtp_user", ""),
            smtp_pass=email.get("smtp_pass", ""),
            from_addr=email.get("from_addr", ""),
            to_addrs=email.get("to_addrs") or [],
            slack_enabled=slack.get("enabled", False),
            slack_webhook=slack.get("webhook_url", ""),
        )


def dispatch_alerts(
    alerts: list,
    config: AlertConfig | None = None,
    run_date: str = "",
) -> dict[str, bool]:
    """
    Send alerts to all enabled channels.

    Returns a dict of {channel_name: success_bool}.
    If no alerts, returns {} without sending anything.
    """
    if not alerts:
        return {}

    cfg = config or AlertConfig.from_env()
    results: dict[str, bool] = {}

    if cfg.email_enabled:
        from .email import send_email_alert
        ok = send_email_alert(
            alerts,
            smtp_host=cfg.smtp_host,
            smtp_port=cfg.smtp_port,
            smtp_user=cfg.smtp_user,
            smtp_pass=cfg.smtp_pass,
            from_addr=cfg.from_addr,
            to_addrs=cfg.to_addrs,
            run_date=run_date,
        )
        results["email"] = ok
        print(f"  Email alert → {'sent ✓' if ok else 'FAILED ✗'}")

    if cfg.slack_enabled:
        from .slack import send_slack_alert
        ok = send_slack_alert(
            alerts,
            webhook_url=cfg.slack_webhook,
            run_date=run_date,
        )
        results["slack"] = ok
        print(f"  Slack alert → {'sent ✓' if ok else 'FAILED ✗'}")

    return results
