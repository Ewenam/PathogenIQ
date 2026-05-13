"""
alerting/slack.py
Send alert messages to Slack via an incoming webhook.

Configure:
  PATHOGENIQ_SLACK_WEBHOOK — Slack incoming webhook URL
  (or pass webhook_url= directly)
"""
from __future__ import annotations

import json
import os
import urllib.request


def send_slack_alert(
    alerts: list,
    webhook_url: str = "",
    run_date: str = "",
) -> bool:
    """POST a Block Kit alert message to a Slack webhook. Returns True on success."""
    webhook_url = webhook_url or os.environ.get("PATHOGENIQ_SLACK_WEBHOOK", "")
    if not webhook_url:
        return False

    level_emoji = {
        "CRITICAL": ":rotating_light:",
        "HIGH": ":warning:",
        "MODERATE": ":large_yellow_circle:",
    }

    date_str = f"  •  Run date: *{run_date}*" if run_date else ""
    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f":microbe: PathogenIQ — {len(alerts)} site(s) flagged",
            },
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"*{len(alerts)}* alert{'s' if len(alerts) != 1 else ''} detected{date_str}",
                }
            ],
        },
        {"type": "divider"},
    ]

    for r in alerts:
        emoji = level_emoji.get(r.level, ":white_circle:")
        top = r.detected_pathogens[0]["taxon"] if r.detected_pathogens else "—"
        disease = r.detected_pathogens[0]["disease"] if r.detected_pathogens else "—"
        blocks.append(
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Site*\n{r.sample_name}"},
                    {"type": "mrkdwn", "text": f"*Level*\n{emoji} {r.level}"},
                    {"type": "mrkdwn", "text": f"*Score*\n`{r.score:.3f}`"},
                    {"type": "mrkdwn", "text": f"*Top Pathogen*\n{top} ({disease})"},
                ],
            }
        )

    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "_PathogenIQ v0.1.0 — Biosurveillance Intelligence Platform_",
                }
            ],
        }
    )

    payload = json.dumps({"blocks": blocks}).encode()
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as exc:
        print(f"  [slack alert] Failed: {exc}")
        return False
