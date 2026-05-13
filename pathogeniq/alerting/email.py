"""
alerting/email.py
Send HTML alert emails via SMTP.

Configure via environment variables or AlertConfig:
  PATHOGENIQ_SMTP_USER   — sender email address
  PATHOGENIQ_SMTP_PASS   — SMTP password / app password
  PATHOGENIQ_ALERT_EMAILS — comma-separated recipient list
"""
from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email_alert(
    alerts: list,
    smtp_host: str = "smtp.gmail.com",
    smtp_port: int = 587,
    smtp_user: str = "",
    smtp_pass: str = "",
    from_addr: str = "",
    to_addrs: list[str] | None = None,
    run_date: str = "",
) -> bool:
    """Send an HTML alert digest email. Returns True on success."""
    smtp_user = smtp_user or os.environ.get("PATHOGENIQ_SMTP_USER", "")
    smtp_pass = smtp_pass or os.environ.get("PATHOGENIQ_SMTP_PASS", "")
    from_addr = from_addr or smtp_user

    if not to_addrs:
        env_addrs = os.environ.get("PATHOGENIQ_ALERT_EMAILS", "")
        to_addrs = [e.strip() for e in env_addrs.split(",") if e.strip()]

    if not smtp_user or not smtp_pass or not to_addrs:
        return False

    n = len(alerts)
    subject = (
        f"[PathogenIQ] {n} Alert{'s' if n != 1 else ''} Detected"
        + (f" — {run_date}" if run_date else "")
    )

    level_color = {"HIGH": "#e67e22", "CRITICAL": "#e74c3c", "MODERATE": "#f39c12"}
    rows = ""
    for r in alerts:
        color = level_color.get(r.level, "#95a5a6")
        top = r.detected_pathogens[0]["taxon"] if r.detected_pathogens else "—"
        rows += f"""
        <tr>
          <td style="padding:10px 14px;border-bottom:1px solid #eee">
            <strong>{r.sample_name}</strong></td>
          <td style="padding:10px 14px;border-bottom:1px solid #eee;
            color:{color};font-weight:bold">{r.level}</td>
          <td style="padding:10px 14px;border-bottom:1px solid #eee">{r.score:.3f}</td>
          <td style="padding:10px 14px;border-bottom:1px solid #eee">{top}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html><body style="font-family:-apple-system,Helvetica,sans-serif;color:#222;
  margin:0;padding:30px;background:#f8f9fa">
<div style="max-width:640px;margin:0 auto;background:#fff;border-radius:10px;
  box-shadow:0 2px 12px rgba(0,0,0,.10);overflow:hidden">
  <div style="background:#1a1a2e;padding:24px 28px">
    <h2 style="color:#fff;margin:0">PathogenIQ Biosurveillance Alert</h2>
    <p style="color:#aaa;margin:6px 0 0">
      {n} site{'s' if n != 1 else ''} triggered alert{('s' if n != 1 else '')}
      {('— ' + run_date) if run_date else ''}
    </p>
  </div>
  <div style="padding:24px 28px">
    <table style="width:100%;border-collapse:collapse">
      <tr style="background:#f0f2f5">
        <th style="padding:10px 14px;text-align:left;font-size:12px;
          text-transform:uppercase;letter-spacing:.5px;color:#555">Site</th>
        <th style="padding:10px 14px;text-align:left;font-size:12px;
          text-transform:uppercase;letter-spacing:.5px;color:#555">Level</th>
        <th style="padding:10px 14px;text-align:left;font-size:12px;
          text-transform:uppercase;letter-spacing:.5px;color:#555">Score</th>
        <th style="padding:10px 14px;text-align:left;font-size:12px;
          text-transform:uppercase;letter-spacing:.5px;color:#555">Top Pathogen</th>
      </tr>
      {rows}
    </table>
  </div>
  <div style="padding:16px 28px;background:#f8f9fa;border-top:1px solid #eee">
    <p style="color:#999;font-size:12px;margin:0">
      PathogenIQ v0.1.0 — Biosurveillance Intelligence Platform
    </p>
  </div>
</div>
</body></html>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(from_addr, to_addrs, msg.as_string())
        return True
    except Exception as exc:
        print(f"  [email alert] Failed: {exc}")
        return False
