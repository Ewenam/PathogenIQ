"""
reporting/report.py
Generate JSON and HTML surveillance reports from pipeline results.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ..scoring.risk import RiskScore


def _risk_color(level: str) -> str:
    return {"LOW": "#27ae60", "MODERATE": "#f39c12", "HIGH": "#e67e22", "CRITICAL": "#e74c3c"}.get(level, "#95a5a6")


def to_dict(risk_scores: list[RiskScore], meta: dict | None = None) -> dict:
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "metadata": meta or {},
        "summary": {
            "total_samples": len(risk_scores),
            "critical": sum(1 for r in risk_scores if r.level == "CRITICAL"),
            "high": sum(1 for r in risk_scores if r.level == "HIGH"),
            "moderate": sum(1 for r in risk_scores if r.level == "MODERATE"),
            "low": sum(1 for r in risk_scores if r.level == "LOW"),
        },
        "samples": [
            {
                "name": r.sample_name,
                "score": round(r.score, 4),
                "level": r.level,
                "detected_pathogens": r.detected_pathogens[:5],
                "community_signal": round(r.community_signal, 4),
                "novelty_signal": round(r.novelty_signal, 4),
                "breakdown": r.breakdown,
            }
            for r in risk_scores
        ],
    }


def save_json(risk_scores: list[RiskScore], output_path: str | Path, meta: dict | None = None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(to_dict(risk_scores, meta), f, indent=2)
    print(f"  JSON report → {output_path}")


def save_html(risk_scores: list[RiskScore], output_path: str | Path, meta: dict | None = None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = to_dict(risk_scores, meta)
    summary = data["summary"]

    rows = ""
    for s in data["samples"]:
        color = _risk_color(s["level"])
        pathogens = ", ".join(p["taxon"] for p in s["detected_pathogens"]) or "—"
        rows += f"""
        <tr>
          <td><strong>{s["name"]}</strong></td>
          <td><span style="color:{color};font-weight:bold">{s["level"]}</span></td>
          <td>{s["score"]:.3f}</td>
          <td style="font-size:12px">{pathogens}</td>
          <td>{s["community_signal"]:.3f}</td>
          <td>{s["novelty_signal"]:.3f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PathogenIQ Surveillance Report</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 40px; background: #f8f9fa; color: #222; }}
  h1 {{ color: #1a1a2e; }} h2 {{ color: #16213e; border-bottom: 2px solid #eee; padding-bottom: 8px; }}
  .summary {{ display: flex; gap: 16px; margin: 24px 0; }}
  .card {{ background: white; border-radius: 8px; padding: 20px 28px; box-shadow: 0 2px 8px rgba(0,0,0,.08); text-align: center; }}
  .card .num {{ font-size: 2em; font-weight: bold; }}
  .card .label {{ font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 1px; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px;
           overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
  th {{ background: #1a1a2e; color: white; padding: 12px 16px; text-align: left; font-size: 13px; }}
  td {{ padding: 10px 16px; border-bottom: 1px solid #f0f0f0; font-size: 13px; }}
  tr:hover td {{ background: #f5f8ff; }}
  .generated {{ color: #999; font-size: 12px; margin-top: 32px; }}
</style>
</head>
<body>
<h1>PathogenIQ Biosurveillance Report</h1>
<p>Generated: {data["generated_at"]} &nbsp;|&nbsp; Samples: {summary["total_samples"]}</p>

<h2>Summary</h2>
<div class="summary">
  <div class="card"><div class="num" style="color:#e74c3c">{summary["critical"]}</div><div class="label">Critical</div></div>
  <div class="card"><div class="num" style="color:#e67e22">{summary["high"]}</div><div class="label">High</div></div>
  <div class="card"><div class="num" style="color:#f39c12">{summary["moderate"]}</div><div class="label">Moderate</div></div>
  <div class="card"><div class="num" style="color:#27ae60">{summary["low"]}</div><div class="label">Low</div></div>
</div>

<h2>Sample Risk Scores</h2>
<table>
  <tr>
    <th>Sample</th><th>Risk Level</th><th>Score</th>
    <th>Detected Pathogens</th><th>Community Signal</th><th>Novelty Signal</th>
  </tr>
  {rows}
</table>

<p class="generated">PathogenIQ v0.1.0 — Biosurveillance Intelligence Platform</p>
</body>
</html>"""

    output_path.write_text(html)
    print(f"  HTML report → {output_path}")
