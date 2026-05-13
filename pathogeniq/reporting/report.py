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


def to_dict(risk_scores: list[RiskScore], meta: dict | None = None,
            baselines=None, cusum_results=None, trend_results=None,
            graph_data: dict | None = None,
            cluster_data: dict | None = None,
            abundance_matrix: dict | None = None) -> dict:
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
        "network": graph_data or {"nodes": [], "edges": []},
        "clusters": cluster_data or {"n_clusters": 0, "assignments": {}, "differential": []},
        "abundance_matrix": abundance_matrix or {"taxa": [], "samples": [], "values": []},
        "samples": [
            {
                "name": r.sample_name,
                "score": round(r.score, 4),
                "level": r.level,
                "detected_pathogens": r.detected_pathogens[:5],
                "community_signal": round(r.community_signal, 4),
                "novelty_signal": round(r.novelty_signal, 4),
                "breakdown": r.breakdown,
                "temporal": {
                    "z_score": round(baselines[r.sample_name].z_score, 3) if baselines and r.sample_name in baselines else None,
                    "pct_above_baseline": baselines[r.sample_name].pct_above_baseline if baselines and r.sample_name in baselines else None,
                    "baseline_anomaly": baselines[r.sample_name].is_anomaly if baselines and r.sample_name in baselines else None,
                    "cusum": round(cusum_results[r.sample_name].cusum_upper, 3) if cusum_results and r.sample_name in cusum_results else None,
                    "cusum_alert": cusum_results[r.sample_name].alert if cusum_results and r.sample_name in cusum_results else None,
                    "cusum_signal": cusum_results[r.sample_name].signal_strength if cusum_results and r.sample_name in cusum_results else None,
                    "trend": trend_results[r.sample_name].trend if trend_results and r.sample_name in trend_results else None,
                    "trend_tau": round(trend_results[r.sample_name].tau, 3) if trend_results and r.sample_name in trend_results else None,
                    "forecast_next": trend_results[r.sample_name].forecast_next if trend_results and r.sample_name in trend_results else None,
                    "trend_summary": trend_results[r.sample_name].summary if trend_results and r.sample_name in trend_results else None,
                },
            }
            for r in risk_scores
        ],
    }


def save_json(risk_scores: list[RiskScore], output_path: str | Path, meta: dict | None = None,
              baselines=None, cusum_results=None, trend_results=None,
              graph_data: dict | None = None,
              cluster_data: dict | None = None,
              abundance_matrix: dict | None = None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(to_dict(risk_scores, meta, baselines, cusum_results, trend_results,
                          graph_data, cluster_data, abundance_matrix), f, indent=2)
    print(f"  JSON report → {output_path}")


def save_html(risk_scores: list[RiskScore], output_path: str | Path, meta: dict | None = None,
              baselines=None, cusum_results=None, trend_results=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = to_dict(risk_scores, meta, baselines, cusum_results, trend_results)
    summary = data["summary"]

    rows = ""
    for s in data["samples"]:
        color = _risk_color(s["level"])
        pathogens = ", ".join(p["taxon"] for p in s["detected_pathogens"]) or "—"
        t = s.get("temporal") or {}
        z = f"{t['z_score']:+.2f}" if t.get("z_score") is not None else "—"
        cusum = f"{t['cusum']:.2f}" if t.get("cusum") is not None else "—"
        cusum_flag = " ⚠" if t.get("cusum_alert") else ""
        trend = t.get("trend", "—")
        trend_arrow = {"increasing": "↑", "decreasing": "↓", "stable": "→"}.get(trend, "—")
        trend_color = {"increasing": "#e74c3c", "decreasing": "#27ae60", "stable": "#888"}.get(trend, "#888")
        forecast = f"{t['forecast_next']:.3f}" if t.get("forecast_next") is not None else "—"
        rows += f"""
        <tr>
          <td><strong>{s["name"]}</strong></td>
          <td><span style="color:{color};font-weight:bold">{s["level"]}</span></td>
          <td>{s["score"]:.3f}</td>
          <td style="font-size:12px">{pathogens}</td>
          <td>{z}</td>
          <td>{cusum}{cusum_flag}</td>
          <td style="color:{trend_color};font-weight:bold">{trend_arrow} {trend}</td>
          <td>{forecast}</td>
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
    <th>Detected Pathogens</th><th>Z-Score</th><th>CUSUM</th><th>Trend</th><th>Forecast</th>
  </tr>
  {rows}
</table>

<p class="generated">PathogenIQ v0.1.0 — Biosurveillance Intelligence Platform</p>
</body>
</html>"""

    output_path.write_text(html)
    print(f"  HTML report → {output_path}")
