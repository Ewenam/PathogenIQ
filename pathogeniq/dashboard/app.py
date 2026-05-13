"""
dashboard/app.py
PathogenIQ web dashboard — FastAPI server.

Reads from the SQLite history store and the latest report JSON
to serve a real-time biosurveillance dashboard.

Run:
    python -m pathogeniq.dashboard.app
    # or via CLI:
    python cli.py dashboard
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from ..temporal.store import TimeSeriesStore, DEFAULT_DB

app = FastAPI(title="PathogenIQ Dashboard", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_DB_PATH = Path(os.environ.get("PATHOGENIQ_DB", str(DEFAULT_DB)))
_REPORT_PATH = Path(os.environ.get("PATHOGENIQ_REPORT", "./reports/report.json"))
_HTML_PATH = Path(__file__).parent / "index.html"


def _store() -> TimeSeriesStore:
    return TimeSeriesStore(_DB_PATH)


def _latest_report() -> dict:
    if _REPORT_PATH.exists():
        try:
            return json.loads(_REPORT_PATH.read_text())
        except Exception:
            pass
    return {}


# ── API routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    if not _HTML_PATH.exists():
        raise HTTPException(500, "Dashboard HTML not found")
    return HTMLResponse(_HTML_PATH.read_text())


@app.get("/api/summary")
async def summary():
    report = _latest_report()
    store = _store()
    return {
        "generated_at": report.get("generated_at", "—"),
        "total_runs": store.run_count(),
        "total_sites": len(store.all_sites()),
        "summary": report.get("summary", {}),
    }


@app.get("/api/sites")
async def sites():
    report = _latest_report()
    return {"sites": report.get("samples", [])}


@app.get("/api/alerts")
async def alerts():
    report = _latest_report()
    samples = report.get("samples", [])
    alert_levels = {"HIGH", "CRITICAL"}
    cusum_alerts = [
        s for s in samples
        if s.get("temporal", {}).get("cusum_alert")
        or (s.get("temporal", {}).get("z_score") or 0) > 2.0
        or s.get("level") in alert_levels
    ]
    return {"alerts": cusum_alerts}


@app.get("/api/site/{site_name}/history")
async def site_history(site_name: str, weeks: int = 16):
    store = _store()
    history = store.get_site_history(site_name, last_n=weeks)
    if not history:
        return {"site": site_name, "history": []}
    return {"site": site_name, "history": history}


@app.get("/api/site/{site_name}/pathogens")
async def site_pathogens(site_name: str):
    report = _latest_report()
    for s in report.get("samples", []):
        if s["name"] == site_name:
            return {
                "site": site_name,
                "pathogens": s.get("detected_pathogens", []),
                "temporal": s.get("temporal", {}),
            }
    raise HTTPException(404, f"Site '{site_name}' not found in latest report")


@app.get("/api/site/{site_name}/cusum")
async def site_cusum(site_name: str, weeks: int = 16):
    """Return CUSUM series data for charting."""
    store = _store()
    history = store.get_site_history(site_name, last_n=weeks)
    scores = [h["risk_score"] for h in history]
    dates = [h["run_date"] for h in history]

    if len(scores) < 2:
        return {"site": site_name, "dates": dates, "cusum": [], "scores": scores}

    from ..temporal.cusum import run_cusum
    result = run_cusum(site_name, scores)
    return {
        "site": site_name,
        "dates": dates,
        "scores": [round(s, 4) for s in scores],
        "cusum": result.cusum_series,
        "threshold": result.threshold,
        "alert": result.alert,
        "signal_strength": result.signal_strength,
    }


def _get_cusum_for_site(site_name: str, weeks: int = 16) -> dict:
    from ..temporal.cusum import run_cusum
    store = _store()
    history = store.get_site_history(site_name, last_n=weeks)
    scores = [h["risk_score"] for h in history]
    dates = [h["run_date"] for h in history]
    if len(scores) < 2:
        return {"site": site_name, "dates": dates, "cusum": [], "scores": scores, "threshold": 4.0}
    result = run_cusum(site_name, scores)
    return {
        "site": site_name,
        "dates": dates,
        "scores": [round(s, 4) for s in scores],
        "cusum": result.cusum_series,
        "threshold": result.threshold,
        "alert": result.alert,
        "signal_strength": result.signal_strength,
    }


app.get("/api/site/{site_name}/cusum_data")(
    lambda site_name, weeks=16: JSONResponse(_get_cusum_for_site(site_name, weeks))
)


if __name__ == "__main__":
    import uvicorn, sys
    db = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_DB)
    report = sys.argv[2] if len(sys.argv) > 2 else "./reports/report.json"
    os.environ["PATHOGENIQ_DB"] = db
    os.environ["PATHOGENIQ_REPORT"] = report
    print(f"PathogenIQ Dashboard → http://localhost:8765")
    print(f"  DB:     {db}")
    print(f"  Report: {report}")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="warning")
