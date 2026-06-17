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

import base64
import json
import os
import secrets
import shutil
import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from typing import List

from ..temporal.store import TimeSeriesStore, DEFAULT_DB

# ── Auth config (env vars must be set before this module is imported) ──────────
_AUTH_ENABLED = os.environ.get("PATHOGENIQ_DASH_AUTH", "true").lower() not in ("false", "0", "no")
_DASH_USER = os.environ.get("PATHOGENIQ_DASH_USER", "admin")
_DASH_PASS = os.environ.get("PATHOGENIQ_DASH_PASS", "pathogeniq")


class _BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not _AUTH_ENABLED:
            return await call_next(request)
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Basic "):
            try:
                decoded = base64.b64decode(auth[6:]).decode("utf-8")
                user, _, pwd = decoded.partition(":")
                if (secrets.compare_digest(user, _DASH_USER) and
                        secrets.compare_digest(pwd, _DASH_PASS)):
                    return await call_next(request)
            except Exception:
                pass
        return Response(
            "Authentication required",
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="PathogenIQ Dashboard"'},
        )


app = FastAPI(title="PathogenIQ Dashboard", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(_BasicAuthMiddleware)  # outermost: runs first, before CORS

_DB_PATH = Path(os.environ.get("PATHOGENIQ_DB", str(DEFAULT_DB)))
_REPORT_PATH = Path(os.environ.get("PATHOGENIQ_REPORT", "./reports/report.json"))
_HTML_PATH = Path(__file__).parent / "index.html"
_LOCATIONS_PATH = Path(__file__).parent / "site_locations.json"


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


@app.get("/api/locations")
async def locations():
    """Return site GPS coordinates from site_locations.json (keys starting with _ are stripped)."""
    if _LOCATIONS_PATH.exists():
        try:
            data = json.loads(_LOCATIONS_PATH.read_text())
            return JSONResponse({k: v for k, v in data.items() if not k.startswith("_")})
        except Exception:
            pass
    return JSONResponse({})


@app.get("/api/network")
async def network():
    """Return co-occurrence network (nodes + edges) from the latest report."""
    report = _latest_report()
    return JSONResponse(report.get("network", {"nodes": [], "edges": []}))


@app.get("/api/clusters")
async def clusters():
    """Return unsupervised cluster assignments and differential abundance results."""
    report = _latest_report()
    return JSONResponse(report.get("clusters", {
        "n_clusters": 0, "assignments": {}, "silhouette": 0.0, "differential": []
    }))


@app.get("/api/abundance_matrix")
async def abundance_matrix():
    """Return the top-50-taxa × all-samples abundance matrix for heatmap rendering."""
    report = _latest_report()
    return JSONResponse(report.get("abundance_matrix", {
        "taxa": [], "samples": [], "values": [], "cluster_assignments": {}
    }))


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
                "score": s.get("score"),
                "level": s.get("level"),
                "breakdown": s.get("breakdown", {}),
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


@app.get("/api/export/csv")
async def export_csv():
    """Download all samples as a flat CSV file."""
    import csv, io
    report = _latest_report()
    samples = report.get("samples", [])
    buf = io.StringIO()
    fields = [
        "name", "level", "score", "top_pathogen", "top_pathogen_abundance",
        "amr_genera", "amr_who_priority", "novelty_signal", "community_signal",
        "cusum_alert", "trend", "trend_tau", "baseline_anomaly", "generated_at",
    ]
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    generated_at = report.get("generated_at", "")
    for s in samples:
        top = s.get("detected_pathogens", [{}])[0] if s.get("detected_pathogens") else {}
        amr = s.get("amr_annotations", [])
        t = s.get("temporal", {})
        w.writerow({
            "name": s.get("name", ""),
            "level": s.get("level", ""),
            "score": s.get("score", ""),
            "top_pathogen": top.get("taxon", ""),
            "top_pathogen_abundance": round(top.get("abundance", 0), 4) if top else "",
            "amr_genera": "; ".join(a["genus"] for a in amr),
            "amr_who_priority": amr[0]["who_priority"] if amr else "",
            "novelty_signal": s.get("novelty_signal", ""),
            "community_signal": s.get("community_signal", ""),
            "cusum_alert": t.get("cusum_alert", ""),
            "trend": t.get("trend", ""),
            "trend_tau": t.get("trend_tau", ""),
            "baseline_anomaly": t.get("baseline_anomaly", ""),
            "generated_at": generated_at,
        })
    csv_text = buf.getvalue()
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=pathogeniq_report.csv"},
    )


@app.get("/api/export/json")
async def export_json():
    """Download the full report JSON."""
    report = _latest_report()
    return Response(
        content=json.dumps(report, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=pathogeniq_report.json"},
    )


@app.get("/api/diff")
async def diff_samples(a: str, b: str):
    """
    Compare two samples: returns taxa that appeared, disappeared,
    or changed significantly in relative abundance.
    """
    report = _latest_report()
    samples = {s["name"]: s for s in report.get("samples", [])}
    if a not in samples:
        raise HTTPException(404, f"Sample '{a}' not found")
    if b not in samples:
        raise HTTPException(404, f"Sample '{b}' not found")

    sa, sb = samples[a], samples[b]

    abundance_matrix = report.get("abundance_matrix", {})
    taxa = abundance_matrix.get("taxa", [])
    sample_names = abundance_matrix.get("samples", [])
    values = abundance_matrix.get("values", [])

    def _abund_map(sample_name: str) -> dict[str, float]:
        if sample_name not in sample_names:
            return {}
        idx = sample_names.index(sample_name)
        return {taxa[i]: values[i][idx] for i in range(len(taxa))}

    abund_a = _abund_map(a)
    abund_b = _abund_map(b)
    all_taxa = set(abund_a) | set(abund_b)

    THRESHOLD = 0.005  # minimum absolute change to report
    appeared, disappeared, changed = [], [], []

    for taxon in sorted(all_taxa):
        va = abund_a.get(taxon, 0.0)
        vb = abund_b.get(taxon, 0.0)
        delta = vb - va
        if abs(delta) < THRESHOLD:
            continue
        entry = {"taxon": taxon, "abundance_a": round(va, 5), "abundance_b": round(vb, 5),
                 "delta": round(delta, 5)}
        if va == 0:
            appeared.append(entry)
        elif vb == 0:
            disappeared.append(entry)
        else:
            changed.append(entry)

    changed.sort(key=lambda x: abs(x["delta"]), reverse=True)
    appeared.sort(key=lambda x: x["abundance_b"], reverse=True)
    disappeared.sort(key=lambda x: x["abundance_a"], reverse=True)

    return {
        "sample_a": {"name": a, "score": sa.get("score"), "level": sa.get("level")},
        "sample_b": {"name": b, "score": sb.get("score"), "level": sb.get("level")},
        "appeared": appeared[:30],
        "disappeared": disappeared[:30],
        "changed": changed[:30],
        "score_delta": round((sb.get("score", 0) or 0) - (sa.get("score", 0) or 0), 4),
    }


@app.get("/api/site/{site_name}/amr")
async def site_amr(site_name: str):
    """Return AMR annotations for a specific site from the latest report."""
    report = _latest_report()
    for s in report.get("samples", []):
        if s["name"] == site_name:
            return {"site": site_name, "amr_annotations": s.get("amr_annotations", [])}
    raise HTTPException(404, f"Site '{site_name}' not found")


# ── Upload & Run ──────────────────────────────────────────────────────────────
# In-memory job store (fine for single-process; persists until restart)
_JOBS: dict[str, dict] = {}


def _pipeline_worker(job_id: str, upload_dir: Path, rank: str, output_dir: Path):
    """Run the pipeline in a background thread, logging progress into _JOBS."""
    job = _JOBS[job_id]
    try:
        from ..pipeline.runner import run as _run, PipelineConfig
        job["status"] = "ingesting"
        job["log"] += "Ingesting Kraken2 reports…\n"

        cfg = PipelineConfig(
            output_dir=str(output_dir),
            alert_threshold=0.6,
            db_path=str(_DB_PATH),
        )

        job["status"] = "scoring"
        job["log"] += "Running pipeline (scoring, AMR, lineage, temporal)…\n"

        _run(input_path=upload_dir, config=cfg, rank=rank, run_characterization=False, quiet=True)

        # Update the global report path so the dashboard reloads the new result
        global _REPORT_PATH
        _REPORT_PATH = output_dir / "report.json"
        os.environ["PATHOGENIQ_REPORT"] = str(_REPORT_PATH)

        job["status"] = "done"
        job["log"] += f"Done. Report → {_REPORT_PATH}\n"
    except Exception as exc:
        job["status"] = "error"
        job["log"] += f"\nError: {exc}\n"
    finally:
        # Clean up uploaded files after pipeline finishes
        shutil.rmtree(upload_dir, ignore_errors=True)


@app.post("/api/upload-run")
async def upload_run(
    files: List[UploadFile] = File(...),
    rank: str = Form("G"),
):
    """
    Accept uploaded Kraken2 .report files (or a single .tsv/.csv count matrix),
    save them to a temp directory, and run the full PathogenIQ pipeline in a
    background thread.  Returns a job_id for polling /api/run-status/{job_id}.
    """
    job_id = str(uuid.uuid4())[:8]
    upload_dir = Path(tempfile.mkdtemp(prefix=f"piq_{job_id}_"))
    output_dir = _REPORT_PATH.parent  # write results next to existing report

    _JOBS[job_id] = {"status": "uploading", "log": f"Job {job_id} — saving {len(files)} file(s)…\n"}

    # Save uploaded files
    for uf in files:
        dest = upload_dir / (uf.filename or f"upload_{uuid.uuid4().hex[:6]}.report")
        content = await uf.read()
        dest.write_bytes(content)
        _JOBS[job_id]["log"] += f"  saved: {dest.name} ({len(content)//1024} KB)\n"

    _JOBS[job_id]["status"] = "ingesting"
    t = threading.Thread(
        target=_pipeline_worker,
        args=(job_id, upload_dir, rank, output_dir),
        daemon=True,
    )
    t.start()

    return JSONResponse({"job_id": job_id, "status": "started", "files": len(files)})


@app.get("/api/run-status/{job_id}")
async def run_status(job_id: str):
    """Poll the status and log of a running pipeline job."""
    if job_id not in _JOBS:
        raise HTTPException(404, f"Job '{job_id}' not found")
    job = _JOBS[job_id]
    return JSONResponse({"job_id": job_id, "status": job["status"], "log": job["log"]})


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
