"""
dashboard/app.py
PathogenIQ web dashboard — FastAPI server.

Reads from the SQLite history store and the latest report JSON
to serve a real-time biosurveillance dashboard.

Two modes, selected by PATHOGENIQ_MODE:
  selfhosted (default) — single global SQLite store + Basic Auth, exactly as
                          before. Used by the CLI/HPC research pipeline.
  saas                  — Postgres-backed, org-scoped data + JWT auth. Each
                          request resolves to an org_id (via saas.auth) which
                          scopes both the history store and the report path.

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

from fastapi import Depends, FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from typing import List

from ..temporal.store import TimeSeriesStore, DEFAULT_DB

_MODE = os.environ.get("PATHOGENIQ_MODE", "selfhosted").lower()

# ── Auth config (env vars must be set before this module is imported) ──────────
_AUTH_ENABLED = os.environ.get("PATHOGENIQ_DASH_AUTH", "true").lower() not in ("false", "0", "no")
_DASH_USER = os.environ.get("PATHOGENIQ_DASH_USER", "admin")
_DASH_PASS = os.environ.get("PATHOGENIQ_DASH_PASS", "pathogeniq")

if _MODE != "saas" and _AUTH_ENABLED and _DASH_PASS == "pathogeniq":
    print(
        "WARNING: PATHOGENIQ_DASH_PASS is unset and using the default password. "
        "Set PATHOGENIQ_DASH_PASS before exposing this dashboard beyond localhost.",
        flush=True,
    )


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
# Cross-origin access is opt-in: the dashboard's own JS talks to this API same-origin,
# so CORS only needs to be opened up for external integrations (set explicitly via env).
_CORS_ORIGINS = [o.strip() for o in os.environ.get("PATHOGENIQ_CORS_ORIGINS", "").split(",") if o.strip()]
if _CORS_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=_CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
if _MODE != "saas":
    # saas mode authenticates per-request via JWT (see _org_dep below), not a
    # single shared Basic Auth password.
    app.add_middleware(_BasicAuthMiddleware)  # outermost: runs first, before CORS

_DB_PATH = Path(os.environ.get("PATHOGENIQ_DB", str(DEFAULT_DB)))
_REPORT_PATH = Path(os.environ.get("PATHOGENIQ_REPORT", "./reports/report.json"))
_HTML_PATH = Path(__file__).parent / "index.html"
_LOGIN_HTML_PATH = Path(__file__).parent / "login.html"
_LOCATIONS_PATH = Path(__file__).parent / "site_locations.json"
_SAAS_REPORTS_DIR = Path(os.environ.get("PATHOGENIQ_SAAS_REPORTS_DIR", "./reports"))


def _org_dep(request: Request) -> str | None:
    """
    FastAPI dependency. In selfhosted mode, returns None — every helper below
    treats None as "use the single global SQLite store/report", i.e. today's
    unchanged behavior. In saas mode, verifies the request's Bearer JWT and
    resolves it to an org_id; raises 401/403 on failure.
    """
    if _MODE != "saas":
        return None
    from ..saas.auth import get_current_org
    return get_current_org(request)


def _actor_dep(request: Request) -> str | None:
    """
    FastAPI dependency mirroring _org_dep: returns None in selfhosted mode
    (callers substitute a generic default), or the JWT's user_id in saas mode —
    recorded as the `actor` in each run's provenance manifest.
    """
    if _MODE != "saas":
        return None
    from ..saas.auth import get_current_user_id
    return get_current_user_id(request)


def _store_for(org_id: str | None):
    if org_id is not None:
        from ..saas.db import PostgresOrgStore
        return PostgresOrgStore(org_id)
    return TimeSeriesStore(_DB_PATH)


def _report_path_for(org_id: str | None) -> Path:
    if org_id is not None:
        return _SAAS_REPORTS_DIR / org_id / "report.json"
    return _REPORT_PATH


def _latest_report_for(org_id: str | None) -> dict:
    path = _report_path_for(org_id)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


# ── API routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    if not _HTML_PATH.exists():
        raise HTTPException(500, "Dashboard HTML not found")
    return HTMLResponse(_HTML_PATH.read_text())


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    if not _LOGIN_HTML_PATH.exists():
        raise HTTPException(404, "Login page not configured")
    html = _LOGIN_HTML_PATH.read_text()
    html = html.replace("{{SUPABASE_URL}}", os.environ.get("PATHOGENIQ_SUPABASE_URL", ""))
    html = html.replace("{{SUPABASE_ANON_KEY}}", os.environ.get("PATHOGENIQ_SUPABASE_ANON_KEY", ""))
    return HTMLResponse(html)


@app.post("/api/saas/bootstrap-org")
async def bootstrap_org_endpoint(request: Request, org_name: str = Form(...)):
    """First-login signup: create (or fetch) the caller's organization."""
    if _MODE != "saas":
        raise HTTPException(404, "Not available outside PATHOGENIQ_MODE=saas")
    from ..saas.auth import get_current_user_id
    from ..saas.db import get_engine
    from ..saas.orgs import bootstrap_org

    user_id = get_current_user_id(request)
    org_id = bootstrap_org(get_engine(), user_id, org_name)
    return {"org_id": org_id}


@app.get("/api/summary")
async def summary(org_id: str | None = Depends(_org_dep)):
    report = _latest_report_for(org_id)
    store = _store_for(org_id)
    return {
        "generated_at": report.get("generated_at", "—"),
        "total_runs": store.run_count(),
        "total_sites": len(store.all_sites()),
        "summary": report.get("summary", {}),
    }


@app.get("/api/sites")
async def sites(org_id: str | None = Depends(_org_dep)):
    report = _latest_report_for(org_id)
    return {"sites": report.get("samples", [])}


@app.get("/api/alerts")
async def alerts(org_id: str | None = Depends(_org_dep)):
    report = _latest_report_for(org_id)
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
async def network(org_id: str | None = Depends(_org_dep)):
    """Return co-occurrence network (nodes + edges) from the latest report."""
    report = _latest_report_for(org_id)
    return JSONResponse(report.get("network", {"nodes": [], "edges": []}))


@app.get("/api/clusters")
async def clusters(org_id: str | None = Depends(_org_dep)):
    """Return unsupervised cluster assignments and differential abundance results."""
    report = _latest_report_for(org_id)
    return JSONResponse(report.get("clusters", {
        "n_clusters": 0, "assignments": {}, "silhouette": 0.0, "differential": []
    }))


@app.get("/api/abundance_matrix")
async def abundance_matrix(org_id: str | None = Depends(_org_dep)):
    """Return the top-50-taxa × all-samples abundance matrix for heatmap rendering."""
    report = _latest_report_for(org_id)
    return JSONResponse(report.get("abundance_matrix", {
        "taxa": [], "samples": [], "values": [], "cluster_assignments": {}
    }))


@app.get("/api/outbreak_clusters")
async def outbreak_clusters(org_id: str | None = Depends(_org_dep)):
    """Return the sample-to-sample similarity dendrogram and outbreak-source clusters."""
    report = _latest_report_for(org_id)
    return JSONResponse(report.get("outbreak_clusters", {"dendrogram": None, "clusters": []}))


@app.get("/api/plan")
async def plan_endpoint(prevalence: float, depth: int = 0, min_reads: int = 1,
                         cost_per_million_reads: float = 5.0, sample_prep_cost: float = 50.0):
    """Stateless sequencing sensitivity/cost planning calculator (no org scoping needed)."""
    from ..planning.calculator import SequencingPlan, plan_report
    try:
        p = SequencingPlan(prevalence=prevalence, depth=depth, min_reads=min_reads,
                            cost_per_million_reads=cost_per_million_reads, sample_prep_cost=sample_prep_cost)
        return JSONResponse(plan_report(p))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/site/{site_name}/history")
async def site_history(site_name: str, weeks: int = 16, org_id: str | None = Depends(_org_dep)):
    store = _store_for(org_id)
    history = store.get_site_history(site_name, last_n=weeks)
    if not history:
        return {"site": site_name, "history": []}
    return {"site": site_name, "history": history}


@app.get("/api/site/{site_name}/pathogens")
async def site_pathogens(site_name: str, org_id: str | None = Depends(_org_dep)):
    report = _latest_report_for(org_id)
    for s in report.get("samples", []):
        if s["name"] == site_name:
            return {
                "site": site_name,
                "pathogens": s.get("detected_pathogens", []),
                "temporal": s.get("temporal", {}),
                "score": s.get("score"),
                "level": s.get("level"),
                "breakdown": s.get("breakdown", {}),
                "external_validation": s.get("external_validation"),
            }
    raise HTTPException(404, f"Site '{site_name}' not found in latest report")


@app.get("/api/site/{site_name}/cusum")
async def site_cusum(site_name: str, weeks: int = 16, org_id: str | None = Depends(_org_dep)):
    """Return CUSUM series data for charting."""
    store = _store_for(org_id)
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


@app.get("/api/site/{site_name}/cusum_data")
async def site_cusum_data(site_name: str, weeks: int = 16, org_id: str | None = Depends(_org_dep)):
    from ..temporal.cusum import run_cusum
    store = _store_for(org_id)
    history = store.get_site_history(site_name, last_n=weeks)
    scores = [h["risk_score"] for h in history]
    dates = [h["run_date"] for h in history]
    if len(scores) < 2:
        return JSONResponse({"site": site_name, "dates": dates, "cusum": [], "scores": scores, "threshold": 4.0})
    result = run_cusum(site_name, scores)
    return JSONResponse({
        "site": site_name,
        "dates": dates,
        "scores": [round(s, 4) for s in scores],
        "cusum": result.cusum_series,
        "threshold": result.threshold,
        "alert": result.alert,
        "signal_strength": result.signal_strength,
    })


@app.get("/api/export/csv")
async def export_csv(org_id: str | None = Depends(_org_dep)):
    """Download all samples as a flat CSV file."""
    import csv, io
    report = _latest_report_for(org_id)
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
async def export_json(org_id: str | None = Depends(_org_dep)):
    """Download the full report JSON."""
    report = _latest_report_for(org_id)
    return Response(
        content=json.dumps(report, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=pathogeniq_report.json"},
    )


@app.get("/api/diff")
async def diff_samples(a: str, b: str, org_id: str | None = Depends(_org_dep)):
    """
    Compare two samples: returns taxa that appeared, disappeared,
    or changed significantly in relative abundance.
    """
    report = _latest_report_for(org_id)
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
async def site_amr(site_name: str, org_id: str | None = Depends(_org_dep)):
    """Return AMR annotations for a specific site from the latest report."""
    report = _latest_report_for(org_id)
    for s in report.get("samples", []):
        if s["name"] == site_name:
            return {"site": site_name, "amr_annotations": s.get("amr_annotations", [])}
    raise HTTPException(404, f"Site '{site_name}' not found")


# ── Upload & Run ──────────────────────────────────────────────────────────────
# In-memory job store (fine for single-process; persists until restart)
_JOBS: dict[str, dict] = {}

_ALLOWED_UPLOAD_SUFFIXES = {".report", ".tsv", ".csv", ".txt", ".bracken", ".metaphlan"}
_MAX_UPLOAD_FILES = 500
_MAX_FILE_BYTES = 50 * 1024 * 1024       # 50 MB per file
_MAX_TOTAL_BYTES = 1024 * 1024 * 1024    # 1 GB per upload batch


def _safe_upload_name(filename: str | None, fallback: str) -> str:
    """
    Reduce an attacker-controlled filename to a bare, extension-checked basename.
    Path(...).name strips any directory components (so "../../etc/passwd" or an
    absolute path can't escape the upload directory), and the suffix is checked
    against an allowlist so only formats the ingestion layer understands are saved.
    """
    name = Path(filename or "").name or fallback
    if Path(name).suffix.lower() not in _ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            400,
            f"Unsupported file type '{name}'. Allowed: {', '.join(sorted(_ALLOWED_UPLOAD_SUFFIXES))}",
        )
    return name


def _pipeline_worker(job_id: str, upload_dir: Path, rank: str, output_dir: Path, org_id: str | None,
                      format: str = "auto", actor: str | None = None,
                      external_signals: Path | None = None,
                      alert_threshold: float = 0.6, min_prevalence: float = 0.1,
                      min_total_reads: int = 50, use_vqvae: bool = False,
                      rank_bump: bool = False):
    """Run the pipeline in a background thread, logging progress into _JOBS."""
    job = _JOBS[job_id]
    try:
        from ..pipeline.runner import run as _run, PipelineConfig
        job["status"] = "ingesting"
        job["log"] += "Ingesting classifier reports…\n"

        store = _store_for(org_id)
        cfg = PipelineConfig(
            output_dir=str(output_dir),
            alert_threshold=alert_threshold,
            min_prevalence=min_prevalence,
            min_total_reads=min_total_reads,
            use_vqvae=use_vqvae,
            rank_bump=rank_bump,
            db_path=str(_DB_PATH),
        )

        job["status"] = "scoring"
        job["log"] += "Running pipeline (scoring, AMR, lineage, temporal)…\n"

        _run(input_path=upload_dir, config=cfg, rank=rank, format=format,
             run_characterization=False, quiet=True, store=store,
             actor=actor or "dashboard-upload", external_signals=external_signals)

        report_path = output_dir / "report.json"
        if org_id is None:
            # Selfhosted mode: update the global pointer so the dashboard reloads
            # the new result (unchanged behavior).
            global _REPORT_PATH
            _REPORT_PATH = report_path
            os.environ["PATHOGENIQ_REPORT"] = str(_REPORT_PATH)
        # In saas mode, report_path is already the deterministic per-org path
        # (reports/{org_id}/report.json), so no global state needs updating.

        job["status"] = "done"
        job["log"] += f"Done. Report → {report_path}\n"
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
    format: str = Form("auto"),
    external_signals: UploadFile | None = File(None),
    alert_threshold: float = Form(0.6),
    min_prevalence: float = Form(0.1),
    min_total_reads: int = Form(50),
    use_vqvae: bool = Form(False),
    rank_bump: bool = Form(False),
    org_id: str | None = Depends(_org_dep),
    actor: str | None = Depends(_actor_dep),
):
    """
    Accept uploaded classifier report files (Kraken2/Bracken/MetaPhlAn, or a
    single .tsv/.csv count matrix), save them to a temp directory, and run the
    full PathogenIQ pipeline in a background thread. Returns a job_id for
    polling /api/run-status/{job_id}.
    """
    if len(files) > _MAX_UPLOAD_FILES:
        raise HTTPException(400, f"Too many files ({len(files)}); max is {_MAX_UPLOAD_FILES}")

    job_id = str(uuid.uuid4())[:8]
    upload_dir = Path(tempfile.mkdtemp(prefix=f"piq_{job_id}_"))
    output_dir = _report_path_for(org_id).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    _JOBS[job_id] = {"status": "uploading", "log": f"Job {job_id} — saving {len(files)} file(s)…\n"}

    # Save uploaded files (filename sanitized + extension-checked; size-capped to prevent disk abuse)
    total_bytes = 0
    seen_names: set[str] = set()
    for uf in files:
        safe_name = _safe_upload_name(uf.filename, f"upload_{uuid.uuid4().hex[:6]}.report")
        if safe_name in seen_names:
            safe_name = f"{uuid.uuid4().hex[:6]}_{safe_name}"
        seen_names.add(safe_name)

        content = await uf.read()
        total_bytes += len(content)
        if len(content) > _MAX_FILE_BYTES or total_bytes > _MAX_TOTAL_BYTES:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise HTTPException(400, "Upload too large")

        dest = upload_dir / safe_name
        dest.write_bytes(content)
        _JOBS[job_id]["log"] += f"  saved: {dest.name} ({len(content)//1024} KB)\n"

    ext_signals_path = None
    if external_signals is not None:
        safe_ext_name = _safe_upload_name(external_signals.filename, "external_signals.csv")
        ext_signals_path = upload_dir / f"__external__{safe_ext_name}"
        ext_signals_path.write_bytes(await external_signals.read())
        _JOBS[job_id]["log"] += f"  saved external signals: {ext_signals_path.name}\n"

    _JOBS[job_id]["status"] = "ingesting"
    t = threading.Thread(
        target=_pipeline_worker,
        args=(job_id, upload_dir, rank, output_dir, org_id, format, actor, ext_signals_path,
              alert_threshold, min_prevalence, min_total_reads, use_vqvae, rank_bump),
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
    print(f"  Mode:   {_MODE}")
    print(f"  DB:     {db}")
    print(f"  Report: {report}")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="warning")
