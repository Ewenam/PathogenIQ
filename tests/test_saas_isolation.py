"""
Proves org-scoped data isolation for PATHOGENIQ_MODE=saas: two organizations
running the pipeline against different sample sets must never see each
other's sites, scores, or history — through both the store layer directly
and the dashboard's HTTP API.

Requires a reachable Postgres (DATABASE_URL). Skipped if unavailable, since
selfhosted/SQLite installs never need this.
"""
import os
import uuid

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("psycopg")
pytest.importorskip("jwt")

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://pathogeniq:pathogeniq_dev@localhost:5432/pathogeniq_saas",
)
JWT_SECRET = "test-secret-for-saas-isolation-tests"


@pytest.fixture(scope="module")
def engine():
    from pathogeniq.saas.db import get_engine, init_schema

    try:
        eng = get_engine(DATABASE_URL)
        init_schema(eng)
    except Exception as exc:
        pytest.skip(f"Postgres not reachable at {DATABASE_URL}: {exc}")
    return eng


def _write_synthetic_reports(tmp_path, names: list[str], taxa_reads: dict[str, list[int]]):
    """
    Write minimal Kraken2-format .report files (one per name in `names`) where
    taxa_reads maps taxon name -> per-sample read counts, aligned with `names`.
    Names deliberately avoid the "<base>_1"/"<base>_2" pattern so the reader's
    paired-end merge logic doesn't collapse them into one combined sample.
    """
    for i, sample_name in enumerate(names):
        lines = []
        for taxon, reads in taxa_reads.items():
            r = reads[i]
            lines.append(f"10.00\t{r}\t{r}\tG\t1\t  {taxon}")
        (tmp_path / f"{sample_name}.report").write_text("\n".join(lines) + "\n")


def test_two_orgs_never_see_each_others_sites(tmp_path, engine):
    from pathogeniq.pipeline.runner import run, PipelineConfig
    from pathogeniq.saas.db import PostgresOrgStore
    from pathogeniq.saas.orgs import bootstrap_org

    user_a, user_b = f"user-{uuid.uuid4().hex[:8]}", f"user-{uuid.uuid4().hex[:8]}"
    org_a = bootstrap_org(engine, user_a, "Org A")
    org_b = bootstrap_org(engine, user_b, "Org B")
    assert org_a != org_b

    dir_a, dir_b = tmp_path / "org_a_input", tmp_path / "org_b_input"
    dir_a.mkdir()
    dir_b.mkdir()
    _write_synthetic_reports(dir_a, ["siteA_north", "siteA_south"], {
        "Escherichia": [4000, 3500], "Bacteroides": [2000, 2200],
        "Salmonella": [800, 100], "Vibrio": [50, 0], "Klebsiella": [300, 250],
    })
    _write_synthetic_reports(dir_b, ["siteB_east", "siteB_west"], {
        "Pseudomonas": [3000, 2800], "Faecalibacterium": [2500, 2600],
        "Yersinia": [60, 90], "Clostridium": [400, 380], "Bacteroides": [200, 220],
    })

    store_a = PostgresOrgStore(org_a, engine=engine)
    store_b = PostgresOrgStore(org_b, engine=engine)

    run(input_path=dir_a, config=PipelineConfig(output_dir=str(tmp_path / "out_a")),
        rank="G", quiet=True, store=store_a)
    run(input_path=dir_b, config=PipelineConfig(output_dir=str(tmp_path / "out_b")),
        rank="G", quiet=True, store=store_b)

    sites_a, sites_b = set(store_a.all_sites()), set(store_b.all_sites())
    assert sites_a == {"siteA_north", "siteA_south"}
    assert sites_b == {"siteB_east", "siteB_west"}
    assert sites_a.isdisjoint(sites_b)

    # The real guarantee: org A's history queries never return org B's runs.
    for site in sites_b:
        assert store_a.get_site_history(site) == [], (
            f"org A's store leaked history for org B's site '{site}'"
        )
    for site in sites_a:
        assert store_b.get_site_history(site) == [], (
            f"org B's store leaked history for org A's site '{site}'"
        )

    assert store_a.run_count() >= 1
    assert store_b.run_count() >= 1


def test_isolation_through_dashboard_api(tmp_path, engine, monkeypatch):
    """Same isolation guarantee, verified through the actual FastAPI routes."""
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("PATHOGENIQ_JWT_SECRET", JWT_SECRET)
    monkeypatch.setenv("PATHOGENIQ_MODE", "saas")
    monkeypatch.setenv("PATHOGENIQ_SAAS_REPORTS_DIR", str(tmp_path / "reports"))

    import importlib
    from pathogeniq.dashboard import app as app_module
    importlib.reload(app_module)

    from fastapi.testclient import TestClient
    import jwt as pyjwt

    client = TestClient(app_module.app)

    user_a, user_b = f"user-{uuid.uuid4().hex[:8]}", f"user-{uuid.uuid4().hex[:8]}"
    token_a = pyjwt.encode({"sub": user_a}, JWT_SECRET, algorithm="HS256")
    token_b = pyjwt.encode({"sub": user_b}, JWT_SECRET, algorithm="HS256")

    r = client.get("/api/sites")
    assert r.status_code == 401

    r = client.post("/api/saas/bootstrap-org",
                     headers={"Authorization": f"Bearer {token_a}"},
                     data={"org_name": "API Org A"})
    assert r.status_code == 200
    org_a = r.json()["org_id"]

    r = client.post("/api/saas/bootstrap-org",
                     headers={"Authorization": f"Bearer {token_b}"},
                     data={"org_name": "API Org B"})
    assert r.status_code == 200
    org_b = r.json()["org_id"]
    assert org_a != org_b

    # Write org A's report directly into its scoped path, then confirm org B
    # can never read it back through the API.
    report_dir_a = tmp_path / "reports" / org_a
    report_dir_a.mkdir(parents=True)
    (report_dir_a / "report.json").write_text(
        '{"generated_at": "now", "summary": {}, "samples": '
        '[{"name": "secret_site_a", "score": 0.9, "level": "CRITICAL"}]}'
    )

    r = client.get("/api/sites", headers={"Authorization": f"Bearer {token_a}"})
    assert r.status_code == 200
    assert r.json()["sites"][0]["name"] == "secret_site_a"

    r = client.get("/api/sites", headers={"Authorization": f"Bearer {token_b}"})
    assert r.status_code == 200
    assert r.json()["sites"] == [], "org B must not see org A's report data"
