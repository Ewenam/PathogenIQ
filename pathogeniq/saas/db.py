"""
saas/db.py
Org-scoped Postgres data layer for PATHOGENIQ_MODE=saas.

Mirrors the public interface of temporal.store.TimeSeriesStore (record_run,
get_site_history, all_sites, run_count) so pipeline.runner.run() can use
either store interchangeably. The selfhosted/SQLite path in temporal/store.py
is untouched by this module.

Schema:
  organizations — one row per tenant
  memberships   — which users belong to which org (role: currently just 'owner')
  runs          — one row per pipeline run, scoped to an org
  scores        — risk score per (run, site), scoped to an org
  abundances    — per-taxon relative abundance per (run, site), scoped to an org
"""
from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS organizations (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memberships (
    user_id     TEXT NOT NULL,
    org_id      TEXT NOT NULL REFERENCES organizations(id),
    role        TEXT NOT NULL DEFAULT 'owner',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, org_id)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id      SERIAL PRIMARY KEY,
    org_id      TEXT NOT NULL REFERENCES organizations(id),
    run_date    TEXT NOT NULL,
    run_ts      TIMESTAMPTZ NOT NULL,
    input_path  TEXT,
    rank        TEXT DEFAULT 'G',
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS scores (
    id                  SERIAL PRIMARY KEY,
    run_id              INTEGER NOT NULL REFERENCES runs(run_id),
    org_id              TEXT NOT NULL REFERENCES organizations(id),
    site                TEXT NOT NULL,
    risk_score          REAL NOT NULL,
    risk_level          TEXT NOT NULL,
    abundance_score     REAL,
    community_signal    REAL,
    novelty_signal      REAL,
    top_pathogen        TEXT,
    top_pathogen_abund  REAL
);

CREATE TABLE IF NOT EXISTS abundances (
    id          SERIAL PRIMARY KEY,
    run_id      INTEGER NOT NULL REFERENCES runs(run_id),
    org_id      TEXT NOT NULL REFERENCES organizations(id),
    site        TEXT NOT NULL,
    taxon       TEXT NOT NULL,
    rel_abund   REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_org           ON runs(org_id);
CREATE INDEX IF NOT EXISTS idx_scores_org_site    ON scores(org_id, site);
CREATE INDEX IF NOT EXISTS idx_scores_run         ON scores(run_id);
CREATE INDEX IF NOT EXISTS idx_abund_org_site_tax ON abundances(org_id, site, taxon);

-- Audit/provenance columns added to `runs` after the original schema shipped.
-- ADD COLUMN IF NOT EXISTS is natively idempotent in Postgres, so this is
-- safe to re-run against an already-migrated database.
ALTER TABLE runs ADD COLUMN IF NOT EXISTS pathogeniq_version TEXT;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS classifier_format TEXT;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS input_hash TEXT;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS config_hash TEXT;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS actor TEXT;
"""


def get_engine(db_url: str | None = None) -> Engine:
    url = db_url or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. PATHOGENIQ_MODE=saas requires a Postgres "
            "connection string (see .env.example)."
        )
    return create_engine(url, pool_pre_ping=True, future=True)


def init_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(_SCHEMA_SQL))


@contextmanager
def _tx(engine: Engine) -> Iterator:
    with engine.begin() as conn:
        yield conn


class PostgresOrgStore:
    """
    Org-scoped equivalent of temporal.store.TimeSeriesStore. Every method is
    implicitly filtered to self.org_id so a request authenticated for one
    organization can never read or write another organization's rows.
    """

    def __init__(self, org_id: str, db_url: str | None = None, engine: Engine | None = None):
        self.org_id = org_id
        self.engine = engine or get_engine(db_url)
        init_schema(self.engine)

    def record_run(
        self,
        risk_scores: list,
        sampleset,
        run_date: str | None = None,
        input_path: str = "",
        rank: str = "G",
        notes: str = "",
        manifest: dict | None = None,
    ) -> int:
        now = datetime.now(tz=timezone.utc)
        d = run_date or now.date().isoformat()
        manifest = manifest or {}

        with _tx(self.engine) as conn:
            run_id = conn.execute(
                text(
                    """INSERT INTO runs (org_id, run_date, run_ts, input_path, rank, notes,
                                          pathogeniq_version, classifier_format, input_hash, config_hash, actor)
                       VALUES (:org_id, :run_date, :run_ts, :input_path, :rank, :notes,
                               :pathogeniq_version, :classifier_format, :input_hash, :config_hash, :actor)
                       RETURNING run_id"""
                ),
                {
                    "org_id": self.org_id,
                    "run_date": d,
                    "run_ts": now,
                    "input_path": input_path,
                    "rank": rank,
                    "notes": notes,
                    "pathogeniq_version": manifest.get("pathogeniq_version"),
                    "classifier_format": manifest.get("classifier_format"),
                    "input_hash": manifest.get("input_hash"),
                    "config_hash": manifest.get("config_hash"),
                    "actor": manifest.get("actor"),
                },
            ).scalar_one()

            for rs in risk_scores:
                top = rs.detected_pathogens[0] if rs.detected_pathogens else {}
                conn.execute(
                    text(
                        """INSERT INTO scores
                           (run_id, org_id, site, risk_score, risk_level,
                            abundance_score, community_signal, novelty_signal,
                            top_pathogen, top_pathogen_abund)
                           VALUES (:run_id, :org_id, :site, :risk_score, :risk_level,
                                   :abundance_score, :community_signal, :novelty_signal,
                                   :top_pathogen, :top_pathogen_abund)"""
                    ),
                    {
                        "run_id": run_id,
                        "org_id": self.org_id,
                        "site": rs.sample_name,
                        "risk_score": rs.score,
                        "risk_level": rs.level,
                        "abundance_score": rs.breakdown.get("abundance_score", 0),
                        "community_signal": rs.community_signal,
                        "novelty_signal": rs.novelty_signal,
                        "top_pathogen": top.get("taxon", ""),
                        "top_pathogen_abund": top.get("abundance", 0.0),
                    },
                )

            if sampleset and sampleset.relative_abundance is not None:
                rel = sampleset.relative_abundance
                rows = []
                for taxon in rel.index:
                    for site in rel.columns:
                        val = float(rel.loc[taxon, site])
                        if val > 0:
                            rows.append({
                                "run_id": run_id, "org_id": self.org_id,
                                "site": site, "taxon": taxon, "rel_abund": val,
                            })
                if rows:
                    conn.execute(
                        text(
                            """INSERT INTO abundances (run_id, org_id, site, taxon, rel_abund)
                               VALUES (:run_id, :org_id, :site, :taxon, :rel_abund)"""
                        ),
                        rows,
                    )

        return run_id

    def get_site_history(self, site: str, last_n: int = 52) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """SELECT r.run_date, r.run_ts, s.risk_score, s.risk_level,
                              s.abundance_score, s.community_signal, s.novelty_signal,
                              s.top_pathogen, s.top_pathogen_abund
                       FROM scores s
                       JOIN runs r ON r.run_id = s.run_id
                       WHERE s.org_id = :org_id AND s.site = :site
                       ORDER BY r.run_ts ASC
                       LIMIT :last_n"""
                ),
                {"org_id": self.org_id, "site": site, "last_n": last_n},
            ).mappings().all()
        return [dict(r) for r in rows]

    def all_sites(self) -> list[str]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("SELECT DISTINCT site FROM scores WHERE org_id = :org_id ORDER BY site"),
                {"org_id": self.org_id},
            ).all()
        return [r[0] for r in rows]

    def run_count(self) -> int:
        with self.engine.connect() as conn:
            return conn.execute(
                text("SELECT COUNT(*) FROM runs WHERE org_id = :org_id"),
                {"org_id": self.org_id},
            ).scalar_one()


def create_organization(engine: Engine, name: str, org_id: str | None = None) -> str:
    org_id = org_id or uuid.uuid4().hex
    with _tx(engine) as conn:
        conn.execute(
            text("INSERT INTO organizations (id, name) VALUES (:id, :name)"),
            {"id": org_id, "name": name},
        )
    return org_id


def add_membership(engine: Engine, user_id: str, org_id: str, role: str = "owner") -> None:
    with _tx(engine) as conn:
        conn.execute(
            text(
                """INSERT INTO memberships (user_id, org_id, role)
                   VALUES (:user_id, :org_id, :role)
                   ON CONFLICT (user_id, org_id) DO NOTHING"""
            ),
            {"user_id": user_id, "org_id": org_id, "role": role},
        )


def get_org_for_user(engine: Engine, user_id: str) -> str | None:
    """Return the first org_id this user belongs to, or None."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT org_id FROM memberships WHERE user_id = :user_id ORDER BY created_at LIMIT 1"),
            {"user_id": user_id},
        ).first()
    return row[0] if row else None
