"""
temporal/store.py
SQLite-backed time series store for PathogenIQ.

Persists risk scores, pathogen abundances, and novelty scores across runs
so temporal models can compare new samples against historical baselines.

Schema:
  runs        — one row per pipeline run (timestamp, input path, config)
  scores      — risk score per (run, site/sample)
  abundances  — per-taxon relative abundance per (run, site)
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date
from pathlib import Path
from typing import Iterator


DEFAULT_DB = Path.home() / ".pathogeniq" / "history.db"


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def _tx(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    conn = _connect(db_path)
    with _tx(conn):
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date    TEXT NOT NULL,          -- ISO date YYYY-MM-DD
                run_ts      TEXT NOT NULL,          -- full ISO timestamp
                input_path  TEXT,
                rank        TEXT DEFAULT 'G',
                notes       TEXT
            );

            CREATE TABLE IF NOT EXISTS scores (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      INTEGER NOT NULL REFERENCES runs(run_id),
                site        TEXT NOT NULL,          -- sample/site name
                risk_score  REAL NOT NULL,
                risk_level  TEXT NOT NULL,
                abundance_score   REAL,
                community_signal  REAL,
                novelty_signal    REAL,
                top_pathogen      TEXT,
                top_pathogen_abund REAL
            );

            CREATE TABLE IF NOT EXISTS abundances (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      INTEGER NOT NULL REFERENCES runs(run_id),
                site        TEXT NOT NULL,
                taxon       TEXT NOT NULL,
                rel_abund   REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_scores_site    ON scores(site);
            CREATE INDEX IF NOT EXISTS idx_scores_run     ON scores(run_id);
            CREATE INDEX IF NOT EXISTS idx_abund_site_tax ON abundances(site, taxon);
        """)
    return conn


class TimeSeriesStore:
    def __init__(self, db_path: Path | str = DEFAULT_DB):
        self.db_path = Path(db_path)
        self.conn = init_db(self.db_path)

    def record_run(
        self,
        risk_scores: list,
        sampleset,
        run_date: str | None = None,
        input_path: str = "",
        rank: str = "G",
        notes: str = "",
    ) -> int:
        """
        Persist a completed pipeline run. Returns the run_id.
        """
        from datetime import timezone
        now = datetime.now(tz=timezone.utc)
        d = run_date or now.date().isoformat()

        with _tx(self.conn):
            cur = self.conn.execute(
                "INSERT INTO runs (run_date, run_ts, input_path, rank, notes) VALUES (?,?,?,?,?)",
                (d, now.isoformat(), input_path, rank, notes),
            )
            run_id = cur.lastrowid

            for rs in risk_scores:
                top = rs.detected_pathogens[0] if rs.detected_pathogens else {}
                self.conn.execute(
                    """INSERT INTO scores
                       (run_id, site, risk_score, risk_level,
                        abundance_score, community_signal, novelty_signal,
                        top_pathogen, top_pathogen_abund)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        run_id,
                        rs.sample_name,
                        rs.score,
                        rs.level,
                        rs.breakdown.get("abundance_score", 0),
                        rs.community_signal,
                        rs.novelty_signal,
                        top.get("taxon", ""),
                        top.get("abundance", 0.0),
                    ),
                )

            if sampleset and sampleset.relative_abundance is not None:
                rel = sampleset.relative_abundance
                rows = []
                for taxon in rel.index:
                    for site in rel.columns:
                        val = float(rel.loc[taxon, site])
                        if val > 0:
                            rows.append((run_id, site, taxon, val))
                self.conn.executemany(
                    "INSERT INTO abundances (run_id, site, taxon, rel_abund) VALUES (?,?,?,?)",
                    rows,
                )

        return run_id

    def get_site_history(self, site: str, last_n: int = 52) -> list[dict]:
        """
        Return the last N run records for a specific site, ordered oldest→newest.
        """
        rows = self.conn.execute(
            """SELECT r.run_date, r.run_ts, s.risk_score, s.risk_level,
                      s.abundance_score, s.community_signal, s.novelty_signal,
                      s.top_pathogen, s.top_pathogen_abund
               FROM scores s
               JOIN runs r ON r.run_id = s.run_id
               WHERE s.site = ?
               ORDER BY r.run_ts ASC
               LIMIT ?""",
            (site, last_n),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_taxon_history(self, site: str, taxon: str, last_n: int = 52) -> list[dict]:
        """Return abundance history for a specific taxon at a site."""
        rows = self.conn.execute(
            """SELECT r.run_date, a.rel_abund
               FROM abundances a
               JOIN runs r ON r.run_id = a.run_id
               WHERE a.site = ? AND a.taxon = ?
               ORDER BY r.run_ts ASC
               LIMIT ?""",
            (site, taxon, last_n),
        ).fetchall()
        return [dict(r) for r in rows]

    def all_sites(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT site FROM scores ORDER BY site"
        ).fetchall()
        return [r["site"] for r in rows]

    def run_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
