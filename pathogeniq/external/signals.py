"""
external/signals.py
Cross-validation of internal risk-score temporal trends against an
external signal feed (qPCR, ddPCR, case counts, etc.).

Loads a site/date/value CSV or TSV, joins it against each site's internal
risk-score history on exact normalized (ISO) date, and computes a Pearson
correlation as a concordance check.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

_REQUIRED_COLUMNS = {"site", "date", "value"}


@dataclass
class ExternalValidationResult:
    site: str
    signal_type: str
    n_matched: int
    pearson_r: float | None
    pearson_p: float | None
    concordant: bool | None
    summary: str


def load_external_signals(path: str | Path) -> pd.DataFrame:
    """
    Load a site/date/value (+ optional signal_type) CSV or TSV.

    Raises ValueError if required columns are missing.
    """
    path = Path(path)
    sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    df = pd.read_csv(path, sep=sep)

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"External signals file {path} is missing required column(s): "
            f"{sorted(missing)}. Expected columns: site, date, value "
            f"(optional: signal_type)."
        )

    if "signal_type" not in df.columns:
        df["signal_type"] = "external"

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def validate_site(
    site: str,
    external_df: pd.DataFrame,
    history: list[dict],
    current_score: float,
    current_date: str,
    signal_type: str = "external",
) -> ExternalValidationResult:
    """
    Correlate this site's internal risk-score series (past history + the
    current run, in that order) against its external signal series, joined
    on exact normalized date string.
    """
    site_external = external_df[external_df["site"] == site]
    if not site_external.empty and "signal_type" in site_external:
        observed_type = site_external["signal_type"].iloc[0]
        signal_type = observed_type

    internal_by_date = {h["run_date"]: h["risk_score"] for h in history}
    internal_by_date[current_date] = current_score

    matched_internal = []
    matched_external = []
    for _, row in site_external.iterrows():
        d = row["date"]
        if d in internal_by_date:
            matched_internal.append(internal_by_date[d])
            matched_external.append(row["value"])

    n_matched = len(matched_internal)
    if n_matched < 3:
        return ExternalValidationResult(
            site=site,
            signal_type=signal_type,
            n_matched=n_matched,
            pearson_r=None,
            pearson_p=None,
            concordant=None,
            summary=f"Insufficient overlapping dates ({n_matched} < 3) for correlation.",
        )

    r, p = pearsonr(matched_internal, matched_external)
    r, p = float(r), float(p)
    concordant = r > 0.5 and p < 0.05
    summary = (
        f"r={r:.2f}, p={p:.3f} over {n_matched} matched dates — "
        f"{'concordant' if concordant else 'not concordant'} with internal signal."
    )

    return ExternalValidationResult(
        site=site,
        signal_type=signal_type,
        n_matched=n_matched,
        pearson_r=round(r, 4),
        pearson_p=round(p, 4),
        concordant=concordant,
        summary=summary,
    )


def validate_all_sites(
    external_df: pd.DataFrame,
    store,
    risk_scores: list,
    run_date: str,
    window: int = 16,
) -> dict[str, ExternalValidationResult]:
    """
    Validate every site present in both the external feed and the current
    run's risk scores. History is pulled before the current run is
    recorded (past-only), matching the temporal module's convention.
    """
    external_sites = set(external_df["site"].unique())
    results = {}
    for rs in risk_scores:
        if rs.sample_name not in external_sites:
            continue
        history = store.get_site_history(rs.sample_name, last_n=window)
        results[rs.sample_name] = validate_site(
            site=rs.sample_name,
            external_df=external_df,
            history=history,
            current_score=rs.score,
            current_date=run_date,
        )
    return results
