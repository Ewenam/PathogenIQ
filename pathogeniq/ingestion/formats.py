"""
ingestion/formats.py
Classifier-agnostic report parsing. Each loader returns a pd.Series of read
counts (or count-like abundance values) indexed by taxon name, named after
the source file's stem — the same contract as reader.load_kraken_report, so
callers don't need to know which classifier produced a given file.

Supported formats:
  kraken2   — headerless 6-column report (reader.load_kraken_report)
  bracken   — TSV with a `name/taxonomy_id/taxonomy_lvl/.../new_est_reads` header
  metaphlan — TSV with `#`-prefixed comment lines and pipe-delimited clade_name
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

FormatName = Literal["kraken2", "bracken", "metaphlan"]

_BRACKEN_HEADER_COLS = {
    "name", "taxonomy_id", "taxonomy_lvl", "kraken_assigned_reads",
    "added_reads", "new_est_reads", "fraction_total_reads",
}

# Kraken2/Bracken rank letter -> MetaPhlAn clade prefix
_METAPHLAN_RANK_PREFIX = {"F": "f__", "G": "g__", "S": "s__"}


def sniff_format(path: Path) -> FormatName:
    """
    Classify a report file by content (not extension, since .tsv is used by
    Bracken, MetaPhlAn, and plain count matrices alike).
    """
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                return "metaphlan"
            first_field = stripped.split("\t")[0].strip().lower()
            if first_field == "clade_name":
                return "metaphlan"
            if first_field == "name":
                cols = {c.strip() for c in stripped.split("\t")}
                if _BRACKEN_HEADER_COLS.issubset(cols):
                    return "bracken"
            # First substantive line isn't a recognized header — assume
            # Kraken2's headerless positional format.
            return "kraken2"
    raise ValueError(f"Could not sniff format: '{path}' appears empty")


def load_bracken_report(path: Path, rank: str = "G") -> pd.Series:
    """Parse a Bracken TSV, using new_est_reads at the requested rank."""
    df = pd.read_csv(path, sep="\t", dtype={"taxonomy_lvl": str})
    missing = _BRACKEN_HEADER_COLS - set(df.columns)
    if missing:
        raise ValueError(f"'{path}' is missing expected Bracken columns: {missing}")
    sub = df[df["taxonomy_lvl"].str.strip() == rank]
    counts = sub.groupby(sub["name"].str.strip())["new_est_reads"].sum()
    return counts.astype(float).rename(path.stem)


def load_metaphlan_report(path: Path, rank: str = "G") -> pd.Series:
    """
    Parse a MetaPhlAn profile TSV. Keeps only rows whose deepest clade segment
    matches the requested rank (so a genus-level row isn't double-counted
    against its child species rows), using relative_abundance as the value.
    """
    prefix = _METAPHLAN_RANK_PREFIX.get(rank)
    if prefix is None:
        raise ValueError(f"MetaPhlAn rank mapping not defined for rank={rank!r}")

    counts: dict[str, float] = {}
    saw_data_row = False
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            clade, _taxid, rel_abund = parts[0], parts[1], parts[2]
            saw_data_row = True
            segments = clade.strip().split("|")
            last = segments[-1]
            if "__" not in last:
                continue
            last_prefix = last.split("__", 1)[0] + "__"
            if last_prefix != prefix:
                continue
            name = last.split("__", 1)[1]
            try:
                counts[name] = counts.get(name, 0.0) + float(rel_abund)
            except ValueError:
                continue

    if not saw_data_row:
        raise ValueError(f"'{path}' does not look like a MetaPhlAn profile (no data rows found)")

    return pd.Series(counts, dtype=float, name=path.stem)


def load_report(path: Path, rank: str = "G", fmt: str = "auto") -> tuple[pd.Series, FormatName]:
    """Dispatch to the correct parser. Returns (series, resolved_format)."""
    resolved: FormatName = sniff_format(path) if fmt == "auto" else fmt  # type: ignore[assignment]

    if resolved == "kraken2":
        from .reader import load_kraken_report
        series = load_kraken_report(path, rank=rank)
    elif resolved == "bracken":
        series = load_bracken_report(path, rank=rank)
    elif resolved == "metaphlan":
        series = load_metaphlan_report(path, rank=rank)
    else:
        raise ValueError(f"Unknown format '{resolved}' for '{path}'")

    return series, resolved
