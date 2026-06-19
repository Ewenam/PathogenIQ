"""
lineage/detector.py
Species-level context annotation for PathogenIQ.

Parses Kraken2 reports at species rank (rank='S') and annotates detected
species with known lineage/variant context from the curated database.

Usage in pipeline:
    from pathogeniq.lineage.detector import annotate_lineage
    annotations = annotate_lineage(species_series, threshold=0.001)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd

from .database import SPECIES_CONTEXT


@dataclass
class LineageAnnotation:
    species: str
    genus: str
    common_name: str
    who_label: str
    abundance: float
    known_variants: list[dict] = field(default_factory=list)
    wbe_utility: str = ""
    risk_elevation: float = 0.0   # additive to sample risk score


def annotate_lineage(
    species_series: pd.Series,
    threshold: float = 0.001,
) -> list[LineageAnnotation]:
    """
    Given a species-level relative abundance Series (from load_kraken_report
    with rank='S'), return LineageAnnotation for any species in the database
    above the abundance threshold.

    Args:
        species_series: relative abundance Series indexed by species name
        threshold: minimum relative abundance to report (default 0.1%)

    Returns:
        List of LineageAnnotation sorted by risk_elevation × abundance.
    """
    total = species_series.sum()
    if total == 0:
        return []

    rel = species_series / total
    results: list[LineageAnnotation] = []

    for species, abund in rel.items():
        if abund < threshold:
            continue
        ctx = SPECIES_CONTEXT.get(str(species))
        if not ctx:
            continue
        results.append(LineageAnnotation(
            species=str(species),
            genus=ctx.get("genus", ""),
            common_name=ctx.get("common_name", ""),
            who_label=ctx.get("who_label", ""),
            abundance=round(float(abund), 5),
            known_variants=ctx.get("known_variants", []),
            wbe_utility=ctx.get("wbe_utility", ""),
            risk_elevation=ctx.get("risk_elevation", 0.0),
        ))

    results.sort(key=lambda a: a.risk_elevation * a.abundance, reverse=True)
    return results


def lineage_annotations_to_dict(annotations: list[LineageAnnotation]) -> list[dict]:
    return [
        {
            "species": a.species,
            "genus": a.genus,
            "common_name": a.common_name,
            "who_label": a.who_label,
            "abundance": a.abundance,
            "known_variants": a.known_variants,
            "wbe_utility": a.wbe_utility,
            "risk_elevation": a.risk_elevation,
        }
        for a in annotations
    ]


def load_species_from_report(report_path, threshold: float = 0.001, fmt: str = "auto") -> list[LineageAnnotation]:
    """
    Convenience: load a single report file at species rank and annotate.
    Useful for standalone lineage analysis without running the full pipeline.
    `fmt` accepts "auto" (content-sniffed), "kraken2", "bracken", or "metaphlan".
    """
    from pathogeniq.ingestion.formats import load_report
    from pathlib import Path
    series, _resolved = load_report(Path(report_path), rank="S", fmt=fmt)
    total = series.sum()
    if total == 0:
        return []
    rel = series / total
    return annotate_lineage(rel, threshold=threshold)
