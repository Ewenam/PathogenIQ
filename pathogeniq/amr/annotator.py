"""
amr/annotator.py
Annotates detected pathogens with AMR resistance profiles from the
curated database. Works at genus level — no raw reads required.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .database import AMR_DB, CRITICAL, HIGH


@dataclass
class AMRAnnotation:
    genus: str
    who_priority: str          # critical / high / medium
    cdc_threat: str
    eskape: bool
    resistances: list[dict]    # [{drug_class, mechanism, prevalence}]
    last_resort_drugs: list[str]
    wbe_note: str
    abundance: float           # relative abundance in this sample
    amr_score: float           # 0–1 urgency score for risk weighting


def _amr_score(entry: dict, abundance: float) -> float:
    """Urgency score: combines WHO priority tier × pathogen abundance."""
    tier = {CRITICAL: 1.0, HIGH: 0.7, "medium": 0.4}.get(entry["who_priority"], 0.3)
    n_resistances = len(entry.get("resistances", []))
    breadth = min(n_resistances / 5, 1.0)  # normalised by 5 drug classes = max breadth
    return round(tier * 0.6 + breadth * 0.2 + float(abundance) * 0.2, 4)


def annotate_amr(detected_pathogens: list[dict]) -> list[AMRAnnotation]:
    """
    Given the detected_pathogens list from a RiskScore (each entry has
    'genus', 'abundance', etc.), return AMR annotations for any genus
    that appears in the AMR database.
    """
    results: list[AMRAnnotation] = []
    seen: set[str] = set()

    for p in detected_pathogens:
        genus = p.get("genus", "").split()[0]
        if not genus or genus in seen:
            continue
        seen.add(genus)

        entry = AMR_DB.get(genus)
        if not entry:
            continue

        abundance = float(p.get("abundance", 0.0))
        results.append(AMRAnnotation(
            genus=genus,
            who_priority=entry["who_priority"],
            cdc_threat=entry.get("cdc_threat", "—"),
            eskape=entry.get("eskape", False),
            resistances=entry.get("resistances", []),
            last_resort_drugs=entry.get("last_resort_drugs", []),
            wbe_note=entry.get("wbe_note", ""),
            abundance=abundance,
            amr_score=_amr_score(entry, abundance),
        ))

    results.sort(key=lambda a: a.amr_score, reverse=True)
    return results


def amr_annotations_to_dict(annotations: list[AMRAnnotation]) -> list[dict]:
    return [
        {
            "genus": a.genus,
            "who_priority": a.who_priority,
            "cdc_threat": a.cdc_threat,
            "eskape": a.eskape,
            "resistances": a.resistances,
            "last_resort_drugs": a.last_resort_drugs,
            "wbe_note": a.wbe_note,
            "abundance": a.abundance,
            "amr_score": a.amr_score,
        }
        for a in annotations
    ]
