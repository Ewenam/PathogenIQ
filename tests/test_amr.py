"""Tests for pathogeniq/amr/ — AMR annotation module."""
import pytest
from pathogeniq.amr.annotator import annotate_amr, amr_annotations_to_dict
from pathogeniq.amr.database import AMR_DB, CRITICAL, HIGH


def test_database_integrity():
    """Every entry in AMR_DB must have required keys."""
    required = {"who_priority", "resistances", "last_resort_drugs", "wbe_note"}
    for genus, entry in AMR_DB.items():
        missing = required - entry.keys()
        assert not missing, f"{genus} missing keys: {missing}"
        assert entry["who_priority"] in (CRITICAL, HIGH, "medium"), genus
        assert isinstance(entry["resistances"], list), genus
        assert len(entry["resistances"]) >= 1, genus


def test_eskape_all_present():
    """All six ESKAPE pathogens must be in the database."""
    eskape = {"Enterococcus", "Staphylococcus", "Klebsiella",
              "Acinetobacter", "Pseudomonas", "Enterobacter"}
    for genus in eskape:
        assert genus in AMR_DB, f"ESKAPE genus {genus} missing from AMR_DB"
        assert AMR_DB[genus]["eskape"] is True


def test_annotate_known_genus():
    pathogens = [{"genus": "Klebsiella", "abundance": 0.25, "taxon": "Klebsiella pneumoniae"}]
    result = annotate_amr(pathogens)
    assert len(result) == 1
    ann = result[0]
    assert ann.genus == "Klebsiella"
    assert ann.who_priority == CRITICAL
    assert ann.eskape is True
    assert ann.amr_score > 0.7   # critical + high abundance → high urgency score
    assert len(ann.resistances) >= 2


def test_annotate_unknown_genus():
    pathogens = [{"genus": "Caulobacter", "abundance": 0.5, "taxon": "Caulobacter crescentus"}]
    result = annotate_amr(pathogens)
    assert result == []


def test_annotate_deduplicates():
    """Same genus listed twice should only produce one annotation."""
    pathogens = [
        {"genus": "Pseudomonas", "abundance": 0.10, "taxon": "Pseudomonas aeruginosa"},
        {"genus": "Pseudomonas", "abundance": 0.05, "taxon": "Pseudomonas fluorescens"},
    ]
    result = annotate_amr(pathogens)
    assert len(result) == 1


def test_sorted_by_score():
    pathogens = [
        {"genus": "Salmonella",  "abundance": 0.05},
        {"genus": "Yersinia",    "abundance": 0.20},
        {"genus": "Haemophilus", "abundance": 0.03},
    ]
    result = annotate_amr(pathogens)
    scores = [a.amr_score for a in result]
    assert scores == sorted(scores, reverse=True)


def test_to_dict_serializable():
    import json
    pathogens = [{"genus": "Vibrio", "abundance": 0.15}]
    dicts = amr_annotations_to_dict(annotate_amr(pathogens))
    # Must be JSON-serializable (no dataclasses, sets, etc.)
    json.dumps(dicts)
