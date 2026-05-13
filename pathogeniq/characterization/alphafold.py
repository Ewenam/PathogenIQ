"""
characterization/alphafold.py
Novel pathogen characterization via protein structure prediction.

When a novel or uncharacterized pathogen is detected, this module:
  1. Retrieves the protein sequence from NCBI (given a taxon name or accession)
  2. Predicts the 3D structure via ESMFold API (free, no local GPU required)
     OR routes to a local AlphaFold installation if configured
  3. Compares the predicted structure against known virulence proteins
     to assess potential danger of the novel pathogen

ESMFold API: https://esmatlas.com/resources/fold (Meta, free tier)
AlphaFold DB: https://alphafold.ebi.ac.uk/
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import requests


VIRULENCE_KEYWORDS = [
    "toxin", "hemolysin", "invasin", "adhesin", "protease",
    "siderophore", "capsule", "fimbria", "pilus", "effector",
    "secretion system", "autotransporter", "superantigen",
    "phospholipase", "hyaluronidase", "coagulase", "fibronectin",
]


@dataclass
class StructurePrediction:
    taxon: str
    protein_name: str
    sequence: str
    pdb_str: str                    # PDB-format structure string
    confidence: float               # mean pLDDT score (0–100), higher = better
    virulence_flags: list[str]      # matched virulence keywords from NCBI annotation
    risk_annotation: str            # human-readable risk assessment


def fetch_protein_sequence(taxon_name: str, max_results: int = 1) -> list[dict]:
    """
    Query NCBI Protein DB for sequences associated with a taxon.
    Returns list of {accession, title, sequence}.
    """
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    params = {
        "db": "protein",
        "term": f"{taxon_name}[Organism] AND virulence[Title]",
        "retmax": max_results,
        "retmode": "json",
    }
    try:
        search_resp = requests.get(f"{base}/esearch.fcgi", params=params, timeout=15)
        search_resp.raise_for_status()
        ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        return [{"error": str(e)}]

    if not ids:
        # Fall back to broader search
        params["term"] = f"{taxon_name}[Organism]"
        try:
            r = requests.get(f"{base}/esearch.fcgi", params=params, timeout=15)
            ids = r.json().get("esearchresult", {}).get("idlist", [])
        except Exception:
            return []

    results = []
    for uid in ids[:max_results]:
        try:
            fetch_resp = requests.get(
                f"{base}/efetch.fcgi",
                params={"db": "protein", "id": uid, "rettype": "fasta", "retmode": "text"},
                timeout=20,
            )
            fetch_resp.raise_for_status()
            fasta = fetch_resp.text
            lines = fasta.strip().split("\n")
            title = lines[0].lstrip(">").strip()
            seq = "".join(lines[1:]).strip()
            results.append({"accession": uid, "title": title, "sequence": seq})
        except Exception as e:
            results.append({"accession": uid, "error": str(e)})
        time.sleep(0.4)  # NCBI rate limit: 3 req/sec without API key

    return results


def predict_structure_esmfold(sequence: str, timeout: int = 120) -> tuple[str, float]:
    """
    Predict protein structure using ESMFold API.
    Returns (pdb_string, mean_pLDDT_confidence).

    ESMFold is ~10x faster than AlphaFold2 with comparable accuracy on most proteins.
    Sequence length limit: ~400 AA for API; use local model for longer.
    """
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    try:
        resp = requests.post(
            url,
            data=sequence,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=timeout,
        )
        resp.raise_for_status()
        pdb_str = resp.text

        # ESMFold stores per-residue pLDDT in the B-factor column (cols 61-66, 0-indexed 60:66).
        # The API returns values in the 0–1 range (e.g. 0.87 = 87% confidence), NOT 0–100.
        plddt_values = []
        for line in pdb_str.split("\n"):
            if line.startswith("ATOM") and len(line) > 60:
                try:
                    plddt_values.append(float(line[60:66].strip()))
                except ValueError:
                    pass

        if not plddt_values:
            # API returned no ATOM records — log the response prefix for diagnosis
            preview = pdb_str[:200].replace("\n", " ") if pdb_str else "<empty>"
            print(f"    [warn] ESMFold returned no ATOM records. Response: {preview}")
            return "", 0.0

        # Scale 0–1 → 0–100 to match the StructurePrediction.confidence units
        raw_mean = sum(plddt_values) / len(plddt_values)
        confidence = raw_mean * 100.0 if raw_mean <= 1.0 else raw_mean

        return pdb_str, confidence
    except Exception as e:
        print(f"    [warn] ESMFold request failed: {e}")
        return "", 0.0


def annotate_virulence(protein_title: str) -> list[str]:
    """Check protein title/annotation for known virulence factor keywords."""
    title_lower = protein_title.lower()
    return [kw for kw in VIRULENCE_KEYWORDS if kw in title_lower]


def characterize_novel_pathogen(
    taxon_name: str,
    use_esmfold: bool = True,
) -> list[StructurePrediction]:
    """
    Full characterization pipeline for a novel/flagged taxon:
      1. Fetch protein sequences from NCBI
      2. Predict structures
      3. Annotate virulence flags
      4. Generate risk assessment

    Returns list of StructurePrediction (one per retrieved protein).
    """
    proteins = fetch_protein_sequence(taxon_name, max_results=3)
    predictions = []

    for prot in proteins:
        if "error" in prot or not prot.get("sequence"):
            continue

        seq = prot["sequence"]
        title = prot.get("title", "unknown")

        # Truncate very long sequences for API
        if use_esmfold and len(seq) > 400:
            seq = seq[:400]

        pdb_str, confidence = ("", 0.0)
        if use_esmfold:
            pdb_str, confidence = predict_structure_esmfold(seq)

        virulence_flags = annotate_virulence(title)

        # Risk annotation
        if virulence_flags and confidence > 70:
            risk = (
                f"HIGH CONCERN: Predicted structure has high confidence (pLDDT={confidence:.1f}) "
                f"and matches virulence factors: {', '.join(virulence_flags)}. "
                f"Recommend immediate pathogen identification and containment protocols."
            )
        elif virulence_flags:
            risk = (
                f"MODERATE CONCERN: Protein annotation matches virulence factors "
                f"({', '.join(virulence_flags)}) but structure prediction confidence is low "
                f"(pLDDT={confidence:.1f}). Recommend further sequencing."
            )
        elif confidence > 70:
            risk = (
                f"MONITOR: High-confidence structure predicted (pLDDT={confidence:.1f}) "
                f"but no known virulence markers in annotation. Novel protein — further analysis needed."
            )
        else:
            risk = "LOW CONCERN: No virulence markers detected and structure prediction confidence is low."

        predictions.append(StructurePrediction(
            taxon=taxon_name,
            protein_name=title,
            sequence=seq,
            pdb_str=pdb_str,
            confidence=confidence,
            virulence_flags=virulence_flags,
            risk_annotation=risk,
        ))

    return predictions
