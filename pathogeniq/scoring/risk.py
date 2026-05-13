"""
scoring/risk.py
Pathogen risk scoring engine.

Combines three signals into a [0, 1] risk score per sample:
  1. Known pathogen abundance  — is a high-risk taxon present and at what level?
  2. Community context         — is the pathogen in a pathogen-enriched community?
  3. Novelty signal            — is there an unusual sequence anomaly?

Risk levels:
  0.0–0.3  LOW      routine environmental background
  0.3–0.6  MODERATE elevated — monitor closely
  0.6–0.8  HIGH     alert — investigate immediately
  0.8–1.0  CRITICAL outbreak-level signal
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ── Pathogen risk database ─────────────────────────────────────────────────────
# Curated from WHO priority pathogens, CDC NNDSS, ESKAPE, and biosafety lists.
# risk_weight: 0.0 (harmless) → 1.0 (highest priority pathogen)

PATHOGEN_DB: dict[str, dict] = {
    # ── Tier 1: Critical / BSL-3-4 / WHO priority ──────────────────────────────
    "Yersinia": {"risk_weight": 0.95, "category": "bacteria", "disease": "Plague"},
    "Francisella": {"risk_weight": 0.95, "category": "bacteria", "disease": "Tularemia"},
    "Bacillus": {"risk_weight": 0.90, "category": "bacteria", "disease": "Anthrax (B.anthracis)"},
    "Brucella": {"risk_weight": 0.90, "category": "bacteria", "disease": "Brucellosis"},
    "Burkholderia": {"risk_weight": 0.90, "category": "bacteria", "disease": "Melioidosis"},
    "Coxiella": {"risk_weight": 0.85, "category": "bacteria", "disease": "Q fever"},
    "Rickettsia": {"risk_weight": 0.85, "category": "bacteria", "disease": "Spotted fever/Typhus"},
    "Clostridioides": {"risk_weight": 0.80, "category": "bacteria", "disease": "C. diff"},
    "Clostridium": {"risk_weight": 0.80, "category": "bacteria", "disease": "Botulism/Tetanus"},
    "Vibrio": {"risk_weight": 0.80, "category": "bacteria", "disease": "Cholera"},

    # ── Tier 2: High priority ───────────────────────────────────────────────────
    "Salmonella": {"risk_weight": 0.75, "category": "bacteria", "disease": "Salmonellosis/Typhoid"},
    "Shigella": {"risk_weight": 0.75, "category": "bacteria", "disease": "Shigellosis"},
    "Campylobacter": {"risk_weight": 0.70, "category": "bacteria", "disease": "Campylobacteriosis"},
    "Listeria": {"risk_weight": 0.70, "category": "bacteria", "disease": "Listeriosis"},
    "Escherichia": {"risk_weight": 0.65, "category": "bacteria", "disease": "E. coli O157:H7"},
    "Staphylococcus": {"risk_weight": 0.60, "category": "bacteria", "disease": "MRSA/Food poisoning"},
    "Streptococcus": {"risk_weight": 0.60, "category": "bacteria", "disease": "Strep/Pneumonia"},
    "Klebsiella": {"risk_weight": 0.65, "category": "bacteria", "disease": "Pneumonia/Sepsis (ESKAPE)"},
    "Acinetobacter": {"risk_weight": 0.65, "category": "bacteria", "disease": "Hospital infections (ESKAPE)"},
    "Pseudomonas": {"risk_weight": 0.60, "category": "bacteria", "disease": "Opportunistic infections"},
    "Enterococcus": {"risk_weight": 0.60, "category": "bacteria", "disease": "VRE (ESKAPE)"},
    "Neisseria": {"risk_weight": 0.65, "category": "bacteria", "disease": "Meningitis/Gonorrhea"},
    "Haemophilus": {"risk_weight": 0.55, "category": "bacteria", "disease": "Meningitis/Pneumonia"},
    "Legionella": {"risk_weight": 0.70, "category": "bacteria", "disease": "Legionnaires disease"},
    "Mycobacterium": {"risk_weight": 0.75, "category": "bacteria", "disease": "Tuberculosis/Leprosy"},
    "Treponema": {"risk_weight": 0.65, "category": "bacteria", "disease": "Syphilis"},
    "Leptospira": {"risk_weight": 0.65, "category": "bacteria", "disease": "Leptospirosis"},
    "Helicobacter": {"risk_weight": 0.55, "category": "bacteria", "disease": "Gastric ulcers"},

    # ── Tier 3: Moderate priority ───────────────────────────────────────────────
    "Enterobacter": {"risk_weight": 0.50, "category": "bacteria", "disease": "Nosocomial infections"},
    "Proteus": {"risk_weight": 0.45, "category": "bacteria", "disease": "UTI"},
    "Serratia": {"risk_weight": 0.50, "category": "bacteria", "disease": "Nosocomial infections"},
    "Stenotrophomonas": {"risk_weight": 0.50, "category": "bacteria", "disease": "Opportunistic"},
    "Fusobacterium": {"risk_weight": 0.45, "category": "bacteria", "disease": "Oral/colon cancer"},
    "Bacteroides": {"risk_weight": 0.35, "category": "bacteria", "disease": "Anaerobic infections"},

    # ── Viruses (detected via Kraken2 viral DB) ─────────────────────────────────
    "Betacoronavirus": {"risk_weight": 0.90, "category": "virus", "disease": "COVID-19/SARS"},
    "Orthomyxovirus": {"risk_weight": 0.85, "category": "virus", "disease": "Influenza"},
    "Norovirus": {"risk_weight": 0.75, "category": "virus", "disease": "Gastroenteritis"},
    "Rotavirus": {"risk_weight": 0.70, "category": "virus", "disease": "Childhood diarrhea"},
    "Enterovirus": {"risk_weight": 0.70, "category": "virus", "disease": "Polio/Meningitis"},
    "Adenovirus": {"risk_weight": 0.60, "category": "virus", "disease": "Respiratory/GI"},
    "Orthopoxvirus": {"risk_weight": 0.85, "category": "virus", "disease": "Mpox/Smallpox"},
    "Flavivirus": {"risk_weight": 0.80, "category": "virus", "disease": "Dengue/Zika/WNV"},
    "Alphavirus": {"risk_weight": 0.75, "category": "virus", "disease": "Chikungunya/EEE"},

    # ── Fungi / parasites ───────────────────────────────────────────────────────
    "Candida": {"risk_weight": 0.65, "category": "fungi", "disease": "Candidiasis (drug-resistant)"},
    "Aspergillus": {"risk_weight": 0.60, "category": "fungi", "disease": "Aspergillosis"},
    "Cryptosporidium": {"risk_weight": 0.70, "category": "parasite", "disease": "Cryptosporidiosis"},
    "Giardia": {"risk_weight": 0.60, "category": "parasite", "disease": "Giardiasis"},
}


@dataclass
class RiskScore:
    sample_name: str
    score: float                        # 0.0 – 1.0
    level: str                          # LOW / MODERATE / HIGH / CRITICAL
    detected_pathogens: list[dict]      # [{taxon, risk_weight, abundance, disease}]
    community_signal: float             # 0–1 community-level enrichment
    novelty_signal: float               # 0–1 anomaly score
    breakdown: dict = field(default_factory=dict)

    def is_alert(self, threshold: float = 0.6) -> bool:
        return self.score >= threshold


def _score_level(score: float) -> str:
    if score >= 0.8:
        return "CRITICAL"
    if score >= 0.6:
        return "HIGH"
    if score >= 0.3:
        return "MODERATE"
    return "LOW"


def score_sample(
    sample_name: str,
    taxa_abundance: pd.Series,
    community_label: int | None = None,
    pathogen_community_labels: set[int] | None = None,
    novelty_score: float = 0.0,
    weights: dict | None = None,
) -> RiskScore:
    """
    Compute a composite risk score for a single sample.

    Args:
        taxa_abundance: relative abundance Series indexed by taxon name
        community_label: which SBM community this sample's dominant taxa fall in
        pathogen_community_labels: set of community labels enriched for pathogens
        novelty_score: 0–1 score from the novelty detector
    """
    w = weights or {"abundance": 0.5, "community": 0.25, "novelty": 0.25}

    # Signal 1: pathogen abundance
    detected = []
    abundance_score = 0.0
    for taxon, abundance in taxa_abundance.items():
        genus = taxon.split()[0]
        if genus in PATHOGEN_DB and abundance > 0:
            info = PATHOGEN_DB[genus]
            contribution = info["risk_weight"] * float(abundance)
            detected.append({
                "taxon": taxon,
                "genus": genus,
                "risk_weight": info["risk_weight"],
                "abundance": float(abundance),
                "disease": info["disease"],
                "category": info["category"],
                "contribution": contribution,
            })
            abundance_score += contribution

    abundance_score = min(abundance_score, 1.0)
    detected.sort(key=lambda x: x["contribution"], reverse=True)

    # Signal 2: community context
    community_signal = 0.0
    if community_label is not None and pathogen_community_labels:
        community_signal = 1.0 if community_label in pathogen_community_labels else 0.0

    # Signal 3: novelty
    novelty = float(np.clip(novelty_score, 0.0, 1.0))

    # Composite score
    score = (
        w["abundance"] * abundance_score +
        w["community"] * community_signal +
        w["novelty"] * novelty
    )
    score = float(np.clip(score, 0.0, 1.0))

    return RiskScore(
        sample_name=sample_name,
        score=score,
        level=_score_level(score),
        detected_pathogens=detected,
        community_signal=community_signal,
        novelty_signal=novelty,
        breakdown={
            "abundance_score": abundance_score,
            "community_signal": community_signal,
            "novelty_signal": novelty,
            "weights": w,
        },
    )


def identify_pathogen_communities(
    sbm_result,
    taxa_abundance: pd.DataFrame,
    min_pathogen_fraction: float = 0.2,
) -> set[int]:
    """
    Identify which SBM communities are enriched for known pathogens.
    A community is flagged if >= min_pathogen_fraction of its members are in PATHOGEN_DB.
    """
    flagged = set()
    labels = sbm_result.labels
    taxa_names = sbm_result.taxa_names

    for k in range(sbm_result.k):
        members = [taxa_names[i] for i, lbl in enumerate(labels) if lbl == k]
        if not members:
            continue
        pathogen_members = [
            t for t in members if t.split()[0] in PATHOGEN_DB
        ]
        fraction = len(pathogen_members) / len(members)
        if fraction >= min_pathogen_fraction:
            flagged.add(k)

    return flagged


def score_all_samples(
    sampleset,
    sbm_result=None,
    novelty_scores: dict[str, float] | None = None,
    alert_threshold: float = 0.6,
) -> list[RiskScore]:
    """
    Score all samples in a SampleSet and return sorted risk scores (highest first).
    """
    pathogen_communities: set[int] = set()
    if sbm_result is not None:
        pathogen_communities = identify_pathogen_communities(sbm_result, sampleset.relative_abundance)

    scores = []
    for sample in sampleset.samples:
        name = sample.name
        rel_abund = sampleset.relative_abundance.get(name, pd.Series(dtype=float))
        novelty = (novelty_scores or {}).get(name, 0.0)

        # Determine dominant community for this sample
        community_label = None
        if sbm_result is not None:
            # Assign community by majority vote of taxa present in this sample
            present_taxa = rel_abund[rel_abund > 0].index.tolist()
            if present_taxa and sbm_result.taxa_names:
                taxon_to_idx = {t: i for i, t in enumerate(sbm_result.taxa_names)}
                community_votes = [
                    sbm_result.labels[taxon_to_idx[t]]
                    for t in present_taxa if t in taxon_to_idx
                ]
                if community_votes:
                    community_label = max(set(community_votes), key=community_votes.count)

        rs = score_sample(
            sample_name=name,
            taxa_abundance=rel_abund,
            community_label=community_label,
            pathogen_community_labels=pathogen_communities,
            novelty_score=novelty,
        )
        scores.append(rs)

    scores.sort(key=lambda x: x.score, reverse=True)
    return scores
