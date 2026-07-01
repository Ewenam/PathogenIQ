"""
benchmark/synthetic.py
Generate synthetic Kraken2-style count matrices with controlled pathogen injections.

Enables reproducible ground-truth evaluation of pipeline sensitivity and specificity.

Design:
  - Background taxa: 20 common environmental genera not in PATHOGEN_DB
  - Pathogen injections: reads added at a controlled fraction of total reads
  - Count distribution: Dirichlet-distributed background (realistic compositional noise)
  - Total reads: 100k–500k per sample (typical wastewater metagenomics)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Environmental genera absent from PATHOGEN_DB — serve as inert background
BACKGROUND_TAXA: list[str] = [
    "Sphingomonas", "Methylobacterium", "Acidovorax", "Comamonas", "Brevundimonas",
    "Pedobacter", "Hymenobacter", "Arthrobacter", "Microbacterium", "Massilia",
    "Chitinophaga", "Devosia", "Mesorhizobium", "Bradyrhizobium", "Caulobacter",
    "Bosea", "Novosphingobium", "Parvibaculum", "Steroidobacter", "Phenylobacterium",
]


@dataclass
class Scenario:
    name: str
    description: str
    # {genus: (min_abundance, max_abundance)} — genus must match PATHOGEN_DB key
    contamination: dict[str, tuple[float, float]] = field(default_factory=dict)
    n_samples: int = 10
    # True → samples are expected to trigger alerts; False → should be LOW/MODERATE
    # None → unknown (novelty-dependent; no precision/recall counted)
    expected_alert: Optional[bool] = None


# ── Standard benchmark scenarios ──────────────────────────────────────────────

SCENARIOS: list[Scenario] = [
    Scenario(
        name="negative_controls",
        description="Clean samples — environmental background only",
        n_samples=12,
        contamination={},
        expected_alert=False,
    ),
    Scenario(
        name="low_contamination",
        description="Single pathogen at low abundance (1–5% Salmonella)",
        n_samples=10,
        contamination={"Salmonella": (0.01, 0.05)},
        expected_alert=False,  # low concentration should stay below alert threshold
    ),
    Scenario(
        name="high_cholera",
        description="Outbreak-level Vibrio cholerae (20–60% abundance)",
        n_samples=8,
        contamination={"Vibrio": (0.20, 0.60)},
        expected_alert=True,
    ),
    Scenario(
        name="critical_yersinia",
        description="Tier-1 BSL-3 pathogen — Yersinia at critical levels (30–60%)",
        n_samples=6,
        contamination={"Yersinia": (0.30, 0.60)},
        expected_alert=True,
    ),
    Scenario(
        name="multi_pathogen",
        description="Multi-pathogen community (Salmonella + E. coli + Listeria, 5–20% each)",
        n_samples=8,
        contamination={
            "Salmonella": (0.05, 0.15),
            "Escherichia": (0.03, 0.10),
            "Listeria": (0.02, 0.08),
        },
        expected_alert=True,
    ),
    Scenario(
        name="novel_agent",
        description="Unrecognized genus at high abundance — tests novelty detector",
        n_samples=8,
        contamination={"Hypotheticus": (0.15, 0.40)},
        expected_alert=None,  # novelty-score dependent; not in PATHOGEN_DB
    ),
]


# ── Hard scenarios near the decision boundary ─────────────────────────────────
# The scenarios above are near-trivially separable (clear outbreak vs clean),
# which is why calibration on them yields AUROC≈1.0. These deliberately sit on
# the boundary so the validation harness produces meaningful ROC/PR and a
# non-degenerate operating point. Kept SEPARATE from SCENARIOS so the canonical
# benchmark / paper figures are unchanged.
HARD_SCENARIOS: list[Scenario] = [
    Scenario(
        name="borderline_outbreak",
        description="Genuine but modest signal — high-risk pathogen just around "
                    "the alert boundary (8–16%). A true positive that is hard to catch.",
        n_samples=12,
        contamination={"Salmonella": (0.08, 0.16)},
        expected_alert=True,
    ),
    Scenario(
        name="cryptic_low_shed",
        description="Real waterborne pathogen at very low shedding (2–5%) — the "
                    "emerging-event case that abundance alone misses.",
        n_samples=10,
        contamination={"Vibrio": (0.02, 0.05)},
        expected_alert=True,
    ),
    Scenario(
        name="endemic_flora_confounder",
        description="High load of endemic opportunists that happen to be in the "
                    "risk DB (Pseudomonas/Streptococcus/Acinetobacter, 8–20% total) — "
                    "routine background that should NOT alert but trips naive "
                    "multi-pathogen load rules.",
        n_samples=12,
        contamination={
            "Pseudomonas": (0.05, 0.10),
            "Streptococcus": (0.03, 0.07),
            "Acinetobacter": (0.02, 0.05),
        },
        expected_alert=False,
    ),
    Scenario(
        name="near_threshold_negative",
        description="A single moderate-risk genus at 4–9% — elevated but below a "
                    "credible outbreak level; should stay under the alert threshold.",
        n_samples=10,
        contamination={"Klebsiella": (0.04, 0.09)},
        expected_alert=False,
    ),
]


@dataclass
class SyntheticDataset:
    scenario: Scenario
    count_matrix: pd.DataFrame          # taxa × samples
    ground_truth: pd.DataFrame          # samples × {is_contaminated, pathogen_fraction}
    pathogen_fractions: dict[str, dict[str, float]]  # sample → {genus: fraction}


def generate_scenario(
    scenario: Scenario,
    seed: int = 42,
    total_reads_range: tuple[int, int] = (100_000, 500_000),
    add_noise: bool = True,
) -> SyntheticDataset:
    """
    Generate a synthetic count matrix for one benchmark scenario.

    Returns a SyntheticDataset with the count matrix and ground-truth labels.

    add_noise: if True (default), apply Poisson sampling noise to every
        taxon's read count, simulating the sequencing-depth variability
        present in real Kraken2 reports (counts are draws from the
        underlying composition, not exact proportions of total_reads).
        Low-abundance taxa are affected proportionally more than
        high-abundance ones, as in real data.
    """
    rng = np.random.default_rng(seed)
    n = scenario.n_samples
    bg = BACKGROUND_TAXA

    # ── Background reads via Dirichlet ────────────────────────────────────────
    alpha = np.ones(len(bg)) * 2.0  # slightly concentrated distribution
    bg_proportions = rng.dirichlet(alpha, size=n)  # (n, n_bg)
    total_reads = rng.integers(*total_reads_range, size=n)

    # ── Inject pathogen reads ─────────────────────────────────────────────────
    pathogen_fractions: dict[str, dict[str, float]] = {
        f"WW_site_{i+1:02d}": {} for i in range(n)
    }
    sample_names = list(pathogen_fractions.keys())

    # Track cumulative pathogen fraction per sample (can't exceed ~0.85)
    cum_fracs = np.zeros(n)
    pathogens: dict[str, np.ndarray] = {}

    for genus, (lo, hi) in scenario.contamination.items():
        fracs = rng.uniform(lo, hi, size=n)
        # Clamp so total pathogen fraction stays < 0.85
        available = np.clip(0.85 - cum_fracs, 0.0, None)
        fracs = np.minimum(fracs, available)
        cum_fracs += fracs
        pathogens[genus] = fracs
        for i, name in enumerate(sample_names):
            pathogen_fractions[name][genus] = round(float(fracs[i]), 5)

    # Scale background proportions down to leave room for pathogens
    bg_reads = np.round(bg_proportions * total_reads[:, None] * (1 - cum_fracs[:, None])).astype(int)

    # ── Build count matrix (taxa × samples) ──────────────────────────────────
    data: dict[str, list[int]] = {}
    for j, taxon in enumerate(bg):
        data[taxon] = bg_reads[:, j].tolist()

    for genus, fracs in pathogens.items():
        pathogen_reads = np.round(total_reads * fracs).astype(int)
        data[genus] = pathogen_reads.tolist()

    count_matrix = pd.DataFrame(data, index=sample_names).T
    count_matrix.index.name = "taxon"

    # ── Sequencing-depth noise ────────────────────────────────────────────────
    # Real Kraken2 counts are draws from the underlying composition, not exact
    # proportions of total_reads. Apply Poisson sampling noise per taxon×sample
    # so low-abundance taxa (e.g. low_contamination pathogens) show realistic
    # run-to-run variability while high-abundance taxa remain stable.
    if add_noise:
        expected = count_matrix.to_numpy().astype(float)
        noisy = rng.poisson(expected)
        count_matrix = pd.DataFrame(noisy, index=count_matrix.index, columns=count_matrix.columns)

    # ── Ground truth labels ───────────────────────────────────────────────────
    total_pathogen_frac = cum_fracs
    is_contaminated = total_pathogen_frac > 0.0
    gt = pd.DataFrame(
        {
            "is_contaminated": is_contaminated,
            "total_pathogen_fraction": total_pathogen_frac.round(4),
            "expected_alert": (
                [scenario.expected_alert] * n if scenario.expected_alert is not None
                else [None] * n
            ),
        },
        index=sample_names,
    )

    return SyntheticDataset(
        scenario=scenario,
        count_matrix=count_matrix,
        ground_truth=gt,
        pathogen_fractions=pathogen_fractions,
    )


def generate_all_scenarios(seed: int = 42) -> list[SyntheticDataset]:
    """Generate datasets for all standard benchmark scenarios."""
    return [generate_scenario(s, seed=seed + i) for i, s in enumerate(SCENARIOS)]


# Approximate theoretical (genomic-DNA-based) composition of the ZymoBIOMICS
# Microbial Community Standard (Zymo Research, cat. #D6300), bacterial
# fraction only — excludes the standard's two yeast components (S. cerevisiae,
# C. neoformans), which fall outside PATHOGEN_DB's bacteria/virus/fungi-genus
# scope. Figures are illustrative/from public documentation — verify against
# Zymo's current Certificate of Analysis before citing in publication.
ZYMOBIOMICS_COMPOSITION: dict[str, float] = {
    "Listeria": 0.141, "Pseudomonas": 0.042, "Bacillus": 0.174,
    "Escherichia": 0.101, "Salmonella": 0.104, "Lactobacillus": 0.184,
    "Enterococcus": 0.099, "Staphylococcus": 0.155,
}


def generate_mock_community(
    composition: Optional[dict[str, float]] = None,
    n_replicates: int = 5,
    total_reads_range: tuple[int, int] = (100_000, 500_000),
    add_noise: bool = True,
    seed: int = 42,
) -> SyntheticDataset:
    """
    Generate replicate samples matching a fully-defined reference community
    composition (every taxon has a known abundance — no separate
    "background"), unlike generate_scenario's injected-pathogen-into-
    background model. Used for ground-truth validation against a known
    mock community (e.g. ZymoBIOMICS), mirroring how MARTi validated
    against the same standard.
    """
    from ..scoring.risk import PATHOGEN_DB

    if composition is None:
        composition = ZYMOBIOMICS_COMPOSITION

    rng = np.random.default_rng(seed)
    taxa = list(composition.keys())
    fracs = np.array(list(composition.values()), dtype=float)
    fracs = fracs / fracs.sum()  # normalize to sum to 1

    total_reads = rng.integers(*total_reads_range, size=n_replicates)
    sample_names = [f"zymo_rep_{i+1:02d}" for i in range(n_replicates)]

    expected = fracs[:, None] * total_reads[None, :]
    if add_noise:
        counts = rng.poisson(expected)
    else:
        counts = np.round(expected).astype(int)

    count_matrix = pd.DataFrame(counts, index=taxa, columns=sample_names)
    count_matrix.index.name = "taxon"

    total_pathogen_fraction = sum(f for g, f in zip(taxa, fracs) if g in PATHOGEN_DB)
    gt = pd.DataFrame(
        {
            "is_contaminated": [True] * n_replicates,
            "total_pathogen_fraction": [round(total_pathogen_fraction, 4)] * n_replicates,
            "expected_alert": [True] * n_replicates,
        },
        index=sample_names,
    )

    intended_fracs = {g: round(float(f), 5) for g, f in zip(taxa, fracs)}
    pathogen_fractions = {name: dict(intended_fracs) for name in sample_names}

    scenario = Scenario(
        name="zymobiomics_mock_community",
        description="ZymoBIOMICS D6300 reference community — ground-truth validation",
        n_samples=n_replicates,
        expected_alert=True,
    )

    return SyntheticDataset(
        scenario=scenario,
        count_matrix=count_matrix,
        ground_truth=gt,
        pathogen_fractions=pathogen_fractions,
    )
