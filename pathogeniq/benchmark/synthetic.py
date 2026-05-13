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
) -> SyntheticDataset:
    """
    Generate a synthetic count matrix for one benchmark scenario.

    Returns a SyntheticDataset with the count matrix and ground-truth labels.
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
