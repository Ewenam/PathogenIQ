"""
pipeline/runner.py
Orchestrates the full PathogenIQ pipeline end-to-end.

Flow:
  1. Ingest  — load Kraken2 reports or raw FASTQs
  2. Filter  — remove rare/low-prevalence taxa
  3. Graph   — build FDR-corrected co-occurrence network
  4. SBM     — fit stochastic block model, find communities
  5. Novelty — score each sample for anomalous profiles
  6. Risk    — compute composite risk score per sample
  7. Report  — save JSON + HTML reports
  8. Alert   — print/return alerts for HIGH/CRITICAL samples
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class PipelineConfig:
    # Graph
    spearman_threshold: float = 0.45
    fdr_alpha: float = 0.05
    min_prevalence: float = 0.1
    min_total_reads: int = 50
    # SBM
    max_k: int = 12
    sbm_n_init: int = 10
    # Novelty
    anomaly_contamination: float = 0.1
    # Risk
    alert_threshold: float = 0.6
    # Reporting
    output_dir: str = "./reports"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(
            spearman_threshold=cfg.get("graph", {}).get("spearman_threshold", 0.45),
            fdr_alpha=cfg.get("graph", {}).get("fdr_alpha", 0.05),
            min_prevalence=cfg.get("graph", {}).get("min_prevalence", 0.1),
            min_total_reads=cfg.get("graph", {}).get("min_total_reads", 50),
            max_k=cfg.get("sbm", {}).get("max_k", 12),
            sbm_n_init=cfg.get("sbm", {}).get("n_init", 10),
            alert_threshold=cfg.get("risk", {}).get("alert_threshold", 0.6),
            output_dir=cfg.get("reporting", {}).get("output_dir", "./reports"),
        )


def run(
    input_path: str | Path,
    config: PipelineConfig | None = None,
    rank: str = "G",
    run_characterization: bool = False,
    quiet: bool = False,
) -> dict:
    """
    Run the full pipeline on a directory of Kraken2 reports or a count matrix TSV.

    Args:
        input_path: directory with *.report files OR path to a TSV/CSV count matrix
        config: PipelineConfig (uses defaults if None)
        rank: taxonomic rank — 'G'=genus, 'S'=species
        run_characterization: if True, runs ESMFold on flagged novel pathogens
        quiet: suppress progress output

    Returns:
        dict with keys: sampleset, sbm_result, risk_scores, alerts, report_paths
    """
    from rich.console import Console
    from rich.progress import track

    cfg = config or PipelineConfig()
    console = Console(quiet=quiet)
    input_path = Path(input_path)

    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    console.rule("[bold]PathogenIQ[/bold] — Step 1: Ingestion")
    from ..ingestion.reader import load_sample_directory, load_count_matrix, filter_taxa

    if input_path.is_dir():
        console.print(f"Loading Kraken2 reports from [cyan]{input_path}[/cyan]")
        sampleset = load_sample_directory(input_path, rank=rank)
    else:
        console.print(f"Loading count matrix from [cyan]{input_path}[/cyan]")
        sampleset = load_count_matrix(input_path)

    console.print(f"  {len(sampleset.samples)} samples, {sampleset.taxa_matrix.shape[0]} taxa")

    # ── Step 2: Filter ────────────────────────────────────────────────────────
    console.rule("Step 2: Taxa Filtering")
    sampleset = filter_taxa(
        sampleset,
        min_prevalence=cfg.min_prevalence,
        min_total_reads=cfg.min_total_reads,
    )
    console.print(f"  After filtering: {sampleset.taxa_matrix.shape[0]} taxa retained")

    n_taxa = sampleset.taxa_matrix.shape[0]
    sbm_result = None

    if n_taxa < 3:
        console.print("[yellow]Too few taxa for community analysis — skipping graph/SBM[/yellow]")
    else:
        # ── Step 3: Build co-occurrence graph ─────────────────────────────────
        console.rule("Step 3: Co-occurrence Graph (FDR-corrected)")
        from ..community.graph import build_cooccurrence_graph
        adj, taxa_names, rho_matrix = build_cooccurrence_graph(
            sampleset,
            spearman_threshold=cfg.spearman_threshold,
            fdr_alpha=cfg.fdr_alpha,
        )

        # ── Step 4: Fit SBM ───────────────────────────────────────────────────
        console.rule("Step 4: Stochastic Block Model")
        from ..community.sbm import fit_sbm
        from ..ingestion.reader import SampleSet

        # Build a minimal sampleset with just the filtered taxa for SBM
        sbm_sampleset = SampleSet(
            samples=sampleset.samples,
            taxa_matrix=sampleset.taxa_matrix.loc[taxa_names],
            relative_abundance=sampleset.relative_abundance.loc[taxa_names],
        )
        sbm_result = fit_sbm(
            adj,
            taxa_names=taxa_names,
            max_k=min(cfg.max_k, n_taxa - 1),
            n_init=cfg.sbm_n_init,
        )
        console.print(f"  Communities found: K={sbm_result.k}")

    # ── Step 5: Novelty detection ─────────────────────────────────────────────
    console.rule("Step 5: Novelty Detection")
    from ..novelty.detector import compute_novelty_scores
    novelty_scores = compute_novelty_scores(
        sampleset,
        contamination=cfg.anomaly_contamination,
    )
    for name, score in sorted(novelty_scores.items(), key=lambda x: -x[1]):
        if score > 0.5:
            console.print(f"  [yellow]Novelty flag:[/yellow] {name} (score={score:.3f})")

    # ── Step 6: Risk scoring ──────────────────────────────────────────────────
    console.rule("Step 6: Risk Scoring")
    from ..scoring.risk import score_all_samples
    risk_scores = score_all_samples(
        sampleset,
        sbm_result=sbm_result,
        novelty_scores=novelty_scores,
        alert_threshold=cfg.alert_threshold,
    )

    alerts = [r for r in risk_scores if r.is_alert(cfg.alert_threshold)]
    for r in risk_scores:
        color = {"LOW": "green", "MODERATE": "yellow", "HIGH": "red", "CRITICAL": "bold red"}.get(r.level, "white")
        console.print(f"  [{color}]{r.level:8s}[/{color}]  {r.sample_name:<30s}  score={r.score:.3f}")

    # ── Step 7: Characterization (optional) ───────────────────────────────────
    characterization_results = {}
    if run_characterization and alerts:
        console.rule("Step 7: AlphaFold Characterization")
        from ..characterization.alphafold import characterize_novel_pathogen
        flagged_taxa = set()
        for alert in alerts:
            for p in alert.detected_pathogens[:2]:
                flagged_taxa.add(p["taxon"])
        for taxon in flagged_taxa:
            console.print(f"  Characterizing [cyan]{taxon}[/cyan] via ESMFold...")
            preds = characterize_novel_pathogen(taxon)
            characterization_results[taxon] = preds
            for pred in preds:
                console.print(f"    pLDDT={pred.confidence:.1f}  flags={pred.virulence_flags}")

    # ── Step 8: Report ────────────────────────────────────────────────────────
    console.rule("Step 8: Reporting")
    from ..reporting.report import save_json, save_html
    out_dir = Path(cfg.output_dir)
    meta = {"input": str(input_path), "rank": rank, "n_samples": len(sampleset.samples)}
    json_path = out_dir / "report.json"
    html_path = out_dir / "report.html"
    save_json(risk_scores, json_path, meta=meta)
    save_html(risk_scores, html_path, meta=meta)

    if alerts:
        console.rule(f"[bold red]ALERTS ({len(alerts)} samples)[/bold red]")
        for r in alerts:
            console.print(f"  [bold red]ALERT[/bold red] {r.sample_name}: {r.level} (score={r.score:.3f})")
            for p in r.detected_pathogens[:3]:
                console.print(f"    ↳ {p['taxon']} — {p['disease']} (abundance={p['abundance']:.4f})")
    else:
        console.print("[green]No alerts — all samples within expected range[/green]")

    return {
        "sampleset": sampleset,
        "sbm_result": sbm_result,
        "risk_scores": risk_scores,
        "novelty_scores": novelty_scores,
        "alerts": alerts,
        "characterization": characterization_results,
        "report_paths": {"json": str(json_path), "html": str(html_path)},
    }
