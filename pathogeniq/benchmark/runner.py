"""
benchmark/runner.py
Run all benchmark scenarios and produce a validation report.

Usage:
    pathogeniq benchmark
    pathogeniq benchmark --quick --seed 42 --output benchmark.json
    python -m pathogeniq.benchmark.runner
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .synthetic import SCENARIOS, generate_scenario, SyntheticDataset
from .metrics import compute_metrics, aggregate_metrics, ScenarioMetrics


def _run_scenario_pipeline(
    dataset: SyntheticDataset,
    alert_threshold: float = 0.6,
    quick: bool = False,
) -> list:
    """
    Run the PathogenIQ scoring pipeline on one synthetic dataset.

    quick=True: skip graph/SBM (much faster, suitable for risk-scoring validation)
    quick=False: run full pipeline including co-occurrence graph + SBM
    """
    from pathogeniq.ingestion.reader import SampleSet, Sample, filter_taxa
    import pandas as pd

    mat = dataset.count_matrix
    rel = mat.div(mat.sum(axis=0), axis=1).fillna(0)
    samples = [Sample(name=col) for col in mat.columns]
    sampleset = SampleSet(samples=samples, taxa_matrix=mat, relative_abundance=rel)

    # Filter as the pipeline would
    sampleset = filter_taxa(sampleset, min_prevalence=0.05, min_total_reads=10)

    sbm_result = None
    if not quick and sampleset.taxa_matrix.shape[0] >= 3:
        try:
            from pathogeniq.community.graph import build_cooccurrence_graph
            from pathogeniq.community.sbm import fit_sbm

            n_taxa = sampleset.taxa_matrix.shape[0]
            adj, taxa_names, _ = build_cooccurrence_graph(sampleset)
            sbm_sampleset = SampleSet(
                samples=sampleset.samples,
                taxa_matrix=sampleset.taxa_matrix.loc[taxa_names],
                relative_abundance=sampleset.relative_abundance.loc[taxa_names],
            )
            sbm_result = fit_sbm(
                adj, taxa_names=taxa_names,
                max_k=min(8, n_taxa - 1), n_init=3,
            )
        except Exception:
            pass  # fall back to no-SBM scoring

    from pathogeniq.novelty.detector import compute_novelty_scores
    novelty_scores = compute_novelty_scores(sampleset)

    from pathogeniq.scoring.risk import score_all_samples
    return score_all_samples(
        sampleset,
        sbm_result=sbm_result,
        novelty_scores=novelty_scores,
        alert_threshold=alert_threshold,
    )


def run_benchmark(
    alert_threshold: float = 0.6,
    quick: bool = False,
    seed: int = 42,
    quiet: bool = False,
) -> dict:
    """
    Run all benchmark scenarios and return a results dict.

    Returns:
        {
          "generated_at": ...,
          "alert_threshold": ...,
          "scenarios": [ScenarioMetrics ...],
          "aggregate": {...},
        }
    """
    console = Console(quiet=quiet)
    console.rule("[bold]PathogenIQ Benchmark[/bold] — Ground Truth Validation")
    console.print(
        f"  Mode: {'quick (no SBM)' if quick else 'full pipeline'}  |  "
        f"Alert threshold: {alert_threshold}  |  Seed: {seed}\n"
    )

    all_metrics: list[ScenarioMetrics] = []

    for i, scenario in enumerate(SCENARIOS):
        t0 = time.perf_counter()
        dataset = generate_scenario(scenario, seed=seed + i)

        risk_scores = _run_scenario_pipeline(dataset, alert_threshold, quick)

        metrics = compute_metrics(
            scenario_name=scenario.name,
            description=scenario.description,
            risk_scores=risk_scores,
            ground_truth=dataset.ground_truth,
            alert_threshold=alert_threshold,
        )
        all_metrics.append(metrics)
        elapsed = time.perf_counter() - t0

        if not quiet:
            _print_scenario_summary(console, metrics, dataset, risk_scores, elapsed)

    agg = aggregate_metrics(all_metrics)

    if not quiet:
        _print_aggregate(console, agg, all_metrics)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "alert_threshold": alert_threshold,
        "mode": "quick" if quick else "full",
        "seed": seed,
        "scenarios": [_metrics_to_dict(m) for m in all_metrics],
        "aggregate": agg,
    }


def _print_scenario_summary(
    console: Console,
    m: ScenarioMetrics,
    dataset: SyntheticDataset,
    risk_scores: list,
    elapsed: float,
) -> None:
    console.rule(f"[bold]{m.scenario}[/bold]")
    console.print(f"  {m.description}")
    console.print(
        f"  Samples: {m.n_samples}  |  "
        f"Contaminated: {m.n_contaminated}  |  "
        f"Alerts raised: {m.n_alerts}  |  "
        f"Time: {elapsed:.2f}s"
    )

    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Sample", style="bold")
    table.add_column("Level")
    table.add_column("Score", justify="right")
    table.add_column("Pathogen%", justify="right")
    table.add_column("TP/FP/TN/FN")

    score_map = {r.sample_name: r for r in risk_scores}
    level_style = {
        "LOW": "green", "MODERATE": "yellow",
        "HIGH": "red", "CRITICAL": "bold red",
    }

    for sample_name, row in dataset.ground_truth.iterrows():
        rs = score_map.get(sample_name)
        if rs is None:
            continue
        frac = row["total_pathogen_fraction"]
        expected = row["expected_alert"]
        predicted = rs.score >= m.alert_threshold
        truly_pos = expected is True

        if expected is None:
            verdict = "[dim]—[/dim]"
        elif truly_pos and predicted:
            verdict = "[green]TP[/green]"
        elif truly_pos and not predicted:
            verdict = "[red]FN[/red]"
        elif not truly_pos and predicted:
            verdict = "[yellow]FP[/yellow]"
        else:
            verdict = "[green]TN[/green]"

        style = level_style.get(rs.level, "white")
        table.add_row(
            sample_name,
            f"[{style}]{rs.level}[/{style}]",
            f"{rs.score:.3f}",
            f"{frac*100:.1f}%",
            verdict,
        )

    console.print(table)

    if m.sensitivity is not None:
        sens_str = f"{m.sensitivity:.2f}" if m.sensitivity is not None else "—"
        spec_str = f"{m.specificity:.2f}" if m.specificity is not None else "—"
        prec_str = f"{m.precision:.2f}" if m.precision is not None else "—"
        f1_str = f"{m.f1:.2f}" if m.f1 is not None else "—"
        console.print(
            f"\n  Sensitivity={sens_str}  Specificity={spec_str}  "
            f"Precision={prec_str}  F1={f1_str}"
        )
        console.print(
            f"  TP={m.tp}  FP={m.fp}  TN={m.tn}  FN={m.fn}  |  "
            f"Mean score: {m.mean_score:.3f}  Max: {m.max_score:.3f}\n"
        )
    else:
        console.print(
            f"\n  [dim](expected_alert=None — novelty scenario, no TP/FP counted)[/dim]  "
            f"Mean score: {m.mean_score:.3f}  Max: {m.max_score:.3f}\n"
        )


def _print_aggregate(
    console: Console, agg: dict, all_metrics: list[ScenarioMetrics]
) -> None:
    if not agg:
        return

    console.rule("[bold green]Aggregate Benchmark Results[/bold green]")

    summary = Table(show_header=True, title="Overall Performance")
    summary.add_column("Metric", style="bold")
    summary.add_column("Macro-avg", justify="center")
    summary.add_column("Micro-avg", justify="center")

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "—"

    summary.add_row(
        "Sensitivity (Recall)",
        _fmt(agg.get("macro_sensitivity")),
        _fmt(agg.get("micro_sensitivity")),
    )
    summary.add_row(
        "Specificity",
        _fmt(agg.get("macro_specificity")),
        _fmt(agg.get("micro_specificity")),
    )
    summary.add_row(
        "Precision",
        _fmt(agg.get("macro_precision")),
        _fmt(agg.get("micro_precision")),
    )
    summary.add_row("F1 Score", _fmt(agg.get("macro_f1")), "—")
    summary.add_row(
        "Confusion (TP/FP/TN/FN)",
        "—",
        f"{agg['total_tp']}/{agg['total_fp']}/{agg['total_tn']}/{agg['total_fn']}",
    )

    console.print(summary)
    console.print("\n[bold]PathogenIQ v0.1.0[/bold] — Benchmark complete.\n")


def _metrics_to_dict(m: ScenarioMetrics) -> dict:
    return {
        "scenario": m.scenario,
        "description": m.description,
        "n_samples": m.n_samples,
        "n_contaminated": m.n_contaminated,
        "n_alerts": m.n_alerts,
        "tp": m.tp, "fp": m.fp, "tn": m.tn, "fn": m.fn,
        "sensitivity": m.sensitivity,
        "specificity": m.specificity,
        "precision": m.precision,
        "f1": m.f1,
        "mean_score": m.mean_score,
        "max_score": m.max_score,
        "mean_contaminated_score": m.mean_contaminated_score,
        "mean_clean_score": m.mean_clean_score,
    }


if __name__ == "__main__":
    run_benchmark()
