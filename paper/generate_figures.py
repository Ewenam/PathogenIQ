#!/usr/bin/env python3
"""
paper/generate_figures.py
=========================
Run from the project root:

    python paper/generate_figures.py                        # synthetic benchmark only
    python paper/generate_figures.py --report reports/report.json   # + real data overlay

Outputs (all in paper/figures/):
    fig1_pipeline.pdf       — architecture diagram
    fig2_cusum.pdf          — CUSUM temporal trace with lead-time annotation
    fig3_benchmark.pdf      — per-scenario F1 bar chart
    fig4_risk_dist.pdf      — risk score distributions by scenario
    table_results.tex       — filled-in TODO replacements for the paper

Requires: matplotlib, numpy, scipy  (all in the project venv)
"""
from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

# ── ensure project is importable ─────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── IEEE-style global settings ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":         8,
    "axes.titlesize":    9,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "lines.linewidth":   1.2,
    "axes.linewidth":    0.7,
    "grid.linewidth":    0.4,
    "grid.alpha":        0.4,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.02,
})

COLORS = {
    "pathogeniq": "#1a3a6e",
    "baseline1":  "#e07b39",
    "baseline2":  "#6db56d",
    "baseline3":  "#b05cbb",
    "critical":   "#c0392b",
    "high":       "#e67e22",
    "moderate":   "#f1c40f",
    "low":        "#27ae60",
    "cusum":      "#1a3a6e",
    "threshold":  "#c0392b",
    "alert":      "#e74c3c",
    "score":      "#2980b9",
}

ONE_COL = 3.487   # IEEE single column width (inches)
TWO_COL = 7.22    # IEEE double column width (inches)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Pipeline Architecture
# ══════════════════════════════════════════════════════════════════════════════

def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(TWO_COL, 1.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stages = [
        ("Kraken2\nReports",   0.22, "#ecf0f1", "#7f8c8d", False),
        ("Taxa\nFilter",       1.35, "#ecf0f1", "#7f8c8d", False),
        ("Co-occurrence\nGraph", 2.60, "#dce8f5", "#2471a3", True),
        ("SBM\nCommunity",     3.95, "#dce8f5", "#2471a3", True),
        ("Novelty\nDetector",  5.20, "#dce8f5", "#2471a3", True),
        ("Risk\nScorer",       6.45, "#fdebd0", "#ca6f1e", True),
        ("CUSUM\n+ Trend",     7.65, "#fdebd0", "#ca6f1e", True),
        ("ESMFold\nAlert",     8.85, "#fadbd8", "#c0392b", True),
    ]

    BOX_W, BOX_H = 0.95, 0.52
    Y_CENTER = 0.60

    for label, x, bg, border, is_ml in stages:
        rect = FancyBboxPatch(
            (x - BOX_W / 2, Y_CENTER - BOX_H / 2), BOX_W, BOX_H,
            boxstyle="round,pad=0.04",
            facecolor=bg, edgecolor=border,
            linewidth=1.2 if is_ml else 0.7,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(x, Y_CENTER, label, ha="center", va="center",
                fontsize=6.2, fontweight="bold" if is_ml else "normal",
                color="#1a1a1a", zorder=4, linespacing=1.3)

    # Arrows between stages
    for i in range(len(stages) - 1):
        x0 = stages[i][1] + BOX_W / 2
        x1 = stages[i + 1][1] - BOX_W / 2
        ax.annotate("", xy=(x1, Y_CENTER), xytext=(x0, Y_CENTER),
                    arrowprops=dict(arrowstyle="-|>", color="#555",
                                    lw=0.9, mutation_scale=7),
                    zorder=2)

    # Legend
    ml_patch  = mpatches.Patch(facecolor="#dce8f5", edgecolor="#2471a3",
                                linewidth=1.2, label="ML stage")
    det_patch = mpatches.Patch(facecolor="#fdebd0", edgecolor="#ca6f1e",
                                linewidth=1.2, label="Detection / scoring")
    alr_patch = mpatches.Patch(facecolor="#fadbd8", edgecolor="#c0392b",
                                linewidth=1.2, label="Alert / characterization")
    ax.legend(handles=[ml_patch, det_patch, alr_patch],
              loc="lower center", ncol=3,
              bbox_to_anchor=(0.5, -0.04),
              frameon=True, framealpha=0.9,
              edgecolor="#bbb", fontsize=6.5)

    fig.savefig(FIG_DIR / "fig1_pipeline.pdf")
    fig.savefig(FIG_DIR / "fig1_pipeline.png")
    plt.close(fig)
    print("  fig1_pipeline.pdf  ✓")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — CUSUM Temporal Trace
# ══════════════════════════════════════════════════════════════════════════════

def _simulate_cusum_series(seed=7, n=24, outbreak_start=14, outbreak_mag=0.55):
    """
    Produce a realistic risk-score series with a slow-rising outbreak.
    Returns (weeks, scores, cusum_upper, threshold_first_alert_week).
    """
    rng = np.random.default_rng(seed)
    scores = rng.normal(0.18, 0.06, n)
    scores[outbreak_start:] += np.linspace(0, outbreak_mag, n - outbreak_start)
    scores = np.clip(scores, 0, 1)

    mu  = float(np.mean(scores[:outbreak_start]))
    sig = float(np.std(scores[:outbreak_start], ddof=1)) or 1.0
    z   = (scores - mu) / sig

    k, h = 0.5, 4.0
    cusum = np.zeros(n)
    for i in range(1, n):
        cusum[i] = max(0.0, cusum[i - 1] + z[i] - k)

    alert_week = next((i for i in range(n) if cusum[i] >= h), None)
    thresh_week = next((i for i in range(n) if scores[i] >= 0.6), None)

    return np.arange(n), scores, cusum, h, alert_week, thresh_week


def fig2_cusum():
    weeks, scores, cusum, h, cusum_alert, thresh_alert = _simulate_cusum_series()

    fig, axes = plt.subplots(2, 1, figsize=(ONE_COL * 1.85, 2.5),
                             sharex=True, gridspec_kw={"hspace": 0.12})

    # ── top: risk score ───────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(weeks, scores, color=COLORS["score"], lw=1.3, label="Risk score")
    ax.axhline(0.60, color=COLORS["threshold"], lw=0.9,
               ls="--", label="Threshold (0.6)")
    ax.fill_between(weeks, scores, alpha=0.15, color=COLORS["score"])
    if thresh_alert is not None:
        ax.axvline(thresh_alert, color=COLORS["threshold"],
                   lw=0.8, ls=":", alpha=0.7)
    if cusum_alert is not None:
        ax.axvline(cusum_alert, color=COLORS["cusum"], lw=1.0, ls="-", alpha=0.8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Risk score")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax.grid(True, axis="y")
    ax.set_title("CUSUM Early Warning: Simulated Outbreak Site", fontsize=8)

    # ── bottom: CUSUM statistic ───────────────────────────────────────────────
    ax = axes[1]
    ax.plot(weeks, cusum, color=COLORS["cusum"], lw=1.3, label="CUSUM $C_t$")
    ax.axhline(h, color=COLORS["alert"], lw=0.9, ls="--",
               label=f"Decision threshold ($h={h:.0f}$)")
    ax.fill_between(weeks, cusum, alpha=0.15, color=COLORS["cusum"])

    if cusum_alert is not None:
        ax.axvline(cusum_alert, color=COLORS["cusum"],
                   lw=1.0, ls="-", alpha=0.8, label="CUSUM alert")
        ax.scatter([cusum_alert], [cusum[cusum_alert]], color=COLORS["alert"],
                   s=28, zorder=5)

        # Lead-time annotation
        if thresh_alert is not None and thresh_alert > cusum_alert:
            lead = thresh_alert - cusum_alert
            mid = (cusum_alert + thresh_alert) / 2
            ax.annotate(
                f"{lead} wk lead",
                xy=(cusum_alert, h + 0.5),
                xytext=(mid, h + 1.8),
                fontsize=6.5,
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=0.7),
                ha="center", color="#333",
            )
            ax.axvspan(cusum_alert, thresh_alert, alpha=0.10,
                       color=COLORS["cusum"], label="Lead time")

    ax.set_ylim(-0.3, max(cusum) * 1.25 + 1)
    ax.set_xlabel("Surveillance week")
    ax.set_ylabel("CUSUM $C_t$")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, ncol=2)
    ax.grid(True, axis="y")

    fig.savefig(FIG_DIR / "fig2_cusum.pdf")
    fig.savefig(FIG_DIR / "fig2_cusum.png")
    plt.close(fig)
    print("  fig2_cusum.pdf     ✓")


# ══════════════════════════════════════════════════════════════════════════════
#  RUN BENCHMARK  (real pipeline, synthetic data)
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark_all(alert_threshold=0.6, seed=42):
    """
    Run PathogenIQ + three baselines on all synthetic scenarios.
    Returns a dict with per-scenario metrics for each method.
    """
    import pandas as pd
    from pathogeniq.benchmark.synthetic import SCENARIOS, generate_scenario
    from pathogeniq.benchmark.metrics import compute_metrics
    from pathogeniq.ingestion.reader import SampleSet, Sample, filter_taxa
    from pathogeniq.novelty.detector import compute_novelty_scores
    from pathogeniq.scoring.risk import score_all_samples, score_sample, PATHOGEN_DB

    results = {}  # {scenario_name: {method: ScenarioMetrics}}

    for scenario in SCENARIOS:
        dataset = generate_scenario(scenario, seed=seed)
        mat = dataset.count_matrix
        rel = mat.div(mat.sum(axis=0), axis=1).fillna(0)
        samples = [Sample(name=col) for col in mat.columns]
        sampleset = SampleSet(samples=samples, taxa_matrix=mat, relative_abundance=rel)
        sampleset = filter_taxa(sampleset, min_prevalence=0.05, min_total_reads=10)

        gt = dataset.ground_truth

        # ── Baseline 1: simple threshold ──────────────────────────────────────
        def _threshold_scores(ss, thresh=0.01):
            from pathogeniq.scoring.risk import RiskScore
            out = []
            for s in ss.samples:
                ra = ss.relative_abundance.get(s.name, pd.Series(dtype=float))
                hit = any(
                    ra.get(t, 0) >= thresh
                    for t in ra.index
                    if t.split()[0] in PATHOGEN_DB
                )
                out.append(RiskScore(
                    sample_name=s.name,
                    score=0.65 if hit else 0.1,
                    level="HIGH" if hit else "LOW",
                    detected_pathogens=[], community_signal=0,
                    novelty_signal=0,
                ))
            return out

        # ── Baseline 2: abundance-only (no community, no novelty) ────────────
        def _abundance_scores(ss):
            out = []
            for s in ss.samples:
                ra = ss.relative_abundance.get(s.name, pd.Series(dtype=float))
                rs = score_sample(s.name, ra,
                                  community_label=None,
                                  pathogen_community_labels=set(),
                                  novelty_score=0.0)
                out.append(rs)
            out.sort(key=lambda x: x.score, reverse=True)
            return out

        # ── Baseline 3: SBM-only (community flag, no direct detection) ───────
        def _sbm_only_scores(ss):
            try:
                from pathogeniq.community.graph import build_cooccurrence_graph
                from pathogeniq.community.sbm import fit_sbm
                from pathogeniq.scoring.risk import identify_pathogen_communities, RiskScore
                n_taxa = ss.taxa_matrix.shape[0]
                if n_taxa < 3:
                    raise ValueError("too few taxa")
                adj, taxa_names, _ = build_cooccurrence_graph(ss)
                sbm = fit_sbm(adj, taxa_names=taxa_names,
                              max_k=min(6, n_taxa - 1), n_init=3)
                path_comms = identify_pathogen_communities(sbm, ss.relative_abundance)
                out = []
                for s in ss.samples:
                    ra = ss.relative_abundance.get(s.name, pd.Series(dtype=float))
                    present = ra[ra > 0].index.tolist()
                    t2i = {t: i for i, t in enumerate(sbm.taxa_names)}
                    votes = [sbm.labels[t2i[t]] for t in present if t in t2i]
                    comm = max(set(votes), key=votes.count) if votes else None
                    comm_sig = 1.0 if comm in path_comms else 0.0
                    score = 0.65 * comm_sig  # community context only
                    from pathogeniq.scoring.risk import _score_level
                    out.append(RiskScore(
                        sample_name=s.name,
                        score=score,
                        level=_score_level(score),
                        detected_pathogens=[], community_signal=comm_sig,
                        novelty_signal=0,
                    ))
                out.sort(key=lambda x: x.score, reverse=True)
                return out
            except Exception:
                from pathogeniq.scoring.risk import RiskScore, _score_level
                return [RiskScore(s.name, 0.1, "LOW", [], 0, 0)
                        for s in ss.samples]

        # ── PathogenIQ full ───────────────────────────────────────────────────
        novelty = compute_novelty_scores(sampleset)
        sbm_res = None
        try:
            from pathogeniq.community.graph import build_cooccurrence_graph
            from pathogeniq.community.sbm import fit_sbm
            n_taxa = sampleset.taxa_matrix.shape[0]
            if n_taxa >= 3:
                adj, taxa_names, _ = build_cooccurrence_graph(sampleset)
                sbm_res = fit_sbm(adj, taxa_names=taxa_names,
                                  max_k=min(8, n_taxa - 1), n_init=3)
        except Exception:
            pass

        method_scores = {
            "Threshold":      _threshold_scores(sampleset),
            "Abundance-only": _abundance_scores(sampleset),
            "SBM-only":       _sbm_only_scores(sampleset),
            "PathogenIQ":     score_all_samples(sampleset, sbm_result=sbm_res,
                                               novelty_scores=novelty,
                                               alert_threshold=alert_threshold),
        }

        scenario_results = {}
        for method, scores in method_scores.items():
            m = compute_metrics(scenario.name, scenario.description,
                                scores, gt, alert_threshold)
            scenario_results[method] = m
        results[scenario.name] = scenario_results
        print(f"    {scenario.name:30s}  PathogenIQ F1={scenario_results['PathogenIQ'].f1}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Per-scenario F1 bar chart
# ══════════════════════════════════════════════════════════════════════════════

SCENARIO_LABELS = {
    "negative_controls":  "Negative\ncontrol",
    "low_contamination":  "Low\ncontam.",
    "high_cholera":       "High\ncholera",
    "critical_yersinia":  "Critical\nYersinia",
    "multi_pathogen":     "Multi-\npathogen",
    "novel_taxon":        "Novel\ntaxon",
}


def fig3_benchmark(results: dict):
    methods   = ["Threshold", "Abundance-only", "SBM-only", "PathogenIQ"]
    pal       = [COLORS["baseline1"], COLORS["baseline2"],
                 COLORS["baseline3"], COLORS["pathogeniq"]]
    scenarios = list(results.keys())

    n_sc  = len(scenarios)
    n_m   = len(methods)
    x     = np.arange(n_sc)
    width = 0.18

    fig, ax = plt.subplots(figsize=(TWO_COL, 2.3))

    for mi, (method, color) in enumerate(zip(methods, pal)):
        f1_vals = []
        for sc in scenarios:
            f1 = results[sc][method].f1
            f1_vals.append(f1 if f1 is not None else 0.0)
        offset = (mi - (n_m - 1) / 2) * width
        bars = ax.bar(x + offset, f1_vals, width,
                      color=color, alpha=0.85,
                      label=method, edgecolor="white", linewidth=0.4)
        # Annotate best method bars
        if method == "PathogenIQ":
            for rect, val in zip(bars, f1_vals):
                if val > 0:
                    ax.text(rect.get_x() + rect.get_width() / 2,
                            rect.get_height() + 0.015,
                            f"{val:.2f}", ha="center", va="bottom",
                            fontsize=5.5, color=COLORS["pathogeniq"],
                            fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios],
                       fontsize=7)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("F1 Score")
    ax.set_title("Detection Performance: PathogenIQ vs. Baselines", fontsize=8)
    ax.legend(ncol=4, frameon=True, framealpha=0.9, edgecolor="#bbb",
              loc="upper center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(True, axis="y", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(FIG_DIR / "fig3_benchmark.pdf")
    fig.savefig(FIG_DIR / "fig3_benchmark.png")
    plt.close(fig)
    print("  fig3_benchmark.pdf ✓")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Risk score distribution by scenario
# ══════════════════════════════════════════════════════════════════════════════

def fig4_risk_dist(results: dict, alert_threshold=0.6):
    """
    Violin plot of PathogenIQ risk score distributions, coloured by scenario type.
    We need to re-run the pipeline to get per-sample scores.
    """
    import pandas as pd
    from pathogeniq.benchmark.synthetic import SCENARIOS, generate_scenario
    from pathogeniq.ingestion.reader import SampleSet, Sample, filter_taxa
    from pathogeniq.novelty.detector import compute_novelty_scores
    from pathogeniq.scoring.risk import score_all_samples

    SEED = 42
    data_pos, data_neg, labels = [], [], []

    for scenario in SCENARIOS:
        dataset = generate_scenario(scenario, seed=SEED)
        mat = dataset.count_matrix
        rel = mat.div(mat.sum(axis=0), axis=1).fillna(0)
        samples_list = [Sample(name=c) for c in mat.columns]
        ss = SampleSet(samples=samples_list,
                       taxa_matrix=mat, relative_abundance=rel)
        ss = filter_taxa(ss, min_prevalence=0.05, min_total_reads=10)
        novelty = compute_novelty_scores(ss)
        risk_scores = score_all_samples(ss, novelty_scores=novelty)
        gt = dataset.ground_truth

        pos_scores = [r.score for r in risk_scores
                      if gt.loc[r.sample_name, "is_contaminated"]]
        neg_scores = [r.score for r in risk_scores
                      if not gt.loc[r.sample_name, "is_contaminated"]]
        data_pos.append(pos_scores if pos_scores else [0.0])
        data_neg.append(neg_scores if neg_scores else [0.0])
        labels.append(SCENARIO_LABELS.get(scenario.name, scenario.name))

    n = len(labels)
    fig, ax = plt.subplots(figsize=(TWO_COL, 2.5))

    positions_neg = np.arange(n) * 1.6 - 0.28
    positions_pos = np.arange(n) * 1.6 + 0.28

    # Remove empty violins gracefully
    def _violin(data, positions, color, label):
        valid = [(d, p) for d, p in zip(data, positions) if len(d) > 1]
        if not valid:
            return
        d_list, p_list = zip(*valid)
        parts = ax.violinplot(d_list, positions=p_list,
                              widths=0.45, showmedians=True,
                              showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.55)
            pc.set_edgecolor(color)
        for key in ["cmedians", "cmins", "cmaxes", "cbars"]:
            if key in parts:
                parts[key].set_color(color)
                parts[key].set_linewidth(0.9)
        # Proxy for legend
        ax.scatter([], [], color=color, alpha=0.7, s=30, label=label, marker="s")

    _violin(data_neg, positions_neg, COLORS["low"],      "Negative samples")
    _violin(data_pos, positions_pos, COLORS["critical"],  "Contaminated samples")

    ax.axhline(alert_threshold, color="#555", lw=0.9, ls="--",
               label=f"Alert threshold ({alert_threshold})")

    ax.set_xticks(np.arange(n) * 1.6)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Risk score")
    ax.set_title("Risk Score Distributions by Benchmark Scenario", fontsize=8)
    ax.legend(ncol=3, frameon=True, framealpha=0.9, edgecolor="#bbb",
              loc="upper left", fontsize=7)
    ax.grid(True, axis="y", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(FIG_DIR / "fig4_risk_dist.pdf")
    fig.savefig(FIG_DIR / "fig4_risk_dist.png")
    plt.close(fig)
    print("  fig4_risk_dist.pdf ✓")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Ablation: signal contribution (bar chart)
# ══════════════════════════════════════════════════════════════════════════════

def fig5_ablation(results: dict):
    """
    Show aggregate F1 when removing each signal one at a time.
    Re-runs with modified scoring configs.
    """
    import pandas as pd
    from pathogeniq.benchmark.synthetic import SCENARIOS, generate_scenario
    from pathogeniq.ingestion.reader import SampleSet, Sample, filter_taxa
    from pathogeniq.novelty.detector import compute_novelty_scores
    from pathogeniq.scoring.risk import score_all_samples, score_sample, _score_level, RiskScore
    from pathogeniq.benchmark.metrics import compute_metrics

    SEED = 42

    configs = {
        "Full PathogenIQ":           dict(alpha=0.50, beta=0.25, gamma=0.25, direct=True,  temporal=True),
        "w/o Community ($\\beta=0$)": dict(alpha=0.75, beta=0.00, gamma=0.25, direct=True,  temporal=True),
        "w/o Novelty ($\\gamma=0$)":  dict(alpha=0.75, beta=0.25, gamma=0.00, direct=True,  temporal=True),
        "w/o Direct detection":      dict(alpha=0.50, beta=0.25, gamma=0.25, direct=False, temporal=True),
    }

    agg_f1 = {k: [] for k in configs}

    for scenario in SCENARIOS:
        dataset = generate_scenario(scenario, seed=SEED)
        mat = dataset.count_matrix
        rel = mat.div(mat.sum(axis=0), axis=1).fillna(0)
        samples_list = [Sample(name=c) for c in mat.columns]
        ss = SampleSet(samples=samples_list, taxa_matrix=mat, relative_abundance=rel)
        ss = filter_taxa(ss, min_prevalence=0.05, min_total_reads=10)
        novelty = compute_novelty_scores(ss)

        try:
            from pathogeniq.community.graph import build_cooccurrence_graph
            from pathogeniq.community.sbm import fit_sbm
            from pathogeniq.scoring.risk import identify_pathogen_communities
            n_taxa = ss.taxa_matrix.shape[0]
            adj, taxa_names, _ = build_cooccurrence_graph(ss)
            sbm_res = fit_sbm(adj, taxa_names=taxa_names,
                              max_k=min(8, n_taxa - 1), n_init=3)
            path_comms = identify_pathogen_communities(sbm_res, ss.relative_abundance)
        except Exception:
            sbm_res, path_comms = None, set()

        for cfg_name, cfg in configs.items():
            scores_out = []
            for s in ss.samples:
                ra = ss.relative_abundance.get(s.name, pd.Series(dtype=float))
                nov = novelty.get(s.name, 0.0) if cfg["gamma"] > 0 else 0.0

                comm_label = None
                if sbm_res is not None and cfg["beta"] > 0:
                    present = ra[ra > 0].index.tolist()
                    t2i = {t: i for i, t in enumerate(sbm_res.taxa_names)}
                    votes = [sbm_res.labels[t2i[t]] for t in present if t in t2i]
                    if votes:
                        comm_label = max(set(votes), key=votes.count)

                w = {"abundance": cfg["alpha"],
                     "community": cfg["beta"],
                     "novelty":   cfg["gamma"]}
                rs = score_sample(
                    s.name, ra,
                    community_label=comm_label if cfg["beta"] > 0 else None,
                    pathogen_community_labels=path_comms if cfg["beta"] > 0 else set(),
                    novelty_score=nov,
                    weights=w,
                )
                if not cfg["direct"]:
                    # Rebuild without direct pathway (override with composite only)
                    import numpy as _np
                    from pathogeniq.scoring.risk import PATHOGEN_DB
                    ab_score = min(1.0, sum(
                        PATHOGEN_DB[t.split()[0]]["risk_weight"] * float(v)
                        for t, v in ra.items()
                        if t.split()[0] in PATHOGEN_DB and v > 0
                    ))
                    comm_s = 1.0 if comm_label in path_comms else 0.0
                    composite = (cfg["alpha"] * ab_score +
                                 cfg["beta"] * comm_s +
                                 cfg["gamma"] * nov)
                    score = float(_np.clip(composite, 0, 1))
                    rs = RiskScore(
                        sample_name=s.name,
                        score=score,
                        level=_score_level(score),
                        detected_pathogens=rs.detected_pathogens,
                        community_signal=comm_s,
                        novelty_signal=nov,
                    )
                scores_out.append(rs)

            m = compute_metrics(scenario.name, scenario.description,
                                scores_out, dataset.ground_truth, 0.6)
            if m.f1 is not None:
                agg_f1[cfg_name].append(m.f1)

    means = {k: (np.mean(v) if v else 0.0) for k, v in agg_f1.items()}
    full_f1 = means["Full PathogenIQ"]

    labels   = list(means.keys())
    vals     = list(means.values())
    deltas   = [v - full_f1 for v in vals]
    bar_cols = [COLORS["pathogeniq"] if i == 0 else "#c0392b" for i in range(len(labels))]

    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.0),
                             gridspec_kw={"width_ratios": [1.4, 1]})

    # Left: absolute F1
    ax = axes[0]
    bars = ax.barh(labels[::-1], vals[::-1],
                   color=bar_cols[::-1], alpha=0.85,
                   edgecolor="white", linewidth=0.4, height=0.55)
    for rect, val in zip(bars, vals[::-1]):
        ax.text(val + 0.005, rect.get_y() + rect.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=6.5)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Macro-avg F1")
    ax.set_title("Signal Contribution (Ablation)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="x", zorder=0)

    # Right: delta from full
    ax = axes[1]
    d_labels  = labels[1:][::-1]
    d_vals    = deltas[1:][::-1]
    d_colors  = ["#c0392b" if d < 0 else "#27ae60" for d in d_vals]
    ax.barh(d_labels, d_vals, color=d_colors, alpha=0.85,
            edgecolor="white", linewidth=0.4, height=0.55)
    for i, (dv, dl) in enumerate(zip(d_vals, d_labels)):
        ax.text(dv - 0.002 if dv < 0 else dv + 0.002,
                i, f"{dv:+.3f}", va="center",
                ha="right" if dv < 0 else "left", fontsize=6.5)
    ax.axvline(0, color="#555", lw=0.7)
    ax.set_xlabel("$\\Delta$ F1 vs. full")
    ax.set_title("Marginal Contribution", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="x", zorder=0)

    fig.tight_layout(pad=0.5)
    fig.savefig(FIG_DIR / "fig5_ablation.pdf")
    fig.savefig(FIG_DIR / "fig5_ablation.png")
    plt.close(fig)
    print("  fig5_ablation.pdf  ✓")

    return means, deltas, labels


# ══════════════════════════════════════════════════════════════════════════════
#  GENERATE LaTeX TABLE SNIPPET
# ══════════════════════════════════════════════════════════════════════════════

def write_latex_tables(results: dict, ablation_means: dict,
                       ablation_deltas: list, ablation_labels: list,
                       out_path: Path):
    methods = ["Threshold", "Abundance-only", "SBM-only", "PathogenIQ"]

    def _agg(method):
        f1s  = [results[sc][method].f1  or 0 for sc in results]
        sens = [results[sc][method].sensitivity or 0 for sc in results
                if results[sc][method].sensitivity is not None]
        spec = [results[sc][method].specificity or 0 for sc in results
                if results[sc][method].specificity is not None]
        far  = [results[sc][method].fp /
                max(results[sc][method].n_samples, 1) for sc in results]
        return (np.mean(sens) if sens else 0,
                np.mean(spec) if spec else 0,
                np.mean(f1s),
                np.mean(far))

    lines = []
    lines.append("% ── TABLE 1: Main detection results ────────────────────")
    lines.append("% (Replace \\TODO{} in the paper with these values)\n")

    for method in methods:
        s, sp, f1, far = _agg(method)
        bold = method == "PathogenIQ"
        row = (f"{'\\textbf{' if bold else ''}{method}{'}' if bold else ''}"
               f" & {'\\textbf{' if bold else ''}{s*100:.1f}\\%{'}' if bold else ''}"
               f" & {'\\textbf{' if bold else ''}{sp*100:.1f}\\%{'}' if bold else ''}"
               f" & {'\\textbf{' if bold else ''}{f1*100:.1f}\\%{'}' if bold else ''}"
               f" & {'\\textbf{' if bold else ''}{far*100:.1f}\\%{'}' if bold else ''}"
               r" \\")
        lines.append(row)

    lines.append("\n% ── TABLE 2: Synthetic benchmark F1 per scenario ──────")
    SCENARIO_NAMES = {
        "negative_controls": "Negative control",
        "low_contamination":  "Low contamination",
        "high_cholera":       "High contamination",
        "critical_yersinia":  "Critical pathogen spike",
        "multi_pathogen":     "Multi-pathogen community",
        "novel_taxon":        "Novel taxon intrusion",
    }
    for sc in results:
        piq  = results[sc]["PathogenIQ"].f1
        ab   = results[sc]["Abundance-only"].f1
        piq_s  = f"{piq:.3f}" if piq  is not None else "—"
        ab_s   = f"{ab:.3f}"  if ab   is not None else "—"
        name = SCENARIO_NAMES.get(sc, sc)
        lines.append(f"{name} & {piq_s} & {ab_s} \\\\")

    lines.append("\n% ── TABLE 3: Ablation signal contribution ───────────")
    for i, label in enumerate(ablation_labels):
        f1_val = ablation_means.get(label, 0.0)
        delta  = ablation_deltas[i]
        delta_s = f"$-${abs(delta):.3f}" if delta < 0 else f"$+${delta:.3f}"
        if i == 0:
            lines.append(f"Full PathogenIQ & {f1_val:.3f} & — \\\\")
        else:
            lines.append(f"{label} & {f1_val:.3f} & {delta_s} \\\\")

    # Summary line
    piq_agg = _agg("PathogenIQ")
    bl_agg  = _agg("Abundance-only")
    lines.append(f"\n% ── ABSTRACT numbers ──────────────────────────────")
    lines.append(f"% PathogenIQ:    Sens={piq_agg[0]*100:.1f}%  Spec={piq_agg[1]*100:.1f}%  F1={piq_agg[2]*100:.1f}%")
    lines.append(f"% Abundance-only:Sens={bl_agg[0]*100:.1f}%  Spec={bl_agg[1]*100:.1f}%  F1={bl_agg[2]*100:.1f}%")
    lines.append(f"% F1 improvement: {(piq_agg[2]-bl_agg[2])*100:.1f}pp over Abundance-only")

    out_path.write_text("\n".join(lines))
    print(f"  table_results.tex  ✓  ({out_path})")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate PathogenIQ paper figures")
    parser.add_argument("--report", default=None,
                        help="Path to real report.json for overlay on synthetic results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    print("\nPathogenIQ — Paper Figure Generator")
    print("=" * 42)
    print(f"Output directory: {FIG_DIR}\n")

    print("[1/6] Pipeline architecture diagram ...")
    fig1_pipeline()

    print("[2/6] CUSUM temporal trace ...")
    fig2_cusum()

    print("[3/6] Running benchmark (all scenarios × all methods) ...")
    results = run_benchmark_all(alert_threshold=args.threshold, seed=args.seed)

    print("[4/6] Benchmark F1 bar chart ...")
    fig3_benchmark(results)

    print("[5/6] Risk score distribution violins ...")
    fig4_risk_dist(results, alert_threshold=args.threshold)

    print("[6/6] Ablation figure ...")
    ablation_means, ablation_deltas, ablation_labels = fig5_ablation(results)

    print("[7/7] Writing LaTeX table numbers ...")
    write_latex_tables(results, ablation_means, ablation_deltas, ablation_labels,
                       Path(__file__).parent / "table_results.tex")

    print(f"\nDone. Files written to {FIG_DIR}/")
    print("Include in LaTeX with:")
    print(r"  \includegraphics[width=\columnwidth]{figures/fig1_pipeline}")
    print(r"  \includegraphics[width=\columnwidth]{figures/fig2_cusum}")
    print(r"  \includegraphics[width=\linewidth]{figures/fig3_benchmark}")
    print(r"  \includegraphics[width=\columnwidth]{figures/fig4_risk_dist}")
    print(r"  \includegraphics[width=\linewidth]{figures/fig5_ablation}")


if __name__ == "__main__":
    main()
