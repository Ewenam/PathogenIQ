#!/usr/bin/env python3
"""
paper/generate_figures.py
=========================
Run from the project root:

    python paper/generate_figures.py                        # synthetic benchmark only
    python paper/generate_figures.py --report reports/report.json   # + real data overlay

Outputs (all in paper/figures/):
    fig2_cusum.pdf          — CUSUM temporal trace with lead-time annotation
    fig3_benchmark.pdf      — per-scenario F1 bar chart (synthetic-only fallback)
    fig3_sbm_network.pdf    — K=2 SBM co-occurrence network (requires --report)
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
import matplotlib.patheffects as pe

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── IEEE-style global settings ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":         13,
    "axes.titlesize":    13,
    "axes.labelsize":    13,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   11,
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
#  FIGURE 3 (REAL) — SBM co-occurrence network (standalone)
# ══════════════════════════════════════════════════════════════════════════════

def fig3_sbm_network(report_data: dict):
    """
    Force-directed (Fruchterman-Reingold) layout of the real-data
    co-occurrence network: taxa connected by at least one
    FDR-significant Spearman edge (|rho|>=0.45, BH-corrected
    alpha=0.05), colour-coded by SBM community membership (K=2).
    Solid grey edges = positive co-occurrence, dashed red = negative.
    Node size ~ sqrt(mean abundance). Isolated taxa (no significant
    edges) are omitted from the layout and noted in the caption.
    """
    try:
        import networkx as nx
    except ImportError:
        print("  [fig3] networkx not installed — skipping network figure")
        return

    nodes = report_data["network"]["nodes"]
    edges = report_data["network"]["edges"]
    n_total_taxa = len(nodes)
    n_samples = len(report_data.get("samples", []))

    max_ab = max(n.get("mean_abundance", 0.0) for n in nodes)

    def node_size(ab):
        return 30 + 220 * (ab / max_ab) ** 0.5

    G = nx.Graph()
    for n in nodes:
        G.add_node(n["id"], community=n["community"],
                   mean_abundance=n.get("mean_abundance", 0.0))
    for e in edges:
        G.add_edge(e["source"], e["target"], rho=e["rho"], weight=abs(e["rho"]))

    connected = [n_id for n_id in G.nodes() if G.degree(n_id) > 0]
    H = G.subgraph(connected)
    n_isolated = n_total_taxa - len(connected)

    total_by_community = {0: 0, 1: 0}
    for n in nodes:
        total_by_community[n["community"]] += 1
    connected_by_community = {0: [], 1: []}
    for n_id in connected:
        connected_by_community[H.nodes[n_id]["community"]].append(n_id)

    # The smaller community is drawn in the foreground/highlight colour.
    small_comm, large_comm = (0, 1) if total_by_community[0] <= total_by_community[1] else (1, 0)
    small_ids, large_ids = connected_by_community[small_comm], connected_by_community[large_comm]

    # Force-directed layout over the connected subgraph only (isolated
    # taxa with no significant edges are omitted). Each connected
    # component is laid out independently (so small components aren't
    # squeezed by the large one) and then placed in a grid: the largest
    # component spans the top, the remaining small components form a
    # row below it. Positions are finally rescaled to [-1, 1]^2.
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    pos: dict = {}
    if len(comps) == 1:
        pos = nx.spring_layout(H, seed=10, k=0.7, iterations=300, weight="weight")
    else:
        largest, rest = comps[0], comps[1:]
        boxes = [(0.0, 10.0, 3.2, 10.0)]
        w = 10.0 / len(rest)
        for i in range(len(rest)):
            boxes.append((i * w, (i + 1) * w, 0.0, 2.2))
        for comp, (x0, x1, y0, y1) in zip([largest] + rest, boxes):
            sub = H.subgraph(comp)
            n_sub = len(comp)
            if n_sub == 1:
                sub_pos = {next(iter(comp)): np.array([0.5, 0.5])}
            else:
                k = 1.8 / (n_sub ** 0.5)
                sub_pos = nx.spring_layout(sub, seed=10, k=k, iterations=300,
                                            weight="weight")
            xs = [p[0] for p in sub_pos.values()]
            ys = [p[1] for p in sub_pos.values()]
            xr = (max(xs) - min(xs)) or 1.0
            yr = (max(ys) - min(ys)) or 1.0
            xmin, ymin = min(xs), min(ys)
            for node, (x, y) in sub_pos.items():
                pos[node] = np.array([
                    x0 + (x - xmin) / xr * (x1 - x0),
                    y0 + (y - ymin) / yr * (y1 - y0),
                ])
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    xr = (max(all_x) - min(all_x)) or 1.0
    yr = (max(all_y) - min(all_y)) or 1.0
    xmin, ymin = min(all_x), min(all_y)
    pos = {n_id: np.array([(p[0] - xmin) / xr * 2 - 1, (p[1] - ymin) / yr * 2 - 1])
           for n_id, p in pos.items()}

    # ── Draw ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(ONE_COL * 2.1, ONE_COL * 2.1))

    nx.draw_networkx_nodes(H, pos, nodelist=large_ids, ax=ax,
                           node_color=COLORS["baseline1"],
                           node_size=[node_size(H.nodes[n]["mean_abundance"]) for n in large_ids],
                           alpha=0.8, linewidths=0.4, edgecolors="white")

    pos_edges = [(u, v) for u, v, d in H.edges(data=True) if d["rho"] > 0]
    neg_edges = [(u, v) for u, v, d in H.edges(data=True) if d["rho"] < 0]
    nx.draw_networkx_edges(H, pos, edgelist=pos_edges, ax=ax,
                           edge_color="#888", alpha=0.85, width=1.0)
    nx.draw_networkx_edges(H, pos, edgelist=neg_edges, ax=ax,
                           edge_color="#e74c3c", alpha=0.85, width=1.0,
                           style="dashed")

    nx.draw_networkx_nodes(H, pos, nodelist=small_ids, ax=ax,
                           node_color=COLORS["pathogeniq"],
                           node_size=[node_size(H.nodes[n]["mean_abundance"]) for n in small_ids],
                           alpha=0.95, linewidths=0.6, edgecolors="white")

    # Labels: every connected taxon. Each label is offset away from its
    # neighbour(s) — for degree>=2 nodes, perpendicular to the line
    # through the first two neighbours; for degree-1 (leaf) nodes, away
    # from the single neighbour. Horizontal/vertical alignment follows
    # the dominant offset axis so the text extends away from the node
    # rather than centring on top of it.
    for n_id in H.nodes():
        x, y = pos[n_id]
        neighbors = list(H.neighbors(n_id))
        if len(neighbors) >= 2:
            p0, p1 = pos[neighbors[0]], pos[neighbors[1]]
            dx, dy = -(p1[1] - p0[1]), (p1[0] - p0[0])
            if dy < 0:
                dx, dy = -dx, -dy
        else:
            nbr = pos[neighbors[0]]
            dx, dy = x - nbr[0], y - nbr[1]
        norm = (dx * dx + dy * dy) ** 0.5
        if norm < 1e-8:
            dx, dy, norm = 0.0, 1.0, 1.0
        sz = node_size(H.nodes[n_id]["mean_abundance"])
        off = 0.025 + 0.0003 * sz
        lx, ly = x + dx / norm * off, y + dy / norm * off
        if abs(dx) > abs(dy):
            ha, va = ("left" if dx > 0 else "right"), "center"
        else:
            ha, va = "center", ("bottom" if dy > 0 else "top")
        ax.text(lx, ly, n_id, fontsize=10, ha=ha, va=va)

    small_handle = plt.Line2D([0], [0], marker="o", linestyle="None",
                              markerfacecolor=COLORS["pathogeniq"],
                              markeredgecolor="white", markersize=7,
                              label=f"Community {small_comm} ($n$={total_by_community[small_comm]})")
    large_handle = plt.Line2D([0], [0], marker="o", linestyle="None",
                              markerfacecolor=COLORS["baseline1"],
                              markeredgecolor="white", markersize=7,
                              label=f"Community {large_comm} ($n$={total_by_community[large_comm]})")
    pos_line = plt.Line2D([0], [0], color="#888", linewidth=1.0, label="Positive co-occ.")
    neg_line = plt.Line2D([0], [0], color="#e74c3c", linewidth=1.0,
                          linestyle="dashed", label="Negative co-occ.")
    ax.legend(handles=[small_handle, large_handle, pos_line, neg_line],
              loc="lower center", ncol=2, fontsize=9.5,
              frameon=True, framealpha=0.85, edgecolor="#ccc",
              handlelength=1.2, borderpad=0.4, labelspacing=0.4,
              handletextpad=0.4, columnspacing=0.8,
              bbox_to_anchor=(0.5, -0.10))

    ax.set_title(
        rf"K=2 SBM Co-occurrence Network ($N$={n_samples}, {n_total_taxa} taxa)",
        fontsize=13)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.margins(0.16)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.10)
    fig.savefig(FIG_DIR / "fig3_sbm_network.pdf")
    fig.savefig(FIG_DIR / "fig3_sbm_network.png")
    plt.close(fig)
    print(f"  fig3_sbm_network.pdf ✓  (K=2 SBM co-occurrence network, "
          f"{len(connected)}/{n_total_taxa} connected taxa shown)")


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

    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.4),
                             gridspec_kw={"width_ratios": [1.4, 1]})

    # Left: absolute F1
    ax = axes[0]
    bars = ax.barh(labels[::-1], vals[::-1],
                   color=bar_cols[::-1], alpha=0.85,
                   edgecolor="white", linewidth=0.4, height=0.55)
    for rect, val in zip(bars, vals[::-1]):
        ax.text(val + 0.005, rect.get_y() + rect.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Macro-avg F1")
    ax.set_title("Signal Contribution (Ablation)", fontsize=13)
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
                ha="right" if dv < 0 else "left", fontsize=10)
    ax.axvline(0, color="#555", lw=0.7)
    ax.set_xlabel("$\\Delta$ F1 vs. full")
    ax.set_title("Marginal Contribution", fontsize=13)
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
        f1s  = [results[sc][method].f1 for sc in results
                if results[sc][method].f1 is not None]
        sens = [results[sc][method].sensitivity for sc in results
                if results[sc][method].sensitivity is not None]
        spec = [results[sc][method].specificity for sc in results
                if results[sc][method].specificity is not None]
        far  = [results[sc][method].fp /
                max(results[sc][method].n_samples, 1) for sc in results
                if results[sc][method].fp > 0 or results[sc][method].tn > 0]
        return (np.mean(sens) if sens else 0,
                np.mean(spec) if spec else 0,
                np.mean(f1s) if f1s else 0,
                np.mean(far) if far else 0)

    lines = []
    lines.append("% ── TABLE 1: Main detection results ────────────────────")
    lines.append("% (Replace \\TODO{} in the paper with these values)\n")

    for method in methods:
        s, sp, f1, far = _agg(method)
        bold = method == "PathogenIQ"
        b0 = "\\textbf{" if bold else ""
        b1 = "}" if bold else ""
        row = (f"{b0}{method}{b1}"
               f" & {b0}{s*100:.1f}\\%{b1}"
               f" & {b0}{sp*100:.1f}\\%{b1}"
               f" & {b0}{f1*100:.1f}\\%{b1}"
               f" & {b0}{far*100:.1f}\\%{b1}"
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
#  SBM BOOTSTRAP STABILITY (Tier-3 analysis — numbers only, no figure)
# ══════════════════════════════════════════════════════════════════════════════

def sbm_bootstrap_stability(report_data: dict, B: int = 100, seed: int = 42) -> dict:
    """
    Bootstrap co-assignment stability for the SBM community structure.

    For each pair of taxa (i, j), we track whether they are assigned to the
    same community in each bootstrap resample.  Within-community pairs (same
    original community) and between-community pairs (different communities)
    are compared to characterise how stable the K=3 partition is.

    Returns a dict with summary statistics that can be cited in the paper.
    """
    import pandas as pd
    from pathogeniq.community.graph import build_cooccurrence_graph
    from pathogeniq.community.sbm import fit_sbm
    from pathogeniq.ingestion.reader import SampleSet, Sample, filter_taxa

    ab        = report_data["abundance_matrix"]
    taxa_all  = ab["taxa"]
    sample_names = ab["samples"]
    X         = np.array(ab["values"])          # (n_taxa, n_samples)
    N         = X.shape[1]
    T         = len(taxa_all)

    # Original community assignments
    nodes = {n["id"]: n["community"] for n in report_data["network"]["nodes"]}
    orig_comm = np.array([nodes.get(t, -1) for t in taxa_all])

    # Keep only taxa that appear in the network
    mask      = orig_comm >= 0
    X_net     = X[mask]
    taxa_net  = [t for t, m in zip(taxa_all, mask) if m]
    orig_net  = orig_comm[mask]
    n_taxa    = len(taxa_net)

    # Pairwise co-assignment accumulators
    # co_count[i, j] = # runs where taxa i and j were co-assigned
    # pair_count[i, j] = # runs where both taxa were present (in same graph)
    co_count    = np.zeros((n_taxa, n_taxa), dtype=np.int32)
    pair_count  = np.zeros((n_taxa, n_taxa), dtype=np.int32)
    k_hist      = []   # K selected in each run
    valid_runs  = 0

    rng = np.random.default_rng(seed)
    t2i = {t: i for i, t in enumerate(taxa_net)}

    for b in range(B):
        idx    = rng.integers(0, N, size=N)
        X_boot = X_net[:, idx]

        df = pd.DataFrame(X_boot, index=taxa_net,
                          columns=[f"s{i}" for i in range(N)])
        samples_list = [Sample(name=c) for c in df.columns]
        rel = df.div(df.sum(axis=0), axis=1).fillna(0)
        ss  = SampleSet(samples=samples_list, taxa_matrix=df, relative_abundance=rel)
        ss  = filter_taxa(ss, min_prevalence=0.05, min_total_reads=0)

        try:
            adj, tn, _ = build_cooccurrence_graph(ss)
            if len(tn) < 5:
                continue
            sbm = fit_sbm(adj, taxa_names=tn, max_k=min(8, len(tn) - 1), n_init=3)
        except Exception:
            continue

        k_hist.append(len(set(sbm.labels)))
        valid_runs += 1

        # Map bootstrap taxa back to global index
        boot_label = {taxon: sbm.labels[i]
                      for i, taxon in enumerate(sbm.taxa_names)}

        present = [t for t in taxa_net if t in boot_label]
        for i_t, ti in enumerate(present):
            gi = t2i[ti]
            li = boot_label[ti]
            for j_t, tj in enumerate(present):
                if j_t <= i_t:
                    continue
                gj = t2i[tj]
                lj = boot_label[tj]
                pair_count[gi, gj] += 1
                pair_count[gj, gi] += 1
                if li == lj:
                    co_count[gi, gj] += 1
                    co_count[gj, gi] += 1

    if valid_runs == 0:
        print("  [bootstrap] no valid runs — skipping")
        return {}

    # Co-assignment probability matrix (NaN where pair never co-appeared)
    with np.errstate(invalid="ignore"):
        prob = np.where(pair_count > 0, co_count / pair_count, np.nan)

    # Within-community pairs vs. between-community pairs
    within_probs, between_probs = [], []
    for i in range(n_taxa):
        for j in range(i + 1, n_taxa):
            p = prob[i, j]
            if np.isnan(p):
                continue
            if orig_net[i] == orig_net[j]:
                within_probs.append(p)
            else:
                between_probs.append(p)

    k_counts = {k: k_hist.count(k) for k in sorted(set(k_hist))}
    modal_k  = max(k_counts, key=k_counts.get)

    results = {
        "valid_runs":    valid_runs,
        "within_mean":   float(np.mean(within_probs))  if within_probs  else 0.0,
        "within_std":    float(np.std(within_probs))   if within_probs  else 0.0,
        "between_mean":  float(np.mean(between_probs)) if between_probs else 0.0,
        "between_std":   float(np.std(between_probs))  if between_probs else 0.0,
        "k_distribution": k_counts,
        "modal_k":        modal_k,
        "k_modal_frac":   k_counts[modal_k] / valid_runs,
        "prob_matrix":    prob,
        "taxa_net":       taxa_net,
        "orig_comm":      orig_net.tolist(),
    }

    print(f"\n  SBM Bootstrap Stability (B={valid_runs} valid runs):")
    print(f"    Within-community co-assignment:  {results['within_mean']*100:.1f}% ± {results['within_std']*100:.1f}%")
    print(f"    Between-community co-assignment: {results['between_mean']*100:.1f}% ± {results['between_std']*100:.1f}%")
    print(f"    K distribution: {k_counts}  (modal K={modal_k}, {results['k_modal_frac']*100:.0f}% of runs)")

    return results


def fig6_sbm_consensus(boot_results: dict):
    """
    Consensus matrix heatmap — shows co-assignment probability for all taxon pairs,
    taxa ordered by original community.  Stable communities appear as dense blocks.
    """
    if not boot_results:
        return

    prob      = boot_results["prob_matrix"]
    orig_comm = np.array(boot_results["orig_comm"])
    taxa_net  = boot_results["taxa_net"]

    # Sort taxa by community
    order   = np.argsort(orig_comm, kind="stable")
    prob_s  = prob[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(ONE_COL * 1.85, ONE_COL * 1.85))

    im = ax.imshow(prob_s, aspect="auto", cmap="Blues",
                   vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Co-assignment probability")

    # Community boundary lines
    comm_sorted = orig_comm[order]
    for k in range(1, 3):
        boundary = np.where(comm_sorted == k)[0]
        if len(boundary):
            b = boundary[0] - 0.5
            ax.axhline(b, color="white", lw=1.2)
            ax.axvline(b, color="white", lw=1.2)

    # Community labels on axes
    comm_sizes = [np.sum(orig_comm == k) for k in range(3)]
    comm_mids  = np.cumsum([0] + comm_sizes[:-1]) + np.array(comm_sizes) / 2
    ax.set_xticks(comm_mids)
    ax.set_xticklabels([f"C{k}\n(n={s})" for k, s in enumerate(comm_sizes)], fontsize=6)
    ax.set_yticks(comm_mids)
    ax.set_yticklabels([f"C{k}" for k in range(3)], fontsize=6)

    B = boot_results["valid_runs"]
    wm = boot_results["within_mean"] * 100
    bm = boot_results["between_mean"] * 100
    ax.set_title(
        rf"SBM Consensus Matrix ($B={B}$ bootstrap resamples)",
        fontsize=7.5)
    fig.text(0.5, -0.02,
             rf"Within-community co-assignment: {wm:.1f}%; "
             rf"between-community: {bm:.1f}%",
             ha="center", fontsize=6, color="#555", style="italic")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_consensus.pdf")
    fig.savefig(FIG_DIR / "fig6_consensus.png")
    plt.close(fig)
    print("  fig6_consensus.pdf ✓  (SBM bootstrap consensus matrix)")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 (REAL) — CUSUM trace from real wastewater time series
# ══════════════════════════════════════════════════════════════════════════════

def fig2_cusum_real(report_data: dict):
    """
    Plot the CUSUM trace over the 60-run site-207 chronological series
    (recalibrated risk scores; ~2 runs/week over ~30 weeks). The CUSUM
    statistic transiently exceeds the decision boundary around week 20
    before receding — a single-week excursion rather than a sustained
    shift, so no persistent alert fires.
    """
    samples = report_data["samples"]
    if "run_index" not in samples[0]:
        print("  [fig2] report has no 'run_index' field — falling back to simulated.")
        fig2_cusum()
        return

    series   = sorted(samples, key=lambda s: s["run_index"])
    dates    = [s["collection_date"] for s in series]
    scores   = np.array([s["score"] for s in series])
    cusums   = np.array([s.get("cusum", 0.0) for s in series])
    alerts   = np.array([s.get("cusum_alert", False) for s in series])
    h        = report_data.get("temporal", {}).get("h", 4.0)
    x        = np.arange(len(series))

    fig, axes = plt.subplots(2, 1, figsize=(ONE_COL * 1.5, 2.7),
                             sharex=True, gridspec_kw={"hspace": 0.12})

    # ── top: risk score ───────────────────────────────────────────────────────
    ax = axes[0]
    ax.bar(x, scores, color=COLORS["score"], alpha=0.75, width=0.7)
    ax.axhline(0.60, color=COLORS["threshold"], lw=0.9, ls="--")
    for i, (a, sc) in enumerate(zip(alerts, scores)):
        if a:
            ax.bar(x[i], sc, color=COLORS["alert"], alpha=0.9, width=0.7)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Risk score")
    ax.grid(True, axis="y", zorder=0)
    ax.set_title("CUSUM Trace (Site-207, 60 runs)", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)

    # ── bottom: CUSUM statistic ───────────────────────────────────────────────
    ax = axes[1]
    ax.bar(x, cusums, color=COLORS["cusum"], alpha=0.75, width=0.7, zorder=2)
    ax.plot(x, cusums, color=COLORS["high"], lw=1.3, marker="o", markersize=3,
            zorder=4, label="$C_t$ trend")
    ax.axhline(h, color=COLORS["alert"], lw=0.9, ls="--")
    alert_idx = [i for i, a in enumerate(alerts) if a]
    if alert_idx:
        ax.bar([x[i] for i in alert_idx], [cusums[i] for i in alert_idx],
               color=COLORS["alert"], alpha=0.9, width=0.7, zorder=3)
        ax.scatter([x[i] for i in alert_idx], [cusums[i] for i in alert_idx],
                   color=COLORS["alert"], s=28, zorder=5)
    ax.legend(loc="upper left", fontsize=10, frameon=True, framealpha=0.85,
              edgecolor="#ccc", handlelength=1.4, borderpad=0.3)
    ax.set_ylim(0, max(h * 1.5, cusums.max() * 1.1))
    ax.set_ylabel("$C_t$")
    ax.set_xticks(x[::8])
    ax.set_xticklabels([d[2:] for d in dates[::8]], fontsize=9.5,
                       rotation=30, ha="right")
    ax.set_xlabel("Collection date (YY-MM-DD)")
    ax.grid(True, axis="y", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(bottom=0.14, top=0.92, hspace=0.12)
    fig.savefig(FIG_DIR / "fig2_cusum.pdf")
    fig.savefig(FIG_DIR / "fig2_cusum.png")
    plt.close(fig)
    print("  fig2_cusum.pdf  ✓  (real data — site-207 ~30-week, 60-run series)")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 (REAL) — Multi-signal risk breakdown across real samples
# ══════════════════════════════════════════════════════════════════════════════

def fig4_risk_dist_real(report_data: dict, alert_threshold: float = 0.6):
    """
    Signal distribution figure for 27 real wastewater samples.

    Community signal is binary (0 or 1), so it is shown as a bar chart
    (n active / n total) rather than a violin, which would misleadingly
    suggest a continuous distribution.  Abundance, novelty, and composite
    signals are genuinely continuous and shown as violins with jitter.
    """
    samples    = report_data["samples"]
    n_samples  = len(samples)

    breakdown  = [s.get("breakdown", {}) for s in samples]
    ab_vals    = [b.get("abundance_score",  0.0) for b in breakdown]
    comm_vals  = [b.get("community_signal", 0.0) for b in breakdown]
    nov_vals   = [b.get("novelty_signal",   0.0) for b in breakdown]
    total_vals = [s["score"] for s in samples]

    n_active = sum(1 for v in comm_vals if v > 0)

    # Positions: 1=Abundance, 2=Community(bar), 3=Novelty, 4=Composite
    violin_groups = [
        (1, ab_vals,    "Abundance\n(α=0.50)",  "#2980b9"),
        (3, nov_vals,   "Novelty\n(γ=0.25)",    "#27ae60"),
        (4, total_vals, "Composite\nScore",     "#c0392b"),
    ]
    COMM_COLOR = "#e67e22"

    fig, ax = plt.subplots(figsize=(ONE_COL * 1.5, 2.5))

    # ── Violin plots for continuous signals ──────────────────────────────────
    for pos, data, label, color in violin_groups:
        parts = ax.violinplot([data], positions=[pos],
                              widths=0.55, showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.55)
            pc.set_edgecolor(color)
        for key in ["cmedians", "cmins", "cmaxes", "cbars"]:
            if key in parts:
                parts[key].set_color(color)
                parts[key].set_linewidth(0.9)
        rng = np.random.default_rng(pos)
        jitter = rng.uniform(-0.1, 0.1, len(data))
        ax.scatter(np.full(len(data), pos) + jitter, data,
                   color=color, alpha=0.6, s=14, zorder=4)

    # ── Community signal: bar chart (binary 0/1) ──────────────────────────────
    # Show fraction active as a filled bar, fraction inactive as grey remainder
    ax.bar([2], [n_active / n_samples], width=0.55,
           color=COMM_COLOR, alpha=0.75, zorder=3,
           label=f"Active ({n_active}/{n_samples})")
    ax.bar([2], [1 - n_active / n_samples], width=0.55,
           bottom=[n_active / n_samples],
           color="#ddd", alpha=0.75, zorder=3,
           label=f"Inactive ({n_samples - n_active}/{n_samples})")
    ax.text(2, 0.5,
            f"{n_active}/{n_samples}\n({n_active/n_samples*100:.0f}%)",
            ha="center", va="center", fontsize=11,
            color="white", fontweight="bold", zorder=5)

    ax.axhline(alert_threshold, color="#555", lw=0.9, ls="--")
    ax.text(4.42, alert_threshold + 0.03, f"θ={alert_threshold}",
            fontsize=10, color="#555", va="bottom")

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(
        ["Abundance\n(α=0.50)", "Community\n(β=0.25)",
         "Novelty\n(γ=0.25)", "Composite\nScore"],
        fontsize=11)
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel("Signal value")
    ax.set_title(
        f"Risk Signals: {n_samples} Real Samples",
        fontsize=12)
    ax.grid(True, axis="y", zorder=0, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_risk_dist.pdf")
    fig.savefig(FIG_DIR / "fig4_risk_dist.png")
    plt.close(fig)
    print("  fig4_risk_dist.pdf  ✓  (real data — mixed violin/bar, binary community)")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate PathogenIQ paper figures")
    parser.add_argument("--report", default=None,
                        help="Path to real report.json (enables real-data fig2 + fig4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    report_data = None
    if args.report:
        import json as _json
        report_data = _json.load(open(args.report))

    print("\nPathogenIQ — Paper Figure Generator")
    print("=" * 42)
    print(f"Output directory: {FIG_DIR}")
    if report_data:
        print(f"Real data: {args.report} ({len(report_data['samples'])} samples)\n")
    else:
        print("No --report provided: fig2/fig4 will use simulated data.\n")

    print("[1/5] CUSUM temporal trace ...")
    if report_data:
        fig2_cusum_real(report_data)
    else:
        fig2_cusum()

    print("[2/5] Running benchmark (all scenarios × all methods) ...")
    results = run_benchmark_all(alert_threshold=args.threshold, seed=args.seed)

    print("[3/5] SBM network (real data) / F1 bar chart (synthetic fallback) ...")
    if report_data:
        fig3_sbm_network(report_data)
    else:
        fig3_benchmark(results)

    print("[4/5] Risk score distribution ...")
    if report_data:
        fig4_risk_dist_real(report_data, alert_threshold=args.threshold)
    else:
        fig4_risk_dist(results, alert_threshold=args.threshold)

    print("[5/5] Ablation figure ...")
    ablation_means, ablation_deltas, ablation_labels = fig5_ablation(results)

    print("[7/7] Writing LaTeX table numbers ...")
    write_latex_tables(results, ablation_means, ablation_deltas, ablation_labels,
                       Path(__file__).parent / "table_results.tex")

    if report_data:
        print("[8/8] SBM bootstrap stability (B=100 resamples) ...")
        boot_results = sbm_bootstrap_stability(report_data, B=100, seed=42)
        if boot_results:
            fig6_sbm_consensus(boot_results)

    fig3_name = "fig3_sbm_network" if report_data else "fig3_benchmark"
    print(f"\nDone. Files written to {FIG_DIR}/")
    print("Include in LaTeX with:")
    print(r"  \includegraphics[width=\columnwidth]{figures/fig2_cusum}")
    print(rf"  \includegraphics[width=\linewidth]{{figures/{fig3_name}}}")
    print(r"  \includegraphics[width=\columnwidth]{figures/fig4_risk_dist}")
    print(r"  \includegraphics[width=\linewidth]{figures/fig5_ablation}")


if __name__ == "__main__":
    main()
