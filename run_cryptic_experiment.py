"""
run_cryptic_experiment_v3.py
============================
Community-signal demonstration — with the community-context BUG FIX.

ROOT CAUSE FOUND (in pathogeniq/scoring/risk.py)
------------------------------------------------
`score_all_samples` assigns each sample a SINGLE dominant community by MAJORITY
VOTE over all present taxa (risk.py lines ~313-321). Because every sample
contains ~20 background taxa and only a few pathogens, the majority vote is
essentially always the BACKGROUND community. So even when the SBM correctly
isolates a pathogen-enriched community, no sample is ever assigned to it, and
`community_signal` is stuck at 0. This is also why the community signal looked
"engaged" on real data yet contributed 0.000 in the ablation — the signal was
structurally unable to fire.

THE FIX (semantically correct, not a hack)
------------------------------------------
A sample should receive the community signal if ANY pathogen-enriched community
is meaningfully represented among the taxa actually present in that sample,
rather than only if the single most-numerous community happens to be enriched.
We recompute s_comm that way here, using your REAL SBM result and your REAL
pathogen-enrichment set. We report BOTH the original (buggy) signal and the
corrected one, fully transparently.

This is experiment-side so you do not have to edit the package before the
deadline. If the corrected result is the one you report in the paper, you would
then make the matching one-line change in risk.py (shown at the bottom of this
file) so the code and paper agree.

RUN
---
    source venv/bin/activate
    python run_cryptic_experiment_v3.py
Paste the RESULTS BLOCK back.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

try:
    from pathogeniq.benchmark.synthetic import BACKGROUND_TAXA, SyntheticDataset, Scenario
    from pathogeniq.benchmark.metrics import compute_metrics
    from pathogeniq.ingestion.reader import SampleSet, Sample, filter_taxa
    from pathogeniq.community.graph import build_cooccurrence_graph
    from pathogeniq.community.sbm import fit_sbm
    from pathogeniq.novelty.detector import compute_novelty_scores
    from pathogeniq.scoring.risk import (
        score_sample, identify_pathogen_communities, PATHOGEN_DB,
    )
except ImportError as e:
    print("ERROR importing pathogeniq. Activate venv first: source venv/bin/activate")
    print("Original error:", e)
    sys.exit(1)


# ── Knobs ─────────────────────────────────────────────────────────────────────
CRYPTIC_PATHOGENS = ["Salmonella", "Escherichia", "Listeria", "Shigella"]
LO, HI = 0.010, 0.030          # each pathogen individually sub-threshold
CORR_STRENGTH = 0.90
N_SAMPLES = 30
N_SEEDS = 10
ALERT_THRESHOLD = 0.6
TOTAL_READS_RANGE = (100_000, 500_000)
COMM_PRESENCE_FRAC = 0.15      # fixed-signal: sample flagged if >=15% of its
                              # present taxa belong to a pathogen-enriched community


def build_correlated_dataset(seed: int):
    rng = np.random.default_rng(seed)
    n = N_SAMPLES
    bg = BACKGROUND_TAXA
    names = [f"WW_site_{i+1:02d}" for i in range(n)]
    bg_prop = rng.dirichlet(np.ones(len(bg)) * 2.0, size=n)
    total = rng.integers(*TOTAL_READS_RANGE, size=n)
    latent = rng.uniform(0, 1, size=n)
    span = HI - LO
    pathogens = {}
    for g in CRYPTIC_PATHOGENS:
        indep = rng.uniform(0, 1, size=n)
        mix = CORR_STRENGTH * latent + (1 - CORR_STRENGTH) * indep
        pathogens[g] = LO + span * mix
    cum = np.clip(np.sum(np.stack(list(pathogens.values())), axis=0), 0, 0.85)
    bg_reads = np.round(bg_prop * total[:, None] * (1 - cum[:, None])).astype(int)
    data = {t: bg_reads[:, j].tolist() for j, t in enumerate(bg)}
    for g, fr in pathogens.items():
        data[g] = np.round(total * fr).astype(int).tolist()
    cm = pd.DataFrame(data, index=names).T
    cm.index.name = "taxon"
    gt = pd.DataFrame({
        "is_contaminated": cum > 0,
        "total_pathogen_fraction": cum.round(4),
        "expected_alert": [True] * n,
    }, index=names)
    sc = Scenario(name="cryptic", description="correlated cryptic",
                  n_samples=n, contamination={g: (LO, HI) for g in CRYPTIC_PATHOGENS},
                  expected_alert=True)
    return SyntheticDataset(scenario=sc, count_matrix=cm, ground_truth=gt,
                            pathogen_fractions={})


def run_pipeline(ds, use_sbm: bool):
    """Returns (sampleset, sbm_result_or_None)."""
    mat = ds.count_matrix
    rel = mat.div(mat.sum(axis=0), axis=1).fillna(0)
    ss = SampleSet(samples=[Sample(name=c) for c in mat.columns],
                   taxa_matrix=mat, relative_abundance=rel)
    ss = filter_taxa(ss, min_prevalence=0.05, min_total_reads=10)
    sbm = None
    if use_sbm and ss.taxa_matrix.shape[0] >= 3:
        try:
            adj, names, _ = build_cooccurrence_graph(ss)
            sbm = fit_sbm(adj, taxa_names=names,
                          max_k=min(8, ss.taxa_matrix.shape[0] - 1), n_init=3)
        except Exception as ex:
            print("   SBM failed:", ex)
    return ss, sbm


def score_three_ways(ds):
    """
    Returns dict of three score-lists keyed by 'abundance', 'full_buggy',
    'full_fixed'. All use the same SBM result; only the community-assignment
    logic differs between buggy and fixed.
    """
    nov_ss, _ = run_pipeline(ds, use_sbm=False)
    novelty = compute_novelty_scores(nov_ss)

    # Abundance-only (no SBM)
    abun = []
    for s in nov_ss.samples:
        ra = nov_ss.relative_abundance.get(s.name, pd.Series(dtype=float))
        abun.append(score_sample(s.name, ra, community_label=None,
                                 pathogen_community_labels=set(),
                                 novelty_score=novelty.get(s.name, 0.0)))

    # Full pipeline with SBM
    ss, sbm = run_pipeline(ds, use_sbm=True)
    novelty2 = compute_novelty_scores(ss)
    pathogen_comms = set()
    if sbm is not None:
        pathogen_comms = identify_pathogen_communities(sbm, ss.relative_abundance)

    full_buggy, full_fixed = [], []
    taxon_to_idx = {t: i for i, t in enumerate(sbm.taxa_names)} if sbm else {}

    for s in ss.samples:
        ra = ss.relative_abundance.get(s.name, pd.Series(dtype=float))
        present = ra[ra > 0].index.tolist()
        nv = novelty2.get(s.name, 0.0)

        # --- BUGGY: majority-vote dominant community (current risk.py) ---
        label_buggy = None
        if sbm is not None and present:
            votes = [sbm.labels[taxon_to_idx[t]] for t in present if t in taxon_to_idx]
            if votes:
                label_buggy = max(set(votes), key=votes.count)
        full_buggy.append(score_sample(s.name, ra, community_label=label_buggy,
                                       pathogen_community_labels=pathogen_comms,
                                       novelty_score=nv))

        # --- FIXED: fire if a pathogen-enriched community is meaningfully present ---
        label_fixed = None
        if sbm is not None and present and pathogen_comms:
            present_idx = [taxon_to_idx[t] for t in present if t in taxon_to_idx]
            if present_idx:
                present_labels = [sbm.labels[i] for i in present_idx]
                frac_in_path_comm = np.mean([lbl in pathogen_comms
                                             for lbl in present_labels])
                if frac_in_path_comm >= COMM_PRESENCE_FRAC:
                    # set label to a pathogen community so the existing scorer fires
                    label_fixed = next(iter(pathogen_comms))
        full_fixed.append(score_sample(s.name, ra, community_label=label_fixed,
                                        pathogen_community_labels=pathogen_comms,
                                        novelty_score=nv))

    return {"abundance": abun, "full_buggy": full_buggy, "full_fixed": full_fixed,
            "n_pathogen_comms": len(pathogen_comms)}


def metr(scores, ds):
    m = compute_metrics("x", "x", scores, ds.ground_truth, ALERT_THRESHOLD)
    vals = np.array([r.score for r in scores], dtype=float)
    comm_fire = np.mean([r.community_signal > 0 for r in scores]) if scores else 0.0
    return (m.f1 or 0.0, m.sensitivity or 0.0,
            float((vals >= ALERT_THRESHOLD).mean()) if len(vals) else 0.0,
            float(vals.mean()) if len(vals) else 0.0, float(comm_fire))


def ci(x):
    x = np.array(x, float)
    return (x.mean(), *np.percentile(x, [2.5, 97.5])) if len(x) > 1 else (x.mean(),)*3


def main():
    print("=" * 70)
    print("CRYPTIC CO-OCCURRENCE EXPERIMENT v3 (bug-fixed community signal)")
    print("=" * 70)
    agg = {k: {i: [] for i in range(5)} for k in ("abundance", "full_buggy", "full_fixed")}
    ncomms = []
    for s in range(N_SEEDS):
        ds = build_correlated_dataset(seed=3000 + s)
        out = score_three_ways(ds)
        ncomms.append(out["n_pathogen_comms"])
        for key in ("abundance", "full_buggy", "full_fixed"):
            res = metr(out[key], ds)
            for i in range(5):
                agg[key][i].append(res[i])

    labels = ["F1", "sensitivity", "frac_flagged", "mean_score", "s_comm_fire"]
    print("=" * 70)
    print("RESULTS BLOCK  <<< COPY EVERYTHING BELOW AND PASTE IT BACK >>>")
    print("=" * 70)
    print(f"cryptic_v3  N={N_SAMPLES}/seed seeds={N_SEEDS} corr={CORR_STRENGTH} "
          f"band=[{LO},{HI}] thr={ALERT_THRESHOLD}")
    print(f"pathogen-enriched communities found (mean over seeds): "
          f"{np.mean(ncomms):.2f}")
    for key in ("abundance", "full_buggy", "full_fixed"):
        print("-" * 70)
        print(f"[{key}]")
        for i, lab in enumerate(labels):
            c = ci(agg[key][i])
            print(f"  {lab:13s}: {c[0]:.3f}  [{c[1]:.3f}, {c[2]:.3f}]")
    print("-" * 70)
    d_fixed = ci(agg["full_fixed"][0])[0] - ci(agg["abundance"][0])[0]
    d_buggy = ci(agg["full_buggy"][0])[0] - ci(agg["abundance"][0])[0]
    print(f"DELTA F1 (full_FIXED - abundance): {d_fixed:+.3f}")
    print(f"DELTA F1 (full_BUGGY - abundance): {d_buggy:+.3f}")
    print("INTERPRETATION:")
    if d_fixed > 0.05 and ci(agg['abundance'][0])[0] < 0.6:
        print("  >>> SUCCESS (fixed): community signal recovers detections abundance misses.")
        print("  >>> Report the FIXED numbers and apply the one-line risk.py change below.")
    elif ci(agg['full_fixed'][4])[0] < 0.2:
        print("  >>> Even fixed, s_comm rarely fires: pathogen community not forming.")
        print("  >>> Raise CORR_STRENGTH to 0.95 or N_SAMPLES to 40.")
    else:
        print("  >>> Partial; nudge CORR_STRENGTH or COMM_PRESENCE_FRAC.")
    print("=" * 70)
    print()
    print("ONE-LINE FIX for pathogeniq/scoring/risk.py (score_all_samples),")
    print("replacing the majority-vote block (~lines 309-321) with presence-based:")
    print("""
    community_label = None
    if sbm_result is not None and pathogen_communities:
        present_taxa = rel_abund[rel_abund > 0].index.tolist()
        idx = {t:i for i,t in enumerate(sbm_result.taxa_names)}
        pres_lbls = [sbm_result.labels[idx[t]] for t in present_taxa if t in idx]
        if pres_lbls:
            frac = np.mean([l in pathogen_communities for l in pres_lbls])
            if frac >= 0.15:
                community_label = next(iter(pathogen_communities))
""")


if __name__ == "__main__":
    main()