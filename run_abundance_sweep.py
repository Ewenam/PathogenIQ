"""
run_abundance_sweep.py
======================
PATH C: Characterize WHERE the SBM community signal changes detection decisions.

This is the honest, complete version of the cryptic co-occurrence experiment.
Instead of reporting a single (potentially cherry-picked) abundance band, we
sweep the per-pathogen abundance across a range and, at each level, measure the
alert rate of:
    - abundance-only      (no community signal)
    - full PathogenIQ     (with the bug-fixed community signal)

EXPECTED SHAPE OF THE RESULT
----------------------------
  - LOW abundance:        neither fires (both ~0)        -> nothing to detect
  - BORDERLINE abundance: full PathogenIQ fires, abundance-only does NOT
                          <-- this is the community signal's value region
  - HIGH abundance:       both fire (~1)                 -> abundance alone suffices

The width and location of the borderline gap is the paper's headline figure:
it shows the community signal converts sub-threshold co-colonized samples into
alerts, precisely characterized rather than asserted.

Uses the bug-fixed risk.py (presence-based community assignment). Everything
downstream is your real pipeline.

RUN
---
    source venv/bin/activate
    python run_abundance_sweep.py

Outputs:
    - prints a RESULTS TABLE (copy it back)
    - writes abundance_sweep.csv  (raw numbers for the paper)
    - writes abundance_sweep.json (same, structured)
Then run the companion plot script:  python plot_abundance_sweep.py
"""
from __future__ import annotations

import sys
import json
import csv
import numpy as np
import pandas as pd

try:
    from pathogeniq.benchmark.synthetic import BACKGROUND_TAXA, SyntheticDataset, Scenario
    from pathogeniq.ingestion.reader import SampleSet, Sample, filter_taxa
    from pathogeniq.community.graph import build_cooccurrence_graph
    from pathogeniq.community.sbm import fit_sbm
    from pathogeniq.novelty.detector import compute_novelty_scores
    from pathogeniq.scoring.risk import (
        score_sample, identify_pathogen_communities,
    )
except ImportError as e:
    print("ERROR importing pathogeniq. Activate venv first: source venv/bin/activate")
    print("Original error:", e)
    sys.exit(1)


# ── Sweep configuration ───────────────────────────────────────────────────────
CRYPTIC_PATHOGENS = ["Salmonella", "Escherichia", "Listeria", "Shigella"]
# Per-pathogen MEAN abundance levels to sweep (band is +/- HALF_WIDTH around each)
SWEEP_LEVELS = [0.010, 0.020, 0.030, 0.040, 0.050, 0.065, 0.080]
HALF_WIDTH = 0.005
CORR_STRENGTH = 0.90
N_SAMPLES = 30
N_SEEDS = 10
ALERT_THRESHOLD = 0.6
COMM_PRESENCE_FRAC = 0.15
TOTAL_READS_RANGE = (100_000, 500_000)


def build_dataset(level: float, seed: int):
    rng = np.random.default_rng(seed)
    n = N_SAMPLES
    bg = BACKGROUND_TAXA
    names = [f"WW_site_{i+1:02d}" for i in range(n)]
    lo, hi = max(0.0, level - HALF_WIDTH), level + HALF_WIDTH
    bg_prop = rng.dirichlet(np.ones(len(bg)) * 2.0, size=n)
    total = rng.integers(*TOTAL_READS_RANGE, size=n)
    latent = rng.uniform(0, 1, size=n)
    span = hi - lo
    pathogens = {}
    for g in CRYPTIC_PATHOGENS:
        indep = rng.uniform(0, 1, size=n)
        mix = CORR_STRENGTH * latent + (1 - CORR_STRENGTH) * indep
        pathogens[g] = lo + span * mix
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
    sc = Scenario(name="sweep", description="sweep", n_samples=n,
                  contamination={g: (lo, hi) for g in CRYPTIC_PATHOGENS},
                  expected_alert=True)
    return SyntheticDataset(scenario=sc, count_matrix=cm, ground_truth=gt,
                            pathogen_fractions={})


def pipeline(ds, use_sbm):
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
        except Exception:
            pass
    return ss, sbm


def alert_rate(ds, use_community):
    """Fraction of samples flagged (score >= threshold)."""
    if use_community:
        ss, sbm = pipeline(ds, use_sbm=True)
        nov = compute_novelty_scores(ss)
        pcomm = identify_pathogen_communities(sbm, ss.relative_abundance) if sbm else set()
        idx = {t: i for i, t in enumerate(sbm.taxa_names)} if sbm else {}
        flagged = 0
        for s in ss.samples:
            ra = ss.relative_abundance.get(s.name, pd.Series(dtype=float))
            present = ra[ra > 0].index.tolist()
            label = None
            if sbm and pcomm and present:
                pres_lbls = [sbm.labels[idx[t]] for t in present if t in idx]
                if pres_lbls and np.mean([l in pcomm for l in pres_lbls]) >= COMM_PRESENCE_FRAC:
                    label = next(iter(pcomm))
            rs = score_sample(s.name, ra, community_label=label,
                              pathogen_community_labels=pcomm,
                              novelty_score=nov.get(s.name, 0.0))
            if rs.score >= ALERT_THRESHOLD:
                flagged += 1
        return flagged / len(ss.samples) if ss.samples else 0.0
    else:
        ss, _ = pipeline(ds, use_sbm=False)
        nov = compute_novelty_scores(ss)
        flagged = 0
        for s in ss.samples:
            ra = ss.relative_abundance.get(s.name, pd.Series(dtype=float))
            rs = score_sample(s.name, ra, community_label=None,
                              pathogen_community_labels=set(),
                              novelty_score=nov.get(s.name, 0.0))
            if rs.score >= ALERT_THRESHOLD:
                flagged += 1
        return flagged / len(ss.samples) if ss.samples else 0.0


def ci(x):
    x = np.array(x, float)
    return (x.mean(), *np.percentile(x, [2.5, 97.5])) if len(x) > 1 else (x.mean(),)*3


def main():
    print("=" * 70)
    print("ABUNDANCE SWEEP — community signal decision characterization")
    print("=" * 70)
    rows = []
    for level in SWEEP_LEVELS:
        abun_rates, full_rates = [], []
        for s in range(N_SEEDS):
            ds = build_dataset(level, seed=4000 + s)
            abun_rates.append(alert_rate(ds, use_community=False))
            full_rates.append(alert_rate(ds, use_community=True))
        a = ci(abun_rates); f = ci(full_rates)
        rows.append({
            "level": level,
            "abundance_only_alert_rate": round(a[0], 4),
            "abundance_only_lo": round(a[1], 4),
            "abundance_only_hi": round(a[2], 4),
            "full_alert_rate": round(f[0], 4),
            "full_lo": round(f[1], 4),
            "full_hi": round(f[2], 4),
            "community_gain": round(f[0] - a[0], 4),
        })

    # Write outputs
    with open("abundance_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open("abundance_sweep.json", "w") as fh:
        json.dump({"config": {
            "pathogens": CRYPTIC_PATHOGENS, "corr_strength": CORR_STRENGTH,
            "n_samples": N_SAMPLES, "n_seeds": N_SEEDS,
            "alert_threshold": ALERT_THRESHOLD,
            "comm_presence_frac": COMM_PRESENCE_FRAC,
            "half_width": HALF_WIDTH,
        }, "rows": rows}, fh, indent=2)

    print("=" * 70)
    print("RESULTS TABLE  <<< COPY EVERYTHING BELOW AND PASTE IT BACK >>>")
    print("=" * 70)
    print(f"sweep: pathogens={CRYPTIC_PATHOGENS} corr={CORR_STRENGTH} "
          f"N={N_SAMPLES}/seed seeds={N_SEEDS} thr={ALERT_THRESHOLD}")
    print(f"{'level':>7} | {'abund_only':>12} | {'full_PIQ':>12} | {'gain':>7}")
    print("-" * 70)
    max_gain_level = None; max_gain = -1
    for r in rows:
        print(f"{r['level']:>7.3f} | "
              f"{r['abundance_only_alert_rate']:>5.2f} "
              f"[{r['abundance_only_lo']:.2f},{r['abundance_only_hi']:.2f}] | "
              f"{r['full_alert_rate']:>5.2f} "
              f"[{r['full_lo']:.2f},{r['full_hi']:.2f}] | "
              f"{r['community_gain']:>+7.2f}")
        if r["community_gain"] > max_gain:
            max_gain = r["community_gain"]; max_gain_level = r["level"]
    print("-" * 70)
    print(f"PEAK community gain: {max_gain:+.2f} at per-pathogen level "
          f"{max_gain_level:.3f}")
    print("INTERPRETATION:")
    if max_gain > 0.15:
        print("  >>> STRONG borderline window: community signal converts sub-threshold")
        print("  >>> co-colonized samples into alerts that abundance-only misses.")
        print("  >>> This is the headline figure. Run plot_abundance_sweep.py.")
    elif max_gain > 0.05:
        print("  >>> Modest but real borderline gain. Honest, reportable.")
        print("  >>> Consider raising CORR_STRENGTH to 0.95 to widen the window.")
    else:
        print("  >>> Weak separation. The community signal is a minor modifier here;")
        print("  >>> report it as such (Path B framing) rather than a detector.")
    print("=" * 70)
    print("Wrote abundance_sweep.csv and abundance_sweep.json")


if __name__ == "__main__":
    main()
