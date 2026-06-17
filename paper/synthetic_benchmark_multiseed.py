#!/usr/bin/env python3
"""
synthetic_benchmark_multiseed.py
=================================
Runs the synthetic benchmark (pathogeniq.benchmark.synthetic, with Poisson
sampling noise enabled by default) across multiple seeds and reports
mean +/- 95% CI for sensitivity/specificity/F1/FAR per scenario and per
method (Threshold, Abundance-only, SBM-only, PathogenIQ), plus the
macro-averaged Table-I numbers.

Usage:
    python paper/synthetic_benchmark_multiseed.py [--seeds 10] [--base-seed 42]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper"))

from generate_figures import run_benchmark_all  # noqa: E402

METHODS = ["Threshold", "Abundance-only", "SBM-only", "PathogenIQ"]


def ci95(x):
    x = np.array([v for v in x if v is not None], dtype=float)
    if len(x) == 0:
        return None, None, None
    if len(x) == 1:
        return float(x[0]), float(x[0]), float(x[0])
    return float(x.mean()), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.6)
    args = p.parse_args()

    # per_scenario[scenario][method][metric] -> list over seeds
    per_scenario = {}
    macro = {m: {"sens": [], "spec": [], "f1": [], "far": []} for m in METHODS}

    for s in range(args.seeds):
        seed = args.base_seed + s * 1000
        print(f"--- seed {seed} ---")
        results = run_benchmark_all(alert_threshold=args.threshold, seed=seed)

        for sc, methods in results.items():
            per_scenario.setdefault(sc, {m: {"f1": [], "sens": [], "spec": []} for m in METHODS})
            for m in METHODS:
                metrics = methods[m]
                per_scenario[sc][m]["f1"].append(metrics.f1)
                per_scenario[sc][m]["sens"].append(metrics.sensitivity)
                per_scenario[sc][m]["spec"].append(metrics.specificity)

        # macro aggregation per seed (mean over scenarios with a value)
        for m in METHODS:
            f1s = [results[sc][m].f1 for sc in results if results[sc][m].f1 is not None]
            sens = [results[sc][m].sensitivity for sc in results if results[sc][m].sensitivity is not None]
            spec = [results[sc][m].specificity for sc in results if results[sc][m].specificity is not None]
            far = [
                results[sc][m].fp / max(results[sc][m].n_samples, 1)
                for sc in results
                if results[sc][m].fp > 0 or results[sc][m].tn > 0
            ]
            if f1s:
                macro[m]["f1"].append(np.mean(f1s))
            if sens:
                macro[m]["sens"].append(np.mean(sens))
            if spec:
                macro[m]["spec"].append(np.mean(spec))
            if far:
                macro[m]["far"].append(np.mean(far))

    print("\n" + "=" * 78)
    print(f"PER-SCENARIO F1 across {args.seeds} seeds (mean [95% CI])")
    print("=" * 78)
    for sc, methods in per_scenario.items():
        print(f"\n{sc}:")
        for m in METHODS:
            f1_mean, f1_lo, f1_hi = ci95(methods[m]["f1"])
            sens_mean, _, _ = ci95(methods[m]["sens"])
            spec_mean, _, _ = ci95(methods[m]["spec"])
            f1_str = f"{f1_mean:.3f} [{f1_lo:.3f},{f1_hi:.3f}]" if f1_mean is not None else "—"
            sens_str = f"{sens_mean:.3f}" if sens_mean is not None else "—"
            spec_str = f"{spec_mean:.3f}" if spec_mean is not None else "—"
            print(f"  {m:16s} F1={f1_str:24s} sens={sens_str:8s} spec={spec_str:8s}")

    print("\n" + "=" * 78)
    print(f"TABLE I (macro-avg across scenarios, mean [95% CI] over {args.seeds} seeds)")
    print("=" * 78)
    def fmt(mean, lo, hi):
        if mean is None:
            return "—"
        return f"{mean*100:.1f}% [{lo*100:.1f},{hi*100:.1f}]"

    for m in METHODS:
        sens_mean, sens_lo, sens_hi = ci95(macro[m]["sens"])
        spec_mean, spec_lo, spec_hi = ci95(macro[m]["spec"])
        f1_mean, f1_lo, f1_hi = ci95(macro[m]["f1"])
        far_mean, far_lo, far_hi = ci95(macro[m]["far"])
        print(f"{m:16s} "
              f"Sens={fmt(sens_mean, sens_lo, sens_hi)}  "
              f"Spec={fmt(spec_mean, spec_lo, spec_hi)}  "
              f"F1={fmt(f1_mean, f1_lo, f1_hi)}  "
              f"FAR={fmt(far_mean, far_lo, far_hi)}")


if __name__ == "__main__":
    main()
