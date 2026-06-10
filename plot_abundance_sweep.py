"""
plot_abundance_sweep.py
=======================
Turn abundance_sweep.csv into a publication-quality figure for the paper.

Produces fig_community_sweep.pdf (and .png) showing alert rate vs per-pathogen
abundance for abundance-only vs full PathogenIQ, with the borderline window —
where the community signal changes decisions — shaded.

RUN (after run_abundance_sweep.py):
    python plot_abundance_sweep.py
"""
from __future__ import annotations

import csv
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALERT_THRESHOLD = 0.6  # for annotation only

try:
    rows = list(csv.DictReader(open("abundance_sweep.csv")))
except FileNotFoundError:
    print("abundance_sweep.csv not found — run run_abundance_sweep.py first.")
    sys.exit(1)

levels = np.array([float(r["level"]) for r in rows]) * 100  # to %
abun = np.array([float(r["abundance_only_alert_rate"]) for r in rows])
abun_lo = np.array([float(r["abundance_only_lo"]) for r in rows])
abun_hi = np.array([float(r["abundance_only_hi"]) for r in rows])
full = np.array([float(r["full_alert_rate"]) for r in rows])
full_lo = np.array([float(r["full_lo"]) for r in rows])
full_hi = np.array([float(r["full_hi"]) for r in rows])
gain = full - abun

# Uniform palette consistent with the rest of the paper's figures
C_ABUN = "#4C72B0"   # blue
C_FULL = "#C44E52"   # red
C_GAIN = "#DD8452"   # orange shade for the window

fig, ax = plt.subplots(figsize=(3.4, 2.5))  # single IEEE column

# Shade the borderline window where community signal adds the most
if gain.max() > 0.05:
    window = levels[gain > 0.05]
    if len(window):
        ax.axvspan(window.min(), window.max(), color=C_GAIN, alpha=0.15,
                   label="Community-signal window")

ax.plot(levels, abun, "-o", color=C_ABUN, ms=4, lw=1.5, label="Abundance-only")
ax.fill_between(levels, abun_lo, abun_hi, color=C_ABUN, alpha=0.15)
ax.plot(levels, full, "-s", color=C_FULL, ms=4, lw=1.5, label="Full PathogenIQ")
ax.fill_between(levels, full_lo, full_hi, color=C_FULL, alpha=0.15)

ax.set_xlabel("Per-pathogen abundance (%)", fontsize=8)
ax.set_ylabel("Alert rate", fontsize=8)
ax.set_ylim(-0.03, 1.03)
ax.tick_params(labelsize=7)
ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.25, lw=0.5)
fig.tight_layout(pad=0.4)

fig.savefig("fig_community_sweep.pdf", bbox_inches="tight")
fig.savefig("fig_community_sweep.png", dpi=200, bbox_inches="tight")
print("Wrote fig_community_sweep.pdf and fig_community_sweep.png")
print(f"Peak community gain: {gain.max():+.2f} at "
      f"{levels[np.argmax(gain)]:.1f}% per-pathogen abundance")
