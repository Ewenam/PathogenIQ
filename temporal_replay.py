#!/usr/bin/env python3
"""
temporal_replay.py
==================
Feeds your existing time-stamped wastewater samples into the PathogenIQ
history store as separate dated runs, enabling CUSUM and Mann-Kendall
trend analysis in the dashboard.

Each entry below = one WWTP site.  Samples collected at different dates
from the same physical location are processed in chronological order so
the temporal store sees them as a real time series for that site.

Usage:
    python temporal_replay.py

After it finishes, run the pipeline once more on ALL samples (to refresh
report.json), then launch the dashboard — the timeline charts and CUSUM
alerts will be populated.
"""
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

# Directory that contains your classified .report files
REPORTS_DIR = Path("/home/users/razumah1/Desktop/AAB Project/data/reports")

# Where to write temporary per-run report output (safe to delete afterwards)
REPLAY_OUTPUT = Path("/home/users/razumah1/Desktop/PathogenIQ/reports/temporal_replay")

# Python executable (will auto-detect venv)
PYTHON = sys.executable
CLI = Path(__file__).parent / "cli.py"

# ── Site definitions ───────────────────────────────────────────────────────────
# Each site is a list of (collection_date, report_file_stem) tuples.
# Report file stem = the .report filename without extension.
# Dates must be in ISO format YYYY-MM-DD and in chronological order.

SITES = {
    # Wisconsin State Laboratory of Hygiene — 6 time points
    "Wisconsin_WSLH": [
        ("2025-10-02", "cleaned_SRR35939739"),
        ("2025-10-06", "cleaned_SRR35939740"),
        ("2025-10-07", "cleaned_SRR35939738"),
        ("2025-10-13", "cleaned_SRR35939736"),
        ("2025-10-19", "cleaned_SRR35939735"),
        ("2025-10-20", "cleaned_SRR35939737"),
    ],

    # Colorado — each county has only 1 date; useful for per-county baselines
    # Uncomment to include — 1 data point each won't show trends yet but
    # establishes the baseline for future runs on new Colorado data.
    #
    # "Colorado_Boulder": [
    #     ("2025-09-15", "cleaned_SRR35556007"),
    # ],
    # "Colorado_Arapahoe": [
    #     ("2025-09-15", "cleaned_SRR35556008"),
    # ],
    # "Colorado_Adams": [
    #     ("2025-09-15", "cleaned_SRR35556009"),
    # ],
}

# ── Replay logic ───────────────────────────────────────────────────────────────

def run_single(report_path: Path, date: str, out_dir: Path):
    """Run the pipeline on a single .report file stamped with a specific date."""
    cmd = [
        PYTHON, str(CLI), "run",
        str(report_path.parent),       # input = directory containing the report
        "--rank", "G",
        "--output", str(out_dir),
        "--run-date", date,
        "--quiet",
    ]
    # Pass a glob pattern so only this one sample is processed.
    # We do this by creating a temporary directory with just that file.
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / report_path.name).symlink_to(report_path.resolve())
        cmd[3] = str(tmp)   # replace input dir with tmp
        result = subprocess.run(cmd, capture_output=False)
        return result.returncode == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("PathogenIQ Temporal Replay")
    print("=" * 50)

    if not REPORTS_DIR.exists():
        print(f"ERROR: REPORTS_DIR not found: {REPORTS_DIR}")
        sys.exit(1)

    REPLAY_OUTPUT.mkdir(parents=True, exist_ok=True)

    total = sum(len(v) for v in SITES.values())
    done = 0

    for site_name, time_points in SITES.items():
        print(f"\n► Site: {site_name}  ({len(time_points)} time points)")
        for date, stem in time_points:
            # Find the .report file (try .report then .txt)
            report = REPORTS_DIR / f"{stem}.report"
            if not report.exists():
                report = REPORTS_DIR / f"{stem}.txt"
            if not report.exists():
                print(f"  [skip] {stem} — .report file not found in {REPORTS_DIR}")
                continue

            done += 1
            print(f"  [{done}/{total}] {date}  {stem}", end="  ", flush=True)
            ok = run_single(report, date, REPLAY_OUTPUT / site_name / date)
            print("✓" if ok else "✗ FAILED")

    print("\n" + "=" * 50)
    print("Replay complete.")
    print()
    print("Next steps:")
    print("  1. Re-run the full pipeline to refresh report.json:")
    print(f'     python run.py')
    print()
    print("  2. Launch the dashboard:")
    print("     python -m pathogeniq.dashboard.app")
    print()
    print("  The Trend and CUSUM columns in the table will now reflect")
    print("  the Wisconsin time series (Oct 2 → Oct 20).")


if __name__ == "__main__":
    main()
