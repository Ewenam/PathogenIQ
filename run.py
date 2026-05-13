#!/usr/bin/env python3
"""
PathogenIQ — Simple Runner
==========================
Fill in the parameters below, then run:

    python3 run.py

Results are saved to OUTPUT_DIR.
Open the dashboard at http://localhost:8765 after the pipeline finishes.
"""
# Auto-switch to the venv Python if dependencies aren't available
import sys, os
from pathlib import Path

_VENV_CANDIDATES = [
    Path(__file__).parent.parent / ".venv" / "bin" / "python3",  # Desktop/.venv
    Path(__file__).parent / "venv" / "bin" / "python3",          # project/venv
]
try:
    import click  # quick check: if this works we're in the right env
except ImportError:
    for _venv_py in _VENV_CANDIDATES:
        if _venv_py.exists():
            os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)
    sys.exit(
        "ERROR: Could not find a virtual environment with dependencies installed.\n"
        "Run:  pip install -r requirements.txt\n"
        "Or activate your venv first: source /path/to/venv/bin/activate"
    )

# ─────────────────────────────────────────────
#  REQUIRED: set these before running
# ─────────────────────────────────────────────

# Path to your Kraken2 report files (.report) OR a pre-built TSV count matrix.
# Examples:
#   INPUT = "/home/users/razumah1/Desktop/AAB Project/data/reports"   # folder of .report files
#   INPUT = "/path/to/counts.tsv"                                      # pre-built matrix
INPUT = "/home/users/razumah1/Desktop/AAB Project/data/reports"

# Where to save the report.json, report.html, and figures.
OUTPUT_DIR = "/home/users/razumah1/Desktop/PathogenIQ/reports"

# ─────────────────────────────────────────────
#  OPTIONAL: tweak these as needed
# ─────────────────────────────────────────────

# Taxonomic rank: "G" = genus (default), "S" = species, "F" = family
RANK = "G"

# Risk score threshold to trigger an alert (0–1). 0.6 is a good starting point.
ALERT_THRESHOLD = 0.6

# Run ESMFold protein structure prediction + NCBI virulence annotation on flagged taxa?
# Requires internet access. Adds ~30–60s per flagged pathogen.
CHARACTERIZE = True

# Launch the dashboard automatically after the run? (opens http://localhost:8765)
LAUNCH_DASHBOARD = True

# Dashboard port
DASHBOARD_PORT = 8765

# Dashboard login credentials (shown in browser when you open the dashboard)
DASH_USER = "admin"
DASH_PASS = "pathogeniq"   # change this to something private

# Set to False to disable the login prompt entirely (not recommended on shared servers)
DASH_AUTH = True

# For advanced graph/SBM params (spearman_threshold, min_prevalence, etc.)
# edit configs/default.yaml directly.

# ─────────────────────────────────────────────
#  Alerting (optional — leave blank to disable)
# ─────────────────────────────────────────────

# Slack webhook URL for alert notifications
SLACK_WEBHOOK = ""   # e.g. "https://hooks.slack.com/services/..."

# Email recipients (comma-separated) for alert notifications
ALERT_EMAILS = ""    # e.g. "lab@gsu.edu,director@gsu.edu"

# Gmail credentials (use an App Password, not your real password)
SMTP_USER = ""       # e.g. "you@gmail.com"
SMTP_PASS = ""       # App password from Google account settings

# ─────────────────────────────────────────────
#  DO NOT EDIT BELOW THIS LINE
# ─────────────────────────────────────────────
import os, sys, subprocess  # stdlib only — no extra dependencies
from pathlib import Path

PYTHON = Path(sys.executable)   # use whatever python runs this script

PATHOGENIQ = Path(__file__).parent / "cli.py"


def _run_cmd(cmd, env=None):
    e = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, env=e)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    print("\n═══════════════════════════════════════════")
    print("  PathogenIQ Pipeline")
    print("═══════════════════════════════════════════")
    print(f"  Input:     {INPUT}")
    print(f"  Output:    {OUTPUT_DIR}")
    print(f"  Rank:      {RANK}")
    print(f"  Threshold: {ALERT_THRESHOLD}")
    print()

    if not Path(INPUT).exists():
        print(f"ERROR: INPUT path does not exist: {INPUT}")
        sys.exit(1)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    run_cmd = [
        str(PYTHON), str(PATHOGENIQ), "run", INPUT,
        "--rank", RANK,
        "--output", OUTPUT_DIR,
        "--alert-threshold", str(ALERT_THRESHOLD),
        *(["--characterize"] if CHARACTERIZE else []),
    ]

    # Point at the project config so graph params are picked up
    config_path = Path(__file__).parent / "configs" / "default.yaml"
    if config_path.exists():
        run_cmd += ["--config", str(config_path)]

    env_overrides = {}
    if SLACK_WEBHOOK:
        env_overrides["PATHOGENIQ_SLACK_WEBHOOK"] = SLACK_WEBHOOK
        run_cmd.append("--alert-slack")
    if ALERT_EMAILS and SMTP_USER and SMTP_PASS:
        env_overrides["PATHOGENIQ_SMTP_USER"] = SMTP_USER
        env_overrides["PATHOGENIQ_SMTP_PASS"] = SMTP_PASS
        env_overrides["PATHOGENIQ_ALERT_EMAILS"] = ALERT_EMAILS
        run_cmd.append("--alert-email")

    _run_cmd(run_cmd, env_overrides)

    report_json = Path(OUTPUT_DIR) / "report.json"
    report_html = Path(OUTPUT_DIR) / "report.html"
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    if report_json.exists():
        print(f"  JSON report: {report_json}")
    if report_html.exists():
        print(f"  HTML report: {report_html}")

    if LAUNCH_DASHBOARD:
        # Free the port if a previous dashboard is still running
        subprocess.run(["pkill", "-f", f"pathogeniq.*dashboard|uvicorn.*pathogeniq"],
                       capture_output=True)
        print(f"\nLaunching dashboard → http://localhost:{DASHBOARD_PORT}")
        print("  Press Ctrl+C to stop.\n")
        env_overrides["PATHOGENIQ_REPORT"] = str(report_json)
        env_overrides["PATHOGENIQ_DASH_USER"] = DASH_USER
        env_overrides["PATHOGENIQ_DASH_PASS"] = DASH_PASS
        env_overrides["PATHOGENIQ_DASH_AUTH"] = "true" if DASH_AUTH else "false"
        _run_cmd([
            str(PYTHON), str(PATHOGENIQ), "dashboard",
            "--report", str(report_json),
            "--port", str(DASHBOARD_PORT),
        ], env_overrides)


if __name__ == "__main__":
    main()
