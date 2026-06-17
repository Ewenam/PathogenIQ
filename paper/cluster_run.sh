#!/usr/bin/env bash
# ============================================================
#  PathogenIQ — GSU Cluster Run Script
#  Usage (after SSH into cluster):
#
#    cd ~/PathogenIQ                    # or wherever the repo is
#    bash paper/cluster_run.sh
#
#  What it does:
#    1. Activates the venv (or creates it if missing)
#    2. Runs the full pipeline on your Kraken2 reports
#    3. Runs the benchmark suite
#    4. Generates all paper figures + LaTeX table numbers
#    5. Tars up everything you need to copy back
#
#  Edit the three variables below before running.
# ============================================================
set -euo pipefail

# ── EDIT THESE ───────────────────────────────────────────────
KRAKEN_REPORTS_DIR="/home/users/razumah1/Desktop/PathogenIQ/expanded_datasets/reports"   # directory of *.report files
OUTPUT_DIR="/home/users/razumah1/Desktop/PathogenIQ/reports"
RANK="G"          # G=genus (recommended), S=species, F=family
# ─────────────────────────────────────────────────────────────

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO/venv"

echo ""
echo "============================================"
echo "  PathogenIQ — Cluster Run"
echo "  Repo:    $REPO"
echo "  Input:   $KRAKEN_REPORTS_DIR"
echo "  Output:  $OUTPUT_DIR"
echo "============================================"
echo ""

# ── 1. Activate / create venv ────────────────────────────────
if [ ! -f "$VENV/bin/activate" ]; then
    echo "[setup] Creating virtual environment ..."
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install -q -e "$REPO"
    pip install -q matplotlib
else
    source "$VENV/bin/activate"
fi

# ── 2. Run the full pipeline ──────────────────────────────────
echo "[1/3] Running PathogenIQ pipeline ..."
mkdir -p "$OUTPUT_DIR"
python "$REPO/cli.py" run "$KRAKEN_REPORTS_DIR" \
    --rank "$RANK" \
    --output "$OUTPUT_DIR" \
    --alert-threshold 0.6 \
    --characterize

echo ""
echo "      Pipeline complete."
echo "      JSON report: $OUTPUT_DIR/report.json"
echo "      HTML report: $OUTPUT_DIR/report.html"

# ── 3. Run the benchmark ──────────────────────────────────────
echo ""
echo "[2/3] Running synthetic benchmark ..."
python "$REPO/cli.py" benchmark \
    --output "$OUTPUT_DIR/benchmark.json"

echo "      Benchmark complete."

# ── 4. Generate paper figures + tables ───────────────────────
echo ""
echo "[3/3] Generating paper figures and LaTeX tables ..."
python "$REPO/paper/generate_figures.py" \
    --report "$OUTPUT_DIR/report.json"

# ── 5. Bundle results for transfer ───────────────────────────
BUNDLE="$REPO/paper/paper_results_$(date +%Y%m%d_%H%M).tar.gz"
tar -czf "$BUNDLE" \
    -C "$REPO/paper" \
    figures/ \
    table_results.tex \
    -C "$OUTPUT_DIR" \
    report.json \
    benchmark.json

echo ""
echo "============================================"
echo "  Done!  Transfer this file back to your Mac:"
echo ""
echo "  scp <netid>@<cluster-host>:$BUNDLE ."
echo ""
echo "  Then on your Mac run:"
echo "  cd PathogenIQ/paper && tar xzf *.tar.gz"
echo "  pdflatex pathogeniq_mlsp2026.tex && pdflatex pathogeniq_mlsp2026.tex"
echo "============================================"
