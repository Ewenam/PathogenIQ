#!/usr/bin/env bash
# download_datasets.sh
# Downloads and classifies expanded wastewater surveillance datasets from NCBI SRA
# using SRA-tools (prefetch + fasterq-dump) and Kraken2.
#
# Accessions sourced from BioProjects:
#   TX  Verily wastewater surveillance  (PRJNA1234567-ish, SRR36780902/904/908)
#   NY  NWSS New York                   (SRR37567371/372/373)
#   FL  Verily wastewater surveillance  (SRR36780874/907/944)
#   GA  Verily wastewater surveillance  (SRR36861624/625, SRR36780738)
#   UT  UPHL Utah                       (SRR38537823-SRR38537827)
#
# Prerequisites:
#   - SRA Toolkit  (prefetch, fasterq-dump)   conda install -c bioconda sra-tools
#   - Kraken2                                  conda install -c bioconda kraken2
#   - A Kraken2 database  (set KRAKEN_DB below)
#
# Usage:
#   chmod +x download_datasets.sh
#   ./download_datasets.sh [--threads N] [--outdir PATH] [--db PATH]
#
# Output:
#   <OUTDIR>/fastq/   raw paired-end FASTQ files
#   <OUTDIR>/reports/ Kraken2 .report files (one per accession)
#   Copy or symlink <OUTDIR>/reports/ to your PathogenIQ data directory.

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
THREADS=8
OUTDIR="$(pwd)/expanded_datasets"
KRAKEN_DB="${KRAKEN_DB:-/data/kraken2_db/standard}"
CONFIDENCE=0.05          # Kraken2 --confidence threshold (lower = more sensitive)
KEEP_FASTQ=false         # set true to retain raw FASTQs after classification

# ── Parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --threads) THREADS="$2"; shift 2;;
    --outdir)  OUTDIR="$2";  shift 2;;
    --db)      KRAKEN_DB="$2"; shift 2;;
    --keep-fastq) KEEP_FASTQ=true; shift;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

FASTQ_DIR="$OUTDIR/fastq"
REPORT_DIR="$OUTDIR/reports"
mkdir -p "$FASTQ_DIR" "$REPORT_DIR"

# ── Accession list ───────────────────────────────────────────────────────────
declare -A ACCESSIONS=(
  # Texas — Verily wastewater surveillance
  [SRR36780902]="TX"
  [SRR36780904]="TX"
  [SRR36780908]="TX"
  # New York — NWSS
  [SRR37567371]="NY"
  [SRR37567372]="NY"
  [SRR37567373]="NY"
  # Florida — Verily wastewater surveillance
  [SRR36780874]="FL"
  [SRR36780907]="FL"
  [SRR36780944]="FL"
  # Georgia — Verily wastewater surveillance
  [SRR36861624]="GA"
  [SRR36861625]="GA"
  [SRR36780738]="GA"
  # Utah — UPHL
  [SRR38537823]="UT"
  [SRR38537824]="UT"
  [SRR38537825]="UT"
  [SRR38537826]="UT"
  [SRR38537827]="UT"
)

echo "=== PathogenIQ dataset expansion ==="
echo "    Kraken2 DB : $KRAKEN_DB"
echo "    Threads    : $THREADS"
echo "    Output     : $OUTDIR"
echo "    Accessions : ${#ACCESSIONS[@]}"
echo

if [[ ! -d "$KRAKEN_DB" ]]; then
  echo "ERROR: Kraken2 database not found at $KRAKEN_DB"
  echo "  Set the KRAKEN_DB env variable or pass --db <path>"
  echo "  To download the standard database (~60 GB):"
  echo "    kraken2-build --standard --db $KRAKEN_DB --threads $THREADS"
  exit 1
fi

# ── Per-accession download + classify ────────────────────────────────────────
for ACC in "${!ACCESSIONS[@]}"; do
  STATE="${ACCESSIONS[$ACC]}"
  REPORT="$REPORT_DIR/${ACC}.report"

  if [[ -f "$REPORT" ]]; then
    echo "  [skip] $ACC ($STATE) — report already exists"
    continue
  fi

  echo "► $ACC ($STATE)"

  # 1. Prefetch SRA archive
  echo "  Downloading..."
  prefetch --max-size 50G -O "$FASTQ_DIR" "$ACC"

  # 2. Convert to paired-end FASTQ
  echo "  Converting to FASTQ..."
  fasterq-dump \
    --split-files \
    --threads "$THREADS" \
    --outdir "$FASTQ_DIR" \
    "$FASTQ_DIR/$ACC/$ACC.sra" 2>/dev/null \
  || fasterq-dump \
    --split-files \
    --threads "$THREADS" \
    --outdir "$FASTQ_DIR" \
    "$ACC"

  R1="$FASTQ_DIR/${ACC}_1.fastq"
  R2="$FASTQ_DIR/${ACC}_2.fastq"

  # 3. Quality trim with fastp (optional but recommended for wastewater)
  if command -v fastp &>/dev/null; then
    echo "  Trimming with fastp..."
    fastp \
      -i "$R1" -I "$R2" \
      -o "${R1%.fastq}.trimmed.fastq" -O "${R2%.fastq}.trimmed.fastq" \
      --thread "$THREADS" \
      --json /dev/null --html /dev/null \
      --qualified_quality_phred 20 \
      --length_required 50 \
      --detect_adapter_for_pe 2>/dev/null
    R1="${R1%.fastq}.trimmed.fastq"
    R2="${R2%.fastq}.trimmed.fastq"
  fi

  # 4. Classify with Kraken2
  echo "  Running Kraken2..."
  kraken2 \
    --db "$KRAKEN_DB" \
    --paired \
    --threads "$THREADS" \
    --confidence "$CONFIDENCE" \
    --report "$REPORT" \
    --output /dev/null \
    "$R1" "$R2"

  echo "  Report written: $REPORT"

  # 5. Optionally clean up FASTQ to save disk
  if [[ "$KEEP_FASTQ" == false ]]; then
    rm -rf "$FASTQ_DIR/$ACC" "$FASTQ_DIR/${ACC}_1.fastq" "$FASTQ_DIR/${ACC}_2.fastq" \
           "$FASTQ_DIR/${ACC}_1.trimmed.fastq" "$FASTQ_DIR/${ACC}_2.trimmed.fastq" 2>/dev/null || true
  fi

  echo
done

echo "=== Done ==="
echo "Reports saved to: $REPORT_DIR"
echo
echo "Next steps:"
echo "  1. Fetch metadata labels:"
echo "       python fetch_metadata.py"
echo "  2. Run PathogenIQ pipeline:"
echo "       python run.py --input $REPORT_DIR --output reports/expanded_run.json"
echo "  3. Launch dashboard:"
echo "       python -m pathogeniq.dashboard.app"
