#!/usr/bin/env bash
# Phase 1 data pipeline: download + classify the site-207 (Nevada WWTP) longitudinal
# series identified from BioProject PRJNA957477 (Verily Life Sciences).
#
# For each accession in data_site207_accessions.tsv:
#   1. prefetch       -> SRA archive (scratch)
#   2. fasterq-dump   -> paired FASTQ (scratch, --split-files, no FASTA conversion)
#   3. kraken2        -> .report file (kept, written to reports/)
#   4. delete the SRA archive + FASTQ for that accession (disk management)
#
# Timing for each step is recorded to timing_log.tsv for the paper's
# "Experimental Setup" reproducibility / runtime reporting.

set -uo pipefail

export PATH="/data/users3/razumah1/anaconda3/bin:$PATH"

ACC_LIST="/home/users/razumah1/Desktop/PathogenIQ/data_site207_accessions.tsv"
SCRATCH="/data/users3/razumah1/pathogeniq_phase1"
REPORT_DIR="/home/users/razumah1/Desktop/PathogenIQ/data/site207_longitudinal/reports"
TIMING_LOG="/home/users/razumah1/Desktop/PathogenIQ/data/site207_longitudinal/timing_log.tsv"
KRAKEN_DB="/home/users/razumah1"
THREADS=12
CONFIDENCE=0.05

mkdir -p "$SCRATCH/sra" "$SCRATCH/fastq" "$REPORT_DIR"

if [[ ! -f "$TIMING_LOG" ]]; then
  echo -e "run_accession\tcollection_date\tsize_mb\tdownload_sec\tfasterq_sec\tkraken2_sec\ttotal_sec\tstatus" > "$TIMING_LOG"
fi

# Skip header line
tail -n +2 "$ACC_LIST" | while IFS=$'\t' read -r DATE ACC BIOSAMPLE SIZE_MB; do
  REPORT="$REPORT_DIR/${ACC}.report"

  if [[ -f "$REPORT" ]]; then
    echo "[skip] $ACC ($DATE) - report already exists"
    continue
  fi

  echo "==> $ACC  (collected $DATE, ${SIZE_MB}MB)"
  T0=$(date +%s)

  # 1. Download
  prefetch --max-size 50G -O "$SCRATCH/sra" "$ACC" >> "$SCRATCH/log_${ACC}.txt" 2>&1
  T1=$(date +%s)

  # 2. Convert to paired FASTQ (no FASTA conversion)
  fasterq-dump \
    --split-files \
    --threads "$THREADS" \
    --outdir "$SCRATCH/fastq" \
    "$SCRATCH/sra/$ACC/$ACC.sra" >> "$SCRATCH/log_${ACC}.txt" 2>&1

  T2=$(date +%s)

  R1="$SCRATCH/fastq/${ACC}_1.fastq"
  R2="$SCRATCH/fastq/${ACC}_2.fastq"

  if [[ ! -f "$R1" || ! -f "$R2" ]]; then
    echo "    ERROR: FASTQ files missing for $ACC" | tee -a "$TIMING_LOG"
    echo -e "${ACC}\t${DATE}\t${SIZE_MB}\t$((T1-T0))\t$((T2-T1))\t0\t$((T2-T0))\tFASTQ_MISSING" >> "$TIMING_LOG"
    rm -rf "$SCRATCH/sra/$ACC"
    continue
  fi

  # 3. Classify with Kraken2 (genus/species report; full taxonomy in .report)
  kraken2 \
    --db "$KRAKEN_DB" \
    --paired \
    --threads "$THREADS" \
    --confidence "$CONFIDENCE" \
    --report "$REPORT" \
    --output /dev/null \
    "$R1" "$R2" >> "$SCRATCH/log_${ACC}.txt" 2>&1

  T3=$(date +%s)

  echo -e "${ACC}\t${DATE}\t${SIZE_MB}\t$((T1-T0))\t$((T2-T1))\t$((T3-T2))\t$((T3-T0))\tOK" >> "$TIMING_LOG"
  echo "    download=$((T1-T0))s  fasterq=$((T2-T1))s  kraken2=$((T3-T2))s  total=$((T3-T0))s"

  # 4. Cleanup scratch
  rm -rf "$SCRATCH/sra/$ACC" "$R1" "$R2"
  rm -f "$SCRATCH/log_${ACC}.txt"
done

echo "Done. Reports in $REPORT_DIR, timing in $TIMING_LOG"
