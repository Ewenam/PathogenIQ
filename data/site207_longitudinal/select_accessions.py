#!/usr/bin/env python3
"""
Reproducible selection rule for the PathogenIQ site-207 longitudinal dataset.

Source: NCBI BioProject PRJNA957477 (Verily Life Sciences national
wastewater surveillance program), retrieved via the SRA Run Selector bulk
runinfo endpoint.

Selection rule (deterministic, no manual curation):
  1. Single facility: keep runs whose BioSample sample_name matches
     "VLT_207-*" -- site 207, a Nevada WWTP (jurisdiction NV,
     ww_population ~= 1.25M; verified via BioSample efetch).
  2. Sequencing tier: keep only Platform/Model == "Illumina NovaSeq 6000".
     Site 207 also has 108 earlier "NextSeq 2000" amplicon runs
     (2023-09 to 2024-09, ~2.5M reads each) that predate the switch to
     NovaSeq 6000 (~10M reads each); these are excluded as too shallow
     for genus-level Kraken2 classification.
  3. Sort ascending by collection date, parsed from the SampleName suffix
     "VLT_207-YYMMDD" (cross-checked against the BioSample
     collection_date attribute -- they agree).
  4. Take the most recent N=27 runs.

This yields a single-facility, time-ordered series of 27 paired-end runs
spanning 2026-02-25 to 2026-05-27 (twice-weekly cadence, ~13 weeks).

Run:
    python3 select_accessions.py [--runinfo runinfo_full.csv] [-n 27]

If --runinfo is omitted, the bulk runinfo CSV is fetched fresh from NCBI.
Output is written to stdout in the same TSV format as
data_site207_accessions.tsv (collection_date, run_accession, biosample,
size_mb).
"""
from __future__ import annotations

import argparse
import csv
import sys
import urllib.request

BIOPROJECT = "PRJNA957477"
SITE_PREFIX = "VLT_207-"
SEQ_MODEL = "Illumina NovaSeq 6000"
RUNINFO_URL = (
    "https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?acc=" + BIOPROJECT
)


def parse_date(sample_name: str) -> str:
    """'VLT_207-260225' -> '2026-02-25'"""
    yymmdd = sample_name.split("-")[1][:6]
    return f"20{yymmdd[0:2]}-{yymmdd[2:4]}-{yymmdd[4:6]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runinfo", help="local copy of the bulk runinfo CSV")
    ap.add_argument("-n", type=int, default=27, help="number of runs to select")
    args = ap.parse_args()

    if args.runinfo:
        fh = open(args.runinfo, newline="")
    else:
        req = urllib.request.urlopen(RUNINFO_URL, timeout=120)
        fh = (line.decode("utf-8") for line in req)

    reader = csv.DictReader(fh)
    candidates = [
        row
        for row in reader
        if row["SampleName"].startswith(SITE_PREFIX) and row["Model"] == SEQ_MODEL
    ]
    candidates.sort(key=lambda r: parse_date(r["SampleName"]))
    selected = candidates[-args.n :]

    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    writer.writerow(["collection_date", "run_accession", "biosample", "size_mb"])
    for row in selected:
        writer.writerow(
            [parse_date(row["SampleName"]), row["Run"], row["BioSample"], row["size_MB"]]
        )


if __name__ == "__main__":
    main()
