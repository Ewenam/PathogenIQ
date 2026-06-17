# Site-207 Longitudinal Dataset — Provenance

## Source

NCBI BioProject [PRJNA957477](https://www.ncbi.nlm.nih.gov/bioproject/957477)
(Verily Life Sciences national wastewater surveillance program; SRA study
SRP433461; 14,418 runs total across many facilities).

## Facility

**Site 207** — a wastewater treatment plant in Nevada (`geo_loc_name: USA:
Nevada`, `ww_surv_jurisdiction: NV`, `ww_population: 1,250,000`,
`ww_sample_type: composite`, 24h composite, post-grit-removal). Verified via
`efetch` against BioSample SAMN56353621.

Verily encodes facility + collection date in the BioSample `sample_name` /
`ww_surv_system_sample_id` as `VLT_207-YYMMDD` (confirmed to agree with the
`collection_date` attribute).

## Selection rule (deterministic, reproducible)

Implemented in [`select_accessions.py`](select_accessions.py); reproduces
`../../data_site207_accessions.tsv` exactly.

1. **Single facility**: `SampleName` matches `VLT_207-*` (281 runs total for
   site 207 across the whole BioProject).
2. **Sequencing tier**: keep only `Platform/Model == "Illumina NovaSeq
   6000"` (173 of the 281 runs). Site 207 also has 108 earlier `NextSeq
   2000` amplicon runs (2023-09 -- 2024-09, ~2.5M reads/run) that predate a
   switch to NovaSeq 6000 (~10M reads/run); these are excluded as too
   shallow for genus-level Kraken2 classification.
3. **Sort** ascending by collection date parsed from `VLT_207-YYMMDD`.
4. **Take the most recent N=60 runs.**

Result: a single-facility, time-ordered series of 60 paired-end runs,
**2025-11-05 -- 2026-06-03**, sampled twice weekly (~30 weeks).

To re-derive from scratch:

```bash
python3 select_accessions.py -n 60 > data_site207_accessions.tsv
```

(fetches the bulk runinfo CSV from NCBI directly; pass
`--runinfo <local.csv>` to use a cached copy).

## Classification pipeline

Implemented in [`run_pipeline.sh`](run_pipeline.sh). Per accession:

1. `prefetch --max-size 50G` (SRA Toolkit 3.4.1)
2. `fasterq-dump --split-files` -> paired FASTQ (no FASTA conversion)
3. `kraken2 --paired --confidence 0.05 --report <ACC>.report --output /dev/null`
   directly on the paired FASTQ

### Kraken2 database

- Kraken2 **v2.1.6**
- Standard database (RefSeq bacteria/archaea/viral/human/UniVec_Core),
  built **2025-10-15**
- `k=35`, `l=31`, 41,949 taxonomy nodes
- Located at `/home/users/razumah1/{hash,opts,taxo}.k2d`
- Classification parameters: `--confidence 0.05`, `--paired`, 12 threads

### Timing / scalability

Per-run wall-clock timing (download, FASTQ conversion, classification) is
recorded in [`timing_log.tsv`](timing_log.tsv) for the paper's Experimental
Setup reproducibility and runtime/scalability reporting.
