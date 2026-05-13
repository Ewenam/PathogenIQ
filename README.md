# PathogenIQ

**AI-powered biosurveillance platform for detecting, tracking, and characterizing disease-causing pathogens from metagenomic sequencing data.**

PathogenIQ turns raw Kraken2 metagenomic output (wastewater, clinical, environmental) into actionable risk intelligence — identifying 80 known pathogens, detecting novel organisms, tracking outbreak trends over time, and sending automated alerts when thresholds are breached.

---

## What It Does

PathogenIQ answers three questions from a set of sequencing samples:

1. **What's there?** — Which known pathogens are present and at what abundance?
2. **How alarming is it?** — A calibrated 0–1 risk score with four levels (LOW → CRITICAL) per sample
3. **Is it getting worse?** — Week-over-week trend analysis, CUSUM changepoint detection, and short-term forecasting

It was built for **wastewater-based epidemiology (WBE)** — the same methodology used to track COVID-19 through sewage systems globally — generalized to 80 pathogens across bacteria, viruses, fungi, and parasites.

---

## Installation

**Requirements:** Python 3.11+. No GPU required. Internet access needed only for ESMFold characterization (optional).

```bash
git clone https://github.com/Ewenam/PathogenIQ.git
cd PathogenIQ
pip install -r requirements.txt
pip install . --no-build-isolation
```

**Verify installation:**
```bash
pathogeniq benchmark
# Expected: Sensitivity=1.00, Precision=1.00, F1=1.00
```

> **Note for GSU / HPC users:** The dashboard is fully self-contained — Chart.js and Leaflet.js are bundled inline so no external CDN requests are made. Works on restricted or firewalled networks.

---

## Quick Start — `run.py`

The simplest way to run the full pipeline is `run.py` at the project root. Open it, fill in the paths at the top, and run it:

```python
# run.py — edit these lines:
INPUT       = "/path/to/your/kraken_reports/"  # folder of .report files, or a counts.tsv
OUTPUT_DIR  = "./reports"
RANK        = "G"          # G = genus (default), S = species, F = family
ALERT_THRESHOLD = 0.6      # 0–1, triggers HIGH alert
CHARACTERIZE    = False    # True = run ESMFold on flagged pathogens (~30-60s each)
LAUNCH_DASHBOARD = True    # opens http://localhost:8765 when done

# Dashboard login
DASH_USER = "admin"
DASH_PASS = "pathogeniq"   # change this on shared servers
DASH_AUTH = True
```

Then run:
```bash
python3 run.py
```

`run.py` auto-detects the virtual environment — no need to activate it first. Advanced graph/SBM parameters (Spearman threshold, min prevalence, etc.) are in `configs/default.yaml`.

---

## Full Pipeline (8 Steps)

```
Kraken2 .report files  OR  taxa × samples count matrix (TSV/CSV)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1 · Ingestion                                             │
│  Parse Kraken2 reports → taxa × samples count matrix           │
│  Compute relative abundances (each sample sums to 1.0)         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2 · Taxa Filtering                                        │
│  Drop taxa in < 10% of samples OR < 50 total reads             │
│  Removes noise; keeps ecologically meaningful taxa              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3 · Co-occurrence Graph (FDR-corrected)                   │
│  Spearman correlation between all taxon pairs across samples    │
│  Benjamini-Hochberg FDR correction prevents false edges         │
│  Produces a sparse, ecologically meaningful network             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4 · Stochastic Block Model (SBM)                          │
│  Variational EM community detection on the co-occurrence graph  │
│  Finds K=2..12 microbial communities; selects K via ICL         │
│  Identifies "pathogen-enriched" communities                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5 · Novelty Detection                                     │
│  Isolation Forest: samples with unusual taxonomic profiles      │
│  flagged even if no known pathogen is present                   │
│  Catches organisms not yet in the pathogen database             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5.5 · VQ-VAE Sequence Embedding (optional)               │
│  Train on reference genomes; score new sequences by             │
│  reconstruction error — high error = novel / divergent variant  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 6 · Risk Scoring                                          │
│  Composite score: abundance (50%) + community (25%) +           │
│  novelty (25%), plus WBE-calibrated direct detection override   │
│  Assigns LOW / MODERATE / HIGH / CRITICAL per sample            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 6.5 · Temporal Analysis                                   │
│  Baseline Z-score · CUSUM changepoint · Mann-Kendall trend      │
│  Reads from SQLite history DB; records this run for next time   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 7 · Characterization (optional, --characterize)           │
│  For flagged pathogens: fetch protein sequences from NCBI       │
│  → ESMFold 3D structure prediction (pLDDT confidence score)     │
│  → virulence factor annotation (toxin/adhesin/secretion system) │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 8 · Report + Alerts                                       │
│  JSON + HTML reports saved to disk                              │
│  Email / Slack alerts fired for HIGH / CRITICAL samples         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Risk Scoring

Each sample receives a **0–1 composite risk score** from three signals:

| Signal | Weight | What it measures |
|--------|--------|-----------------|
| Pathogen abundance | 50% | Known pathogens present × their calibrated risk weight |
| Community context | 25% | Is the sample dominated by a pathogen-enriched community? |
| Novelty | 25% | How anomalous is this sample's taxonomic profile? |

**Direct detection override (WBE-calibrated):** If a BSL-3 pathogen (Yersinia, Ebola, etc.) reaches ≥5% of reads, or a high-risk pathogen (Vibrio, Salmonella, etc.) reaches ≥15%, the score is forced to CRITICAL/HIGH regardless of community or novelty. This follows CDC/WHO wastewater action thresholds.

| Score | Level | Action |
|-------|-------|--------|
| 0.8 – 1.0 | **CRITICAL** | Outbreak-level — emergency response protocol |
| 0.6 – 0.8 | **HIGH** | Alert — investigate immediately, notify health authorities |
| 0.3 – 0.6 | **MODERATE** | Elevated — increase sampling frequency, monitor closely |
| 0.0 – 0.3 | **LOW** | Routine environmental background |

---

## Pathogen Database

Built-in database of **80 genera** with calibrated risk weights (0–1), curated from WHO priority pathogens, CDC NNDSS, ESKAPE, and BSL classification lists.

**Bacteria (34 genera)**

| Tier | Examples | Risk weight |
|------|---------|------------|
| Critical / BSL-3 | *Yersinia* (plague), *Francisella* (tularemia), *Brucella*, *Burkholderia* (melioidosis) | 0.90–0.95 |
| High | *Vibrio* (cholera), *Salmonella*, *Shigella*, *Mycobacterium* (TB), *Listeria* | 0.70–0.85 |
| ESKAPE | *Klebsiella*, *Acinetobacter*, *Pseudomonas*, *Enterococcus* | 0.60–0.65 |

**Viruses (42 genera)**

| Category | Examples | Risk weight |
|----------|---------|------------|
| Hemorrhagic fever (BSL-4) | *Ebolavirus*, *Marburgvirus*, *Mammarenavirus* (Lassa) | 0.90–0.98 |
| Respiratory | *Betacoronavirus* (COVID/SARS), *Alphainfluenzavirus* (flu A), *Orthopneumovirus* (RSV), *Morbillivirus* (measles) | 0.75–0.90 |
| Arboviral | *Flavivirus* (dengue/Zika/WNV), *Alphavirus* (chikungunya), *Orthonairovirus* (CCHF) | 0.78–0.90 |
| Poxvirus | *Orthopoxvirus* (mpox/smallpox) | 0.88 |
| Bloodborne | *Lentivirus* (HIV), *Orthohepadnavirus* (HBV), *Hepacivirus* (HCV) | 0.75–0.80 |
| GI / Waterborne | *Norovirus*, *Hepatovirus* (HAV), *Orthohepevirus* (HEV), *Rotavirus* | 0.65–0.75 |

**Fungi / Parasites (4 genera):** *Candida*, *Aspergillus*, *Cryptosporidium*, *Giardia*

---

## Temporal Monitoring

PathogenIQ maintains a **SQLite history database** (`~/.pathogeniq/history.db`) and runs three temporal analyses on every pipeline run:

**Baseline Z-score** — How far is today's risk score from the rolling historical mean? Z > 2.0 triggers a baseline anomaly flag.

**CUSUM (Cumulative Sum)** — The CDC/ECDC gold standard for outbreak detection. Unlike a single-point threshold, CUSUM accumulates small consistent increases over time, catching slow-rising outbreaks before they cross any single alert threshold. Signal strengths: none → weak → moderate → strong.

**Mann-Kendall Trend + Forecast** — Non-parametric trend test (no normality assumption). Returns trend direction (increasing / decreasing / stable), Kendall's τ strength coefficient, and a 1-week-ahead forecast with 80% confidence interval via Holt's double exponential smoothing.

All three signals appear in the JSON report, HTML report, and dashboard.

---

## Novel Pathogen Characterization

When `--characterize` is enabled (or `CHARACTERIZE = True` in `run.py`), PathogenIQ runs a three-step characterization on every flagged HIGH/CRITICAL taxon:

1. **NCBI protein fetch** — queries NCBI Protein DB for virulence-associated sequences for the taxon
2. **ESMFold structure prediction** — sends sequences to Meta's ESMFold API (free, no GPU, ~30–60s per protein); returns a PDB-format 3D structure and a pLDDT confidence score (0–100, higher = better)
3. **Virulence annotation** — scans protein titles against a curated keyword list (toxin, hemolysin, invasin, adhesin, protease, secretion system, etc.)

**Risk tiers from characterization:**
- **HIGH CONCERN** — pLDDT > 70 AND virulence keywords matched: high-confidence structure with known virulence factors
- **MODERATE CONCERN** — virulence keywords matched but low structure confidence: recommend further sequencing
- **MONITOR** — high-confidence structure but no virulence keywords: novel protein, further analysis needed
- **LOW CONCERN** — no virulence markers and low structure confidence

Run characterization on a specific taxon directly:
```bash
pathogeniq characterize "Acinetobacter"
pathogeniq characterize "Yersinia pestis"
pathogeniq characterize Salmonella --no-fold   # NCBI lookup only, skip ESMFold
```

> **What this does NOT do:** This is not variant/mutation calling (SNPs, indels). That requires raw reads + a reference genome + a variant caller (e.g. GATK, iVar). PathogenIQ operates at the taxon abundance level. Characterization provides structural and functional context for flagged genera, not nucleotide-level variant analysis.

---

## Web Dashboard

```bash
pathogeniq dashboard --report ./reports/report.json
# → http://localhost:8765
```

Or use `run.py` with `LAUNCH_DASHBOARD = True` — it launches automatically after the pipeline finishes.

**Login:** HTTP Basic Auth is enabled by default. Default credentials: `admin` / `pathogeniq`. Set via env vars or `run.py`:
```bash
export PATHOGENIQ_DASH_USER="admin"
export PATHOGENIQ_DASH_PASS="your-password"
export PATHOGENIQ_DASH_AUTH="false"   # disable auth entirely
```

**Dashboard features:**
- **Summary cards** — Critical / High / Moderate / Low sample counts
- **Surveillance map** — Leaflet.js map with risk-colored markers per site (requires `site_locations.json`)
- **Sites table** — sortable by risk score, Z-score, CUSUM signal, trend, forecast; live search bar and risk level filter pills
- **Keyboard navigation** — ↑/↓ to move between sites, Enter to select, Escape to close
- **Alert feed** — HIGH/CRITICAL sites with top detected pathogens
- **Site detail panel** — animated risk score meter, score history chart, CUSUM changepoint chart, pathogen breakdown
- **Auto-refresh** every 60 seconds

> **Offline / restricted networks:** Chart.js and Leaflet.js are bundled directly into `index.html` — no external requests are made. The dashboard works fully offline.

### Setting up the map view

The easiest way is to run `fetch_metadata.py` (see below) — it fills in `site_locations.json` automatically for any SRA dataset. If your samples are not from SRA, edit the file manually:

```json
{
  "my_site_A": { "lat": 33.748, "lon": -84.387, "label": "Atlanta WWTP North" },
  "my_site_B": { "lat": 33.755, "lon": -84.412, "label": "Atlanta WWTP South" }
}
```

Keys must match the sample names exactly as they appear in the pipeline report (after paired-end merging). Sites without coordinates are still shown in the table — just not on the map.

---

## Fetching Sample Metadata from NCBI (`fetch_metadata.py`)

If your Kraken2 reports came from public SRA data (filenames contain `SRR`, `ERR`, or `DRR` accession IDs), `fetch_metadata.py` automates everything:

- Extracts accession IDs from your report filenames
- Fetches title, study name, geographic location, isolation source, and collection date from NCBI BioSample
- Geocodes location strings to lat/lon (tries OpenStreetMap Nominatim; falls back to a built-in US state/county table for restricted networks)
- Writes `site_locations.json` so the dashboard map has pins immediately
- Writes `reports/metadata.tsv` — a flat metadata table you can use to assign group labels for differential abundance analysis

**Setup** — open `fetch_metadata.py` and set one variable:

```python
INPUT_DIR = "/path/to/your/kraken_reports"   # same as INPUT in run.py
```

**Run:**

```bash
python3 fetch_metadata.py
```

**Output:**

```
Scanning /path/to/reports for SRA accessions...
  Found 9 sample(s), 9 unique accession(s):
    cleaned_SRR35556007  →  SRR35556007
    cleaned_SRR35939735  →  SRR35939735
    ...

Fetching NCBI metadata for 9 accessions...
  Retrieved 9 record(s) from NCBI.

Geocoding locations via OpenStreetMap Nominatim...
  SRR35556007: geocoded 'USA: Colorado, Boulder County' → (40.093, -105.371)
  SRR35939735: geocoded 'USA: Wisconsin' → (43.073, -89.401)
  ...

Wrote site_locations.json → pathogeniq/dashboard/site_locations.json
Wrote metadata.tsv        → reports/metadata.tsv  (9 samples, 49 columns)
```

> **Paired-end files** (`_1.report` / `_2.report`) are automatically collapsed to one entry per accession — matching how the pipeline merges them.

> **Offline / restricted networks:** If Nominatim is unreachable, coordinates are looked up from a built-in table covering all 50 US states and common Colorado counties. Add entries to `_US_FALLBACK` in the script for other locations.

> **NCBI API key (optional):** Add your free NCBI API key to `NCBI_API_KEY` in the script to raise the rate limit from 3 to 10 requests/second. Get one at [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/).

---

## Running PathogenIQ (CLI)

```bash
# Basic run
pathogeniq run ./kraken_reports/

# Species-level
pathogeniq run ./kraken_reports/ --rank S

# With ESMFold characterization of flagged pathogens
pathogeniq run ./kraken_reports/ --characterize

# Full options
pathogeniq run ./kraken_reports/ \
    --rank G \
    --output ./reports \
    --alert-threshold 0.6 \
    --alert-email \
    --alert-slack \
    --characterize

# View a report summary
pathogeniq summary reports/report.json

# Watch a directory for new data
pathogeniq watch ./data --interval 3600 --alert-slack

# Validation benchmark
pathogeniq benchmark
```

---

## Automated Alerts

Configure via environment variables or `run.py`:

```bash
export PATHOGENIQ_SLACK_WEBHOOK="https://hooks.slack.com/services/..."
export PATHOGENIQ_SMTP_USER="you@gmail.com"
export PATHOGENIQ_SMTP_PASS="your-16-char-app-password"
export PATHOGENIQ_ALERT_EMAILS="pi@gsu.edu,labdirector@gsu.edu"

pathogeniq run ./kraken_reports/ --alert-email --alert-slack
```

Or set in `configs/default.yaml` under the `alerting:` section.

---

## Docker Deployment

```bash
cp .env.example .env   # fill in credentials
docker compose up -d   # starts dashboard + automated watcher

# Dashboard at http://your-server:8765
# Drop Kraken2 reports into ./data/ — watcher picks them up hourly
```

---

## Kraken2 Database

PathogenIQ reads Kraken2 output but does not ship a reference database.

| Database | Size | Use case |
|----------|------|----------|
| MiniKraken2 | ~8 GB | Research / development — covers all major genera |
| Standard | ~50 GB | Production surveillance |
| PlusPF | ~80 GB | Maximum sensitivity (adds viruses, fungi, human) |

```bash
wget https://genome-idx.s3.amazonaws.com/kraken/minikraken2_v2_8GB_201904_UPDATE.tgz
tar -xzf minikraken2_v2_8GB_201904_UPDATE.tgz

# Set in configs/default.yaml:
# kraken2:
#   db_path: "/path/to/minikraken2_v2_8GB"
```

---

## Configuration

Edit `configs/default.yaml` to tune the pipeline:

```yaml
graph:
  spearman_threshold: 0.45   # Minimum |rho| for co-occurrence edge
  fdr_alpha: 0.05            # BH-FDR significance cutoff
  min_prevalence: 0.1        # Min fraction of samples taxon must appear in
  min_total_reads: 50        # Min summed reads across all samples

sbm:
  max_k: 12                  # Maximum communities to try
  n_init: 10                 # EM random restarts

risk:
  alert_threshold: 0.6       # Score above which HIGH alert fires

vqvae:
  enabled: false
  model_path: ""             # Path to trained checkpoint

reporting:
  output_dir: "./reports"
```

---

## Module Reference

| Module | Description |
|--------|-------------|
| `run.py` | Simple config-at-the-top runner — edit INPUT/OUTPUT/params, then `python3 run.py` |
| `ingestion/reader.py` | Parses Kraken2 `.report` files or TSV/CSV count matrices into a unified `SampleSet` |
| `detection/kraken.py` | Kraken2 subprocess wrapper — runs classification on raw FASTQ files |
| `community/graph.py` | FDR-corrected Spearman co-occurrence network builder |
| `community/sbm.py` | Variational EM Stochastic Block Model with ICL model selection (K=2–12) |
| `novelty/detector.py` | Isolation Forest + z-score spike detection for taxonomic anomalies |
| `scoring/risk.py` | Composite risk scorer against 80-pathogen database with WBE-calibrated direct detection |
| `temporal/store.py` | SQLite history store — records every run, provides site history queries |
| `temporal/baseline.py` | Rolling baseline Z-score computation |
| `temporal/cusum.py` | CUSUM changepoint detection (CDC/ECDC standard algorithm) |
| `temporal/trend.py` | Mann-Kendall trend test + Holt's exponential smoothing forecast |
| `characterization/alphafold.py` | NCBI protein fetch + ESMFold structure prediction + virulence annotation |
| `embedding/vqvae.py` | VQ-VAE model for sequence-level novelty detection |
| `embedding/train.py` | VQ-VAE training script |
| `alerting/dispatcher.py` | Unified alert dispatcher (email + Slack) |
| `benchmark/runner.py` | Ground-truth benchmark suite with 6 synthetic scenarios |
| `scheduler/watcher.py` | Directory poller for automated pipeline triggering |
| `dashboard/app.py` | FastAPI server — Basic Auth, map, sites, alerts, CUSUM, characterization APIs |
| `dashboard/index.html` | Self-contained SPA — Chart.js + Leaflet.js bundled inline (no CDN) |
| `dashboard/site_locations.json` | GPS coordinates for map view markers (auto-populated by `fetch_metadata.py`) |
| `reporting/report.py` | JSON + HTML report generation with temporal signals |
| `pipeline/runner.py` | End-to-end orchestrator — runs all steps in sequence |
| `clustering/cluster.py` | Unsupervised hierarchical clustering of samples (auto-selects k) |
| `clustering/differential.py` | Kruskal-Wallis differential abundance with BH-FDR correction |
| `fetch_metadata.py` | Fetch NCBI SRA metadata, geocode locations, populate `site_locations.json` and `metadata.tsv` |

---

## Server Deployment (Linux HPC)

**Port forwarding** — access the dashboard from your laptop:
```bash
ssh -L 8765:localhost:8765 username@your-server.edu
# Then open http://localhost:8765
```

**Persistent history** — on shared HPC, point history DB to scratch storage:
```bash
export PATHOGENIQ_DB=/scratch/your_project/pathogeniq.db
```

**VQ-VAE on Linux** — PyTorch available on all Python versions:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pathogeniq train-embedder reference_genomes.fasta --epochs 100
```

---

## Background

PathogenIQ was built on two prior research projects:

1. **Genomic Sequence Detection** — VQ-VAE discrete representation learning for viral variant clustering in wastewater metagenomics
2. **Probabilistic Graph-Based Sequence Reconstruction** — Stochastic Block Model co-occurrence network analysis of Kraken2-classified wastewater samples (see [AAB Project](https://github.com/Ewenam/Probabilistic-Graph-Based-Sequence-Reconstruction))

PathogenIQ unifies and productionizes both, fixing key methodological issues (SBM degeneracy from over-dense graphs, absence of FDR correction, no quantitative risk scoring) and adding temporal outbreak tracking, automated alerting, web dashboard, Docker deployment, and a validated benchmark suite.

---

## Author

Richmond Azumah — [GitHub](https://github.com/Ewenam)
