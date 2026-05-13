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
│  Step 7 · Characterization (optional)                           │
│  For novel flagged pathogens: fetch protein sequences (NCBI)    │
│  + ESMFold structure prediction + virulence annotation          │
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

## Installation

**Requirements:** Python 3.11+, internet access for ESMFold API (optional). No GPU required.

```bash
git clone https://github.com/Ewenam/PathogenIQ.git
cd PathogenIQ
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Verify installation:**
```bash
pathogeniq benchmark --quick
# Expected: Sensitivity=1.00, Precision=1.00, F1=1.00
```

---

## Running PathogenIQ

### Run the pipeline

```bash
# On a directory of Kraken2 .report files
pathogeniq run ./kraken_reports/

# On a pre-built taxa × samples count matrix (TSV)
pathogeniq run counts.tsv

# Species-level classification (default is genus G)
pathogeniq run ./kraken_reports/ --rank S

# Custom output directory
pathogeniq run ./kraken_reports/ --output ./reports

# With automated email + Slack alerts
pathogeniq run ./kraken_reports/ --alert-email --alert-slack

# With ESMFold protein characterization of novel flagged pathogens
pathogeniq run ./kraken_reports/ --characterize

# With VQ-VAE sequence-level novelty detection (requires trained model)
pathogeniq run ./kraken_reports/ --use-vqvae

# Full options
pathogeniq run ./kraken_reports/ \
    --rank G \
    --output ./reports \
    --alert-threshold 0.6 \
    --alert-email \
    --alert-slack \
    --characterize
```

### View a report summary in the terminal

```bash
pathogeniq summary reports/report.json
```

### Characterize a specific pathogen

```bash
pathogeniq characterize "Yersinia pestis"
pathogeniq characterize Salmonella --no-fold   # NCBI lookup only, skip ESMFold
```

### Launch the web dashboard

```bash
pathogeniq dashboard
# → http://localhost:8765
```

The dashboard shows:
- Summary cards: Critical / High / Moderate / Low counts
- Sortable sites table with Z-score, CUSUM signal bar, trend arrow (↑↓→), and forecast
- Alert feed (HIGH/CRITICAL sites with top pathogen)
- Click any site → risk score history chart + CUSUM chart + pathogen breakdown
- Auto-refreshes every 60 seconds

### Watch a directory for new data (automated scheduling)

```bash
# Check ./data every hour; run pipeline automatically on any new reports found
pathogeniq watch ./data --interval 3600

# With alerts and process existing data immediately on start
pathogeniq watch ./data --interval 1800 --alert-email --alert-slack --run-on-start
```

Processed inputs are tracked in `.pathogeniq_processed.json` — the watcher never re-runs the same data.

### Run the validation benchmark

```bash
pathogeniq benchmark                    # full pipeline, all 6 scenarios
pathogeniq benchmark --quick            # skip SBM, faster (risk scoring only)
pathogeniq benchmark --output bench.json
```

Runs 6 controlled synthetic scenarios (negative controls, low contamination, cholera outbreak, critical Yersinia, multi-pathogen community, novel agent) and reports sensitivity, specificity, precision, and F1.

### Train the VQ-VAE sequence embedder (optional)

```bash
# Train on a FASTA file of reference genomes
pathogeniq train-embedder reference_genomes.fasta --epochs 100

# GPU-accelerated training (Apple Silicon or CUDA)
pathogeniq train-embedder reference_genomes.fasta --epochs 200 --device mps

# Then enable in pipeline
pathogeniq run ./kraken_reports/ --use-vqvae
```

---

## Automated Alerts

Configure via environment variables (no code changes needed):

```bash
# Slack
export PATHOGENIQ_SLACK_WEBHOOK="https://hooks.slack.com/services/..."

# Email (Gmail example — use an App Password, not your login password)
export PATHOGENIQ_SMTP_USER="you@gmail.com"
export PATHOGENIQ_SMTP_PASS="your-16-char-app-password"
export PATHOGENIQ_ALERT_EMAILS="pi@gsu.edu,labdirector@gsu.edu"

# Run with alerts enabled
pathogeniq run ./kraken_reports/ --alert-email --alert-slack
```

Or configure in `configs/default.yaml` under the `alerting:` section.

---

## Docker Deployment

```bash
# Copy and fill in credentials
cp .env.example .env

# Start dashboard + automated watcher (both services)
docker compose up -d

# Dashboard only
docker compose up -d dashboard

# View logs
docker compose logs -f
```

- Dashboard available at `http://your-server:8765`
- Drop Kraken2 reports into `./data/` — the watcher picks them up automatically
- Reports and history database persist across container restarts via named volumes

---

## Kraken2 Database

PathogenIQ reads Kraken2 output but does not ship a reference database. You need a Kraken2 database installed separately.

**For research and development (recommended):** The **MiniKraken2** database (~8 GB) works well. It covers all major bacterial and viral genera in the PathogenIQ pathogen database. Sensitivity is slightly reduced at very low abundances, but outbreak-level signals (which trigger HIGH/CRITICAL alerts) are reliably detected.

**For production surveillance:** The **Standard** (~50 GB) or **PlusPF** (~80 GB, adds viruses + fungi + human) databases give maximum sensitivity.

```bash
# Download MiniKraken2 (8 GB)
wget https://genome-idx.s3.amazonaws.com/kraken/minikraken2_v2_8GB_201904_UPDATE.tgz
tar -xzf minikraken2_v2_8GB_201904_UPDATE.tgz

# Set in configs/default.yaml
# kraken2:
#   db_path: "/path/to/minikraken2_v2_8GB"

# Run Kraken2 on a FASTQ file
kraken2 --db /path/to/minikraken2_v2_8GB \
        --report sample.report \
        --classified-out sample_classified.fastq \
        sample.fastq

# Then run PathogenIQ on the resulting reports
pathogeniq run ./reports_dir/
```

---

## Configuration

Edit `configs/default.yaml` to tune the pipeline without changing code:

```yaml
graph:
  spearman_threshold: 0.45   # Minimum |rho| for co-occurrence edge
  fdr_alpha: 0.05            # BH-FDR significance cutoff

sbm:
  max_k: 12                  # Maximum communities to try
  n_init: 10                 # EM random restarts

risk:
  alert_threshold: 0.6       # Score above which HIGH alert is raised

alerting:
  email:
    enabled: false
    smtp_host: smtp.gmail.com
    to_addrs: []
  slack:
    enabled: false
    webhook_url: ""

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
| `embedding/tokenizer.py` | DNA k-mer (k=6) frequency vectorizer for sequence-level analysis |
| `embedding/vqvae.py` | VQ-VAE model with EMA-updated codebook for sequence novelty detection |
| `embedding/train.py` | VQ-VAE training script with validation and best-checkpoint saving |
| `alerting/dispatcher.py` | Unified alert dispatcher (email + Slack) |
| `benchmark/runner.py` | Ground-truth benchmark suite with 6 synthetic scenarios |
| `scheduler/watcher.py` | Directory poller for automated pipeline triggering |
| `dashboard/app.py` | FastAPI dashboard server (port 8765) |
| `dashboard/index.html` | Single-page Chart.js surveillance interface |
| `reporting/report.py` | JSON + HTML report generation with temporal signals |
| `pipeline/runner.py` | End-to-end orchestrator — runs all steps in sequence |

---

## Server Deployment (Linux HPC)

For deployment on a university or institutional server:

**Port forwarding** — If port 8765 is firewalled, access the dashboard from your laptop with:
```bash
ssh -L 8765:localhost:8765 username@your-server.edu
# Then open http://localhost:8765 in your browser
```

**Singularity** — Many HPC clusters use Singularity instead of Docker (rootless containers). The Dockerfile is compatible:
```bash
singularity build pathogeniq.sif docker://ghcr.io/ewenam/pathogeniq:latest
singularity exec pathogeniq.sif pathogeniq run ./data/
```

**PyTorch on Linux** — Unlike macOS + Python 3.13, Linux servers support PyTorch on all Python versions. VQ-VAE training and inference are available after:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Persistent history** — The SQLite history database lives at `~/.pathogeniq/history.db` by default. On shared HPC systems, set a custom path to persistent storage:
```bash
export PATHOGENIQ_DB=/scratch/your_project/pathogeniq.db
```

---

## Background

PathogenIQ was built on two prior research projects:

1. **Genomic Sequence Detection** — VQ-VAE discrete representation learning for viral variant clustering in wastewater metagenomics
2. **Probabilistic Graph-Based Sequence Reconstruction** — Stochastic Block Model co-occurrence network analysis of Kraken2-classified wastewater samples

PathogenIQ unifies and productionizes both, fixing key methodological issues (SBM degeneracy from over-dense graphs, absence of FDR correction, no quantitative risk scoring) and adding temporal outbreak tracking, automated alerting, web dashboard, Docker deployment, and a validated benchmark suite.

---

## Author

Richmond Azumah — [GitHub](https://github.com/Ewenam)
