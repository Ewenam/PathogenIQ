# PathogenIQ

**AI-powered biosurveillance platform for detecting and characterizing disease-causing pathogens from sequencing data.**

PathogenIQ turns raw metagenomic sequencing data (wastewater, clinical, environmental) into actionable risk intelligence — identifying known pathogens, flagging novel organisms, and characterizing their threat potential using protein structure prediction.

---

## What it does

```
Raw sequencing data (FASTQ / Kraken2 reports)
        ↓
Taxa classification & filtering
        ↓
FDR-corrected co-occurrence graph
        ↓
Stochastic Block Model community detection
        ↓
Isolation Forest novelty detection
        ↓
47-pathogen risk scoring (WHO + CDC priority list)
        ↓
ESMFold structure prediction for novel pathogens
        ↓
JSON + HTML surveillance report with alerts
```

---

## Quickstart

```bash
git clone https://github.com/Ewenam/PathogenIQ.git
cd PathogenIQ
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Run on a directory of Kraken2 reports:**
```bash
python cli.py run ./kraken_reports/
```

**Run on a pre-built taxa × samples count matrix:**
```bash
python cli.py run counts.tsv --rank S --output ./reports
```

**With AlphaFold/ESMFold characterization of flagged pathogens:**
```bash
python cli.py run ./kraken_reports/ --characterize
```

**View a saved report:**
```bash
python cli.py summary reports/report.json
```

**Characterize a specific taxon:**
```bash
python cli.py characterize "Yersinia pestis"
```

---

## Pipeline modules

| Module | Description |
|--------|-------------|
| `ingestion/reader.py` | Parses Kraken2 `.report` files or TSV/CSV count matrices into a unified `SampleSet` |
| `detection/kraken.py` | Kraken2 subprocess wrapper — runs classification on raw FASTQ |
| `community/graph.py` | Builds Spearman co-occurrence graph with **Benjamini-Hochberg FDR correction** (prevents degenerate block models) |
| `community/sbm.py` | Variational EM Stochastic Block Model with ICL model selection (K=2..12) |
| `novelty/detector.py` | Isolation Forest + per-taxon z-score spike detection |
| `scoring/risk.py` | Composite risk scorer against a curated 47-pathogen database (WHO, CDC, ESKAPE) |
| `characterization/alphafold.py` | Fetches protein sequences from NCBI and predicts 3D structure via ESMFold API |
| `reporting/report.py` | Generates JSON and HTML surveillance reports |
| `pipeline/runner.py` | End-to-end orchestrator — runs all steps in sequence |

---

## Risk levels

| Score | Level | Meaning |
|-------|-------|---------|
| 0.8 – 1.0 | **CRITICAL** | Outbreak-level signal — immediate action required |
| 0.6 – 0.8 | **HIGH** | Alert — investigate immediately |
| 0.3 – 0.6 | **MODERATE** | Elevated — monitor closely |
| 0.0 – 0.3 | **LOW** | Routine environmental background |

Risk is a weighted composite of three signals:
- **Pathogen abundance** (50%) — known pathogens from the WHO/CDC priority list weighted by danger level
- **Community context** (25%) — whether the sample's dominant taxa cluster with known pathogenic communities
- **Novelty** (25%) — anomaly score from Isolation Forest detecting unusual taxonomic profiles

---

## Pathogen database

The built-in database covers 47 genera across bacteria, viruses, fungi, and parasites:

- **Tier 1 (critical):** *Yersinia*, *Francisella*, *Brucella*, *Burkholderia*, *Vibrio*, *Clostridioides*
- **Tier 2 (high):** *Salmonella*, *Shigella*, *Listeria*, *Mycobacterium*, *Klebsiella*, *Acinetobacter* (ESKAPE)
- **Viral:** *Betacoronavirus* (COVID/SARS), *Orthomyxovirus* (Influenza), *Flavivirus* (Dengue/Zika), *Orthopoxvirus* (Mpox)
- **Fungi/parasites:** *Candida*, *Cryptosporidium*, *Giardia*

---

## Configuration

Edit `configs/default.yaml` to tune the pipeline:

```yaml
graph:
  spearman_threshold: 0.45   # Minimum |rho| for co-occurrence edge
  fdr_alpha: 0.05            # BH-FDR significance cutoff

sbm:
  max_k: 12                  # Maximum communities to try

risk:
  alert_threshold: 0.6       # Score above which an alert is raised

kraken2:
  db_path: ""                # Path to your Kraken2 database
  confidence: 0.1
```

---

## Requirements

- Python 3.9+
- Kraken2 (optional — can use pre-computed reports)
- Internet access for ESMFold API and NCBI lookups (optional — only needed for `--characterize`)

No GPU required. ESMFold runs via API.

---

## Roadmap

- [ ] Temporal modeling — track risk trends and detect slow-rising outbreaks week-over-week
- [ ] Web dashboard — real-time surveillance interface
- [ ] Viral Kraken2 database integration — full respiratory/enteric virus coverage
- [ ] VQ-VAE sequence embedding — variant-level anomaly detection within species
- [ ] Automated alerts — email/Slack notifications on threshold breach
- [ ] Ground truth validation — benchmarking on spiked positive controls

---

## Background

PathogenIQ was built on two prior research projects:

1. **Genomic Sequence Detection** — VQ-VAE discrete representation learning for viral variant clustering in wastewater metagenomics
2. **Probabilistic Graph-Based Sequence Reconstruction** — SBM co-occurrence network analysis of Kraken2-classified wastewater samples

PathogenIQ unifies and productionizes both, fixing key issues (SBM degeneracy from over-dense graphs, lack of FDR correction, no quantitative risk scoring) and adding the novel pathogen characterization layer via protein structure prediction.

---

## Author

Richmond Azumah — [GitHub](https://github.com/Ewenam)
