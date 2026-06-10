---
name: project-paper-context
description: Research context for PathogenIQ paper — VQ-VAE vs ESMFold distinction, MaskedVQ-Seq sister paper, MLSP 2026 target
metadata:
  type: project
---

User co-authored a separate paper "Reference-Free Variant Detection in Wastewater Genomic Surveillance via Masked Vector-Quantized Autoencoders" (MaskedVQ-Seq) with a friend — currently under review at Machine Learning for Healthcare. This is NOT PathogenIQ; it's a standalone VQ-VAE paper for SARS-CoV-2 wastewater read-level clustering.

PathogenIQ is a separate system paper targeting MLSP 2026 (deadline June 12, 2026).

**Why:** The VQ-VAE and ESMFold in PathogenIQ are NOT redundant — they operate at different stages on different data types:
- VQ-VAE (embedding/): operates on DNA k-mer frequency vectors from Kraken2-classified taxa → novelty detection signal at the sample level
- ESMFold (characterization/): operates on amino acid sequences from NCBI → protein structure → virulence assessment, runs DOWNSTREAM only on already-flagged pathogens

**PathogenIQ paper angle:** System contribution — integrated ML biosurveillance pipeline. NOT another VQ-VAE paper (that's MaskedVQ-Seq). Key ML contributions: SBM community structure, multi-signal risk scoring, CUSUM temporal anomaly detection, ESMFold characterization. VQ-VAE is one component.

**How to apply:** Do not position PathogenIQ as a VQ-VAE paper. Position it as an end-to-end ML surveillance system. For MLSP, emphasize the ML pipeline integration and the "Applied ML in Healthcare" angle.
