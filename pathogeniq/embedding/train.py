"""
embedding/train.py
Train a VQ-VAE on reference genomic sequences for novelty detection.

The trained model learns a discrete codebook of "normal" sequence patterns.
At inference time, sequences that diverge from known references yield high
reconstruction error — this is the novelty signal.

Recommended reference databases:
  - NCBI RefSeq complete bacterial genomes
  - NCBI Viral RefSeq (for viral surveillance)
  - PathogenIQ ships without a pretrained model; run this once on your target DB.

Usage:
    pathogeniq train-embedder reference_genomes.fasta --epochs 100
    python -m pathogeniq.embedding.train reference_genomes.fasta
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as _e:
    raise ImportError(
        "PyTorch is required for VQ-VAE training.\n"
        "Install it: pip install torch\n"
        "See https://pytorch.org/get-started for platform-specific wheels."
    ) from _e

from .tokenizer import tokenize_fasta, VOCAB_SIZE
from .vqvae import VQVAE

DEFAULT_OUTPUT = Path.home() / ".pathogeniq" / "vqvae.pt"


def train(
    fasta_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT,
    # Model architecture
    hidden_dim: int = 256,
    latent_dim: int = 64,
    num_embeddings: int = 512,
    commitment_cost: float = 0.25,
    # Training
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.1,
    device: str = "cpu",
    quiet: bool = False,
) -> VQVAE:
    """
    Train a VQ-VAE on a FASTA file of reference sequences.

    Args:
        fasta_path:    FASTA file with reference genomes / contigs
        output_path:   where to save the .pt checkpoint
        epochs:        training epochs
        val_split:     fraction of data held out for validation loss
        device:        "cpu", "cuda", or "mps"

    Returns:
        Trained VQVAE model (also saved to output_path).
    """
    fasta_path = Path(fasta_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print(f"  Loading sequences from {fasta_path} ...")
    kmer_vecs = tokenize_fasta(fasta_path, min_length=100)
    if not quiet:
        print(f"  {len(kmer_vecs)} sequences tokenized")

    if len(kmer_vecs) < 10:
        raise ValueError(
            f"Need >= 10 sequences (min_length=100 bp), found {len(kmer_vecs)}. "
            "Check that the FASTA contains DNA sequences ≥ 100 bp."
        )

    vectors = list(kmer_vecs.values())
    X = torch.tensor(np.stack(vectors), dtype=torch.float32)

    # Train / validation split
    n_val = max(1, int(len(X) * val_split))
    perm = torch.randperm(len(X))
    X_train = X[perm[n_val:]]
    X_val = X[perm[:n_val]]

    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)

    device = torch.device(device)
    model = VQVAE(
        vocab_size=VOCAB_SIZE,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if not quiet:
        print(
            f"  Training VQ-VAE: {len(X_train)} train / {len(X_val)} val "
            f"| epochs={epochs} batch={batch_size} lr={lr}"
        )
        print(f"  Architecture: hidden={hidden_dim} latent={latent_dim} "
              f"codebook={num_embeddings}")

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, loss, _ = model(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if not quiet and (epoch % 10 == 0 or epoch == 1):
            model.eval()
            with torch.no_grad():
                X_val_d = X_val.to(device)
                _, val_loss, _ = model(X_val_d)
            avg_train = total_loss / len(train_loader)
            print(f"  Epoch {epoch:4d}/{epochs}  "
                  f"train={avg_train:.4f}  val={float(val_loss):.4f}")

        # Track best checkpoint
        model.eval()
        with torch.no_grad():
            _, val_loss, _ = model(X_val.to(device))
        val_loss_f = float(val_loss)
        if val_loss_f < best_val_loss:
            best_val_loss = val_loss_f
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    config = {
        "vocab_size": VOCAB_SIZE,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "num_embeddings": num_embeddings,
        "commitment_cost": commitment_cost,
        "best_val_loss": round(best_val_loss, 6),
        "n_train": len(X_train),
        "fasta": str(fasta_path),
    }
    torch.save({"model": model.state_dict(), "config": config}, output_path)
    if not quiet:
        print(f"  Best val loss: {best_val_loss:.4f}")
        print(f"  Checkpoint saved → {output_path}")

    return model
