"""
embedding/embedder.py
Sequence-level novelty detection using a trained VQ-VAE checkpoint.

Workflow:
  1. Train VQ-VAE on reference sequences (pathogeniq train-embedder)
  2. Score new sequences — high reconstruction loss = novel / divergent

Usage:
    from pathogeniq.embedding.embedder import score_sequences

    results = score_sequences({"SARS-CoV-2": "ATCGATCG..."})
    for taxon, emb in results.items():
        print(taxon, emb.novelty_score, emb.is_novel)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tokenizer import tokenize_sequence, VOCAB_SIZE
from .vqvae import VQVAE, _TORCH_AVAILABLE

DEFAULT_MODEL_PATH = Path.home() / ".pathogeniq" / "vqvae.pt"
_DEFAULT_THRESHOLD = 0.05   # reconstruction MSE above this → novel


@dataclass
class SequenceEmbedding:
    taxon: str
    codebook_index: int          # nearest codebook vector index
    embedding: list[float]       # quantized latent vector
    reconstruction_loss: float   # MSE reconstruction error (novelty proxy)
    is_novel: bool               # True if recon_loss > threshold
    novelty_score: float         # 0–1 normalized score


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> VQVAE | None:
    """Load a trained VQ-VAE checkpoint. Returns None if file not found or torch unavailable."""
    if not _TORCH_AVAILABLE:
        return None
    import torch
    model_path = Path(model_path)
    if not model_path.exists():
        return None
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    cfg = state.get("config", {})
    model = VQVAE(
        vocab_size=cfg.get("vocab_size", VOCAB_SIZE),
        hidden_dim=cfg.get("hidden_dim", 256),
        latent_dim=cfg.get("latent_dim", 64),
        num_embeddings=cfg.get("num_embeddings", 512),
    )
    model.load_state_dict(state["model"])
    model.eval()
    return model


def embed_sequence(
    taxon: str,
    sequence: str,
    model: VQVAE,
    threshold: float = _DEFAULT_THRESHOLD,
) -> SequenceEmbedding:
    """Embed a single sequence and compute its novelty score."""
    import torch
    x = torch.tensor(tokenize_sequence(sequence), dtype=torch.float32).unsqueeze(0)
    z_q, indices = model.encode(x)
    recon = model.reconstruction_loss(x)
    # Normalize to 0–1: loss of 2× threshold → score 1.0
    novelty_score = float(np.clip(recon / (threshold * 2), 0.0, 1.0))
    return SequenceEmbedding(
        taxon=taxon,
        codebook_index=int(indices[0].item()),
        embedding=z_q[0].tolist(),
        reconstruction_loss=round(recon, 6),
        is_novel=recon > threshold,
        novelty_score=round(novelty_score, 4),
    )


def score_sequences(
    sequences: dict[str, str],
    model_path: str | Path = DEFAULT_MODEL_PATH,
    threshold: float = _DEFAULT_THRESHOLD,
) -> dict[str, SequenceEmbedding]:
    """
    Score multiple sequences for novelty using a trained VQ-VAE.

    Args:
        sequences: {taxon_name: dna_or_rna_sequence}
        model_path: path to .pt checkpoint (from `pathogeniq train-embedder`)
        threshold: reconstruction MSE above which a sequence is flagged novel

    Returns:
        {} if no model found, else {taxon_name: SequenceEmbedding}
    """
    model = load_model(model_path)
    if model is None:
        return {}

    results = {}
    for taxon, seq in sequences.items():
        if seq and len(seq) >= 6:
            results[taxon] = embed_sequence(taxon, seq, model, threshold)
    return results


def novelty_scores_from_embeddings(
    embeddings: dict[str, SequenceEmbedding],
) -> dict[str, float]:
    """Extract {taxon: novelty_score} from embedding results."""
    return {taxon: emb.novelty_score for taxon, emb in embeddings.items()}
