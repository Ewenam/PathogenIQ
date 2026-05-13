"""
embedding/vqvae.py
VQ-VAE (Vector Quantized Variational Autoencoder) for DNA sequence embedding.

Architecture:
  Encoder:    k-mer freq vector → latent z  (Linear + LayerNorm + GELU stack)
  Quantizer:  z → z_q  (EMA-updated codebook; straight-through gradient)
  Decoder:    z_q → reconstructed k-mer freqs

Training objective:
  L = L_recon + β · L_commitment
  L_recon     = MSE(x̂, x)
  L_commitment = ||sg[z_q] - z||²   (commitment cost; codebook updated via EMA)

Reference: van den Oord et al. (2017) "Neural Discrete Representation Learning"
           Razavi et al. (2019) "Generating Diverse High-Fidelity Images with
                                  VQ-VAE-2"
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for VQ-VAE embedding.\n"
            "Install it for your platform:\n"
            "  pip install torch  (Linux / macOS Python ≤3.12)\n"
            "  See https://pytorch.org/get-started for other platforms."
        )


if _TORCH_AVAILABLE:
    _Module = nn.Module
else:
    _Module = object  # fallback so class definitions don't error at import


class VectorQuantizerEMA(_Module):
    """
    EMA-updated vector quantizer with Laplace smoothing.
    Codebook vectors are updated with exponential moving averages — more stable
    than gradient-based updates and avoids codebook collapse.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        _require_torch()
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        w = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding", w)
        self.register_buffer("cluster_size", torch.ones(num_embeddings))
        self.register_buffer("ema_embed", w.clone())

    def forward(self, z: torch.Tensor):
        # z: (B, D)
        distances = (
            torch.sum(z**2, dim=1, keepdim=True)
            + torch.sum(self.embedding**2, dim=1)
            - 2.0 * z @ self.embedding.T
        )  # (B, K)

        indices = distances.argmin(dim=1)  # (B,)
        z_q = self.embedding[indices]      # (B, D)

        if self.training:
            one_hot = F.one_hot(indices, self.num_embeddings).float()  # (B, K)
            n_i = one_hot.sum(0)  # (K,)
            self.cluster_size.mul_(self.decay).add_(n_i * (1 - self.decay))

            ema_embed_new = one_hot.T @ z.detach()  # (K, D)
            self.ema_embed.mul_(self.decay).add_(ema_embed_new * (1 - self.decay))

            # Laplace-smoothed normalisation
            n = self.cluster_size.sum()
            smoothed = (
                (self.cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )
            self.embedding.copy_(self.ema_embed / smoothed.unsqueeze(1))

        vq_loss = self.commitment_cost * F.mse_loss(z_q.detach(), z)
        # Straight-through estimator: gradients pass through as if z_q == z
        z_q_st = z + (z_q - z).detach()

        return z_q_st, vq_loss, indices


class VQVAE(_Module):
    """
    VQ-VAE for genomic sequence embedding and novelty detection.

    Input:  k-mer frequency vector  (vocab_size,)
    Output: reconstructed k-mer frequencies  (vocab_size,)

    Novelty proxy: reconstruction MSE — sequences unlike the training corpus
    yield high reconstruction error.
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
    ):
        _require_torch()
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

        self.quantizer = VectorQuantizerEMA(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
            decay=decay,
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size),
            nn.Softmax(dim=-1),  # output is a frequency distribution
        )

    def forward(self, x: torch.Tensor):
        """Returns (x_hat, total_loss, codebook_indices)."""
        z = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z)
        x_hat = self.decoder(z_q)
        recon_loss = F.mse_loss(x_hat, x)
        return x_hat, recon_loss + vq_loss, indices

    def encode(self, x):
        """Return (quantized_embedding, codebook_indices) without gradients."""
        import torch
        with torch.no_grad():
            self.eval()
            z = self.encoder(x)
            z_q, _, indices = self.quantizer(z)
            return z_q, indices

    def reconstruction_loss(self, x) -> float:
        """Return scalar MSE reconstruction loss (novelty proxy)."""
        import torch
        import torch.nn.functional as F_
        with torch.no_grad():
            self.eval()
            x_hat, _, _ = self(x)
            return float(F_.mse_loss(x_hat, x).item())
