"""
embedding/maskedvq.py
Vendored MaskedVQ-Seq architecture (conv VQ-VAE over ordered k-mer tokens).

This is the trained read-level model from the `genomic_sequence_detection`
project (Azumah 2026) — NOT the placeholder frequency-vector VQ-VAE in
`embedding/vqvae.py`. It is reproduced verbatim so the released checkpoints
(`masked_vq_best_model.pt`, `contrastive_best_model.pt`) load with matching
state-dict keys.

Model summary (from checkpoint `args`):
  vocab_size=4099 (4^6 canonical 6-mers + PAD/UNK/MASK), max_seq_length=150,
  num_codes=512, code_dim=64, embed_dim=128, hidden_dim=256, ~450K params.

Novelty is read-level:
  * reconstruction cross-entropy of the decoder's token logits, and
  * mean quantization distance to the learned codebook,
  * plus (contrastive head) distance of the SimCLR embedding from a reference
    lineage distribution.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # torch is an optional dependency
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    _Module = nn.Module
else:  # allow import without torch; construction will raise
    _Module = object


class VectorQuantizerEMA(_Module):
    """EMA-updated vector quantizer (per-position). Inference-time forward only
    needs the codebook; EMA branch is training-only and skipped in eval()."""

    def __init__(self, num_codes=512, code_dim=64,
                 commitment_cost=0.1, decay=0.90, eps=1e-5):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Embedding(num_codes, code_dim)
        self.embedding.weight.data.uniform_(-1 / num_codes, 1 / num_codes)

        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

    def forward(self, z_e):  # (B, L, D)
        B, L, D = z_e.shape
        flat = z_e.reshape(-1, D)
        emb = self.embedding.weight
        dists = (flat.pow(2).sum(dim=1, keepdim=True)
                 - 2 * flat @ emb.t()
                 + emb.pow(2).sum(dim=1, keepdim=True).t())  # (N, K)
        codes = torch.argmin(dists, dim=1)
        z_q = F.embedding(codes, emb).view(B, L, D)
        z_q_st = z_e + (z_q - z_e).detach()
        commit = F.mse_loss(z_e, z_q.detach())
        loss_vq = self.commitment_cost * commit
        # min distance per position = quantization error (novelty proxy)
        min_dist = dists.gather(1, codes.unsqueeze(1)).view(B, L)
        return z_q_st, loss_vq, codes.view(B, L), min_dist


class Encoder(_Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 code_dim=64, pad_id=None):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, code_dim, kernel_size=1),
        )
        self.post = nn.Sequential(nn.LayerNorm(code_dim), nn.Dropout(0.1))

    def forward(self, x):  # (B, L)
        emb = self.token_emb(x)
        h = emb.transpose(1, 2)
        h = self.conv(h)
        h = h.transpose(1, 2)
        return self.post(h)


class Decoder(_Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 code_dim=64, pad_id=None):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(code_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, embed_dim, kernel_size=1),
            nn.ReLU(),
        )
        self.out_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, z_q):  # (B, L, D)
        h = z_q.transpose(1, 2)
        h = self.conv(h)
        h = h.transpose(1, 2)
        h = self.out_norm(h)
        return self.output_proj(h)  # (B, L, V)


class MaskedVQVAE(_Module):
    """Full conv VQ-VAE. `forward` runs deterministically in eval() (no noise)."""

    def __init__(self, vocab_size, pad_id, num_codes=512, code_dim=64,
                 embed_dim=128, hidden_dim=256, commitment_cost=0.1, decay=0.95):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim, code_dim, pad_id)
        self.vq = VectorQuantizerEMA(num_codes, code_dim,
                                     commitment_cost=commitment_cost, decay=decay)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim, code_dim, pad_id)

    def forward(self, x):
        z_e = self.encoder(x)
        if self.training:
            z_e = z_e + 0.1 * torch.randn_like(z_e)
        z_q, loss_vq, codes, min_dist = self.vq(z_e)
        logits = self.decoder(z_q)
        return logits, loss_vq, codes, min_dist


class ContrastiveHead(_Module):
    """Pretrained encoder + SimCLR projection head. Produces unit-norm
    embeddings for lineage separation / distribution-anomaly novelty."""

    def __init__(self, encoder, embed_dim=64, proj_dim=64):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, tokens):
        z_e = self.encoder(tokens)
        z_mean = z_e.mean(dim=1)
        z_proj = self.proj(z_mean)
        return F.normalize(z_proj, dim=-1)


# ── K-mer tokenizer (self-contained; no BioPython needed for tokenization) ─────
from itertools import product as _product


class KmerTokenizer:
    """Canonical-order k-mer tokenizer matching the trained model.

    Builds all 4^k k-mers in fixed ACGT product order (indices 0..4^k-1), then
    PAD/UNK/MASK. The k-mer→index map is identical regardless of vocab size, so
    reads tokenize consistently for both the masked (vocab 4099) and contrastive
    (vocab 4097) checkpoints. Non-ACGT k-mers are skipped (never emit UNK).
    """

    def __init__(self, k: int = 6, max_len: int = 150):
        self.k = k
        self.max_len = max_len
        self.kmers = ["".join(p) for p in _product("ACGT", repeat=k)]
        self.stoi = {km: i for i, km in enumerate(self.kmers)}
        for tok in ("<PAD>", "<UNK>", "<MASK>"):
            self.stoi[tok] = len(self.stoi)
        self.pad_id = self.stoi["<PAD>"]
        self.unk_id = self.stoi["<UNK>"]
        self.mask_id = self.stoi["<MASK>"]

    def encode_seq(self, seq: str) -> list[int]:
        seq = seq.upper()
        acgt = set("ACGT")
        toks: list[int] = []
        k = self.k
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            if set(kmer) <= acgt:
                toks.append(self.stoi[kmer])
            if len(toks) >= self.max_len:
                break
        if len(toks) < self.max_len:
            toks += [self.pad_id] * (self.max_len - len(toks))
        return toks[:self.max_len]
