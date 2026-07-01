"""
embedding/vqvae_novelty.py
Read-level reference-free novelty scoring with the trained MaskedVQ-Seq model.

Two complementary novelty signals, both computed from raw reads:

  1. Reconstruction novelty (masked_vq checkpoint)
       per-read token cross-entropy of the decoder + mean codebook distance.
       Reads unlike the training corpus reconstruct poorly / quantize far from
       the codebook.

  2. Embedding-distribution novelty (contrastive checkpoint)
       SimCLR embeddings are fit to a REFERENCE read set (known/background
       lineages) with an IsolationForest; query reads far from that manifold
       score as anomalous. This is the "emerging variant" detector.

The per-sample novelty score is the fraction of a sample's reads flagged
anomalous (or a normalized mean), mapped to [0, 1] to feed the composite
risk score's novelty term.

CPU-only friendly. torch/biopython are optional; import fails gracefully.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .maskedvq import _TORCH_AVAILABLE, KmerTokenizer

# Default checkpoint locations (the genomic_sequence_detection project).
_CKPT_DIR = Path.home() / "Desktop" / "genomic_sequence_detection" / "ckpts"
DEFAULT_MASKED_CKPT = _CKPT_DIR / "masked_vq_best_model.pt"
DEFAULT_CONTRASTIVE_CKPT = _CKPT_DIR / "contrastive_best_model.pt"


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for MaskedVQ-Seq novelty scoring.\n"
            "  pip install --index-url https://download.pytorch.org/whl/cpu torch"
        )


# ── checkpoint loading ─────────────────────────────────────────────────────────
def _strip_prefix(state: dict, prefix: str = "module.") -> dict:
    return {(k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in state.items()}


def load_masked_vq(ckpt_path: str | Path = DEFAULT_MASKED_CKPT):
    """Load the full conv VQ-VAE for reconstruction novelty."""
    _require_torch()
    import torch
    from .maskedvq import MaskedVQVAE

    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    tok = KmerTokenizer(k=a.get("k_mer", 6), max_len=a.get("max_seq_length", 150))
    model = MaskedVQVAE(
        vocab_size=a.get("vocab_size", len(tok.stoi)),
        pad_id=tok.pad_id,
        num_codes=a.get("num_codes", 512),
        code_dim=a.get("code_dim", 64),
        embed_dim=a.get("embed_dim", 128),
        hidden_dim=a.get("hidden_dim", 256),
        commitment_cost=a.get("commitment_cost", 0.1),
    )
    model.load_state_dict(_strip_prefix(ck["model_state_dict"]))
    model.eval()
    return model, tok


def load_contrastive(ckpt_path: str | Path = DEFAULT_CONTRASTIVE_CKPT):
    """Load the encoder + SimCLR projection head for embedding novelty."""
    _require_torch()
    import torch
    from .maskedvq import Encoder, ContrastiveHead

    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    sd = _strip_prefix(ck["model_state_dict"])
    vocab_size = sd["encoder.token_emb.weight"].shape[0]
    code_dim = sd["encoder.post.0.weight"].shape[0]
    embed_dim = sd["encoder.token_emb.weight"].shape[1]
    proj_dim = sd["proj.2.weight"].shape[0]
    tok = KmerTokenizer(k=a.get("k_mer", 6), max_len=a.get("max_seq_length", 150))
    enc = Encoder(vocab_size, embed_dim=embed_dim, hidden_dim=256,
                  code_dim=code_dim, pad_id=tok.pad_id)
    head = ContrastiveHead(enc, embed_dim=code_dim, proj_dim=proj_dim)
    head.load_state_dict(sd)
    head.eval()
    return head, tok


# ── read IO ─────────────────────────────────────────────────────────────────────
def read_fastq(path: str | Path, limit: int | None = None) -> list[str]:
    """Read sequences from a FASTQ (every 4th line), optionally capped."""
    seqs: list[str] = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i % 4 == 1:
                seqs.append(line.strip())
                if limit and len(seqs) >= limit:
                    break
    return seqs


def _batch_tokens(tok: KmerTokenizer, seqs: list[str]):
    import torch
    return torch.tensor([tok.encode_seq(s) for s in seqs], dtype=torch.long)


# ── per-read scoring ─────────────────────────────────────────────────────────────
def reconstruction_scores(model, tok, seqs: list[str], batch_size: int = 512):
    """Per-read (cross_entropy, quant_distance) from the masked VQ-VAE."""
    import torch
    import torch.nn.functional as F
    ce_all, qd_all = [], []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            x = _batch_tokens(tok, seqs[i:i + batch_size])
            logits, _, _, min_dist = model(x)          # (B,L,V), (B,L)
            B, L, V = logits.shape
            ce = F.cross_entropy(
                logits.reshape(-1, V), x.reshape(-1), reduction="none"
            ).view(B, L)
            mask = (x != tok.pad_id).float()           # ignore pad positions
            denom = mask.sum(1).clamp(min=1.0)
            ce_all.append(((ce * mask).sum(1) / denom).cpu().numpy())
            qd_all.append(((min_dist * mask).sum(1) / denom).cpu().numpy())
    return np.concatenate(ce_all), np.concatenate(qd_all)


def embed_reads(head, tok, seqs: list[str], batch_size: int = 512) -> np.ndarray:
    """Per-read unit-norm SimCLR embeddings (N, proj_dim)."""
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            x = _batch_tokens(tok, seqs[i:i + batch_size])
            out.append(head(x).cpu().numpy())
    return np.concatenate(out, axis=0)


# ── reference-based embedding anomaly ────────────────────────────────────────────
@dataclass
class EmbeddingReference:
    """IsolationForest fit on reference (background/known-lineage) embeddings."""
    detector: object
    mean: np.ndarray
    contamination: float = 0.05

    @classmethod
    def fit(cls, ref_embeddings: np.ndarray, contamination: float = 0.05):
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(contamination=contamination, random_state=42,
                              n_estimators=200)
        iso.fit(ref_embeddings)
        return cls(detector=iso, mean=ref_embeddings.mean(0),
                   contamination=contamination)

    def anomaly(self, query: np.ndarray) -> np.ndarray:
        """Per-read anomaly in [0,1] (1 = most anomalous vs reference)."""
        raw = self.detector.decision_function(query)   # higher = more normal
        # map to [0,1] with 1 = most anomalous
        return 1.0 - (raw - raw.min()) / (np.ptp(raw) + 1e-9)


# ── high-level per-sample novelty ────────────────────────────────────────────────
@dataclass
class SampleNovelty:
    n_reads: int
    recon_ce_mean: float
    quant_dist_mean: float
    recon_novelty: float                 # [0,1]
    embedding_novelty: float | None      # [0,1] if a reference was provided
    novelty_score: float                 # fused [0,1]
    detail: dict = field(default_factory=dict)


def score_sample_reads(
    seqs: list[str],
    masked=None,
    contrastive=None,
    reference: EmbeddingReference | None = None,
    ce_ref: tuple[float, float] | None = None,
    anomaly_flag_q: float = 0.90,
) -> SampleNovelty:
    """Score one sample's reads → per-sample novelty.

    ce_ref: (mean, std) of reconstruction CE on a reference read set. If given,
        recon_novelty is a z-score-style deviation; else it is a self-normalized
        spread within this sample (useful only comparatively).
    reference: fitted EmbeddingReference for the contrastive-embedding anomaly.
    anomaly_flag_q: read is 'anomalous' if its embedding anomaly exceeds this
        quantile of the reference (embedding_novelty = fraction flagged).
    """
    if masked is None:
        masked = load_masked_vq()
    model, mtok = masked
    ce, qd = reconstruction_scores(model, mtok, seqs)

    if ce_ref is not None:
        mu, sd = ce_ref
        recon_nov = float(np.clip(((ce.mean() - mu) / (sd + 1e-9)) / 3.0, 0.0, 1.0))
    else:
        recon_nov = float(np.clip((ce.mean() - ce.min()) / (np.ptp(ce) + 1e-9), 0, 1))

    emb_nov = None
    if contrastive is not None and reference is not None:
        head, ctok = contrastive
        emb = embed_reads(head, ctok, seqs)
        anom = reference.anomaly(emb)
        emb_nov = float((anom >= anomaly_flag_q).mean())

    fused = emb_nov if emb_nov is not None else recon_nov
    if emb_nov is not None:
        fused = max(recon_nov, emb_nov)   # max-fusion, matching pipeline convention

    return SampleNovelty(
        n_reads=len(seqs),
        recon_ce_mean=round(float(ce.mean()), 4),
        quant_dist_mean=round(float(qd.mean()), 4),
        recon_novelty=round(recon_nov, 4),
        embedding_novelty=None if emb_nov is None else round(emb_nov, 4),
        novelty_score=round(float(fused), 4),
        detail={"ce_p50": float(np.median(ce)), "ce_p95": float(np.percentile(ce, 95))},
    )
