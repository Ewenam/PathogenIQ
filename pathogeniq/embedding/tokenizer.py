"""
embedding/tokenizer.py
DNA/RNA k-mer frequency tokenizer.

Converts a nucleotide sequence into a fixed-length k-mer frequency vector.
Default k=6 → vocabulary of 4^6 = 4,096 canonical k-mers (ACGT only).
Ambiguous bases (N, R, Y, …) are skipped — only pure ACGT k-mers are counted.
"""
from __future__ import annotations

from itertools import product

import numpy as np

K_DEFAULT = 6
BASES = "ACGT"

# Build canonical vocab once at import time
_KMER_VOCAB: dict[str, int] = {
    "".join(b): i for i, b in enumerate(product(BASES, repeat=K_DEFAULT))
}
VOCAB_SIZE = len(_KMER_VOCAB)  # 4096


def build_vocab(k: int = K_DEFAULT) -> dict[str, int]:
    """Return {kmer: index} for all canonical k-mers of length k."""
    if k == K_DEFAULT:
        return _KMER_VOCAB
    return {"".join(b): i for i, b in enumerate(product(BASES, repeat=k))}


def tokenize_sequence(
    seq: str,
    k: int = K_DEFAULT,
    vocab: dict[str, int] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert a DNA/RNA sequence to a k-mer frequency vector.

    Args:
        seq: nucleotide sequence (case-insensitive; U → T for RNA)
        k: k-mer length
        vocab: optional pre-built vocab dict (uses default 6-mer vocab if None)
        normalize: if True, divide by total valid k-mer count

    Returns:
        float32 array of shape (vocab_size,)
    """
    if vocab is None and k == K_DEFAULT:
        vocab = _KMER_VOCAB
    elif vocab is None:
        vocab = build_vocab(k)

    seq = seq.upper().replace("U", "T")
    counts = np.zeros(len(vocab), dtype=np.float32)
    valid = 0

    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        idx = vocab.get(kmer)
        if idx is not None:
            counts[idx] += 1
            valid += 1

    if normalize and valid > 0:
        counts /= valid

    return counts


def tokenize_fasta(
    path: str,
    k: int = K_DEFAULT,
    min_length: int = 100,
) -> dict[str, np.ndarray]:
    """
    Tokenize all sequences in a FASTA file.

    Returns:
        {header: kmer_freq_vector} for sequences >= min_length bp.
    """
    vocab = build_vocab(k)
    results: dict[str, np.ndarray] = {}
    current_id: str | None = None
    buf: list[str] = []

    def _flush():
        if current_id is None:
            return
        seq = "".join(buf)
        if len(seq) >= min_length:
            results[current_id] = tokenize_sequence(seq, k=k, vocab=vocab)

    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                _flush()
                current_id = line[1:].split()[0]
                buf = []
            elif line:
                buf.append(line)
    _flush()
    return results
