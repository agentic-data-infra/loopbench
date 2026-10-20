"""n-gram contamination check: held-out docs vs the packed training stream."""

from __future__ import annotations

import numpy as np

_P = np.uint64(1_000_003)


def ngram_hashes(tokens: np.ndarray, n: int) -> np.ndarray:
    """Polynomial hashes (mod 2^64) of every length-n window."""
    t = tokens.astype(np.uint64)
    m = len(t) - n + 1
    if m <= 0:
        return np.empty(0, dtype=np.uint64)
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * _P + t[j:j + m]
    return h


def check(train_tokens: np.ndarray, heldout: list[np.ndarray], n: int) -> np.ndarray:
    """Fraction of each held-out doc's n-grams that also occur in the training stream."""
    train = np.unique(ngram_hashes(train_tokens, n))
    out = np.zeros(len(heldout))
    for i, doc in enumerate(heldout):
        h = ngram_hashes(doc, n)
        if h.size:
            pos = np.minimum(np.searchsorted(train, h), train.size - 1)
            out[i] = (train[pos] == h).mean()
    return out
