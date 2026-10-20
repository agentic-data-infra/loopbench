"""Text signatures: word n-gram hashes and MinHash LSH bands.

Reference implementation of velox_ext/TextSig.h; both must stay bit-identical
(tests/test_textsig.py checks this against the Velox functions).
"""

from __future__ import annotations

import re

import numpy as np
import xxhash

PRIME = np.uint64(1099511628211)
GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_WORD = re.compile(rb"[A-Za-z0-9]+")


def mix64(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = z ^ (z >> np.uint64(30))
        z = z * np.uint64(0xBF58476D1CE4E5B9)
        z = z ^ (z >> np.uint64(27))
        z = z * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def word_hashes(text: str) -> np.ndarray:
    words = _WORD.findall(text.encode("utf-8"))
    return np.fromiter((xxhash.xxh64_intdigest(w.lower()) for w in words), dtype=np.uint64, count=len(words))


def ngram_hashes(words: np.ndarray, n: int) -> np.ndarray:
    m = len(words) - n + 1
    if n <= 0 or m <= 0:
        return np.empty(0, dtype=np.uint64)
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * PRIME + words[j:j + m]
    return mix64(h)


def word_ngram_hashes(text: str, n: int) -> np.ndarray:
    """int64 view, matching the Velox function's bigint output."""
    return ngram_hashes(word_hashes(text), n).view(np.int64)


def minhash_bands(text: str, bands: int, rows: int, shingle: int, seed: int) -> np.ndarray:
    s = np.unique(ngram_hashes(word_hashes(text), shingle))
    if s.size == 0:
        return np.empty(0, dtype=np.int64)
    with np.errstate(over="ignore"):
        p = np.arange(1, bands * rows + 1, dtype=np.uint64) * GOLDEN
        a = mix64(np.uint64(seed & 0xFFFFFFFFFFFFFFFF) ^ p)
        mins = mix64(s[None, :] ^ a[:, None]).min(axis=1).reshape(bands, rows)
        h = mix64(np.arange(1, bands + 1, dtype=np.uint64) * GOLDEN)
        for r in range(rows):
            h = mix64(h ^ mins[:, r])
    return h.view(np.int64)
