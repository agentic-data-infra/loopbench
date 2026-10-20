"""n-gram contamination check: held-out docs vs the full prepped training pool.

Word n-grams (ASCII alnum words, lowercased; see loopbench.textsig) so the check does
not depend on the tokenizer or on which docs happened to be packed.
"""

from __future__ import annotations

import numpy as np


def check(engine, pool, heldout, ids: list[str], n: int) -> np.ndarray:
    """Fraction of each held-out doc's distinct word n-grams that occur in ``pool``,
    aligned with ``ids`` (0 for docs shorter than n words)."""
    res = engine.contamination(pool, heldout, n)
    return np.array([res[i][1] / res[i][0] if i in res and res[i][0] else 0.0 for i in ids])
