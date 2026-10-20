"""Per-slice aggregation and candidate-vs-baseline diff with paired bootstrap CIs."""

from __future__ import annotations

from collections import Counter

import numpy as np


def bootstrap_ci(d: np.ndarray, n_boot: int, ci: float, rng: np.random.Generator) -> tuple[float, float]:
    if d.size < 2:
        return float("nan"), float("nan")
    means = d[rng.integers(0, d.size, (n_boot, d.size))].mean(1)
    a = (1 - ci) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def slice_values(values: list, top_k: int) -> list[str]:
    """Keep the top_k most frequent values; everything else becomes (other)."""
    keep = {v for v, _ in Counter(values).most_common(top_k)}
    return [str(v) if v in keep else "(other)" for v in values]


def per_slice_diff(slices: dict[str, list], base: np.ndarray, cand: np.ndarray, *, top_k: int,
                   min_docs: int, n_boot: int, ci: float, seed: int) -> list[dict]:
    """Rows of (slice, value, n, base, cand, diff, ci_lo, ci_hi); diff = cand - base (negative = better)."""
    rng = np.random.default_rng(seed)
    ok = ~(np.isnan(base) | np.isnan(cand))
    groups: list[tuple[str, str, np.ndarray]] = [("overall", "all", ok)]
    for col, vals in slices.items():
        vals = np.asarray(slice_values(vals, top_k))
        for v in sorted(set(vals)):
            groups.append((col, v, ok & (vals == v)))
    rows = []
    for col, v, m in groups:
        n = int(m.sum())
        if n < min_docs and col != "overall":
            continue
        d = cand[m] - base[m]
        lo, hi = bootstrap_ci(d, n_boot, ci, rng)
        rows.append({"slice": col, "value": v, "n_docs": n, "loss_baseline": float(base[m].mean()),
                     "loss_candidate": float(cand[m].mean()), "diff": float(d.mean()),
                     "ci_lo": lo, "ci_hi": hi, "significant": bool(hi < 0 or lo > 0)})
    return rows
