"""Per-slice aggregation and candidate-vs-baseline diff with Poisson-bootstrap CIs.

The aggregation runs in the engine over a long table (id, slice, value, base, cand);
bootstrap weights are Poisson(1) drawn from md5(id:replicate), so every engine
resamples identically.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def slice_values(values: list, top_k: int) -> list[str]:
    """Keep the top_k most frequent values; everything else becomes (other)."""
    keep = {v for v, _ in Counter(values).most_common(top_k)}
    return [str(v) if v in keep else "(other)" for v in values]


def write_long(path: Path, ids: list[str], slices: dict[str, list], base: np.ndarray, cand: np.ndarray,
               top_k: int) -> Path:
    ok = ~(np.isnan(base) | np.isnan(cand))
    cols = {"id": [], "s": [], "v": [], "base": [], "cand": []}
    groups = [("overall", ["all"] * len(ids))] + [(c, slice_values(v, top_k)) for c, v in slices.items()]
    for col, vals in groups:
        for i in np.flatnonzero(ok):
            cols["id"].append(ids[i]); cols["s"].append(col); cols["v"].append(vals[i])
            cols["base"].append(float(base[i])); cols["cand"].append(float(cand[i]))
    pq.write_table(pa.table(cols), path)
    return path


def per_slice_diff(engine, long_path: Path, slice_order: list[str], *, min_docs: int, n_boot: int,
                   ci: float) -> list[dict]:
    """Rows of (slice, value, n, base, cand, diff, ci_lo, ci_hi, significant); diff = cand - base."""
    rows = [r for r in engine.slice_bootstrap(long_path, n_boot, ci)
            if r["slice"] == "overall" or r["n_docs"] >= min_docs]
    order = {s: i for i, s in enumerate(["overall", *slice_order])}
    rows.sort(key=lambda r: (order.get(r["slice"], 99), r["value"]))
    for r in rows:
        r["significant"] = bool(r["ci_hi"] < 0 or r["ci_lo"] > 0)
    return rows
