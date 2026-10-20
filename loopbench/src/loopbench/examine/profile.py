"""Dataset profile: what a prep step removed, per slice."""

from __future__ import annotations


def removed_by_slice(engine, before, after, column: str, top_k: int) -> list[dict]:
    """Docs/tokens before vs after a step, grouped by ``column`` (top_k groups by tokens, rest -> (other))."""
    b, a = engine.group_counts(before, column), engine.group_counts(after, column)
    ranked = sorted(b, key=lambda g: -b[g][1])
    agg: dict[str, list[int]] = {}
    for rank, g in enumerate(ranked):
        key = ("(null)" if g is None else g) if rank < top_k else "(other)"
        acc = agg.setdefault(key, [0, 0, 0, 0])
        di, ti = b[g]
        do, to = a.get(g, (0, 0))
        for i, x in enumerate((di, ti, do, to)):
            acc[i] += x
    rows = [{"slice": column, "value": k, "docs_in": di, "tokens_in": ti, "docs_out": do, "tokens_out": to,
             "tokens_removed_frac": 1 - to / ti if ti else 0.0} for k, (di, ti, do, to) in agg.items()]
    return sorted(rows, key=lambda r: -r["tokens_in"])
