"""Relational prep ops (filter, dedup, mix), expressed as SQL over an input relation."""

from __future__ import annotations


def filter_sql(params: dict, src: str, engine) -> str:
    return f"SELECT * FROM {src} WHERE ({params['where']})"


def dedup_sql(params: dict, src: str, engine) -> str:
    # exact dedup on md5 of the key; keep the lowest id per group (deterministic)
    return (f"SELECT * FROM {src} "
            f"QUALIFY row_number() OVER (PARTITION BY md5({params['key']}) ORDER BY id) = 1")


def mix_sql(params: dict, src: str, engine) -> str:
    """Resample groups of ``by`` to target token shares ``weights`` (normalised).

    The output is the largest mixture reachable without upsampling: each group keeps
    a deterministic hash-selected fraction of its documents.
    """
    by, default = params["by"], float(params.get("default_weight", 0.0))
    groups = engine.query(f"SELECT {by}::VARCHAR, sum(token_count) FROM {src} GROUP BY 1")
    w = {g: float(params["weights"].get(g, default)) for g, _ in groups}
    total_w = sum(w.values())
    if total_w <= 0:
        raise ValueError("mix: all weights are zero")
    tok = {g: int(t) for g, t in groups}
    budget = min(tok[g] * total_w / w[g] for g in w if w[g] > 0)
    frac = {g: (w[g] / total_w) * budget / tok[g] if tok[g] else 0.0 for g in w}
    cases = " ".join(f"WHEN {by}::VARCHAR = '{g.replace(chr(39), chr(39) * 2)}' THEN {f!r}"
                     for g, f in frac.items() if g is not None)
    return f"SELECT * FROM {src} WHERE hash01(id) < (CASE {cases} ELSE 0.0 END)"


SQL_OPS = {"filter": filter_sql, "dedup": dedup_sql, "mix": mix_sql}


def to_sql(op: str, params: dict, src: str, engine) -> str:
    return SQL_OPS[op](params, src, engine)
