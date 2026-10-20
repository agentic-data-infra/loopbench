"""Engine-neutral semantics of the prep ops.

  filter         keep rows matching a SQL predicate
  dedup          exact dedup on md5(key); keep the lowest id per group
  mix            resample groups of ``by`` to target token shares (no upsampling);
                 a row of group g survives iff hash01(id) < frac[g]
  minhash_dedup  near-dup removal: MinHash LSH over word shingles; a doc is dropped
                 iff some band bucket it falls in holds a smaller id

hash01(x) = int(md5(x)[:8 hex], 16) / 2^32, identical in every engine.
"""

from __future__ import annotations

MINHASH_DEFAULTS = {"bands": 14, "rows": 8, "shingle": 5, "seed": 0}


def mix_fractions(params: dict, group_tokens: dict[str | None, int]) -> dict[str | None, float]:
    default = float(params.get("default_weight", 0.0))
    w = {g: float(params["weights"].get(g, default)) for g in group_tokens}
    total_w = sum(w.values())
    if total_w <= 0:
        raise ValueError("mix: all weights are zero")
    budget = min(group_tokens[g] * total_w / w[g] for g in w if w[g] > 0)
    return {g: (w[g] / total_w) * budget / group_tokens[g] if group_tokens[g] else 0.0 for g in w}


def sql_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def mix_predicate(by: str, frac: dict[str | None, float], hash01: str) -> str:
    cases = " ".join(f"WHEN CAST({by} AS VARCHAR) = {sql_str(g)} THEN {f!r}"
                     for g, f in frac.items() if g is not None)
    return f"{hash01} < (CASE {cases} ELSE 0.0 END)" if cases else "false"


def minhash_params(params: dict) -> dict:
    return {**MINHASH_DEFAULTS, **params}


def poisson_case(u: str) -> str:
    from ..engines.base import POISSON1_CDF
    whens = " ".join(f"WHEN {u} < {p} THEN {k}" for k, p in enumerate(POISSON1_CDF))
    return f"(CASE {whens} ELSE {len(POISSON1_CDF)} END)"
