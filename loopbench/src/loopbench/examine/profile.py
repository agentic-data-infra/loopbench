"""Dataset profile: what a prep step removed, per slice."""

from __future__ import annotations


def removed_by_slice(engine, before: str, after: str, column: str, top_k: int) -> list[dict]:
    """Docs/tokens before vs after a step, grouped by ``column`` (top_k groups by tokens, rest -> (other))."""
    sql = f"""
    WITH b AS (SELECT {column}::VARCHAR AS g, count(*) AS docs, sum(token_count) AS tokens FROM {before} GROUP BY 1),
         a AS (SELECT {column}::VARCHAR AS g, count(*) AS docs, sum(token_count) AS tokens FROM {after} GROUP BY 1),
         j AS (SELECT b.g, b.docs AS docs_in, b.tokens AS tokens_in, coalesce(a.docs, 0) AS docs_out,
                      coalesce(a.tokens, 0) AS tokens_out,
                      row_number() OVER (ORDER BY b.tokens DESC) AS rk
               FROM b LEFT JOIN a ON b.g IS NOT DISTINCT FROM a.g)
    SELECT CASE WHEN rk <= {top_k} THEN coalesce(g, '(null)') ELSE '(other)' END AS value,
           sum(docs_in), sum(tokens_in), sum(docs_out), sum(tokens_out)
    FROM j GROUP BY 1 ORDER BY 3 DESC"""
    rows = []
    for value, di, ti, do, to in engine.query(sql):
        rows.append({"slice": column, "value": value, "docs_in": int(di), "tokens_in": int(ti),
                     "docs_out": int(do), "tokens_out": int(to),
                     "tokens_removed_frac": 1 - int(to) / int(ti) if ti else 0.0})
    return rows
