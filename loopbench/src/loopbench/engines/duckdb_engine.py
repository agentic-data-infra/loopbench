"""DuckDB engine: the reference implementation (SQL, plus Python UDFs for text signatures)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import duckdb
from duckdb import sqltypes
import pyarrow as pa

from .. import textsig
from ..prep import ops
from .base import Engine, Rel

HASH01 = "(('0x' || substr(md5({x}), 1, 8))::UBIGINT / 4294967296.0)"


def _hash01(x: str) -> str:
    return HASH01.format(x=x)


def _as_list_array(texts, fn) -> pa.Array:
    return pa.array([fn(t).tolist() if t is not None else None for t in texts.to_pylist()],
                    type=pa.list_(pa.int64()))


def _ngram_udf(texts, n):
    k = n[0].as_py()
    return _as_list_array(texts, lambda t: textsig.word_ngram_hashes(t, k))


def _minhash_udf(texts, bands, rows, shingle, seed):
    b, r, k, sd = (x[0].as_py() for x in (bands, rows, shingle, seed))
    return _as_list_array(texts, lambda t: textsig.minhash_bands(t, b, r, k, sd))


class DuckDBEngine(Engine):
    name = "duckdb"

    def __init__(self, threads: int = 16, memory_limit: str = "32GB"):
        self.con = duckdb.connect()
        self.con.execute(f"SET threads={threads}; SET memory_limit='{memory_limit}'; "
                         "SET preserve_insertion_order=false")
        big = duckdb.list_type(sqltypes.BIGINT)
        I, V = sqltypes.INTEGER, sqltypes.VARCHAR
        self.con.create_function("word_ngram_hashes", _ngram_udf, [V, I], big, type="arrow")
        self.con.create_function("minhash_bands", _minhash_udf, [V, I, I, I, sqltypes.BIGINT], big, type="arrow")

    # -- relations ----------------------------------------------------------------------------
    def sql(self, rel: Rel) -> str:
        files = ", ".join(f"'{f}'" for f in rel.files)
        cols = "*" if not rel.drop else f"* EXCLUDE ({', '.join(rel.drop)})"
        where = f" WHERE {rel.where}" if rel.where else ""
        return f"(SELECT {cols} FROM read_parquet([{files}]){where})"

    def _write(self, query: str, out_dir: Path) -> Rel:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.con.execute(f"COPY ({query}) TO '{out_dir / 'part-0.parquet'}' "
                         "(FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 4096)")
        return Rel.of_dir(out_dir)

    # -- ops ----------------------------------------------------------------------------------
    def run_op(self, op: str, params: dict, src: Rel, out_dir: Path) -> Rel:
        s = self.sql(src)
        if op == "filter":
            q = f"SELECT * FROM {s} WHERE ({params['where']})"
        elif op == "dedup":
            q = (f"SELECT * FROM {s} WHERE id IN "
                 f"(SELECT min(id) FROM {s} GROUP BY md5({params['key']}))")
        elif op == "mix":
            frac = ops.mix_fractions(params, {g: t for g, (_, t) in self.group_counts(src, params["by"]).items()})
            q = f"SELECT * FROM {s} WHERE {ops.mix_predicate(params['by'], frac, _hash01('id'))}"
        elif op == "minhash_dedup":
            p = ops.minhash_params(params)
            q = f"""
            WITH b AS MATERIALIZED (
              SELECT id, unnest(minhash_bands(text, {p['bands']}, {p['rows']}, {p['shingle']}, {p['seed']})) AS h
              FROM {s}),
            k AS (SELECT h, min(id) AS keep FROM b GROUP BY h HAVING count(*) > 1),
            d AS (SELECT DISTINCT b.id FROM b JOIN k USING (h) WHERE b.id <> k.keep)
            SELECT * FROM {s} WHERE id NOT IN (SELECT id FROM d)"""
        else:
            raise ValueError(f"duckdb: unsupported op {op}")
        return self._write(q, out_dir)

    # -- queries ------------------------------------------------------------------------------
    def count(self, rel: Rel) -> tuple[int, int]:
        n, t = self.con.execute(f"SELECT count(*), coalesce(sum(token_count), 0) FROM {self.sql(rel)}").fetchone()
        return int(n), int(t)

    def group_counts(self, rel: Rel, column: str) -> dict[str, tuple[int, int]]:
        rows = self.con.execute(f"SELECT CAST({column} AS VARCHAR), count(*), coalesce(sum(token_count), 0) "
                                f"FROM {self.sql(rel)} GROUP BY 1").fetchall()
        return {g: (int(n), int(t)) for g, n, t in rows}

    def ordered_batches(self, rel: Rel, columns: list[str], seed: int,
                        limit: int | None = None) -> Iterator[pa.RecordBatch]:
        lim = f" LIMIT {limit}" if limit else ""
        q = f"SELECT {', '.join(columns)} FROM {self.sql(rel)} ORDER BY md5(id || '{seed}'){lim}"
        yield from self.con.execute(q).to_arrow_reader(4096)

    def contamination(self, pool: Rel, heldout: Rel, n: int) -> dict[str, tuple[int, int]]:
        q = f"""
        WITH hg AS MATERIALIZED (
          SELECT DISTINCT id, unnest(word_ngram_hashes(text, {n})) AS h FROM {self.sql(heldout)}),
        hh AS (SELECT DISTINCT h FROM hg),
        m AS (SELECT DISTINCT p.h FROM (SELECT unnest(word_ngram_hashes(text, {n})) AS h FROM {self.sql(pool)}) p
              SEMI JOIN hh USING (h))
        SELECT hg.id, count(*), count(m.h) FROM hg LEFT JOIN m USING (h) GROUP BY 1"""
        return {i: (int(a), int(b)) for i, a, b in self.con.execute(q).fetchall()}

    def slice_bootstrap(self, long_path: Path, n_boot: int, ci: float) -> list[dict]:
        a = (1 - ci) / 2
        w = ops.poisson_case(_hash01("id || ':' || CAST(b AS VARCHAR)"))
        q = f"""
        WITH l AS (SELECT * FROM read_parquet('{long_path}')),
        pt AS (SELECT s, v, count(*) AS n, avg(base) AS base, avg(cand) AS cand FROM l GROUP BY 1, 2),
        bs AS (SELECT s, v, b, sum(w * (cand - base)) / sum(w) AS m
               FROM (SELECT l.*, r.range AS b, {w} AS w FROM l, range({n_boot}) r) GROUP BY 1, 2, 3),
        q AS (SELECT s, v, quantile_cont(m, {a}) AS lo, quantile_cont(m, {1 - a}) AS hi FROM bs GROUP BY 1, 2)
        SELECT pt.s, pt.v, n, base, cand, lo, hi FROM pt JOIN q USING (s, v)"""
        return _boot_rows(self.con.execute(q).fetchall())


def _boot_rows(rows) -> list[dict]:
    out = []
    for s, v, n, base, cand, lo, hi in rows:
        out.append({"slice": s, "value": v, "n_docs": int(n), "loss_baseline": float(base),
                    "loss_candidate": float(cand), "diff": float(cand - base),
                    "ci_lo": float(lo) if lo is not None else float("nan"),
                    "ci_hi": float(hi) if hi is not None else float("nan")})
    return out
