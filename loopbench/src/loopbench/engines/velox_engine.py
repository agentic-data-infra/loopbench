"""Velox engine: plans are JSON step lists executed by the lbvelox extension
(velox_ext/lbvelox.cpp) with Velox's PlanBuilder, multi-threaded."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.cffi import ffi

from ..prep import ops
from .base import Engine, Rel

HASH01 = "(cast(from_base(substr(to_hex(md5(to_utf8({x}))), 1, 8), 16) as double) / 4294967296.0)"

_TYPES = {pa.string(): "varchar", pa.large_string(): "varchar", pa.int64(): "bigint", pa.int32(): "integer",
          pa.float64(): "double", pa.bool_(): "boolean", pa.float32(): "real"}


def _hash01(x: str) -> str:
    return HASH01.format(x=x)


def _lbvelox():
    try:
        import lbvelox
    except ImportError as e:  # pragma: no cover
        raise ImportError("lbvelox extension not built: uv run python velox_ext/build.py "
                          "--velox-build ~/source/velox/_build/release-py") from e
    return lbvelox


class VeloxEngine(Engine):
    name = "velox"

    def __init__(self, threads: int = 16):
        self.lb = _lbvelox()
        self.lb.init(threads)
        self.threads = threads
        self.last_stats = ""

    # -- plumbing -----------------------------------------------------------------------------
    def scan(self, rel: Rel, columns: list[str] | None = None) -> list[dict]:
        """Scan step (+ projection that hides dropped/unrequested columns)."""
        full = pq.read_schema(rel.files[0])
        cols = [[f.name, _TYPES[f.type]] for f in full]
        keep = columns or [n for n in full.names if n not in rel.drop]
        steps = [{"op": "scan", "files": list(rel.files), "columns": cols}]
        if rel.where:
            steps.append({"op": "filter", "expr": rel.where})
        steps.append({"op": "project", "exprs": keep})
        return steps

    def execute(self, plan: list[dict], stats: bool = False) -> pa.Table:
        res = self.lb.execute(json.dumps(plan), self.threads, stats)
        self.last_stats = res.stats
        batches = []
        for i in range(res.num_batches()):
            arr, sch = ffi.new("struct ArrowArray*"), ffi.new("struct ArrowSchema*")
            res.export_batch(i, int(ffi.cast("uintptr_t", arr)), int(ffi.cast("uintptr_t", sch)))
            batches.append(pa.RecordBatch._import_from_c(int(ffi.cast("uintptr_t", arr)),
                                                         int(ffi.cast("uintptr_t", sch))))
        if not batches:
            return pa.table({})
        return pa.Table.from_batches(batches)

    def _write(self, plan: list[dict], out_dir: Path) -> Rel:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        self.execute(plan + [{"op": "write", "dir": str(out_dir)}])
        return Rel.of_dir(out_dir)

    # -- ops ----------------------------------------------------------------------------------
    def run_op(self, op: str, params: dict, src: Rel, out_dir: Path) -> Rel:
        if op == "filter":
            return self._write(self.scan(src) + [{"op": "filter", "expr": params["where"]}], out_dir)
        if op == "dedup":
            keep = self.scan(src, ["id", params["key"]]) + [
                {"op": "project", "exprs": [f"md5(to_utf8({params['key']})) AS h", "id"]},
                {"op": "agg", "keys": ["h"], "aggs": ["min(id) AS keep_id"]}]
            return self._write(self.scan(src) + [
                {"op": "join", "type": "semi", "left": ["id"], "right": ["keep_id"], "build": keep,
                 "output": src.columns()}], out_dir)
        if op == "mix":
            frac = ops.mix_fractions(params, {g: t for g, (_, t) in self.group_counts(src, params["by"]).items()})
            pred = ops.mix_predicate(params["by"], frac, _hash01("id"))
            return self._write(self.scan(src) + [{"op": "filter", "expr": pred}], out_dir)
        if op == "minhash_dedup":
            p = ops.minhash_params(params)
            with tempfile.TemporaryDirectory(dir=out_dir.parent) as tmp:
                bands = self._write(self.scan(src, ["id", "text"]) + [
                    {"op": "project", "exprs": [
                        "id", f"minhash_bands(text, {p['bands']}, {p['rows']}, {p['shingle']}, "
                              f"cast({p['seed']} as bigint)) AS hs"]},
                    {"op": "unnest", "replicate": ["id"], "columns": ["hs"], "names": ["h"]}],
                    Path(tmp) / "bands")
                buckets = self.scan(bands) + [
                    {"op": "agg", "keys": ["h"], "aggs": ["min(id) AS keep_id", "count(1) AS c"]},
                    {"op": "filter", "expr": "c > 1"},
                    {"op": "project", "exprs": ["h AS kh", "keep_id"]}]
                drops = self.scan(bands) + [
                    {"op": "join", "type": "inner", "left": ["h"], "right": ["kh"], "build": buckets,
                     "output": ["id", "keep_id"]},
                    {"op": "filter", "expr": "id <> keep_id"},
                    {"op": "agg", "keys": ["id"], "aggs": []},
                    {"op": "project", "exprs": ["id AS drop_id"]}]
                return self._write(self.scan(src) + [
                    {"op": "join", "type": "anti", "left": ["id"], "right": ["drop_id"], "build": drops,
                     "output": src.columns()}], out_dir)
        raise ValueError(f"velox: unsupported op {op}")

    # -- queries ------------------------------------------------------------------------------
    def count(self, rel: Rel) -> tuple[int, int]:
        t = self.execute(self.scan(rel, ["token_count"]) + [
            {"op": "agg", "keys": [], "aggs": ["count(1) AS n", "sum(token_count) AS t"]}])
        return int(t["n"][0].as_py()), int(t["t"][0].as_py() or 0)

    def group_counts(self, rel: Rel, column: str) -> dict[str, tuple[int, int]]:
        t = self.execute(self.scan(rel, [column, "token_count"]) + [
            {"op": "project", "exprs": [f"cast({column} as varchar) AS g", "token_count"]},
            {"op": "agg", "keys": ["g"], "aggs": ["count(1) AS n", "sum(token_count) AS t"]}])
        return {g: (int(n), int(tk or 0)) for g, n, tk in zip(*(t[c].to_pylist() for c in ("g", "n", "t")))}

    def ordered_batches(self, rel: Rel, columns: list[str], seed: int,
                        limit: int | None = None) -> Iterator[pa.RecordBatch]:
        plan = self.scan(rel, columns) + [
            {"op": "project", "exprs": columns + [f"md5(to_utf8(concat(id, '{seed}'))) AS ord_key"]},
            {"op": "order_by", "keys": ["ord_key"]}]
        if limit:
            plan.append({"op": "limit", "n": limit})
        yield from self.execute(plan).select(columns).to_batches(max_chunksize=4096)

    def contamination(self, pool: Rel, heldout: Rel, n: int) -> dict[str, tuple[int, int]]:
        def held_grams():
            return self.scan(heldout, ["id", "text"]) + [
                {"op": "project", "exprs": ["id AS hid", f"word_ngram_hashes(text, {n}) AS hs"]},
                {"op": "unnest", "replicate": ["hid"], "columns": ["hs"], "names": ["h"]},
                {"op": "agg", "keys": ["hid", "h"], "aggs": []}]
        distinct_held = held_grams() + [{"op": "agg", "keys": ["h"], "aggs": []},
                                        {"op": "project", "exprs": ["h AS hh"]}]
        matched = self.scan(pool, ["text"]) + [
            {"op": "project", "exprs": [f"word_ngram_hashes(text, {n}) AS ps"]},
            {"op": "unnest", "replicate": [], "columns": ["ps"], "names": ["p"]},
            {"op": "join", "type": "semi", "left": ["p"], "right": ["hh"], "build": distinct_held, "output": ["p"]},
            {"op": "agg", "keys": ["p"], "aggs": []},
            {"op": "project", "exprs": ["p AS m"]}]
        t = self.execute(held_grams() + [
            {"op": "join", "type": "left", "left": ["h"], "right": ["m"], "build": matched,
             "output": ["hid", "m"]},
            {"op": "project", "exprs": ["hid", "if(m IS NULL, 0, 1) AS hit"]},
            {"op": "agg", "keys": ["hid"], "aggs": ["count(1) AS n", "sum(hit) AS hits"]}])
        return {i: (int(a), int(b)) for i, a, b in zip(*(t[c].to_pylist() for c in ("hid", "n", "hits")))}

    def slice_bootstrap(self, long_path: Path, n_boot: int, ci: float) -> list[dict]:
        a = (1 - ci) / 2
        rel = Rel((str(long_path),))
        w = ops.poisson_case(_hash01("concat(id, ':', cast(b as varchar))"))
        point = self.execute(self.scan(rel) + [
            {"op": "agg", "keys": ["s", "v"], "aggs": ["count(1) AS n", "avg(base) AS base", "avg(cand) AS cand"]}])
        boot = self.execute(self.scan(rel) + [
            {"op": "project", "exprs": ["id", "s", "v", "cand - base AS d",
                                        f"sequence(cast(0 as bigint), cast({n_boot - 1} as bigint)) AS bs"]},
            {"op": "unnest", "replicate": ["id", "s", "v", "d"], "columns": ["bs"], "names": ["b"]},
            {"op": "project", "exprs": ["s", "v", "b", "d", f"cast({w} as double) AS w"]},
            {"op": "project", "exprs": ["s", "v", "b", "w * d AS wd", "w"]},  # aggregate args must be columns
            {"op": "agg", "keys": ["s", "v", "b"], "aggs": ["sum(wd) AS sw", "sum(w) AS ww"]},
            {"op": "project", "exprs": ["s", "v", "sw / ww AS m"]},
            {"op": "agg", "keys": ["s", "v"], "aggs": [f"approx_percentile(m, {a}) AS lo",
                                                     f"approx_percentile(m, {1 - a}) AS hi"]}])
        ci_map = {(s, v): (lo, hi) for s, v, lo, hi in zip(*(boot[c].to_pylist() for c in ("s", "v", "lo", "hi")))}
        from .duckdb_engine import _boot_rows
        rows = [(s, v, n, b, c, *ci_map.get((s, v), (None, None)))
                for s, v, n, b, c in zip(*(point[k].to_pylist() for k in ("s", "v", "n", "base", "cand")))]
        return _boot_rows(rows)
