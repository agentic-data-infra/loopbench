from __future__ import annotations

from pathlib import Path

import duckdb

from ..prep import ops
from .base import Engine, OpResult


class DuckDBEngine(Engine):
    name = "duckdb"

    def __init__(self, threads: int = 16, memory_limit: str = "32GB"):
        self.con = duckdb.connect()
        self.con.execute(f"SET threads={threads}; SET memory_limit='{memory_limit}'; "
                         "SET preserve_insertion_order=false")
        self.con.execute("CREATE MACRO hash01(x) AS (md5_number(x) % 1000000007)::DOUBLE / 1000000007")

    def source(self, docs_path: Path, split: str) -> str:
        return f"(SELECT * EXCLUDE (split) FROM read_parquet('{docs_path}') WHERE split = '{split}')"

    def relation(self, path: Path) -> str:
        return f"read_parquet('{path}')"

    def query(self, sql: str) -> list[tuple]:
        return self.con.execute(sql).fetchall()

    def _count(self, rel: str) -> tuple[int, int]:
        n, t = self.con.execute(f"SELECT count(*), coalesce(sum(token_count), 0) FROM {rel}").fetchone()
        return int(n), int(t)

    def validate(self, op: str, params: dict, src: str) -> None:
        self.con.execute(f"EXPLAIN {ops.to_sql(op, params, src, self)}")

    def run_op(self, op: str, params: dict, src: str, out: Path) -> OpResult:
        rows_in, tokens_in = self._count(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.parquet")
        self.con.execute(f"COPY ({ops.to_sql(op, params, src, self)}) TO '{tmp}' "
                         "(FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 50000)")
        tmp.rename(out)
        rows_out, tokens_out = self._count(self.relation(out))
        return OpResult(rows_in, rows_out, tokens_in, tokens_out)
