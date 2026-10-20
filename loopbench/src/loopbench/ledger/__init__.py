"""Run ledger: append-only Parquet files per table, queried through DuckDB."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from ..paths import LEDGER

TABLES = ("runs", "nodes", "slices", "profiles", "spans")


class Ledger:
    def __init__(self, root: Path = LEDGER):
        self.root = root

    def append(self, table: str, rows: list[dict]) -> Path | None:
        if table not in TABLES:
            raise ValueError(f"unknown ledger table {table!r}")
        if not rows:
            return None
        d = self.root / table
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        return path

    def connect(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect()
        for t in TABLES:
            if any((self.root / t).glob("*.parquet")):
                con.execute(f"CREATE VIEW {t} AS SELECT * FROM "
                            f"read_parquet('{self.root / t}/*.parquet', union_by_name = true)")
        return con

    def query(self, sql: str):
        return self.connect().sql(sql)
