"""Engine interface.

An engine executes the relational parts of the loop over Parquet: prep ops, counts,
per-slice profiles, ordered scans for packing, contamination and the per-slice
bootstrap. Every engine must produce the same rows for the same inputs; hashing is
md5-based so DuckDB and Velox agree bit for bit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

# Poisson(1) inverse CDF, used for the bootstrap weights in every engine.
POISSON1_CDF = (0.36787944, 0.73575888, 0.91969860, 0.98101184, 0.99634015, 0.99940582, 0.99991676)


@dataclass(frozen=True)
class Rel:
    """Parquet files plus an optional row predicate (portable SQL expression)."""
    files: tuple[str, ...]
    where: str | None = None
    drop: tuple[str, ...] = ()  # columns hidden from consumers (e.g. the split column)

    @staticmethod
    def of_dir(d: Path) -> "Rel":
        files = sorted(str(p) for p in d.iterdir() if p.is_file() and not p.name.startswith((".", "_")))
        if not files:
            raise FileNotFoundError(f"no data files in {d}")
        return Rel(tuple(files))

    def schema(self) -> list[tuple[str, pa.DataType]]:
        s = pq.read_schema(self.files[0])
        return [(f.name, f.type) for f in s if f.name not in self.drop]

    def columns(self) -> list[str]:
        return [n for n, _ in self.schema()]


@dataclass
class OpResult:
    rows_in: int
    rows_out: int
    tokens_in: int
    tokens_out: int


class Engine(ABC):
    name: str

    def source(self, docs_path: Path, split: str) -> Rel:
        return Rel((str(docs_path),), f"split = '{split}'", drop=("split",))

    @abstractmethod
    def run_op(self, op: str, params: dict, src: Rel, out_dir: Path) -> Rel:
        """Apply ``op`` to ``src``; write Parquet file(s) into ``out_dir``."""

    @abstractmethod
    def count(self, rel: Rel) -> tuple[int, int]:
        """(rows, sum(token_count))."""

    @abstractmethod
    def group_counts(self, rel: Rel, column: str) -> dict[str, tuple[int, int]]:
        """value -> (rows, tokens), values cast to varchar (NULL -> None)."""

    @abstractmethod
    def ordered_batches(self, rel: Rel, columns: list[str], seed: int,
                        limit: int | None = None) -> Iterator[pa.RecordBatch]:
        """Rows ordered by md5(id || seed) (byte order)."""

    @abstractmethod
    def contamination(self, pool: Rel, heldout: Rel, n: int) -> dict[str, tuple[int, int]]:
        """held-out id -> (distinct word n-grams, how many occur anywhere in ``pool``)."""

    @abstractmethod
    def slice_bootstrap(self, long_path: Path, n_boot: int, ci: float) -> list[dict]:
        """Per (slice, value): n, mean base/cand loss and a Poisson-bootstrap CI of the
        mean difference. ``long_path`` has columns id, s, v, base, cand."""

    def run_and_count(self, op: str, params: dict, src: Rel, out_dir: Path) -> tuple[Rel, OpResult]:
        rows_in, tokens_in = self.count(src)
        out = self.run_op(op, params, src, out_dir)
        rows_out, tokens_out = self.count(out)
        return out, OpResult(rows_in, rows_out, tokens_in, tokens_out)
