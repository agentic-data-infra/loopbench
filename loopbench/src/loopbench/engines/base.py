"""Engine interface: executes relational prep ops over Parquet datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OpResult:
    rows_in: int
    rows_out: int
    tokens_in: int
    tokens_out: int


class Engine(ABC):
    name: str

    @abstractmethod
    def source(self, docs_path: Path, split: str) -> str:
        """Relation expression for one split of a prepared dataset."""

    @abstractmethod
    def run_op(self, op: str, params: dict, src: str, out: Path) -> OpResult:
        """Apply ``op`` to relation ``src`` and materialise the result as Parquet at ``out``."""

    @abstractmethod
    def relation(self, path: Path) -> str:
        """Relation expression for a materialised op output."""

    @abstractmethod
    def query(self, sql: str) -> list[tuple]:
        ...
