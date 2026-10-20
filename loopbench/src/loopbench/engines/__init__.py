from .base import Engine, OpResult
from .duckdb_engine import DuckDBEngine

ENGINES = {"duckdb": DuckDBEngine}


def get_engine(name: str = "duckdb", **kw) -> Engine:
    return ENGINES[name](**kw)


__all__ = ["Engine", "OpResult", "DuckDBEngine", "get_engine"]
