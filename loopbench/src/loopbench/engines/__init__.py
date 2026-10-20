from .base import Engine, OpResult, Rel


def get_engine(name: str = "duckdb", **kw) -> Engine:
    if name == "duckdb":
        from .duckdb_engine import DuckDBEngine
        return DuckDBEngine(**kw)
    if name == "velox":
        from .velox_engine import VeloxEngine
        return VeloxEngine(**kw)
    raise ValueError(f"unknown engine {name!r}")


ENGINES = ("duckdb", "velox")

__all__ = ["Engine", "OpResult", "Rel", "get_engine", "ENGINES"]
