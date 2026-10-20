"""Plans and plan hashing.

A node's hash is ``hash(op, params, code version, input snapshot)`` where the input
snapshot is the hash of the upstream node (or the source dataset manifest). Equal
hashes mean equal outputs, so results are cached by hash in the artifact store.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from .. import __version__
from .schema import HypothesisSpec

# op -> modules whose source defines its behaviour ({engine} is substituted)
_ENGINE_CODE = ["loopbench.prep.ops", "loopbench.engines.base", "loopbench.engines.{engine}_engine",
                "loopbench.textsig"]
_EXT_FILES = {"velox": ["lbvelox.cpp", "TextSig.h"]}
OP_CODE = {
    "filter": _ENGINE_CODE,
    "dedup": _ENGINE_CODE,
    "mix": _ENGINE_CODE,
    "minhash_dedup": _ENGINE_CODE,
    "pack": ["loopbench.prep.tokenize", *_ENGINE_CODE],
    "train": ["loopbench.train.model", "loopbench.train.trainer"],
    "score": ["loopbench.examine.scoring", "loopbench.train.model", "loopbench.prep.tokenize"],
}


def stable_hash(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()


@cache
def code_version(op: str, engine: str = "duckdb") -> str:
    h = hashlib.sha256(__version__.encode())
    mods = OP_CODE[op]
    for mod in mods:
        h.update(Path(importlib.util.find_spec(mod.format(engine=engine)).origin).read_bytes())
    if any("{engine}" in m for m in mods):
        ext = Path(__file__).resolve().parents[3] / "velox_ext"
        for f in _EXT_FILES.get(engine, []):
            h.update((ext / f).read_bytes())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class PlanNode:
    op: str
    params: dict
    input_hash: str
    code: str
    hash: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "hash", stable_hash(
            {"op": self.op, "params": self.params, "code": self.code, "input": self.input_hash}))


def node(op: str, params: dict, input_hash: str, engine: str | None = None) -> PlanNode:
    """Data-processing nodes carry the engine in their params, so each engine has its own cache."""
    if engine is not None:
        return PlanNode(op, {**params, "_engine": engine}, input_hash, code_version(op, engine))
    return PlanNode(op, params, input_hash, code_version(op))


@dataclass
class Plan:
    arm: str
    engine: str
    source_hash: str
    prep: list[PlanNode]
    pack: PlanNode
    train: PlanNode

    @property
    def hash(self) -> str:
        return self.train.hash

    def nodes(self) -> list[PlanNode]:
        return [*self.prep, self.pack, self.train]


def build_plan(spec: HypothesisSpec, arm: str, source_hash: str, engine: str = "duckdb") -> Plan:
    prev = source_hash
    prep = []
    for step in spec.arms()[arm]:
        n = node(step.op, step.params, prev, engine)
        prep.append(n)
        prev = n.hash
    pack = node("pack", vars(spec.pack), prev, engine)
    train = node("train", vars(spec.train), pack.hash)
    return Plan(arm, engine, source_hash, prep, pack, train)
