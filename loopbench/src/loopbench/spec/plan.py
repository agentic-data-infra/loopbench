"""Plans and plan hashing.

A node's hash is ``hash(op, params, code version, input snapshot)`` where the input
snapshot is the hash of the upstream node (or the source dataset manifest). Equal
hashes mean equal outputs, so results are cached by hash in the artifact store.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from .. import __version__
from .schema import HypothesisSpec

# op -> modules whose source defines its behaviour
OP_CODE = {
    "filter": ["loopbench.prep.ops", "loopbench.engines.duckdb_engine"],
    "dedup": ["loopbench.prep.ops", "loopbench.engines.duckdb_engine"],
    "mix": ["loopbench.prep.ops", "loopbench.engines.duckdb_engine"],
    "pack": ["loopbench.prep.tokenize"],
    "train": ["loopbench.train.model", "loopbench.train.trainer"],
    "score": ["loopbench.examine.scoring", "loopbench.train.model", "loopbench.prep.tokenize"],
}


def stable_hash(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()


@cache
def code_version(op: str) -> str:
    h = hashlib.sha256(__version__.encode())
    for mod in OP_CODE[op]:
        h.update(Path(importlib.import_module(mod).__file__).read_bytes())
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


def node(op: str, params: dict, input_hash: str) -> PlanNode:
    return PlanNode(op, params, input_hash, code_version(op))


@dataclass
class Plan:
    arm: str
    source_hash: str
    prep: list[PlanNode]
    pack: PlanNode
    train: PlanNode

    @property
    def hash(self) -> str:
        return self.train.hash

    def nodes(self) -> list[PlanNode]:
        return [*self.prep, self.pack, self.train]


def build_plan(spec: HypothesisSpec, arm: str, source_hash: str) -> Plan:
    prev = source_hash
    prep = []
    for step in spec.arms()[arm]:
        n = node(step.op, step.params, prev)
        prep.append(n)
        prev = n.hash
    pack = node("pack", vars(spec.pack), prev)
    train = node("train", vars(spec.train), pack.hash)
    return Plan(arm, source_hash, prep, pack, train)
