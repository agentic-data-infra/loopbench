"""Hypothesis spec: a YAML document describing a baseline and a candidate data recipe.

Example::

    name: edu_score_ge_3
    dataset: fineweb-edu-m0
    baseline:  {prep: [{op: dedup, params: {key: text}}]}
    candidate: {prep: [{op: dedup, params: {key: text}}, {op: filter, params: {where: "score >= 3.0"}}]}
    pack:    {max_tokens: 2000000}
    train:   {steps: 300}
    examine: {heldout_docs: 2000}
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


class SpecError(ValueError):
    pass


# op name -> {param: (type, required)}
OP_PARAMS: dict[str, dict[str, tuple[type | tuple, bool]]] = {
    "filter": {"where": (str, True)},
    "dedup": {"key": (str, True)},
    "mix": {"by": (str, True), "weights": (dict, True), "default_weight": ((int, float), False)},
    "minhash_dedup": {"bands": (int, False), "rows": (int, False), "shingle": (int, False), "seed": (int, False)},
}
DEDUP_KEYS = ("text", "url")


@dataclass(frozen=True)
class Step:
    op: str
    params: dict

    def to_dict(self) -> dict:
        return {"op": self.op, "params": self.params}


@dataclass
class PackConfig:
    max_tokens: int = 2_000_000
    seed: int = 0


@dataclass
class TrainConfig:
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 96
    ctx: int = 256
    vocab_cap: int = 8192
    batch_size: int = 16
    steps: int = 300
    lr: float = 3e-3
    warmup: int = 30
    weight_decay: float = 0.1
    seed: int = 0
    threads: int = 16


@dataclass
class ExamineConfig:
    heldout_docs: int = 2000
    score_tokens: int = 256
    slices: list[str] = field(default_factory=lambda: ["edu_bucket", "len_bucket", "lang_bucket",
                                                        "dump", "tld", "domain"])
    top_k: int = 15
    min_docs: int = 30
    bootstrap: int = 1000
    ci: float = 0.95
    ngram: int = 13
    seed: int = 0


@dataclass
class HypothesisSpec:
    name: str
    dataset: str
    baseline: list[Step]
    candidate: list[Step]
    description: str = ""
    pack: PackConfig = field(default_factory=PackConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    examine: ExamineConfig = field(default_factory=ExamineConfig)
    source: str | None = None

    def arms(self) -> dict[str, list[Step]]:
        return {"baseline": self.baseline, "candidate": self.candidate}

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("source")
        return d


def _dc(cls, raw, where: str):
    raw = raw or {}
    if not isinstance(raw, dict):
        raise SpecError(f"{where}: expected a mapping")
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise SpecError(f"{where}: unknown keys {sorted(unknown)}")
    obj = cls(**raw)
    for f in fields(cls):
        default = getattr(cls(), f.name)
        val = getattr(obj, f.name)
        if isinstance(default, float) and isinstance(val, int):
            setattr(obj, f.name, float(val))
        elif not isinstance(val, type(default)):
            raise SpecError(f"{where}.{f.name}: expected {type(default).__name__}, got {val!r}")
    return obj


def _steps(raw, where: str) -> list[Step]:
    if not isinstance(raw, dict) or not isinstance(raw.get("prep", []), list):
        raise SpecError(f"{where}: expected {{prep: [steps]}}")
    out = []
    for i, s in enumerate(raw.get("prep", [])):
        w = f"{where}.prep[{i}]"
        if not isinstance(s, dict) or "op" not in s:
            raise SpecError(f"{w}: expected {{op, params}}")
        op, params = s["op"], s.get("params", {}) or {}
        if op not in OP_PARAMS:
            raise SpecError(f"{w}: unknown op {op!r} (known: {sorted(OP_PARAMS)})")
        schema = OP_PARAMS[op]
        for k, v in params.items():
            if k not in schema:
                raise SpecError(f"{w}: unknown param {k!r} for op {op}")
            if not isinstance(v, schema[k][0]):
                raise SpecError(f"{w}.{k}: bad type {type(v).__name__}")
        for k, (_, req) in schema.items():
            if req and k not in params:
                raise SpecError(f"{w}: missing required param {k!r}")
        if op == "dedup" and params["key"] not in DEDUP_KEYS:
            raise SpecError(f"{w}: dedup key must be one of {DEDUP_KEYS}")
        if op == "filter" and ";" in params["where"]:
            raise SpecError(f"{w}: filter predicate must be a single SQL expression")
        out.append(Step(op, dict(params)))
    return out


def parse_spec(raw: dict, source: str | None = None) -> HypothesisSpec:
    if not isinstance(raw, dict):
        raise SpecError("spec must be a mapping")
    allowed = {"name", "description", "dataset", "baseline", "candidate", "pack", "train", "examine"}
    unknown = set(raw) - allowed
    if unknown:
        raise SpecError(f"unknown top-level keys {sorted(unknown)}")
    for k in ("name", "dataset", "baseline", "candidate"):
        if k not in raw:
            raise SpecError(f"missing required key {k!r}")
    spec = HypothesisSpec(
        name=str(raw["name"]), dataset=str(raw["dataset"]), description=str(raw.get("description", "")),
        baseline=_steps(raw["baseline"], "baseline"), candidate=_steps(raw["candidate"], "candidate"),
        pack=_dc(PackConfig, raw.get("pack"), "pack"), train=_dc(TrainConfig, raw.get("train"), "train"),
        examine=_dc(ExamineConfig, raw.get("examine"), "examine"), source=source)
    if spec.train.d_model % spec.train.n_head:
        raise SpecError("train.d_model must be divisible by train.n_head")
    if spec.baseline == spec.candidate:
        raise SpecError("baseline and candidate recipes are identical")
    return spec


def load_spec(path: str | Path) -> HypothesisSpec:
    path = Path(path)
    return parse_spec(yaml.safe_load(path.read_text()), source=str(path))
