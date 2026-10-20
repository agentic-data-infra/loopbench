"""Runs one hypothesis end to end: (1) hypothesis, (2) prep, (3) train, (4) examine."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import datasets
from .engines import Rel, get_engine
from .examine import contamination, diff, profile, scoring
from .ledger import Ledger
from .paths import WORK_ROOT, artifact_dir
from .prep import tokenize
from .spec import HypothesisSpec, Plan, PlanNode, build_plan, load_spec, node
from .trace import Tracer
from .train import trainer

DONE = "_SUCCESS"


class Runner:
    def __init__(self, engine: str = "velox", use_cache: bool = True, tracer: Tracer | None = None,
                 ledger: Ledger | None = None):
        self.engine = get_engine(engine)
        self.use_cache = use_cache
        self.tracer = tracer or Tracer()
        self.ledger = ledger or Ledger()
        self.node_rows: list[dict] = []
        self._this_run: set[str] = set()  # nodes computed in this run are reused even with use_cache=False

    # -- artifact cache -------------------------------------------------------------------------
    def _cached(self, n: PlanNode) -> Path | None:
        d = artifact_dir(n.hash)
        reuse = self.use_cache or n.hash in self._this_run
        return d if reuse and (d / DONE).exists() else None

    def _fresh_dir(self, n: PlanNode) -> Path:
        d = artifact_dir(n.hash)
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        return d

    def _finish(self, n: PlanNode, d: Path, arm: str, seconds: float, stats: dict, cached: bool):
        self._this_run.add(n.hash)
        if not cached:
            (d / "node.json").write_text(json.dumps({"op": n.op, "params": n.params, "code": n.code,
                                                     "input": n.input_hash, "stats": stats}))
            (d / DONE).touch()
        self.node_rows.append({"run_id": self.tracer.run_id, "arm": arm, "op": n.op, "hash": n.hash,
                               "input_hash": n.input_hash, "code": n.code, "params": json.dumps(n.params),
                               "cached": cached, "seconds": seconds, "stats": json.dumps(stats)})

    # -- steps 2 and 3: prep, pack, train ---------------------------------------------------------
    def prep(self, plan: Plan, docs: Path) -> list[Rel]:
        """Run (or reuse) the prep nodes of ``plan``; returns the relation after each step."""
        t, eng = self.tracer, self.engine
        rel = eng.source(docs, "train")
        rels = [rel]
        for n in plan.prep:
            t0 = time.perf_counter()
            d, cached = self._cached(n), True
            params = {k: v for k, v in n.params.items() if not k.startswith("_")}
            if d:
                stats = json.loads((d / "node.json").read_text())["stats"]
            else:
                d, cached = self._fresh_dir(n), False
                with t.span(f"prep.{n.op}[{plan.arm}]", "dp.prep", engine=eng.name, hash=n.hash[:12]):
                    _, res = eng.run_and_count(n.op, params, rel, d / "out")
                    stats = asdict(res)
            self._finish(n, d, plan.arm, time.perf_counter() - t0, stats, cached)
            rel = Rel.of_dir(d / "out")
            rels.append(rel)
        return rels

    def execute(self, spec: HypothesisSpec, plan: Plan, docs: Path) -> dict:
        t, eng = self.tracer, self.engine
        rels = self.prep(plan, docs)
        rel = rels[-1]
        n = plan.pack
        t0 = time.perf_counter()
        pack_dir, cached = self._cached(n), True
        if not pack_dir:
            pack_dir, cached = self._fresh_dir(n), False
            with t.span(f"prep.tokenize_pack[{plan.arm}]", "dp.prep", engine=eng.name, hash=n.hash[:12]):
                stats = tokenize.pack(eng, rel, pack_dir, spec.pack.max_tokens, spec.pack.seed)
        else:
            stats = json.loads((pack_dir / "meta.json").read_text())
        self._finish(n, pack_dir, plan.arm, time.perf_counter() - t0, stats, cached)

        n = plan.train
        t0 = time.perf_counter()
        model_dir, cached = self._cached(n), True
        if not model_dir:
            model_dir, cached = self._fresh_dir(n), False
            with t.span(f"io.load_tokens[{plan.arm}]", "io"):
                tokens = tokenize.load_tokens(pack_dir)
            with t.span(f"train[{plan.arm}]", "train", hash=n.hash[:12]):
                print(f"  training {plan.arm} ({len(tokens):,} packed tokens)", flush=True)
                model, stats = trainer.train(tokens, spec.train)
            with t.span(f"io.save_model[{plan.arm}]", "io"):
                trainer.save(model, stats, model_dir)
        else:
            stats = json.loads((model_dir / "train_stats.json").read_text())
        self._finish(n, model_dir, plan.arm, time.perf_counter() - t0, stats, cached)
        return {"rels": rels, "pack_dir": pack_dir, "model_dir": model_dir, "train_stats": stats}

    # -- step 4: examine --------------------------------------------------------------------------
    def heldout(self, spec: HypothesisSpec, docs: Path) -> tuple[list[str], dict[str, list], list[np.ndarray]]:
        ex = spec.examine
        cols = ["id", "text", *ex.slices]
        batches = list(self.engine.ordered_batches(self.engine.source(docs, "heldout"), cols, ex.seed,
                                                   limit=ex.heldout_docs))
        tbl = pa.Table.from_batches(batches)
        ids = tbl["id"].to_pylist()
        slices = {c: tbl[c].to_pylist() for c in ex.slices}
        toks = tokenize.encode_batch(tbl["text"].to_pylist())
        return ids, slices, [np.asarray(t, dtype=np.uint16) for t in toks]

    def score(self, spec: HypothesisSpec, plan: Plan, source_hash: str, model_dir: Path,
              heldout_tokens: list[np.ndarray]) -> np.ndarray:
        ex = spec.examine
        n = node("score", {"heldout_docs": ex.heldout_docs, "score_tokens": ex.score_tokens, "seed": ex.seed},
                 f"{plan.train.hash}:{source_hash}")
        t0 = time.perf_counter()
        d, cached = self._cached(n), True
        if d:
            with self.tracer.span(f"io.load_scores[{plan.arm}]", "io"):
                losses = pq.read_table(d / "scores.parquet").column("loss").to_numpy()
        else:
            d, cached = self._fresh_dir(n), False
            with self.tracer.span(f"io.load_model[{plan.arm}]", "io"):
                model, _ = trainer.load(spec.train, model_dir)
            with self.tracer.span(f"score_heldout[{plan.arm}]", "infer.examine", docs=len(heldout_tokens)):
                losses = scoring.score_docs(model, heldout_tokens, spec.train.vocab_cap, ex.score_tokens,
                                            threads=spec.train.threads)
            with self.tracer.span(f"io.save_scores[{plan.arm}]", "io"):
                pq.write_table(pa.table({"loss": losses}), d / "scores.parquet")
        self._finish(n, d, plan.arm, time.perf_counter() - t0, {"mean_loss": float(np.nanmean(losses))}, cached)
        return losses

    def _scratch(self) -> Path:
        d = WORK_ROOT / "runs" / self.tracer.run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- driver -------------------------------------------------------------------------------------
    def run(self, spec_path: str | Path) -> dict:
        t = self.tracer
        with t.span("run", "overhead"):
            with t.span("hypothesis.load_spec", "agent"):
                spec = load_spec(spec_path)
            with t.span("io.dataset_manifest", "io"):
                manifest = datasets.load_manifest(spec.dataset)
                docs = datasets.docs_path(spec.dataset)
            source_hash = manifest["sha256"]
            plans = {arm: build_plan(spec, arm, source_hash, self.engine.name) for arm in spec.arms()}
            print(f"[{spec.name}] run {t.run_id}  engine={self.engine.name}")
            for arm, p in plans.items():
                print(f"  {arm:<9} plan {p.hash[:12]}  " + " -> ".join(n.op for n in p.nodes()))

            outs = {arm: self.execute(spec, p, docs) for arm, p in plans.items()}

            ex = spec.examine
            prof_rows = []
            with t.span("examine.profile_removed", "dp.examine"):
                for arm, p in plans.items():
                    rels = outs[arm]["rels"]
                    for i, n in enumerate(p.prep):
                        for col in ex.slices:
                            for r in profile.removed_by_slice(self.engine, rels[i], rels[i + 1], col, ex.top_k):
                                prof_rows.append({"run_id": t.run_id, "arm": arm, "step": i, "op": n.op,
                                                  "hash": n.hash, **r})
            with t.span("examine.heldout_fetch_tokenize", "dp.examine"):
                ids, slices, held = self.heldout(spec, docs)
            contam = {}
            heldout_rel = self.engine.source(docs, "heldout")
            with t.span("examine.contamination", "dp.examine", n=ex.ngram):
                for arm in plans:
                    contam[arm] = contamination.check(self.engine, outs[arm]["rels"][-1], heldout_rel, ids, ex.ngram)
            losses = {arm: self.score(spec, p, source_hash, outs[arm]["model_dir"], held)
                      for arm, p in plans.items()}
            with t.span("examine.slice_diff", "dp.examine"):
                long_path = diff.write_long(self._scratch() / "slices_long.parquet", ids, slices,
                                            losses["baseline"], losses["candidate"], ex.top_k)
                slice_rows = diff.per_slice_diff(self.engine, long_path, ex.slices, min_docs=ex.min_docs,
                                                 n_boot=ex.bootstrap, ci=ex.ci)
            with t.span("io.ledger_write", "io"):
                run_row = {"run_id": t.run_id, "hypothesis": spec.name, "dataset": spec.dataset,
                           "engine": self.engine.name,
                           "source_hash": source_hash, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                           "spec": json.dumps(spec.to_dict()),
                           "baseline_plan": plans["baseline"].hash, "candidate_plan": plans["candidate"].hash,
                           "overall_diff": slice_rows[0]["diff"], "overall_ci_lo": slice_rows[0]["ci_lo"],
                           "overall_ci_hi": slice_rows[0]["ci_hi"],
                           **{f"contam_{arm}": float((c > 0).mean()) for arm, c in contam.items()},
                           **{f"train_loss_{arm}": o["train_stats"]["final_loss"] for arm, o in outs.items()}}
                self.ledger.append("runs", [run_row])
                self.ledger.append("nodes", self.node_rows)
                self.ledger.append("slices", [{"run_id": t.run_id, **r} for r in slice_rows])
                self.ledger.append("profiles", prof_rows)
        self.ledger.append("spans", t.rows())
        return {"spec": spec, "plans": plans, "outs": outs, "slices": slice_rows, "profiles": prof_rows,
                "contamination": contam, "run": run_row}
