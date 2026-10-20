"""Engine parity: run a spec's prep steps on every engine and compare the results.

Checks, per arm and step: (rows, tokens), per-slice (rows, tokens), and the exact id
sets; then the pack order (first 2000 ids) and the contamination result on the final
candidate pool.
"""

from __future__ import annotations

import time

import duckdb

from . import datasets
from .engines import ENGINES
from .runner import Runner
from .spec import build_plan, load_spec
from .trace import Tracer


def _ids(rel) -> set[str]:
    files = ", ".join(f"'{f}'" for f in rel.files)
    where = f" WHERE {rel.where}" if rel.where else ""
    return {r[0] for r in duckdb.sql(f"SELECT id FROM read_parquet([{files}]){where}").fetchall()}


def parity(spec_path, engines=ENGINES) -> bool:
    spec = load_spec(spec_path)
    docs = datasets.docs_path(spec.dataset)
    source_hash = datasets.load_manifest(spec.dataset)["sha256"]
    runners = {e: Runner(engine=e, tracer=Tracer()) for e in engines}
    ref, others = engines[0], engines[1:]
    ok = True

    def check(label, a, b):
        nonlocal ok
        same = a == b
        ok &= same
        print(f"  {'OK ' if same else 'DIFF'} {label}" + ("" if same else f": {a} vs {b}"))

    rels = {}
    for arm in spec.arms():
        for e, r in runners.items():
            t0 = time.perf_counter()
            rels[arm, e] = r.prep(build_plan(spec, arm, source_hash, e), docs)
            print(f"[{arm}] {e}: prep {time.perf_counter() - t0:.2f}s")
        for i in range(1, len(rels[arm, ref])):
            step = spec.arms()[arm][i - 1]
            ra = rels[arm, ref][i]
            for e in others:
                rb = rels[arm, e][i]
                check(f"{arm} step {i} {step.op}: rows,tokens {runners[ref].engine.count(ra)}",
                      runners[ref].engine.count(ra), runners[e].engine.count(rb))
                check(f"{arm} step {i} {step.op}: id set", _ids(ra), _ids(rb))
                for col in spec.examine.slices:
                    check(f"{arm} step {i} {step.op}: by {col}",
                          runners[ref].engine.group_counts(ra, col), runners[e].engine.group_counts(rb, col))

    order = {e: [i for b in r.engine.ordered_batches(rels["candidate", e][-1], ["id"], spec.pack.seed, limit=2000)
                 for i in b.column("id").to_pylist()] for e, r in runners.items()}
    for e in others:
        check("pack order (first 2000 ids)", order[ref], order[e])
    heldout = runners[ref].engine.source(docs, "heldout")
    contam = {}
    for e, r in runners.items():
        t0 = time.perf_counter()
        contam[e] = r.engine.contamination(rels["candidate", e][-1], heldout, spec.examine.ngram)
        print(f"contamination {e}: {time.perf_counter() - t0:.2f}s")
    for e in others:
        check(f"contamination (candidate pool, {len(contam[ref])} held-out docs)", contam[ref], contam[e])
    print("PARITY OK" if ok else "PARITY FAILED")
    return ok
