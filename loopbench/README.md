# loopbench

Benchmark harness for the agentic data-discovery loop: a hypothesis spec names a
baseline and a candidate data recipe; loopbench runs (1) hypothesis, (2) data prep,
(3) a small training run per arm and (4) examination, traces every step by category
and records everything in a Parquet ledger.

```
uv sync
uv run loopbench prepare-data --raw ~/data/fineweb-edu/raw/sample/10BT/000_00000.parquet \
    --name fineweb-edu-m0 --tokens 100000000
uv run loopbench run hypotheses/edu_score_ge_3.yaml          # Velox; --engine duckdb, --no-cache
uv run loopbench parity hypotheses/edu_score_ge_3.yaml       # compare engines step by step
uv run loopbench ledger "SELECT hypothesis, overall_diff, overall_ci_lo, overall_ci_hi FROM runs"
```

Data, artifacts and the ledger live under `~/data` (override with `LOOPBENCH_DATA`), never in the repo.

## Layout

| package    | what                                                                                    |
|------------|-----------------------------------------------------------------------------------------|
| `spec/`    | YAML hypothesis schema + validation; plan hashing `hash(op, params, code version, input)` |
| `engines/` | engine interface over Parquet `Rel`s; **Velox** (default) and DuckDB (reference)         |
| `prep/`    | filter, dedup, mix, minhash_dedup; tokenize-pack (GPT-2 BPE, uint16 stream)               |
| `train/`   | nanoGPT-style CPU trainer (~1.3M params, vocab capped to 8192, fixed seeds)              |
| `examine/` | removed-by-slice profile, word 13-gram contamination vs the full prepped pool, per-doc held-out loss, per-slice diff + Poisson-bootstrap CIs (in the engine) |
| `ledger/`  | append-only Parquet tables (`runs`, `nodes`, `slices`, `profiles`, `spans`) queried via DuckDB |
| `trace/`   | spans with categories `agent, dp.prep, train, dp.examine, infer.examine, io, overhead`; breakdown is self time |

Every plan node (prep op, pack, train, score) is cached in the artifact store by its hash,
so re-running a hypothesis only recomputes what changed.

## Dataset `fineweb-edu-m0`

~100M tokens hash-sampled from `HuggingFaceFW/fineweb-edu` `sample/10BT/000_00000.parquet`.
Slice columns: `domain`, `tld`, `dump`, `lang_bucket`, `len_bucket`, `edu_bucket`; fixed held-out
split = 2% of docs by `md5('heldout:' || id)`. Note FineWeb-Edu is already filtered to
`score >= 2.5`, and one shard covers only 11 CC dumps.

## Velox engine

`velox_ext/` is a small C++ pybind module (`lbvelox`) linked against Velox's shared
library. It

* registers `word_ngram_hashes(text, n)` and `minhash_bands(text, bands, rows, shingle, seed)`
  (`TextSig.h`; `loopbench/textsig.py` is the bit-identical Python reference used by DuckDB),
  plus the Parquet reader/writer;
* executes plans written as JSON step lists (`scan`, `filter`, `project`, `agg`, `join`,
  `unnest`, `order_by`, `limit`, `write`) with Velox's PlanBuilder: aggregations are
  partial -> local exchange -> final, so all 16 drivers are used; results come back through
  the Arrow C data interface.

PyVelox's own plan builder has no local exchanges or semi/anti joins, so multi-driver
aggregations would be wrong; that is why plans are built in C++.

Build (once Velox is built with `VELOX_BUILD_SHARED=ON VELOX_BUILD_PYTHON_PACKAGE=ON`):

```
cmake -S ~/source/velox -B ~/source/velox/_build/release-py -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DVELOX_BUILD_SHARED=ON -DVELOX_MONO_LIBRARY=ON -DVELOX_BUILD_PYTHON_PACKAGE=ON -DVELOX_ENABLE_ARROW=ON \
  -DVELOX_ENABLE_PARQUET=ON -DVELOX_BUILD_TESTING=OFF -DTREAT_WARNINGS_AS_ERRORS=OFF \
  -DPython_EXECUTABLE=$PWD/.venv/bin/python -Dpybind11_DIR=$(.venv/bin/python -m pybind11 --cmakedir)
ninja -C ~/source/velox/_build/release-py
uv run python velox_ext/build.py --velox-build ~/source/velox/_build/release-py
```

`build.py` reuses the exact compile/link flags of PyVelox's `plan_builder` module, swaps the
system's non-PIC `libgflags.a` for `libgflags.so` and drops `libunwind.so` (its `_Unwind_*`
otherwise replace libgcc's and C++ exceptions abort instead of reaching Python). It writes
`.venv/.../loopbench_velox.pth` so `lbvelox` and the staged PyVelox import without installing.
Set `LBVELOX_BACKTRACE=1` to print a native backtrace on SIGSEGV/SIGABRT.

Engine parity: hashing (`hash01`, pack/held-out ordering, bootstrap weights) is md5-based and
identical in both engines; `loopbench parity` checks rows, tokens, id sets and per-slice counts
for every step, the pack order and the contamination result.
