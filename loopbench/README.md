# loopbench

Benchmark harness for the agentic data-discovery loop: a hypothesis spec names a
baseline and a candidate data recipe; loopbench runs (1) hypothesis, (2) data prep,
(3) a small training run per arm and (4) examination, traces every step by category
and records everything in a Parquet ledger.

```
uv sync
uv run loopbench prepare-data --raw ~/data/fineweb-edu/raw/sample/10BT/000_00000.parquet \
    --name fineweb-edu-m0 --tokens 100000000
uv run loopbench run hypotheses/edu_score_ge_3.yaml          # --no-cache for a cold run
uv run loopbench ledger "SELECT hypothesis, overall_diff, overall_ci_lo, overall_ci_hi FROM runs"
```

Data, artifacts and the ledger live under `~/data` (override with `LOOPBENCH_DATA`), never in the repo.

## Layout

| package    | what                                                                                    |
|------------|-----------------------------------------------------------------------------------------|
| `spec/`    | YAML hypothesis schema + validation; plan hashing `hash(op, params, code version, input)` |
| `engines/` | engine interface; DuckDB adapter                                                        |
| `prep/`    | filter, dedup, mix (SQL ops), tokenize-pack (GPT-2 BPE, uint16 stream)                   |
| `train/`   | nanoGPT-style CPU trainer (~1.3M params, vocab capped to 8192, fixed seeds)              |
| `examine/` | removed-by-slice profile, 13-gram contamination, per-doc held-out loss, slice diff + bootstrap CIs |
| `ledger/`  | append-only Parquet tables (`runs`, `nodes`, `slices`, `profiles`, `spans`) queried via DuckDB |
| `trace/`   | spans with categories `agent, dp.prep, train, dp.examine, infer.examine, io, overhead`; breakdown is self time |

Every plan node (prep op, pack, train, score) is cached in the artifact store by its hash,
so re-running a hypothesis only recomputes what changed.

## Dataset `fineweb-edu-m0`

~100M tokens hash-sampled from `HuggingFaceFW/fineweb-edu` `sample/10BT/000_00000.parquet`.
Slice columns: `domain`, `tld`, `dump`, `lang_bucket`, `len_bucket`, `edu_bucket`; fixed held-out
split = 2% of docs by `md5('heldout:' || id)`. Note FineWeb-Edu is already filtered to
`score >= 2.5`, and one shard covers only 11 CC dumps.
