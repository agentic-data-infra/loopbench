"""loopbench CLI.

  loopbench prepare-data --raw <parquet>... --name fineweb-edu-m0 --tokens 100000000
  loopbench run hypotheses/edu_score_ge_3.yaml [--no-cache]
  loopbench ledger "SELECT * FROM runs"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _print_results(res: dict, tracer) -> None:
    print("\n== train ==")
    for arm, o in res["outs"].items():
        s = o["train_stats"]
        print(f"  {arm:<9} params {s['n_params']:,}  final train loss {s['final_loss']:.4f}  "
              f"{s['tokens_per_s']:,.0f} tok/s  epochs {s['epochs']:.2f}")
    print("\n== removed by candidate prep (tokens, by edu_bucket) ==")
    for r in res["profiles"]:
        if r["arm"] == "candidate" and r["slice"] == "edu_bucket":
            print(f"  step {r['step']} {r['op']:<7} {r['value']:<12} {r['tokens_in']:>12,} -> "
                  f"{r['tokens_out']:>12,}  removed {100 * r['tokens_removed_frac']:5.1f}%")
    print("\n== contamination (held-out docs sharing any 13-gram with packed train stream) ==")
    for arm, c in res["contamination"].items():
        print(f"  {arm:<9} {100 * (c > 0).mean():5.2f}% of docs, mean overlap {100 * c.mean():.3f}%")
    print("\n== held-out loss, candidate - baseline (negative = candidate better), 95% bootstrap CI ==")
    print(f"  {'slice':<12}{'value':<26}{'n':>6}{'base':>8}{'cand':>8}{'diff':>9}   ci")
    for r in res["slices"]:
        mark = " *" if r["significant"] else ""
        print(f"  {r['slice']:<12}{r['value'][:25]:<26}{r['n_docs']:>6}{r['loss_baseline']:>8.3f}"
              f"{r['loss_candidate']:>8.3f}{r['diff']:>+9.4f}   [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]{mark}")
    print("\n== trace ==")
    print(tracer.format_tree(min_s=0.05))
    print("\n== time by category (self time) ==")
    print(tracer.format_breakdown())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="loopbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare-data")
    p.add_argument("--raw", nargs="+", type=Path, required=True)
    p.add_argument("--name", default="fineweb-edu-m0")
    p.add_argument("--tokens", type=int, default=100_000_000)
    p.add_argument("--heldout-frac", type=float, default=0.02)
    r = sub.add_parser("run")
    r.add_argument("spec", type=Path)
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--engine", default="duckdb")
    lq = sub.add_parser("ledger")
    lq.add_argument("sql")
    args = ap.parse_args(argv)

    if args.cmd == "prepare-data":
        from .datasets import prepare
        print(json.dumps(prepare(args.raw, args.name, args.tokens, args.heldout_frac), indent=2))
    elif args.cmd == "run":
        from .runner import Runner
        runner = Runner(engine=args.engine, use_cache=not args.no_cache)
        res = runner.run(args.spec)
        _print_results(res, runner.tracer)
    elif args.cmd == "ledger":
        from .ledger import Ledger
        Ledger().query(args.sql).show(max_width=200)


if __name__ == "__main__":
    main()
