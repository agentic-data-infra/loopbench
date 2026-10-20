import numpy as np
import pytest

from loopbench.examine import contamination, diff
from loopbench.spec import SpecError, build_plan
from loopbench.spec.schema import parse_spec
from loopbench.trace import Tracer

SPEC = {
    "name": "t", "dataset": "d",
    "baseline": {"prep": [{"op": "dedup", "params": {"key": "text"}}]},
    "candidate": {"prep": [{"op": "dedup", "params": {"key": "text"}},
                           {"op": "filter", "params": {"where": "score >= 3"}}]},
}


def test_plan_hash_shares_prefix_and_depends_on_input():
    spec = parse_spec(SPEC)
    b, c = build_plan(spec, "baseline", "src"), build_plan(spec, "candidate", "src")
    assert b.prep[0].hash == c.prep[0].hash
    assert b.hash != c.hash
    assert build_plan(spec, "baseline", "other").prep[0].hash != b.prep[0].hash
    assert build_plan(parse_spec(SPEC), "candidate", "src").hash == c.hash


@pytest.mark.parametrize("bad", [
    {"op": "nope"}, {"op": "filter", "params": {}}, {"op": "dedup", "params": {"key": "score"}},
    {"op": "filter", "params": {"where": "1=1; DROP TABLE x"}},
])
def test_spec_validation_rejects(bad):
    with pytest.raises(SpecError):
        parse_spec({**SPEC, "candidate": {"prep": [bad]}})


def test_trace_self_time():
    t = Tracer()
    with t.span("root", "overhead"):
        with t.span("a", "train"):
            sum(range(100000))
    bd = t.breakdown()
    assert bd["train"] > 0 and abs(sum(bd.values()) - t.total_s()) < 1e-9


def test_contamination():
    rng = np.random.default_rng(0)
    train = rng.integers(0, 50000, 10000).astype(np.uint16)
    held = [train[100:200].copy(), rng.integers(0, 50000, 100).astype(np.uint16)]
    out = contamination.check(train, held, 13)
    assert out[0] == 1.0 and out[1] == 0.0


def test_slice_diff_ci():
    rng = np.random.default_rng(0)
    base = rng.normal(5, 0.1, 500)
    rows = diff.per_slice_diff({"s": ["x"] * 250 + ["y"] * 250}, base, base - 0.05, top_k=5,
                               min_docs=10, n_boot=200, ci=0.95, seed=0)
    assert rows[0]["slice"] == "overall" and rows[0]["significant"] and rows[0]["ci_hi"] < 0
    assert {r["value"] for r in rows} == {"all", "x", "y"}
