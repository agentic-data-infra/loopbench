import numpy as np
import pytest

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
