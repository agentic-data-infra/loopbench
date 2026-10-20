"""Engine tests on a small synthetic dataset: every engine must agree with the DuckDB reference."""

import importlib.util

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from loopbench import textsig
from loopbench.engines import Rel, get_engine
from loopbench.examine import diff

HAVE_VELOX = importlib.util.find_spec("lbvelox") is not None
ENGINES = ["duckdb"] + (["velox"] if HAVE_VELOX else [])

WORDS = "the quick brown fox jumps over a lazy dog while seven bright stars watch from far away".split()


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    rng = np.random.default_rng(0)
    d = tmp_path_factory.mktemp("data")
    texts = [" ".join(rng.choice(WORDS, size=rng.integers(3, 40))) for _ in range(400)]
    texts += texts[:30]  # exact dups
    texts += [t + " extra" for t in texts[30:60]]  # near dups
    n = len(texts)
    tbl = pa.table({
        "id": [f"doc{i:05d}" for i in range(n)], "text": texts,
        "token_count": [len(t.split()) for t in texts],
        "score": rng.uniform(2.5, 4.5, n).round(3), "edu_bucket": rng.choice(["a", "b", "c"], n),
        "split": np.where(rng.uniform(size=n) < 0.1, "heldout", "train")})
    pq.write_table(tbl, d / "docs.parquet", row_group_size=64)
    return d / "docs.parquet"


@pytest.mark.parametrize("engine", ENGINES)
def test_ops_match_reference(engine, data, tmp_path):
    ref, eng = get_engine("duckdb", threads=4), get_engine(engine, threads=4)
    src = ref.source(data, "train")
    for op, params in [("filter", {"where": "score >= 3.0"}), ("dedup", {"key": "text"}),
                       ("mix", {"by": "edu_bucket", "weights": {"a": 1, "b": 1, "c": 2}}),
                       ("minhash_dedup", {"bands": 8, "rows": 2, "shingle": 3})]:
        a = ref.run_op(op, params, src, tmp_path / f"ref_{op}")
        b = eng.run_op(op, params, src, tmp_path / f"{engine}_{op}")
        assert ref.count(a) == eng.count(b), op
        assert ref.group_counts(a, "edu_bucket") == eng.group_counts(b, "edu_bucket"), op
    dd = ref.run_op("dedup", {"key": "text"}, src, tmp_path / "chk")
    assert ref.count(dd)[0] < ref.count(src)[0]


@pytest.mark.parametrize("engine", ENGINES)
def test_order_contamination_bootstrap(engine, data, tmp_path):
    ref, eng = get_engine("duckdb", threads=4), get_engine(engine, threads=4)
    train, held = ref.source(data, "train"), ref.source(data, "heldout")
    ids = lambda e: [i for b in e.ordered_batches(train, ["id"], 7, limit=50) for i in b.column("id").to_pylist()]
    assert ids(ref) == ids(eng)
    assert ref.contamination(train, held, 4) == eng.contamination(train, held, 4)
    rng = np.random.default_rng(1)
    n = 300
    base = rng.normal(5, 0.1, n)
    path = diff.write_long(tmp_path / "long.parquet", [f"d{i}" for i in range(n)],
                           {"s": ["x"] * 150 + ["y"] * 150}, base, base - 0.05, top_k=5)
    ra = {(r["slice"], r["value"]): r for r in ref.slice_bootstrap(path, 200, 0.95)}
    rb = {(r["slice"], r["value"]): r for r in eng.slice_bootstrap(path, 200, 0.95)}
    assert ra.keys() == rb.keys() == {("overall", "all"), ("s", "x"), ("s", "y")}
    for k in ra:
        assert ra[k]["n_docs"] == rb[k]["n_docs"]
        assert ra[k]["diff"] == pytest.approx(rb[k]["diff"])
        assert ra[k]["ci_hi"] < 0 and rb[k]["ci_hi"] < 0
        assert ra[k]["ci_lo"] == pytest.approx(rb[k]["ci_lo"], abs=2e-3)


def test_textsig_reference():
    h = textsig.word_ngram_hashes("A b, C d! a B c d", 2)
    assert h.dtype == np.int64 and len(h) == 7 and h[0] == h[4]  # "a b" == "A b" after lowercasing
    assert len(textsig.minhash_bands("one two", 4, 2, 3, 0)) == 0
    a = textsig.minhash_bands(" ".join(WORDS), 6, 3, 3, 1)
    assert len(a) == 6 and (a == textsig.minhash_bands(" ".join(WORDS).upper(), 6, 3, 3, 1)).all()


@pytest.mark.skipif(not HAVE_VELOX, reason="lbvelox not built")
def test_velox_functions_bit_identical(data):
    eng = get_engine("velox", threads=4)
    rel = Rel((str(data),))
    t = eng.execute(eng.scan(rel, ["id", "text"]) + [{"op": "project", "exprs": [
        "id", "text", "word_ngram_hashes(text, 3) AS g", "minhash_bands(text, 5, 3, 2, cast(42 as bigint)) AS m"]}])
    for text, g, m in zip(t["text"].to_pylist(), t["g"].to_pylist(), t["m"].to_pylist()):
        assert g == textsig.word_ngram_hashes(text, 3).tolist()
        assert m == textsig.minhash_bands(text, 5, 3, 2, 42).tolist()
