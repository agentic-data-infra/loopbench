"""Build a prepared dataset from raw FineWeb-Edu shards, as Velox plans.

Output: ``<dataset>/docs/*.parquet`` + ``manifest.json``. Adds slice columns (domain,
tld, lang/length/edu buckets; dump is native) and a fixed held-out split:

* the split is keyed by the normalised URL (scheme and ``www.`` stripped), so the same
  page crawled in several dumps lands on one side;
* held-out docs that are MinHash near-duplicates of any train doc (they share an LSH
  band bucket) are moved to train, so the held-out set has no near-dups in training.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

from .engines import Rel, get_engine
from .engines.velox_engine import _hash01
from .prep.ops import minhash_params
from .paths import dataset_dir

SLICE_COLUMNS = ["domain", "tld", "lang_bucket", "len_bucket", "edu_bucket", "dump"]
RAW_COLUMNS = ["id", "text", "url", "dump", "language_score", "token_count", "score", "int_score"]
NEARDUP = minhash_params({})
SPLIT_HASH = _hash01("concat('heldout:', url_key)")

SLICES = [
    "id", "text", "url", "dump", "domain",
    "coalesce(regexp_extract(domain, '\\.([a-z0-9-]+)$', 1), '?') AS tld",
    "language_score",
    "CASE WHEN language_score < 0.90 THEN 'a:<0.90' WHEN language_score < 0.95 THEN 'b:0.90-0.95' "
    "WHEN language_score < 0.98 THEN 'c:0.95-0.98' ELSE 'd:>=0.98' END AS lang_bucket",
    "token_count",
    "CASE WHEN token_count < 256 THEN 'a:<256' WHEN token_count < 1024 THEN 'b:256-1k' "
    "WHEN token_count < 4096 THEN 'c:1k-4k' ELSE 'd:>=4k' END AS len_bucket",
    "score", "int_score",
    "CASE WHEN score < 3.0 THEN 'a:2.5-3.0' WHEN score < 3.5 THEN 'b:3.0-3.5' "
    "WHEN score < 4.0 THEN 'c:3.5-4.0' ELSE 'd:>=4.0' END AS edu_bucket",
    "url_key",
]


def files_sha256(files: list[str], bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(Path(f).name.encode())
        with open(f, "rb") as fh:
            while chunk := fh.read(bufsize):
                h.update(chunk)
    return h.hexdigest()


def prepare(raw_files: list[Path], name: str, target_tokens: int, heldout_frac: float = 0.02,
            threads: int = 16) -> dict:
    eng = get_engine("velox", threads=threads)
    out_dir = dataset_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = Rel(tuple(str(p) for p in raw_files))
    t0 = time.perf_counter()
    total = eng.count(raw)[1]
    keep = min(1.0, target_tokens / total * 1.0001)
    p = NEARDUP
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        tmp = Path(tmp)
        # 1. sample, derive slice columns, URL-keyed split
        staged = eng._write(eng.scan(raw, RAW_COLUMNS) + [
            {"op": "filter", "expr": f"{_hash01('id')} < {keep!r}"},
            {"op": "project", "exprs": [
                *RAW_COLUMNS,
                "lower(regexp_replace(regexp_extract(url, '^[a-zA-Z]+://([^/:?#]+)', 1), '^www\\.', '')) AS domain",
                "lower(regexp_replace(url, '^[a-zA-Z]+://(www\\.)?', '')) AS url_key"]},
            {"op": "project", "exprs": SLICES + [
                f"CASE WHEN {SPLIT_HASH} < {heldout_frac!r} "
                "THEN 'heldout' ELSE 'train' END AS split0"]}], tmp / "staged")
        # 2. LSH band buckets that contain both splits -> their held-out docs move to train
        bands = eng._write(eng.scan(staged, ["id", "text", "split0"]) + [
            {"op": "project", "exprs": ["id", "split0", f"minhash_bands(text, {p['bands']}, {p['rows']}, "
                                                        f"{p['shingle']}, cast({p['seed']} as bigint)) AS hs"]},
            {"op": "unnest", "replicate": ["id", "split0"], "columns": ["hs"], "names": ["h"]}], tmp / "bands")
        mixed = eng.scan(bands) + [
            {"op": "project", "exprs": ["h", "if(split0 = 'heldout', 1, 0) AS is_h", "if(split0 = 'train', 1, 0) AS is_t"]},
            {"op": "agg", "keys": ["h"], "aggs": ["sum(is_h) AS nh", "sum(is_t) AS nt"]},
            {"op": "filter", "expr": "nh > 0 AND nt > 0"},
            {"op": "project", "exprs": ["h AS mh"]}]
        moved = eng.scan(bands) + [
            {"op": "filter", "expr": "split0 = 'heldout'"},
            {"op": "join", "type": "semi", "left": ["h"], "right": ["mh"], "build": mixed, "output": ["id"]},
            {"op": "agg", "keys": ["id"], "aggs": []},
            {"op": "project", "exprs": ["id AS moved_id"]}]
        cols = [c.split(" AS ")[-1] for c in SLICES if not c.endswith("url_key")]
        docs_dir = out_dir / "docs"
        if docs_dir.exists():
            shutil.rmtree(docs_dir)
        docs = eng._write(eng.scan(staged) + [
            {"op": "join", "type": "left", "left": ["id"], "right": ["moved_id"], "build": moved,
             "output": [*cols, "split0", "moved_id"]},
            {"op": "project", "exprs": [*cols, "if(moved_id IS NULL, split0, 'train') AS split"]}], docs_dir)
        n_moved = eng.count(Rel(staged.files, "split0 = 'heldout'"))[0] - eng.count(eng.source(docs_dir, "heldout"))[0]
    tr, ho = eng.count(eng.source(docs_dir, "train")), eng.count(eng.source(docs_dir, "heldout"))
    manifest = {
        "name": name, "raw_files": [str(p) for p in raw_files], "raw_tokens": int(total), "keep_frac": keep,
        "heldout_frac": heldout_frac, "split_key": "url_key", "neardup_moved_to_train": n_moved,
        "neardup_params": p, "slice_columns": SLICE_COLUMNS, "engine": "velox",
        "stats": {"docs": tr[0] + ho[0], "tokens": tr[1] + ho[1], "heldout_docs": ho[0], "heldout_tokens": ho[1]},
        "build_s": time.perf_counter() - t0, "sha256": files_sha256(list(docs.files)),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_manifest(name: str) -> dict:
    return json.loads((dataset_dir(name) / "manifest.json").read_text())


def docs_path(name: str) -> Path:
    d = dataset_dir(name)
    return d / "docs" if (d / "docs").is_dir() else d / "docs.parquet"
