"""Build a prepared dataset (docs.parquet + manifest.json) from raw FineWeb-Edu shards.

Adds slice columns (domain, tld, lang/length/edu buckets; dump is native) and a fixed
held-out split chosen by hash of the document id.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import duckdb

from .paths import dataset_dir

SLICE_COLUMNS = ["domain", "tld", "lang_bucket", "len_bucket", "edu_bucket", "dump"]

PREPARE_SQL = """
COPY (
  WITH src AS (
    SELECT id, text, url, dump, language_score, token_count, score, int_score
    FROM read_parquet({raw})
    WHERE (md5_number(id) % 1000000) < {keep_ppm}
  ), d AS (
    SELECT *, lower(regexp_replace(regexp_extract(url, '^[a-zA-Z]+://([^/:?#]+)', 1), '^www\\.', '')) AS domain
    FROM src
  )
  SELECT id, text, url, dump, domain,
    coalesce(nullif(regexp_extract(domain, '\\.([a-z0-9-]+)$', 1), ''), '?') AS tld,
    language_score,
    CASE WHEN language_score < 0.90 THEN 'a:<0.90' WHEN language_score < 0.95 THEN 'b:0.90-0.95'
         WHEN language_score < 0.98 THEN 'c:0.95-0.98' ELSE 'd:>=0.98' END AS lang_bucket,
    token_count,
    CASE WHEN token_count < 256 THEN 'a:<256' WHEN token_count < 1024 THEN 'b:256-1k'
         WHEN token_count < 4096 THEN 'c:1k-4k' ELSE 'd:>=4k' END AS len_bucket,
    score, int_score,
    CASE WHEN score < 3.0 THEN 'a:2.5-3.0' WHEN score < 3.5 THEN 'b:3.0-3.5'
         WHEN score < 4.0 THEN 'c:3.5-4.0' ELSE 'd:>=4.0' END AS edu_bucket,
    CASE WHEN (md5_number('heldout:' || id) % 10000) < {heldout_bp} THEN 'heldout' ELSE 'train' END AS split
  FROM d
) TO '{out}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 50000)
"""


def file_sha256(path: Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def prepare(raw_files: list[Path], name: str, target_tokens: int, heldout_frac: float = 0.02,
            threads: int = 16) -> dict:
    out_dir = dataset_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "docs.parquet"
    con = duckdb.connect()
    con.execute(f"SET threads={threads}")
    raw = "[" + ", ".join(f"'{p}'" for p in raw_files) + "]"
    total = con.execute(f"SELECT sum(token_count) FROM read_parquet({raw})").fetchone()[0]
    keep_ppm = min(1_000_000, int(1_000_000 * target_tokens / total) + 1)
    t0 = time.perf_counter()
    con.execute(PREPARE_SQL.format(raw=raw, keep_ppm=keep_ppm, heldout_bp=int(heldout_frac * 10000),
                                   out=out))
    stats = dict(zip(["docs", "tokens", "heldout_docs", "heldout_tokens"], con.execute(
        f"SELECT count(*), sum(token_count), count(*) FILTER (split = 'heldout'), "
        f"sum(token_count) FILTER (split = 'heldout') FROM '{out}'").fetchone()))
    manifest = {
        "name": name, "raw_files": [str(p) for p in raw_files], "raw_tokens": int(total),
        "keep_ppm": keep_ppm, "heldout_frac": heldout_frac, "slice_columns": SLICE_COLUMNS,
        "stats": {k: int(v) for k, v in stats.items()}, "build_s": time.perf_counter() - t0,
        "sha256": file_sha256(out),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_manifest(name: str) -> dict:
    return json.loads((dataset_dir(name) / "manifest.json").read_text())


def docs_path(name: str) -> Path:
    return dataset_dir(name) / "docs.parquet"
