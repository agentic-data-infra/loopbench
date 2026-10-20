"""Tokenize-pack: GPT-2 BPE over a deterministic document order, concatenated with EOT.

Tokens are stored as raw GPT-2 ids (uint16); the trainer applies its vocab cap.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken

EOT = 50256


@cache
def encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")


def encode_batch(texts: list[str], threads: int = 16) -> list[list[int]]:
    return encoder().encode_ordinary_batch(texts, num_threads=threads)


def cap_vocab(tokens: np.ndarray, vocab_cap: int) -> np.ndarray:
    """Map raw GPT-2 ids into [0, vocab_cap): EOT -> cap-2, ids >= cap-2 -> cap-1 (UNK).

    GPT-2 BPE ids are roughly ordered by merge frequency, so low ids are the common ones.
    """
    t = tokens.astype(np.int64)
    out = np.where(t >= vocab_cap - 2, vocab_cap - 1, t)
    out[t == EOT] = vocab_cap - 2
    return out


def pack(engine, src: str, out_dir: Path, max_tokens: int, seed: int, batch_docs: int = 4096) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    order = f"md5(id || '{seed}')"
    reader = engine.con.execute(f"SELECT id, text FROM {src} ORDER BY {order}").fetch_record_batch(batch_docs)
    chunks, ids, n = [], [], 0
    for rb in reader:
        texts, bids = rb.column("text").to_pylist(), rb.column("id").to_pylist()
        for doc_id, toks in zip(bids, encode_batch(texts)):
            toks.append(EOT)
            chunks.append(np.asarray(toks, dtype=np.uint16))
            ids.append((doc_id, len(toks)))
            n += len(toks)
            if n >= max_tokens:
                break
        if n >= max_tokens:
            break
    tokens = np.concatenate(chunks)[:max_tokens]
    tokens.tofile(out_dir / "tokens.bin")
    pq.write_table(pa.table({"id": [i for i, _ in ids], "n_tokens": [k for _, k in ids]}),
                   out_dir / "packed_docs.parquet")
    meta = {"n_tokens": int(tokens.size), "n_docs": len(ids), "short": n < max_tokens}
    (out_dir / "meta.json").write_text(json.dumps(meta))
    return meta


def load_tokens(out_dir: Path) -> np.ndarray:
    return np.fromfile(out_dir / "tokens.bin", dtype=np.uint16)
