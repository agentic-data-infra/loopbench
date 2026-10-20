"""Per-document held-out loss scoring."""

from __future__ import annotations

import numpy as np
import torch

from ..prep.tokenize import cap_vocab


@torch.inference_mode()
def score_docs(model, docs: list[np.ndarray], vocab_cap: int, max_tokens: int, batch_size: int = 64,
               threads: int = 16) -> np.ndarray:
    """Mean next-token loss over the first ``max_tokens`` predictions of each doc (NaN if < 2 tokens)."""
    torch.set_num_threads(threads)
    model.eval()
    T = min(max_tokens, model.ctx)
    out = np.full(len(docs), np.nan)
    order = sorted(range(len(docs)), key=lambda i: len(docs[i]))
    for s in range(0, len(order), batch_size):
        ids = [i for i in order[s:s + batch_size] if len(docs[i]) >= 2]
        if not ids:
            continue
        L = min(T + 1, max(len(docs[i]) for i in ids))
        x = torch.zeros(len(ids), L - 1, dtype=torch.long)
        y = torch.full((len(ids), L - 1), -1, dtype=torch.long)
        for r, i in enumerate(ids):
            d = torch.from_numpy(cap_vocab(docs[i][:L], vocab_cap))
            x[r, :len(d) - 1] = d[:-1]
            y[r, :len(d) - 1] = d[1:]
        loss = model.loss(x, y, reduction="none")
        mask = (y >= 0).float()
        per_doc = (loss * mask).sum(1) / mask.sum(1)
        out[ids] = per_doc.numpy()
    return out
