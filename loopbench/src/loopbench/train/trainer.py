"""CPU trainer with fixed seeds. Samples random windows from the packed token stream."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ..prep.tokenize import cap_vocab
from ..spec import TrainConfig
from .model import GPT


def build_model(cfg: TrainConfig) -> GPT:
    return GPT(cfg.vocab_cap, cfg.ctx, cfg.d_model, cfg.n_layer, cfg.n_head)


def train(tokens: np.ndarray, cfg: TrainConfig, tracer=None, log_every: int = 50) -> tuple[GPT, dict]:
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    data = torch.from_numpy(cap_vocab(tokens, cfg.vocab_cap))
    model = build_model(cfg)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr, betas=(0.9, 0.95))

    def lr_at(step):
        if step < cfg.warmup:
            return cfg.lr * (step + 1) / cfg.warmup
        p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
        return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    hi = len(data) - cfg.ctx - 1
    if hi <= 0:
        raise ValueError(f"packed stream too short ({len(data)} tokens) for ctx {cfg.ctx}")
    losses = []
    t0 = time.perf_counter()
    model.train()
    for step in range(cfg.steps):
        starts = torch.from_numpy(rng.integers(0, hi, cfg.batch_size))
        idx = starts[:, None] + torch.arange(cfg.ctx + 1)
        batch = data[idx]
        x, y = batch[:, :-1], batch[:, 1:]
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss = model.loss(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if log_every and (step % log_every == 0 or step == cfg.steps - 1):
            print(f"    step {step:5d}  loss {losses[-1]:.4f}  lr {lr_at(step):.2e}", flush=True)
    dt = time.perf_counter() - t0
    tokens_seen = cfg.steps * cfg.batch_size * cfg.ctx
    stats = {"n_params": model.n_params(), "final_loss": float(np.mean(losses[-20:])),
             "tokens_seen": tokens_seen, "tokens_per_s": tokens_seen / dt,
             "epochs": tokens_seen / len(data)}
    return model, stats


def save(model: GPT, stats: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "train_stats.json").write_text(json.dumps(stats))


def load(cfg: TrainConfig, out_dir: Path) -> tuple[GPT, dict]:
    model = build_model(cfg)
    model.load_state_dict(torch.load(out_dir / "model.pt", weights_only=True))
    return model.eval(), json.loads((out_dir / "train_stats.json").read_text())
