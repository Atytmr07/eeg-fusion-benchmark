"""Bonn training loop: deterministic, and the same protocol for every model.

- The seed is set right before each model is built, from (repeat, fold, model), so no
  result depends on call order.
- Data stays in memory as tensors; there is no DataLoader or file reading.
- Early stopping looks only at validation macro F1. The test fold is never used for
  any decision.
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from .config import Config
from .models import build

DEVICE = torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def class_weights(y: np.ndarray, ncls: int) -> torch.Tensor:
    """Inverse-frequency class weights, normalised to sum to ncls."""
    counts = np.bincount(y, minlength=ncls).astype(np.float64)
    w = 1.0 / np.maximum(counts, 1.0)
    w = w / w.sum() * ncls
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict(model: nn.Module, x1d: torch.Tensor, x2d: torch.Tensor,
            batch_size: int = 128) -> np.ndarray:
    """Raw logits (before softmax)."""
    model.eval()
    outs = []
    for i in range(0, len(x1d), batch_size):
        outs.append(model(x1d[i:i + batch_size], x2d[i:i + batch_size]).cpu().numpy())
    return np.concatenate(outs)


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_one(model_name: str, cfg: Config, ncls: int, seed: int,
              tr: tuple, va: tuple, fusion_budget: int | None = None) -> nn.Module:
    """Train one model and return it with its best-validation weights.

    tr, va: (x1d, x2d, y) tensor triples.
    """
    set_seed(seed)
    model = build(model_name, ncls, cfg.embed_dim, cfg.dropout,
                  fusion_budget=fusion_budget).to(DEVICE)

    x1_tr, x2_tr, y_tr = tr
    x1_va, x2_va, y_va = va
    y_tr_np = y_tr.numpy()

    w = class_weights(y_tr_np, ncls).to(DEVICE) if cfg.class_weighted_loss else None
    criterion = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    n = len(y_tr)
    g = torch.Generator().manual_seed(seed)
    best_f1, best_state, wait = -1.0, None, 0

    for ep in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x1_tr[idx], x2_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()

        logits = predict(model, x1_va, x2_va)
        f1 = f1_score(y_va.numpy(), logits.argmax(1), average="macro")
        if f1 > best_f1 + 1e-6:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        # Early stopping only after min_epochs: before that, a model can still be in
        # the trivial single-class solution with flat validation F1 (see Config).
        if ep >= cfg.min_epochs and wait >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def best_blend_weight(logit1: np.ndarray, logit2: np.ndarray, y: np.ndarray,
                      ncls: int, grid=None) -> float:
    """Score-fusion weight w for p = w*softmax(logit1) + (1-w)*softmax(logit2).

    Chosen on the VALIDATION set only, by minimising log loss over a 41-point grid.
    """
    from sklearn.metrics import log_loss
    if grid is None:
        grid = np.linspace(0.0, 1.0, 41)
    p1, p2 = softmax_np(logit1), softmax_np(logit2)
    best_w, best_ll = 0.5, np.inf
    for w in grid:
        ll = log_loss(y, w * p1 + (1 - w) * p2, labels=list(range(ncls)))
        if ll < best_ll:
            best_ll, best_w = ll, float(w)
    return best_w
