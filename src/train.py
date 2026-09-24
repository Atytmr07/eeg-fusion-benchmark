"""Eğitim döngüsü — deterministik ve tüm modeller için birebir aynı protokol.

Notebook'a göre farklar:
- Seed her model kurulmadan hemen önce (tekrar, fold, model) üçlüsünden türetilerek
  set edilir; hücre/çağrı sırasına bağımlılık yoktur.
- Veri bellekte tensör olarak durur; DataLoader ve dosya okuma yükü yoktur.
- Early stopping yalnızca validation macro-F1 üzerinde; test setine hiçbir aşamada
  bakılmaz.
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
    counts = np.bincount(y, minlength=ncls).astype(np.float64)
    w = 1.0 / np.maximum(counts, 1.0)
    w = w / w.sum() * ncls
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict(model: nn.Module, x1d: torch.Tensor, x2d: torch.Tensor,
            batch_size: int = 128) -> np.ndarray:
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
    """tr/va: (x1d, x2d, y) tensör üçlüleri."""
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
        # Erken durdurma yalnızca min_epochs'tan sonra devreye girer. Aksi hâlde model
        # henüz önemsiz çözümdeyken (tek sınıf tahmini) doğrulama F1'i düz kalır, sayaç
        # dolar ve çökmüş ağırlıklar "en iyi" olarak dondurulur.
        if ep >= cfg.min_epochs and wait >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def best_blend_weight(logit1: np.ndarray, logit2: np.ndarray, y: np.ndarray,
                      ncls: int, grid=None) -> float:
    """Score-level füzyon ağırlığı — YALNIZCA validation üzerinde seçilir."""
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
