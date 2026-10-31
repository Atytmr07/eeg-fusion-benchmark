"""Backbones and fusion operators, sized to a common parameter budget.

Design principle: fusion operators should differ only in *how* they combine the two
branches, not in capacity. Therefore:

- Every two-branch model uses the same CNN1D + CNN2D backbones.
- Every fusion head is sized to the same budget (FUSION_BUDGET); its hidden width is
  solved per operator so that the total parameter count matches.
- Dropout, classifier head and embedding size are identical across models.
- raw1d_wide / spec2d_wide are single-branch controls widened to the same total
  parameter count as the fusion models. They separate "fusion helped" from "more
  parameters helped".

Run `python -m src.models` to print the parameter count of every model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

FUSION_BUDGET = 40_000  # target parameter count of each fusion head


def _solve_hidden(kind: str, dim: int, ncls: int, budget: int) -> int:
    """Hidden width that brings a fusion head closest to `budget` parameters.

    Each formula is the head's parameter count written as a linear function of the
    hidden width h and solved for h.
    """
    if kind == "late":      # 2d -> h -> ncls
        h = (budget - ncls) / (2 * dim + 1 + ncls)
    elif kind == "gated":   # 2d -> h -> d (sigmoid gate), then d -> ncls
        h = (budget - dim - (dim * ncls + ncls)) / (2 * dim + 1 + dim)
    elif kind == "attention":  # 2d -> h -> 2 (softmax weights), then d -> ncls
        h = (budget - 2 - (dim * ncls + ncls)) / (2 * dim + 1 + 2)
    else:
        raise ValueError(kind)
    return max(8, int(round(h)))


class CNN1D(nn.Module):
    """Raw-signal backbone: 3 x (Conv1d k=7, BatchNorm, ReLU, MaxPool 2), then a
    linear projection to a `dim`-dimensional embedding.

    in_ch is the number of input channels (1 for Bonn, 18 for CHB-MIT). Only the first
    convolution depends on it, so Bonn parameter counts are unchanged at in_ch=1.
    """
    def __init__(self, dim: int = 128, width: int = 16, in_ch: int = 1):
        super().__init__()
        def blk(ci, co, k=7):
            return nn.Sequential(nn.Conv1d(ci, co, k, padding=k // 2),
                                 nn.BatchNorm1d(co), nn.ReLU(), nn.MaxPool1d(2))
        c1, c2, c3 = width, width * 2, width * 4
        self.net = nn.Sequential(blk(in_ch, c1), blk(c1, c2), blk(c2, c3))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(c3, dim))

    def forward(self, x):
        return self.head(self.net(x))


class CNN2D(nn.Module):
    """Spectrogram backbone: 3 x (Conv2d 3x3, BatchNorm, ReLU, MaxPool 2x2), then a
    linear projection to a `dim`-dimensional embedding. in_ch as in CNN1D (CHB-MIT
    stacks one spectrogram per EEG channel)."""
    def __init__(self, dim: int = 128, width: int = 16, in_ch: int = 1):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1),
                                 nn.BatchNorm2d(co), nn.ReLU(), nn.MaxPool2d(2))
        c1, c2, c3 = width, width * 2, width * 4
        self.net = nn.Sequential(blk(in_ch, c1), blk(c1, c2), blk(c2, c3))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c3, dim))

    def forward(self, x):
        return self.head(self.net(x))


# --- Single-branch models -------------------------------------------------------

class Unimodal1D(nn.Module):
    def __init__(self, dim=128, ncls=3, p_drop=0.3, width=16, in_ch=1):
        super().__init__()
        self.f = CNN1D(dim, width, in_ch)
        self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(dim, ncls))

    def forward(self, x1d, x2d=None):
        return self.head(self.f(x1d))


class Unimodal2D(nn.Module):
    def __init__(self, dim=128, ncls=3, p_drop=0.3, width=16, in_ch=1):
        super().__init__()
        self.f = CNN2D(dim, width, in_ch)
        self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(dim, ncls))

    def forward(self, x1d=None, x2d=None):
        return self.head(self.f(x2d))


# --- Fusion operators -------------------------------------------------------------
# z1 = raw-branch embedding, z2 = spectrogram-branch embedding.

class LateFusion(nn.Module):
    """Feature-level fusion: concatenate [z1, z2] -> MLP -> classifier."""
    def __init__(self, dim=128, ncls=3, p_drop=0.3, budget=FUSION_BUDGET, in_ch=1):
        super().__init__()
        h = _solve_hidden("late", dim, ncls, budget)
        self.f1, self.f2 = CNN1D(dim, in_ch=in_ch), CNN2D(dim, in_ch=in_ch)
        self.fuse = nn.Sequential(nn.Dropout(p_drop), nn.Linear(2 * dim, h), nn.ReLU())
        self.cls = nn.Linear(h, ncls)

    def forward(self, x1d, x2d):
        z = torch.cat([self.f1(x1d), self.f2(x2d)], dim=1)
        return self.cls(self.fuse(z))


class GatedFusion(nn.Module):
    """Learned per-dimension gate: g = sigmoid(MLP([z1, z2])), z = g*z1 + (1-g)*z2."""
    def __init__(self, dim=128, ncls=3, p_drop=0.3, budget=FUSION_BUDGET, in_ch=1):
        super().__init__()
        h = _solve_hidden("gated", dim, ncls, budget)
        self.f1, self.f2 = CNN1D(dim, in_ch=in_ch), CNN2D(dim, in_ch=in_ch)
        self.gate = nn.Sequential(nn.Dropout(p_drop), nn.Linear(2 * dim, h), nn.ReLU(),
                                  nn.Linear(h, dim), nn.Sigmoid())
        self.cls = nn.Linear(dim, ncls)

    def forward(self, x1d, x2d):
        z1, z2 = self.f1(x1d), self.f2(x2d)
        g = self.gate(torch.cat([z1, z2], dim=1))
        return self.cls(g * z1 + (1 - g) * z2)

    def fusion_weights(self, x1d, x2d):
        z1, z2 = self.f1(x1d), self.f2(x2d)
        return self.gate(torch.cat([z1, z2], dim=1))


class AttentionFusion(nn.Module):
    """Modality-level softmax attention: (a1, a2) = softmax(MLP([z1, z2])),
    z = a1*z1 + a2*z2 (one weight per branch, not per dimension)."""
    def __init__(self, dim=128, ncls=3, p_drop=0.3, budget=FUSION_BUDGET, in_ch=1):
        super().__init__()
        h = _solve_hidden("attention", dim, ncls, budget)
        self.f1, self.f2 = CNN1D(dim, in_ch=in_ch), CNN2D(dim, in_ch=in_ch)
        self.att = nn.Sequential(nn.Dropout(p_drop), nn.Linear(2 * dim, h), nn.ReLU(),
                                 nn.Linear(h, 2))
        self.cls = nn.Linear(dim, ncls)

    def forward(self, x1d, x2d):
        z1, z2 = self.f1(x1d), self.f2(x2d)
        a = torch.softmax(self.att(torch.cat([z1, z2], dim=1)), dim=1)
        return self.cls(a[:, 0:1] * z1 + a[:, 1:2] * z2)

    def fusion_weights(self, x1d, x2d):
        z1, z2 = self.f1(x1d), self.f2(x2d)
        return torch.softmax(self.att(torch.cat([z1, z2], dim=1)), dim=1)


class EarlyFusion(nn.Module):
    """Fusion close to the input.

    A 1D signal and a 2D spectrogram cannot be concatenated at the raw input level
    (different shapes), so what the literature calls "early fusion" is strictly
    intermediate fusion: the signal passes through a shallow strided encoder, is
    aligned to the spectrogram's time axis, broadcast along frequency, and appended
    to the spectrogram as extra channels. One 2D CNN then processes both. The
    manuscript names it accordingly. width=26 is calibrated so the total parameter
    count matches the other fusion models.
    """
    def __init__(self, dim=128, ncls=3, p_drop=0.3, wave_ch=8, width=26,
                 budget=FUSION_BUDGET, in_ch=1):
        super().__init__()
        # Downsample the signal towards the spectrogram's time resolution first.
        # Broadcasting all 4097 samples along frequency would create a huge tensor
        # (B, 9, 59, 2048) and implicitly upsample the spectrogram.
        self.wave_enc = nn.Sequential(
            nn.Conv1d(in_ch, 8, 7, stride=4, padding=3), nn.BatchNorm1d(8), nn.ReLU(),
            nn.Conv1d(8, 8, 5, stride=4, padding=2), nn.BatchNorm1d(8), nn.ReLU(),
            nn.Conv1d(8, wave_ch, 3, stride=2, padding=1), nn.BatchNorm1d(wave_ch), nn.ReLU(),
        )
        def blk(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1),
                                 nn.BatchNorm2d(co), nn.ReLU(), nn.MaxPool2d(2))
        c1, c2, c3 = width, width * 2, width * 4
        self.backbone = nn.Sequential(blk(in_ch + wave_ch, c1), blk(c1, c2), blk(c2, c3))
        self.proj = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c3, dim))
        h = _solve_hidden("late", dim // 2, ncls, budget // 2)
        self.cls = nn.Sequential(nn.Dropout(p_drop), nn.Linear(dim, h), nn.ReLU(),
                                 nn.Linear(h, ncls))

    def forward(self, x1d, x2d):
        B, _, Fdim, T = x2d.shape
        wf = self.wave_enc(x1d)                              # [B, wave_ch, T'']
        wf = F.adaptive_avg_pool1d(wf, T)                    # align to spectrogram time axis
        wf2d = wf.unsqueeze(2).expand(B, wf.shape[1], Fdim, T)
        return self.cls(self.proj(self.backbone(torch.cat([x2d, wf2d], dim=1))))


# --- Factory ------------------------------------------------------------------------

# Width multipliers calibrated so the widened single-branch controls have the same
# total parameter count as the two-branch fusion models (see report_params).
WIDE_WIDTH_1D = 34
WIDE_WIDTH_2D = 30


def build(name: str, ncls: int, dim: int = 128, p_drop: float = 0.3,
          fusion_budget: int | None = None, in_ch: int = 1) -> nn.Module:
    """Build a model by name. `fusion_budget` overrides the fusion head budget; it is
    used by the sensitivity analysis to ask whether conclusions depend on the budget.
    Score fusion is not a network: it is built from raw1d and spec2d by the drivers.
    """
    b = FUSION_BUDGET if fusion_budget is None else fusion_budget
    if name == "raw1d":
        return Unimodal1D(dim, ncls, p_drop, in_ch=in_ch)
    if name == "spec2d":
        return Unimodal2D(dim, ncls, p_drop, in_ch=in_ch)
    if name == "raw1d_wide":
        return Unimodal1D(dim, ncls, p_drop, width=WIDE_WIDTH_1D, in_ch=in_ch)
    if name == "spec2d_wide":
        return Unimodal2D(dim, ncls, p_drop, width=WIDE_WIDTH_2D, in_ch=in_ch)
    if name == "late":
        return LateFusion(dim, ncls, p_drop, budget=b, in_ch=in_ch)
    if name == "gated":
        return GatedFusion(dim, ncls, p_drop, budget=b, in_ch=in_ch)
    if name == "attention":
        return AttentionFusion(dim, ncls, p_drop, budget=b, in_ch=in_ch)
    if name == "early":
        return EarlyFusion(dim, ncls, p_drop, budget=b, in_ch=in_ch)
    raise ValueError(f"unknown model: {name}")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def report_params(ncls: int = 3, dim: int = 128, in_ch: int = 1) -> dict[str, int]:
    names = ["raw1d", "spec2d", "raw1d_wide", "spec2d_wide",
             "late", "gated", "attention", "early"]
    out = {n: count_params(build(n, ncls, dim, in_ch=in_ch)) for n in names}
    # score fusion is not a single network: it is the sum of the two unimodal models
    out["score"] = out["raw1d"] + out["spec2d"]
    return out


if __name__ == "__main__":
    for label, ncls, in_ch in (("Bonn T1 (3 classes, 1 channel)", 3, 1),
                               ("CHB-MIT (2 classes, 18 channels)", 2, 18)):
        print(label)
        for k, v in report_params(ncls, in_ch=in_ch).items():
            print(f"  {k:14s} {v:>9,d}")
