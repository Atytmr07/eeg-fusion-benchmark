"""Manuscript figures.

Design choices:
- The main comparison is a forest plot (mean and 95% CI per model): it shows size and
  uncertainty together, which a bar chart would hide.
- Single-series figures have no legend; values are labelled directly.
- Model family is encoded by position and text, never by colour alone.
- The significance matrix has three states (different / equivalent / inconclusive),
  shown both by colour and by symbol.
- Colours come from a colour-blind-checked palette; grid and axes stay in the background.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .stats import summarize

# Colour-blind-checked categorical palette
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b6b0"

FAMILY = {
    "logvar": "Shallow reference", "shallow": "Shallow reference",
    "raw1d": "Unimodal", "spec2d": "Unimodal",
    "raw1d_wide": "Unimodal (matched)", "spec2d_wide": "Unimodal (matched)",
    "early": "Fusion", "late": "Fusion", "gated": "Fusion",
    "attention": "Fusion", "score": "Fusion",
}
DISPLAY = {
    "logvar": "LogVar (1 feat.)", "shallow": "Shallow-LR (7 feat.)",
    "raw1d": "1D-CNN", "spec2d": "2D-CNN",
    "raw1d_wide": "1D-CNN (wide)", "spec2d_wide": "2D-CNN (wide)",
    "early": "Early/Interm.", "late": "Late", "gated": "Gated",
    "attention": "Attention", "score": "Score",
}
FAMILY_ORDER = ["Fusion", "Unimodal (matched)", "Unimodal", "Shallow reference"]


def _style():
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 300,
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.labelcolor": INK2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _save(fig, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved:", outdir / f"{name}.png")


def fig_forest(df: pd.DataFrame, outdir: Path, metric: str = "f1_macro",
               xlabel: str = "Macro-F1", name: str = "F1_forest",
               focus_span: float = 0.12):
    """Forest plot: per-model mean and 95% CI, grouped by model family.

    Family names are drawn as separate header rows (avoids label overlap). Models
    below the focus range (e.g. the one-feature reference) are pinned to the left edge
    with an arrow; otherwise a single outlier would squash the whole axis.
    """
    _style()
    s = summarize(df, metric).set_index("model")

    entries: list[tuple] = []          # (y, kind, payload)
    y = 0.0
    for fam in FAMILY_ORDER:
        members = [m for m in s.index if FAMILY.get(m) == fam]
        if not members:
            continue
        entries.append((y, "header", fam))
        y += 1
        for m in sorted(members, key=lambda m: -s.loc[m, "mean"]):
            entries.append((y, "model", m))
            y += 1
        y += 0.4

    mu_all = s["mean"].to_numpy()
    hi = float(mu_all.max())
    lo_focus = hi - focus_span
    x_lo = float((s.loc[s["mean"] >= lo_focus, "mean"] -
                  s.loc[s["mean"] >= lo_focus, "ci95"]).min()) - 0.012
    x_hi = min(1.005, float((s["mean"] + s["ci95"]).max()) + 0.022)

    fig, ax = plt.subplots(figsize=(7.4, 0.36 * y + 1.4))

    # shaded band: CI of the best model
    best_m = s["mean"].idxmax()
    ax.axvspan(s.loc[best_m, "mean"] - s.loc[best_m, "ci95"],
               s.loc[best_m, "mean"] + s.loc[best_m, "ci95"],
               color=BLUE, alpha=0.08, zorder=0)
    ax.axvline(s.loc[best_m, "mean"], color=BLUE, lw=1, ls="--", alpha=0.55, zorder=1)

    yticks, ylabels = [], []
    for yy, kind, payload in entries:
        yticks.append(yy)
        if kind == "header":
            ylabels.append(payload.upper())
            continue
        ylabels.append("   " + DISPLAY.get(payload, payload))
        m, c = s.loc[payload, "mean"], s.loc[payload, "ci95"]
        if m < lo_focus:                              # outside the focus range: pin to edge
            ax.plot([x_lo + 0.004], [yy], marker="<", ms=8, color=ORANGE, zorder=3)
            ax.text(x_lo + 0.012, yy, f"{m:.3f}  ↩ off-axis",
                    va="center", ha="left", fontsize=8.5, color=ORANGE)
        else:
            ax.errorbar([m], [yy], xerr=[c], fmt="o", ms=6, lw=2, capsize=3,
                        color=BLUE, ecolor=INK2, zorder=3)
            ax.text(m + c + 0.003, yy, f"{m:.3f}", va="center", ha="left",
                    fontsize=9, color=INK2)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9.5)
    for tick, (_, kind, _) in zip(ax.get_yticklabels(), entries):
        if kind == "header":
            tick.set_fontweight("bold")
            tick.set_fontsize(8.5)
            tick.set_color(INK2)

    ax.set_xlabel(xlabel)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y - 0.4, -0.8)
    ax.grid(axis="x", ls="--", alpha=0.3, color=MUTED)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    n = df.groupby("model").size().min()
    ax.set_title(f"{xlabel}: mean and 95% CI ({n} paired measurements)\n"
                 "shaded band: CI of the best model",
                 loc="left", fontsize=10)
    _save(fig, outdir, name)


def fig_significance_corrected(stats_df: pd.DataFrame, outdir: Path,
                               equiv_margin: float, models: list[str] | None = None,
                               metric_label: str = "macro F1",
                               name: str = "significance_matrix_corrected"):
    """Three-state decision matrix drawn from already-corrected statistics.

    It computes nothing itself: it draws a table produced by src/phase0.py or
    src/chbmit_stats.py (columns model_a, model_b, sig_corrected, delta_min_corrected).

    different    (≠): significant under the corrected test with Holm correction
    equivalent   (≡): not significant and delta_min_corrected <= equiv_margin
    inconclusive (?): neither a difference nor equivalence can be shown
    """
    _style()
    if stats_df.empty:
        return
    all_models = sorted(set(stats_df.model_a) | set(stats_df.model_b))
    order = [m for m in (models or all_models) if m in all_models]
    n = len(order)
    idx = {m: i for i, m in enumerate(order)}

    state = np.full((n, n), np.nan)      # 0 inconclusive, 1 equivalent, 2 different
    for _, r in stats_df.iterrows():
        if r.model_a not in idx or r.model_b not in idx:
            continue
        i, j = idx[r.model_a], idx[r.model_b]
        if bool(r.sig_corrected):
            v = 2.0
        elif r.delta_min_corrected <= equiv_margin:
            v = 1.0
        else:
            v = 0.0
        state[i, j] = state[j, i] = v

    cmap = mpl.colors.ListedColormap(["#f0efe9", AQUA, ORANGE])
    fig, ax = plt.subplots(figsize=(0.62 * n + 2.4, 0.62 * n + 2.0))
    ax.imshow(np.ma.masked_invalid(state), cmap=cmap, vmin=-0.5, vmax=2.5)

    labels = {0: "?", 1: "≡", 2: "≠"}
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "-", ha="center", va="center", color=MUTED)
            elif not np.isnan(state[i, j]):
                ax.text(j, i, labels[int(state[i, j])], ha="center", va="center",
                        fontsize=11, color=INK)

    ax.set_xticks(range(n))
    ax.set_xticklabels([DISPLAY.get(m, m) for m in order], rotation=40, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([DISPLAY.get(m, m) for m in order])
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title(f"Pairwise comparison: {metric_label} (corrected test)\n"
                 f"≠ significant difference (corrected+Holm, p<0.05)   "
                 f"≡ equivalent (δmin ≤ {equiv_margin:g})   "
                 f"? inconclusive",
                 loc="left", fontsize=10)
    _save(fig, outdir, name)
