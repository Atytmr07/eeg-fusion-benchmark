"""Figure 1 of the manuscript: the study design and how leakage is prevented.

  (a) main flow: EEG -> preprocessing multiverse (P0-P6c) -> raw waveform and log-STFT
      -> fusion operators -> classifier -> evaluation
  (b) leakage control: person-wise LOSO with a person-wise inner validation split;
      P4's rejection applied to training and validation windows only; ICA / GEDAI
      fitted per recording without labels

Every label states what src/ implements (src/preprocess.py, src/models.py,
src/chbmit_run.py, src/chbmit_corpus.py); a schematic, no data.

Usage:  python -m src.make_figure1
Output: paper/figures_manuscript/figure1_design.png (300 dpi) and .pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path("paper/figures_manuscript")

INK = "#1f2328"
MUTE = "#57606a"
LINE = "#8c959f"
C = {"data": "#dbe9f6", "prep": "#e3f1df", "rep": "#fbefd5", "fuse": "#f3e1ef",
     "clf": "#ece8f6", "eval": "#e8eaed", "test": "#f6c9c4", "train": "#c7dcf0",
     "val": "#cfe8c9", "rej": "#9aa4ae"}


def box(ax, x, y, w, h, color, title, lines=(), fs=6.6, title_fs=7.4, step_pt=8.6):
    """A rounded box with a bold title and left-aligned lines; line spacing in points,
    so it does not depend on the panel's height."""
    from matplotlib.transforms import offset_copy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=color, ec=LINE, lw=0.7))
    ax.text(x + w / 2, y + h, title, ha="center", va="top", fontsize=title_fs,
            fontweight="bold", color=INK,
            transform=offset_copy(ax.transData, ax.figure, y=-4, units="points"))
    for i, s in enumerate(lines):
        ax.text(x + 0.01, y + h, s, ha="left", va="top", fontsize=fs, color=INK,
                transform=offset_copy(ax.transData, ax.figure, y=-16 - i * step_pt,
                                      units="points"))


def arrow(ax, x0, y0, x1, y1, color=INK):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                 lw=0.9, color=color, shrinkA=0, shrinkB=0))


def panel_a(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 1.0, "a", fontsize=10, fontweight="bold", va="top")
    y, h = 0.02, 0.88
    box(ax, 0.01, y, 0.14, h, C["data"], "EEG",
        ["CHB-MIT: 23 persons", "Siena: 14 patients", "18 bipolar channels,", "256 Hz", "",
         "10 s windows;", "4 non-ictal per", "ictal window,", "per person;",
         "same windows in", "every pipeline"])
    box(ax, 0.17, y, 0.225, h, C["prep"], "Preprocessing multiverse",
        ["P0   none (original)", "P1   0.5-40 Hz", "P2   0.5-70 Hz + mains notch",
         "P3   1-40 Hz", "P4   P1 + technical artefact", "        rejection (training only)",
         "P5   P1 + median/IQR scaling", "P6a P1 + Extended Infomax",
         "P6b P1 + GEDAI", "P6c P1 + AMICA", "x 3 seed sets per pipeline"])
    hh = (h - 0.04) / 2
    box(ax, 0.415, y + hh + 0.04, 0.15, hh, C["rep"], "Raw waveform",
        ["1D CNN encoder", "(16, 32, 64 channels)", "-> 128 features"])
    box(ax, 0.415, y, 0.15, hh, C["rep"], "Log-STFT",
        ["spectrogram, 0-64 Hz", "2D CNN encoder,", "matching depth", "-> 128 features"])
    box(ax, 0.585, y, 0.125, h, C["fuse"], "Fusion operator",
        ["early", "late", "gated", "attention", "score", "", "controls: raw1d,", "spec2d, widened",
         "single branch,", "logvar, shallow"])
    box(ax, 0.73, y, 0.09, h, C["clf"], "Classifier",
        ["ictal vs", "non-ictal", "", "parameter", "budget", "matched", "within 1.3 %"])
    box(ax, 0.84, y, 0.15, h, C["eval"], "Evaluation",
        ["LOSO by person", "macro F1, AUPRC", "corrected paired", "t test + Holm",
         "equivalence", "(+-0.05, +-0.02)", "rank stability", "event metrics", "(SzCORE)"])
    mid = y + h / 2
    up, down = y + hh + 0.04 + hh / 2, y + hh / 2
    arrow(ax, 0.15, mid, 0.17, mid)
    arrow(ax, 0.395, mid, 0.415, up)
    arrow(ax, 0.395, mid, 0.415, down)
    arrow(ax, 0.565, up, 0.585, mid)
    arrow(ax, 0.565, down, 0.585, mid)
    arrow(ax, 0.71, mid, 0.73, mid)
    arrow(ax, 0.82, mid, 0.84, mid)


def persons(ax, x, y, roles, w=0.026, h=0.07, gap=0.006):
    for i, r in enumerate(roles):
        ax.add_patch(Rectangle((x + i * (w + gap), y), w, h, fc=C[r], ec=LINE, lw=0.5))


def panel_b(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 0.99, "b", fontsize=10, fontweight="bold", va="top")
    titles = ["LOSO: split by person", "P4: rejection in training only",
              "P6: per recording, no labels"]
    xs = [0.02, 0.35, 0.68]
    for x, t in zip(xs, titles):
        ax.add_patch(FancyBboxPatch((x, 0.05), 0.30, 0.86,
                                    boxstyle="round,pad=0.006,rounding_size=0.012",
                                    fc="white", ec=LINE, lw=0.7))
        ax.text(x + 0.15, 0.87, t, ha="center", va="top", fontsize=7.4, fontweight="bold",
                color=INK)

    # (1) LOSO
    x0 = 0.035
    roles = ["test"] + ["val"] * 2 + ["train"] * 6
    persons(ax, x0, 0.58, roles)
    ax.text(x0, 0.70, "persons of one fold (schematic)", fontsize=6.2, color=MUTE)
    for lab, col, yy in (("test person: all its windows", "test", 0.46),
                         ("validation persons: early stopping,\nscore fusion weight", "val", 0.33),
                         ("training persons", "train", 0.21)):
        ax.add_patch(Rectangle((x0, yy), 0.018, 0.05, fc=C[col], ec=LINE, lw=0.5))
        ax.text(x0 + 0.026, yy + 0.025, lab, fontsize=6.3, va="center", color=INK,
                linespacing=1.1)
    ax.text(x0, 0.11, "chb01 + chb21 (one person) held out together;\n"
            "no window or seizure on both sides (checked)", fontsize=6.0, color=MUTE,
            va="center")

    # (2) P4
    x1 = 0.365
    n = 9
    for i in range(n):
        rej = i in (2, 6)
        ax.add_patch(Rectangle((x1 + i * 0.03, 0.58), 0.024, 0.07,
                               fc=C["rej"] if rej else C["train"], ec=LINE, lw=0.5,
                               hatch="////" if rej else None))
    ax.text(x1, 0.70, "training + validation windows", fontsize=6.2, color=MUTE)
    for i in range(n):
        ax.add_patch(Rectangle((x1 + i * 0.03, 0.36), 0.024, 0.07, fc=C["test"], ec=LINE,
                               lw=0.5))
    ax.text(x1, 0.48, "test windows: unchanged", fontsize=6.2, color=MUTE)
    ax.text(x1, 0.22, "rejected (hatched): flat channel or >= 0.5 s\n"
            "constant segment, detected on the unfiltered\nsignal",
            fontsize=6.0, color=INK, va="center")
    ax.text(x1, 0.10, "same test set in every pipeline -> paired by fold", fontsize=6.0,
            color=MUTE, va="center")

    # (3) P6
    x2 = 0.695
    for k, (lab, yy) in enumerate((("recording 1", 0.62), ("recording 2", 0.50),
                                   ("recording 3", 0.38))):
        ax.add_patch(Rectangle((x2, yy), 0.085, 0.07, fc=C["data"], ec=LINE, lw=0.5))
        ax.text(x2 + 0.0425, yy + 0.035, lab, fontsize=5.8, ha="center", va="center")
        arrow(ax, x2 + 0.088, yy + 0.035, x2 + 0.118, yy + 0.035, color=MUTE)
        ax.add_patch(Rectangle((x2 + 0.12, yy), 0.15, 0.07, fc=C["prep"], ec=LINE, lw=0.5))
        ax.text(x2 + 0.195, yy + 0.035, "own ICA / GEDAI fit", fontsize=5.8, ha="center",
                va="center")
    ax.text(x2, 0.23, "continuous signal of one recording, before\n"
            "windowing; seizure labels never used; ocular\n"
            "components by a fixed rule (frontal channels)", fontsize=6.0, color=INK,
            va="center")
    ax.text(x2, 0.10, "no statistics shared across recordings or persons", fontsize=6.0,
            color=MUTE, va="center")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "hatch.linewidth": 0.6})
    fig = plt.figure(figsize=(7.2, 4.0))
    ax_a = fig.add_axes([0.0, 0.5, 1.0, 0.5])
    ax_b = fig.add_axes([0.0, 0.0, 1.0, 0.47])
    panel_a(ax_a)
    panel_b(ax_b)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figure1_design.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"written: {OUT / 'figure1_design.png'} and .pdf")


if __name__ == "__main__":
    main()
