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
# The figure is printed at the full text width, about 0.85 of its drawn width, so the
# smallest text here (7.4 pt) prints at about 6.3 pt and the titles at about 7 pt.
FS, TITLE_FS, STEP_PT = 7.4, 8.2, 9.4
C = {"data": "#dbe9f6", "prep": "#e3f1df", "rep": "#fbefd5", "fuse": "#f3e1ef",
     "clf": "#ece8f6", "eval": "#e8eaed", "test": "#f6c9c4", "train": "#c7dcf0",
     "val": "#cfe8c9", "rej": "#9aa4ae"}


def box(ax, x, y, w, h, color, title, lines=(), fs=FS, title_fs=TITLE_FS, step_pt=STEP_PT):
    """A rounded box with a bold title and left-aligned lines; line spacing in points,
    so it does not depend on the panel's height."""
    from matplotlib.transforms import offset_copy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.012",
                                fc=color, ec=LINE, lw=0.7))
    ax.text(x + w / 2, y + h, title, ha="center", va="top", fontsize=title_fs,
            fontweight="bold", color=INK,
            transform=offset_copy(ax.transData, ax.figure, y=-4, units="points"))
    for i, s in enumerate(lines):
        ax.text(x + 0.01, y + h, s, ha="left", va="top", fontsize=fs, color=INK,
                transform=offset_copy(ax.transData, ax.figure, y=-18 - i * step_pt,
                                      units="points"))


def arrow(ax, x0, y0, x1, y1, color=INK):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                 lw=0.9, color=color, shrinkA=0, shrinkB=0))


def panel_a(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 1.0, "a", fontsize=11, fontweight="bold", va="top")
    y, h = 0.02, 0.88
    xs = {}
    x = 0.006
    for k, w in (("eeg", 0.163), ("prep", 0.232), ("rep", 0.166), ("fuse", 0.13),
                 ("clf", 0.088), ("eval", 0.138)):
        xs[k] = (x, w)
        x += w + 0.014
    box(ax, xs["eeg"][0], y, xs["eeg"][1], h, C["data"], "EEG",
        ["CHB-MIT: 23 persons", "Siena: 14 patients,", "resampled from", "512 Hz to 256 Hz",
         "18 bipolar channels,", "256 Hz", "10 s windows;", "4 non-ictal per", "ictal window,",
         "per person; same", "windows in every", "pipeline"])
    box(ax, xs["prep"][0], y, xs["prep"][1], h, C["prep"], "Preprocessing multiverse",
        ["P0   none (original)", "P1   0.5-40 Hz", "P2   0.5-70 Hz + mains notch",
         "       (CHB-MIT 60, Siena 50 Hz)", "P3   1-40 Hz", "P4   P1 + technical artefact", "       rejection (training only)",
         "P5   P1 + median/IQR scaling", "P6a P1 + Extended Infomax",
         "P6b P1 + GEDAI", "P6c P1 + AMICA", "x 5 seed sets per pipeline"])
    hh = (h - 0.04) / 2
    xr, wr = xs["rep"]
    box(ax, xr, y + hh + 0.04, wr, hh, C["rep"], "Raw waveform",
        ["1D CNN encoder,", "16, 32, 64 channels", "-> 128 features"])
    box(ax, xr, y, wr, hh, C["rep"], "Log-STFT",
        ["spectrogram, 0-64 Hz", "2D CNN encoder,", "matching depth", "-> 128 features"])
    box(ax, xs["fuse"][0], y, xs["fuse"][1], h, C["fuse"], "Fusion",
        ["early", "late", "gated", "attention", "score", "", "controls:", "single branch",
         "(raw1d, spec2d,", "also widened),", "logvar, shallow"])
    box(ax, xs["clf"][0], y, xs["clf"][1], h, C["clf"], "Classifier",
        ["ictal vs", "non-ictal", "", "parameter", "budget", "matched", "within", "1.3 %",
         "(joint", "operators)"])
    box(ax, xs["eval"][0], y, xs["eval"][1], h, C["eval"], "Evaluation",
        ["LOSO by person", "macro F1, AUPRC", "corrected paired", "t test + Holm",
         "equivalence", "(±0.05, ±0.02)", "rank stability", "event metrics",
         "(SzCORE)"])
    mid = y + h / 2
    up, down = y + hh + 0.04 + hh / 2, y + hh / 2

    def right(k):
        return xs[k][0] + xs[k][1]
    arrow(ax, right("eeg"), mid, xs["prep"][0], mid)
    arrow(ax, right("prep"), mid, xr, up)
    arrow(ax, right("prep"), mid, xr, down)
    arrow(ax, right("rep"), up, xs["fuse"][0], mid)
    arrow(ax, right("rep"), down, xs["fuse"][0], mid)
    arrow(ax, right("fuse"), mid, xs["clf"][0], mid)
    arrow(ax, right("clf"), mid, xs["eval"][0], mid)


def persons(ax, x, y, roles, w=0.026, h=0.07, gap=0.006):
    for i, r in enumerate(roles):
        ax.add_patch(Rectangle((x + i * (w + gap), y), w, h, fc=C[r], ec=LINE, lw=0.5))


def panel_b(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 0.99, "b", fontsize=11, fontweight="bold", va="top")
    titles = ["LOSO: split by person", "P4: rejection in training only",
              "P6: per recording, no labels"]
    xs = [0.01, 0.345, 0.68]
    for x, t in zip(xs, titles):
        ax.add_patch(FancyBboxPatch((x, 0.03), 0.31, 0.90,
                                    boxstyle="round,pad=0.006,rounding_size=0.012",
                                    fc="white", ec=LINE, lw=0.7))
        ax.text(x + 0.155, 0.905, t, ha="center", va="top", fontsize=TITLE_FS,
                fontweight="bold", color=INK)
    note = {"fontsize": FS, "va": "center", "linespacing": 1.15}

    # (1) LOSO
    x0 = 0.025
    roles = ["test"] + ["val"] * 2 + ["train"] * 6
    persons(ax, x0, 0.68, roles, w=0.026, h=0.06, gap=0.006)
    ax.text(x0, 0.77, "persons of one fold (schematic)", fontsize=FS, color=MUTE)
    for lab, col, yy in (("test person: all its windows", "test", 0.575),
                         ("validation persons: early\nstopping, score fusion weight", "val",
                          0.455),
                         ("training persons", "train", 0.335)):
        ax.add_patch(Rectangle((x0, yy - 0.022), 0.018, 0.044, fc=C[col], ec=LINE, lw=0.5))
        ax.text(x0 + 0.026, yy, lab, color=INK, **note)
    ax.text(x0, 0.15, "chb01 + chb21 (one person) held out\n"
            "together; no window or seizure on\nboth sides (checked)", color=MUTE, **note)

    # (2) P4
    x1 = 0.36
    n = 9
    for i in range(n):
        rej = i in (2, 6)
        ax.add_patch(Rectangle((x1 + i * 0.032, 0.68), 0.026, 0.06,
                               fc=C["rej"] if rej else C["train"], ec=LINE, lw=0.5,
                               hatch="////" if rej else None))
    ax.text(x1, 0.77, "training + validation windows", fontsize=FS, color=MUTE)
    for i in range(n):
        ax.add_patch(Rectangle((x1 + i * 0.032, 0.48), 0.026, 0.06, fc=C["test"], ec=LINE,
                               lw=0.5))
    ax.text(x1, 0.57, "test windows: unchanged", fontsize=FS, color=MUTE)
    ax.text(x1, 0.33, "rejected (hatched): flat channel\n"
            "or ≥ 0.5 s constant segment,\ndetected on the unfiltered signal",
            color=INK, **note)
    ax.text(x1, 0.13, "same test set in every pipeline\n-> paired by fold", color=MUTE,
            **note)

    # (3) P6
    x2 = 0.695
    for k, (lab, yy) in enumerate((("recording 1", 0.72), ("recording 2", 0.625),
                                   ("recording 3", 0.53))):
        ax.add_patch(Rectangle((x2, yy), 0.10, 0.065, fc=C["data"], ec=LINE, lw=0.5))
        ax.text(x2 + 0.05, yy + 0.0325, lab, fontsize=FS, ha="center", va="center")
        arrow(ax, x2 + 0.103, yy + 0.0325, x2 + 0.122, yy + 0.0325, color=MUTE)
        ax.add_patch(Rectangle((x2 + 0.125, yy), 0.16, 0.065, fc=C["prep"], ec=LINE, lw=0.5))
        ax.text(x2 + 0.205, yy + 0.0325, "own ICA / GEDAI fit", fontsize=FS, ha="center",
                va="center")
    ax.text(x2, 0.315, "continuous signal of one recording,\n"
            "before windowing; seizure labels\nnever used. "
            "Infomax, AMICA: ocular\ncomponents by a fixed rule (frontal\n"
            "channels as EOG proxies). GEDAI: its\nown criterion, departure from a\n"
            "leadfield reference covariance.", color=INK, **note)
    ax.text(x2, 0.095, "no statistics shared across\nrecordings or persons", color=MUTE,
            **note)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "hatch.linewidth": 0.6})
    ha, hb = 2.3, 2.7                                   # panel heights, inches
    fig = plt.figure(figsize=(7.2, ha + hb))
    ax_a = fig.add_axes([0.0, hb / (ha + hb), 1.0, ha / (ha + hb)])
    ax_b = fig.add_axes([0.0, 0.0, 1.0, hb / (ha + hb)])
    panel_a(ax_a)
    panel_b(ax_b)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figure1_design.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"written: {OUT / 'figure1_design.png'} and .pdf")


if __name__ == "__main__":
    main()
