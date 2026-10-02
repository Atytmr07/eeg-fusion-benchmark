"""Manuscript figures for Bonn and CHB-MIT, drawn from the corrected statistics.

The significance matrices only draw tables that already went through the corrected
test (results_v2/phase0/*.csv and results_v2/chbmit/phase0_<run>/*.csv); nothing is
recomputed here. CHB-MIT figures use the 23-fold person-grouped run (CHB_RUN).

The equivalence margin is fixed at 0.03 macro F1 for both corpora. Under the same
margin Bonn shows most fusion pairs as equivalent (≡) while CHB-MIT shows no fusion
pair as equivalent, since their delta_min values are all above 0.045. That is the
intended picture: equivalence is established on Bonn and not on CHB-MIT.

Usage:   python -m src.make_manuscript_figures
Output:  paper/figures_manuscript/*.png, *.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from . import figures
from .config import RESULTS_ROOT

OUT = Path("paper/figures_manuscript")
EQUIV_MARGIN = 0.03
CHB_RUN = "loso_grouped"

MODEL_ORDER = ["late", "gated", "attention", "score", "early",
               "spec2d_wide", "raw1d_wide", "spec2d", "raw1d", "shallow", "logvar"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- Bonn T1 ---
    bonn_perfold = pd.read_csv(RESULTS_ROOT / "T1_3class__N2_z_zspec__a45d0ab077"
                               / "perfold.csv")
    figures.fig_forest(bonn_perfold, OUT, metric="f1_macro",
                       xlabel="Macro-F1", name="bonn_T1_forest")

    bonn_stats = pd.read_csv(RESULTS_ROOT / "phase0" / "T1_N2.csv")
    figures.fig_significance_corrected(
        bonn_stats, OUT, equiv_margin=EQUIV_MARGIN, models=MODEL_ORDER,
        metric_label="macro F1 (Bonn T1)", name="bonn_T1_significance_corrected")

    # --- CHB-MIT LOSO ---
    chb_perfold = pd.read_csv(RESULTS_ROOT / "chbmit" / CHB_RUN / "perfold.csv")
    figures.fig_forest(chb_perfold, OUT, metric="f1_macro",
                       xlabel="Macro-F1", name="chbmit_forest", focus_span=0.25)

    chb_stats = pd.read_csv(RESULTS_ROOT / "chbmit" / f"phase0_{CHB_RUN}"
                           / "chbmit_f1_macro.csv")
    figures.fig_significance_corrected(
        chb_stats, OUT, equiv_margin=EQUIV_MARGIN, models=MODEL_ORDER,
        metric_label="macro F1 (CHB-MIT LOSO)", name="chbmit_significance_corrected")

    print(f"\nall saved: {OUT}")


if __name__ == "__main__":
    main()
