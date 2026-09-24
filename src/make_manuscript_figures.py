"""Manuscript için figürler: Bonn ve CHB-MIT, İngilizce, düzeltilmiş istatistikle.

Bilerek kullanılmayan şey: `fig_significance` / `pairwise_table`. O yol naif
Wilcoxon+Holm hesaplıyor (bkz. src/figures.py'deki not), ve bu manuscript'in tam
tersini iddia eder. Onun yerine `fig_significance_corrected`, zaten düzeltilmiş
testten geçmiş `results_v2/phase0/*.csv` ve `results_v2/chbmit/phase0/*.csv`
dosyalarını doğrudan çizer, kendi hesaplama yapmaz.

Denklik eşiği (equiv_margin) her iki korpus için de **0.03** olarak sabitlendi.
Bonn bu eşik altında çoğu füzyon çiftini "denk" (≡) gösterecek; CHB-MIT'te aynı
eşikle hiçbir çift "≡" görünmeyecek, çünkü delta_min zaten 0.055'in üzerinde
(bkz. docs/13). Bu bilinçli bir seçim: aynı eşik iki korpusta farklı sonuç
üreterek "Bonn'da denklik kanıtlanmış, CHB-MIT'te kanıtlanamamış" ayrımını
görsel olarak da doğru yansıtıyor.

Kullanım:  python -m src.make_manuscript_figures
Çıktı:     paper/figures_manuscript/*.png, *.pdf
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

MODEL_ORDER = ["late", "gated", "attention", "score", "early",
              "spec2d_wide", "raw1d_wide", "spec2d", "raw1d", "shallow", "logvar"]


def main() -> None:
    figures.set_lang("en")
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
    chb_perfold = pd.read_csv(RESULTS_ROOT / "chbmit" / "loso_main" / "perfold.csv")
    figures.fig_forest(chb_perfold, OUT, metric="f1_macro",
                       xlabel="Macro-F1", name="chbmit_forest", focus_span=0.25)

    chb_stats = pd.read_csv(RESULTS_ROOT / "chbmit" / "phase0" / "chbmit_f1_macro.csv")
    figures.fig_significance_corrected(
        chb_stats, OUT, equiv_margin=EQUIV_MARGIN, models=MODEL_ORDER,
        metric_label="macro F1 (CHB-MIT LOSO)", name="chbmit_significance_corrected")

    print(f"\ntamamı kaydedildi: {OUT}")


if __name__ == "__main__":
    main()
