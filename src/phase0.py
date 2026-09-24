"""Faz 0: mevcut sonuçların tekrarlı CV korelasyonunu hesaba katarak yeniden analizi.

Yeni deney koşmaz. Her koşunun perfold.csv dosyasındaki model çiftlerini şu testlerle
yeniden değerlendirir:

  naive_wilcoxon   : eski analiz (bağımsızlık varsayar, yanlış pozitif oranı şişik)
  corrected_t      : düzeltilmiş tekrarlı k-fold t testi (birincil)
  bayes            : korelasyonlu Bayesçi t testi, ROPE içinde/dışında olasılıklar
  sign             : işaret testi (simetri varsaymaz, duyarlılık kontrolü)
  delta_min        : düzeltilmiş varyansla denklik sınırı

Test/eğitim oranı:
  birincil  : 1/(k-1)            (yöntemin literatürdeki standart kullanımı)
  duyarlılık: n_test / n_fit     (iç doğrulama ayrıldıktan sonra fiilen eğitime giren
                                  örnek sayısıyla; daha temkinli)

Kullanım:  python -m src.phase0
Çıktı:     results_v2/phase0/*.csv
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .config import RESULTS_ROOT, Config
from .stats import (corrected_equivalence_bound, corrected_ttest, correlated_bayes,
                    holm, sign_test)

OUT = RESULTS_ROOT / "phase0"
ROPES = (0.01, 0.02)


def ratios(n_folds: int, val_ratio: float) -> tuple[float, float]:
    primary = 1.0 / (n_folds - 1)
    fit_frac = (1 - 1 / n_folds) * (1 - val_ratio)     # eğitime fiilen giren oran
    conservative = (1 / n_folds) / fit_frac
    return primary, conservative


def analyze(df: pd.DataFrame, n_folds: int, val_ratio: float = 0.2,
            metric: str = "f1_macro") -> pd.DataFrame:
    r_pri, r_con = ratios(n_folds, val_ratio)
    piv = df.pivot_table(index=["repeat", "fold"], columns="model", values=metric).sort_index()
    rows = []
    for m1, m2 in combinations(piv.columns, 2):
        a, b = piv[m1].to_numpy(float), piv[m2].to_numpy(float)
        ok = ~(np.isnan(a) | np.isnan(b))
        a, b = a[ok], b[ok]
        d = a - b
        if len(d) < 3:
            continue
        identical = bool(np.allclose(d, 0))
        try:
            p_w = 1.0 if identical else stats.wilcoxon(a, b).pvalue
        except ValueError:
            p_w = 1.0
        _, p_c = corrected_ttest(a, b, r_pri)
        _, p_c2 = corrected_ttest(a, b, r_con)
        row = {
            "model_a": m1, "model_b": m2, "n": len(d),
            "mean_a": a.mean(), "mean_b": b.mean(), "mean_diff": d.mean(),
            "p_naive_wilcoxon": p_w,
            "p_corrected": p_c,
            "p_corrected_conservative": p_c2,
            "p_sign": sign_test(a, b),
            "delta_min_naive": abs(d.mean()) + stats.t.ppf(0.95, len(d) - 1)
                               * d.std(ddof=1) / np.sqrt(len(d)),
            "delta_min_corrected": corrected_equivalence_bound(a, b, r_pri),
        }
        for rope in ROPES:
            pl, pr, pg = correlated_bayes(a, b, r_pri, rope)
            tag = f"{int(rope*100):02d}"
            row[f"bayes_p_b_better_r{tag}"] = pl
            row[f"bayes_p_equiv_r{tag}"] = pr
            row[f"bayes_p_a_better_r{tag}"] = pg
        rows.append(row)

    out = pd.DataFrame(rows)
    for col in ("p_naive_wilcoxon", "p_corrected", "p_corrected_conservative", "p_sign"):
        out[col + "_holm"] = holm(out[col].to_numpy())
    out["sig_naive"] = out["p_naive_wilcoxon_holm"] < 0.05
    out["sig_corrected"] = out["p_corrected_holm"] < 0.05
    return out


def load(glob: str) -> pd.DataFrame | None:
    """Kalıba uyan koşuyu yükler.

    Config'e alan eklendiğinde hash değiştiği için aynı görev/normalizasyon kalıbına
    birden fazla dizin uyabilir. Sessizce birini seçmek yanlış sonucu analiz etme
    riski taşıdığı için burada en yeni koşu seçilir ve durum bildirilir.
    """
    cands = [d for d in sorted(RESULTS_ROOT.glob(glob)) if (d / "perfold.csv").exists()]
    if not cands:
        return None
    if len(cands) > 1:
        cands.sort(key=lambda d: (d / "perfold.csv").stat().st_mtime, reverse=True)
        print(f"uyarı: {glob} kalıbına {len(cands)} koşu uyuyor, en yenisi seçildi: "
              f"{cands[0].name}")
    return pd.read_csv(cands[0] / "perfold.csv")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    runs = {
        "T1_N2": ("T1_3class__N2_z_zspec__*", cfg.n_folds),
        "T2_N2": ("T2_intracranial_binary__N2_z_zspec__*", cfg.n_folds),
        "T3_N2": ("T3_classic_binary__N2_z_zspec__*", cfg.n_folds),
        "sens_stft_fine": ("sensitivity/stft__fine", 5),
        "sens_stft_coarse": ("sensitivity/stft__coarse", 5),
        "sens_budget_small": ("sensitivity/budget__small", 5),
        "sens_budget_large": ("sensitivity/budget__large", 5),
    }
    summary = []
    for name, (glob, k) in runs.items():
        df = load(glob)
        if df is None:
            print(f"atlandı: {name}")
            continue
        res = analyze(df, k)
        res.to_csv(OUT / f"{name}.csv", index=False)
        summary.append({"run": name, "pairs": len(res),
                        "sig_naive": int(res.sig_naive.sum()),
                        "sig_corrected": int(res.sig_corrected.sum()),
                        "sig_corrected_conservative":
                            int((res.p_corrected_conservative_holm < 0.05).sum())})
        print(f"{name:20s} çift={len(res):3d}  anlamlı: naif={int(res.sig_naive.sum()):3d}  "
              f"düzeltilmiş={int(res.sig_corrected.sum()):3d}")
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)

    # Normalizasyon ablasyonu: iki kol aynı bölmeleri paylaşır, eşleştirilmiş karşılaştırma
    n1, n2 = load("T1_3class__N1_z_rawspec__*"), load("T1_3class__N2_z_zspec__*")
    if n1 is not None and n2 is not None:
        r_pri, _ = ratios(cfg.n_folds, cfg.val_ratio)
        p1 = n1.pivot_table(index=["repeat", "fold"], columns="model", values="f1_macro").sort_index()
        p2 = n2.pivot_table(index=["repeat", "fold"], columns="model", values="f1_macro").sort_index()
        rows = []
        for m in p1.columns:
            a, b = p1[m].to_numpy(float), p2[m].to_numpy(float)
            d = a - b
            if np.allclose(d, 0):
                rows.append({"model": m, "mean_diff": 0.0, "identical": True})
                continue
            try:
                p_w = stats.wilcoxon(a, b).pvalue
            except ValueError:
                p_w = 1.0
            rows.append({"model": m, "mean_diff": d.mean(), "identical": False,
                         "p_naive_wilcoxon": p_w,
                         "p_corrected": corrected_ttest(a, b, r_pri)[1],
                         "delta_min_corrected": corrected_equivalence_bound(a, b, r_pri)})
        ab = pd.DataFrame(rows)
        t = ~ab.identical
        ab.loc[t, "p_naive_holm"] = holm(ab.loc[t, "p_naive_wilcoxon"].to_numpy())
        ab.loc[t, "p_corrected_holm"] = holm(ab.loc[t, "p_corrected"].to_numpy())
        ab.to_csv(OUT / "ablation_N1_vs_N2.csv", index=False)
        print("\nNormalizasyon ablasyonu (N1 - N2):")
        print(ab.round(4).to_string(index=False))

    print(f"\nkaydedildi: {OUT}")


if __name__ == "__main__":
    main()
