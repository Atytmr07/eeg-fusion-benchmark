"""CHB-MIT LOSO sonuçlarını Faz 0'daki düzeltilmiş çerçeveyle analiz eder.

Bonn'daki phase0.py 5 kat x 5 tekrarlı CV için yazılmıştı (repeat, fold) çiftleri
üzerinden pivot alıyordu. CHB-MIT LOSO'da yapı farklı: 24 katman var, her biri farklı
bir *denek* dışarıda bırakılarak üretiliyor, tekrar yok. Bu yüzden ayrı bir modül.

Test/eğitim oranı: LOSO'da her katmanda 1 denek test, kalan 23 denek eğitimde (iç
doğrulama ayrıldıktan sonra biraz daha az). docs/11_CHBMIT_PLANI.md §10.4'te
belirtildiği gibi oran 1/(n_katman-1) = 1/23 alınır. Bu, k-katlı CV için türetilen
Nadeau-Bengio formülünün doğrudan uzantısıdır: k yerine denek sayısı girer, çünkü
düzeltmenin kaynağı katmanların eğitim verisini paylaşması, düzeltme spesifik olarak
"k-fold" biçimine değil bu paylaşıma bağlıdır.

ROPE (pratik denklik bölgesi) sabitleri Faz 0 ile aynı tutuldu ki iki korpus
karşılaştırılabilir kalsın.

Kullanım:  python -m src.chbmit_stats
Çıktı:     results_v2/chbmit/phase0/*.csv
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

from .config import RESULTS_ROOT
from .stats import (corrected_equivalence_bound, corrected_ttest, correlated_bayes,
                    holm, sign_test)

IN_CSV = RESULTS_ROOT / "chbmit" / "loso_main" / "perfold.csv"
OUT = RESULTS_ROOT / "chbmit" / "phase0"
FUSION = ("early", "late", "gated", "attention", "score")
ROPE_BY_METRIC = {
    "f1_macro": (0.01, 0.02),
    "log_loss": (0.02, 0.05),
    "brier": (0.01, 0.02),
}
LOWER_IS_BETTER = {"log_loss", "brier"}
# chbmit_run.inner_split'in val_frac'ı ile aynı olmalı.
VAL_FRAC = 0.2


def ratios(n_folds: int, val_frac: float = VAL_FRAC) -> tuple[float, float]:
    """Birincil 1/(k-1); temkinli oran iç doğrulamaya ayrılan denekleri eğitimden düşer."""
    n_train = n_folds - 1
    n_val = max(1, int(round(val_frac * n_train)))
    return 1.0 / n_train, 1.0 / (n_train - n_val)


def analyze_metric(df: pd.DataFrame, metric: str, ratio: float,
                   ratio_con: float) -> pd.DataFrame:
    piv = df.pivot_table(index="fold", columns="model", values=metric).sort_index()
    ropes = ROPE_BY_METRIC[metric]
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
        _, p_c = corrected_ttest(a, b, ratio)
        _, p_c2 = corrected_ttest(a, b, ratio_con)
        row = {
            "metric": metric, "model_a": m1, "model_b": m2, "n": len(d),
            "mean_a": a.mean(), "mean_b": b.mean(), "mean_diff": d.mean(),
            "p_naive_wilcoxon": p_w,
            "p_corrected": p_c,
            "p_corrected_conservative": p_c2,
            "p_sign": sign_test(a, b),
            "delta_min_naive": abs(d.mean()) + stats.t.ppf(0.95, len(d) - 1)
                               * d.std(ddof=1) / np.sqrt(len(d)),
            "delta_min_corrected": corrected_equivalence_bound(a, b, ratio),
        }
        for rope in ropes:
            pl, pr, pg = correlated_bayes(a, b, ratio, rope)
            tag = f"{int(rope * 100):02d}"
            row[f"bayes_p_b_better_r{tag}"] = pl
            row[f"bayes_p_equiv_r{tag}"] = pr
            row[f"bayes_p_a_better_r{tag}"] = pg
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_naive_wilcoxon_holm"] = holm(out["p_naive_wilcoxon"].to_numpy())
    out["p_corrected_holm"] = holm(out["p_corrected"].to_numpy())
    out["p_corrected_conservative_holm"] = holm(out["p_corrected_conservative"].to_numpy())
    out["p_sign_holm"] = holm(out["p_sign"].to_numpy())
    out["sig_naive"] = out["p_naive_wilcoxon_holm"] < 0.05
    out["sig_corrected"] = out["p_corrected_holm"] < 0.05
    return out


def per_model_summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = df.groupby("model")[metric]
    s = pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1), "n": g.size()})
    return s.sort_values("mean", ascending=metric in LOWER_IS_BETTER)


def main() -> None:
    if not IN_CSV.exists():
        print(f"bulunamadı: {IN_CSV}")
        return
    df = pd.read_csv(IN_CSV)
    n_folds = df.fold.nunique()
    ratio, ratio_con = ratios(n_folds)
    print(f"CHB-MIT LOSO: {len(df)} satır, {n_folds} katman (denek), "
          f"{df.model.nunique()} model, test/eğitim oranı = 1/{n_folds - 1} "
          f"(temkinli: 1/{round(1 / ratio_con)})\n")

    OUT.mkdir(parents=True, exist_ok=True)
    all_res, summary_rows = [], []

    for metric in ("f1_macro", "log_loss", "brier"):
        s = per_model_summary(df, metric)
        print(f"=== {metric} (küçük daha iyi: {metric in LOWER_IS_BETTER}) ===")
        print(s.round(4).to_string())
        spread_all = s["mean"].max() - s["mean"].min()
        fus = s.loc[[m for m in FUSION if m in s.index], "mean"]
        spread_fus = fus.max() - fus.min() if len(fus) else float("nan")
        print(f"tüm modeller yayılımı: {spread_all:.4f}   "
              f"füzyon operatörleri yayılımı: {spread_fus:.4f}")

        res = analyze_metric(df, metric, ratio, ratio_con)
        if res.empty:
            print("(çift yok, atlanıyor)\n")
            continue
        res.to_csv(OUT / f"chbmit_{metric}.csv", index=False)
        all_res.append(res)

        sig = res[res.sig_corrected]
        print(f"çift={len(res)}  anlamlı: naif={int(res.sig_naive.sum())}  "
              f"düzeltilmiş={int(res.sig_corrected.sum())}")
        if len(sig):
            print("düzeltilmiş testte anlamlı çiftler:")
            print(sig[["model_a", "model_b", "mean_diff", "p_corrected_holm"]]
                  .round(4).to_string(index=False))

        fus_pairs = res[res.model_a.isin(FUSION) & res.model_b.isin(FUSION)]
        rope_lo = ROPE_BY_METRIC[metric][0]
        tag = f"{int(rope_lo * 100):02d}"
        if len(fus_pairs):
            print(f"füzyon çiftleri: n={len(fus_pairs)}  "
                  f"düzeltilmiş anlamlı={int(fus_pairs.sig_corrected.sum())}  "
                  f"delta_min aralığı=[{fus_pairs.delta_min_corrected.min():.4f}, "
                  f"{fus_pairs.delta_min_corrected.max():.4f}]  "
                  f"P(ROPE {rope_lo} içinde denk) aralığı="
                  f"[{fus_pairs[f'bayes_p_equiv_r{tag}'].min():.2f}, "
                  f"{fus_pairs[f'bayes_p_equiv_r{tag}'].max():.2f}]")
        print()

        summary_rows.append({
            "metric": metric, "pairs": len(res),
            "sig_naive": int(res.sig_naive.sum()),
            "sig_corrected": int(res.sig_corrected.sum()),
            "fusion_pairs": len(fus_pairs),
            "fusion_sig_corrected": int(fus_pairs.sig_corrected.sum()) if len(fus_pairs) else 0,
        })

    if all_res:
        pd.concat(all_res).to_csv(OUT / "chbmit_all.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / "summary.csv", index=False)

    # Bonn ile yan yana konabilecek kısa karşılaştırma notu
    print("=== Bonn (T1, N2) ile karşılaştırma referansı ===")
    print("Bonn'da (25 ölçüm, 5x5 CV): T1 füzyon çiftlerinde düzeltilmiş anlamlı=1/10, "
          "delta_min [0.022, 0.130] (bkz. results_v2/phase0/probscores_f1_macro.csv)")
    print(f"CHB-MIT'te (24 ölçüm, LOSO): yukarıdaki tabloya bakın")
    print(f"\nkaydedildi: {OUT}")


if __name__ == "__main__":
    main()
