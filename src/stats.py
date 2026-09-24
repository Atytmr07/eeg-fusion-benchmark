"""İstatistiksel analiz: eşleştirilmiş testler, çoklu karşılaştırma düzeltmesi,
etki büyüklüğü ve denklik (equivalence) testi.

Neden bu kadar katman:

- Tekrarlı CV'de aynı bölmeler tüm modellere uygulandığı için ölçümler *eşleştirilmiştir*;
  bağımsız örneklem testleri geçersizdir. Wilcoxon signed-rank kullanılır.
- 10 model = 45 çift. Düzeltmesiz p-değeri anlamsızdır; Holm-Bonferroni uygulanır.
- "Fark bulamadık" demek yetmez. Denklik testi (TOST), farkın pratik olarak önemsiz bir
  bandın (ROPE) içinde olduğunu *pozitif olarak* gösterir. Negatif sonucu savunulabilir
  kılan şey budur.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni düzeltilmiş p-değerleri."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def paired_cohens_d(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Eşleştirilmiş Cohen's d ve %95 GA (fark dağılımı üzerinden)."""
    d = a - b
    n = len(d)
    sd = d.std(ddof=1)
    if sd < 1e-12:
        return 0.0, 0.0, 0.0
    dz = d.mean() / sd
    se = np.sqrt(1.0 / n + dz**2 / (2 * n))
    t = stats.t.ppf(0.975, n - 1)
    return dz, dz - t * se, dz + t * se


def tost_paired(a: np.ndarray, b: np.ndarray, margin: float) -> float:
    """İki tek-yönlü t testi (TOST). Döndürülen p < alpha ise denklik iddia edilir.

    H0: |mu_d| >= margin   (fark pratik olarak önemli)
    H1: |mu_d| <  margin   (fark pratik olarak önemsiz)
    """
    d = a - b
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    if se < 1e-12:
        return 0.0 if abs(d.mean()) < margin else 1.0
    df = n - 1
    t_lo = (d.mean() + margin) / se     # H0: mu <= -margin
    t_hi = (d.mean() - margin) / se     # H0: mu >= +margin
    p_lo = stats.t.sf(t_lo, df)
    p_hi = stats.t.cdf(t_hi, df)
    return float(max(p_lo, p_hi))


def pairwise_table(df: pd.DataFrame, metric: str = "f1_macro",
                   margin: float = 0.01, alpha: float = 0.05) -> pd.DataFrame:
    """Tüm model çiftleri için eşleştirilmiş test tablosu.

    df: perfold.csv — sütunlar en az {model, repeat, fold, <metric>}
    """
    piv = (df.pivot_table(index=["repeat", "fold"], columns="model", values=metric)
             .sort_index())
    models = list(piv.columns)
    rows = []
    for m1, m2 in combinations(models, 2):
        a, b = piv[m1].to_numpy(), piv[m2].to_numpy()
        ok = ~(np.isnan(a) | np.isnan(b))
        a, b = a[ok], b[ok]
        if len(a) < 3:
            continue
        try:
            w_p = stats.wilcoxon(a, b, zero_method="wilcox").pvalue
        except ValueError:        # tüm farklar sıfır
            w_p = 1.0
        t_p = stats.ttest_rel(a, b).pvalue
        d, lo, hi = paired_cohens_d(a, b)
        rows.append({
            "model_a": m1, "model_b": m2, "n": len(a),
            "mean_a": a.mean(), "mean_b": b.mean(), "mean_diff": (a - b).mean(),
            "wilcoxon_p": w_p, "ttest_p": t_p,
            "cohens_d": d, "d_lo": lo, "d_hi": hi,
            "tost_p": tost_paired(a, b, margin),
            "equiv_bound": equivalence_bound(a, b),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["wilcoxon_p_holm"] = holm(out["wilcoxon_p"].to_numpy())
    out["ttest_p_holm"] = holm(out["ttest_p"].to_numpy())
    out["significant"] = out["wilcoxon_p_holm"] < alpha
    out["equivalent"] = out["tost_p"] < alpha
    return out.sort_values("wilcoxon_p_holm")


def summarize(df: pd.DataFrame, metric: str = "f1_macro") -> pd.DataFrame:
    """Model başına ortalama, std ve %95 GA."""
    rows = []
    for m, g in df.groupby("model"):
        v = g[metric].dropna().to_numpy()
        n = len(v)
        ci = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows.append({"model": m, "n": n, "mean": v.mean(), "std": v.std(ddof=1),
                     "ci95": ci, "lo": v.mean() - ci, "hi": v.mean() + ci})
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


def equivalence_bound(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> float:
    """Denkliğin kurulabildiği en dar marj (δ_min).

    Keyfi bir ROPE seçip "denk / değil" demek yerine, veriden doğrudan okunabilen tek
    sayı: *bu iki model ±δ_min içinde denktir*. Okuyucu kendi pratik anlamlılık eşiğini
    uygulayabilir.

    TOST, marj δ için ancak ve ancak farkın (1−2α) güven aralığı ±δ içinde kalırsa
    reddeder; dolayısıyla

        δ_min = |ortalama fark| + t_{1−α, n−1} · SE

    yani farkın %90 güven aralığının mutlak değerce en uzak ucu (α = 0.05 için).
    """
    d = a - b
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    if n < 2:
        return float("nan")
    return float(abs(d.mean()) + stats.t.ppf(1 - alpha, n - 1) * se)


def tost_power(sd: float, n: int, margin: float = 0.01, alpha: float = 0.05,
               sims: int = 4000, seed: int = 0) -> float:
    """Gerçek fark sıfırken denklik testinin gücü.

    Denklik kurulamaması, denkliğin olmadığı anlamına gelmez — testin o tasarımda
    yeterli gücü olmayabilir. Bu sayı, "denk diyemedik" ifadesinin yanında
    raporlanmalıdır. Simülasyonla hesaplanır (yansız ve varsayımsız).
    """
    rng = np.random.default_rng(seed)
    zeros = np.zeros(n)
    hits = sum(tost_paired(zeros, -rng.normal(0.0, sd, n), margin) < alpha
               for _ in range(sims))
    return hits / sims


def paired_arm_comparison(df_a: pd.DataFrame, df_b: pd.DataFrame,
                          metric: str = "f1_macro", alpha: float = 0.05) -> pd.DataFrame:
    """İki ablasyon kolunu model bazında eşleştirilmiş olarak karşılaştırır.

    Kollar aynı bölmeleri ve aynı model-seed'lerini kullandığı için ölçümler
    (tekrar, fold) düzeyinde eşleşir. Spektrogramı kullanmayan modellerin farkı tam
    olarak sıfır olmalıdır — bu bir determinizm kontrolüdür.
    """
    pa = df_a.pivot_table(index=["repeat", "fold"], columns="model", values=metric).sort_index()
    pb = df_b.pivot_table(index=["repeat", "fold"], columns="model", values=metric).sort_index()
    rows = []
    for m in [c for c in pa.columns if c in pb.columns]:
        a, b = pa[m].to_numpy(), pb[m].to_numpy()
        diff = a - b
        identical = bool(np.allclose(diff, 0.0))
        if identical:
            rows.append({"model": m, "mean_a": a.mean(), "mean_b": b.mean(),
                         "mean_diff": 0.0, "wilcoxon_p": np.nan,
                         "equiv_bound": 0.0, "identical": True})
            continue
        try:
            p = stats.wilcoxon(a, b).pvalue
        except ValueError:
            p = 1.0
        rows.append({"model": m, "mean_a": a.mean(), "mean_b": b.mean(),
                     "mean_diff": diff.mean(), "wilcoxon_p": p,
                     "equiv_bound": equivalence_bound(a, b), "identical": False})
    out = pd.DataFrame(rows)
    tested = out["wilcoxon_p"].notna()
    out.loc[tested, "wilcoxon_p_holm"] = holm(out.loc[tested, "wilcoxon_p"].to_numpy())
    out["significant"] = out["wilcoxon_p_holm"] < alpha
    return out.sort_values("mean_diff", ascending=False).reset_index(drop=True)


def collapse_report(df: pd.DataFrame, auc_f1_gap: float = 0.25) -> pd.DataFrame:
    """Çökmüş (dejenere) koşuları tespit eder.

    İki bağımsız işaret:
      1. Karışıklık matrisinde tüm tahminler tek bir sınıfta toplanmış.
      2. AUC ile macro-F1 arasında büyük uçurum — temsil ayrıştırıcı ama karar
         kuralı bozuk. Bu, doğrulama F1'i üzerinde minimum epoch bütçesi olmayan
         erken durdurmanın tipik imzasıdır (bkz. docs/02_BULGULAR.md, C2).

    Ortalama raporlanmadan ÖNCE çalıştırılmalıdır; boş dönmesi beklenir.
    """
    import ast

    rows = []
    for _, r in df.iterrows():
        cm = r.get("cm")
        single_class = False
        if isinstance(cm, str):
            try:
                m = np.asarray(ast.literal_eval(cm), dtype=float)
                if m.ndim == 2 and m.size:
                    single_class = bool((m.sum(axis=0) > 0).sum() <= 1)
            except (ValueError, SyntaxError):
                pass
        gap = float(r["auc"]) - float(r["f1_macro"]) if pd.notna(r.get("auc")) else np.nan
        if single_class or (pd.notna(gap) and gap > auc_f1_gap):
            rows.append({"model": r["model"], "repeat": r.get("repeat"),
                         "fold": r.get("fold"), "f1_macro": r["f1_macro"],
                         "auc": r.get("auc"), "auc_minus_f1": gap,
                         "single_class_prediction": single_class, "cm": cm})
    return pd.DataFrame(rows)


def min_detectable_effect(df: pd.DataFrame, metric: str = "f1_macro",
                          power: float = 0.8, alpha: float = 0.05) -> float:
    """Bu tasarımla yakalanabilecek en küçük etki (Cohen's dz).

    "Fark bulamadık" iddiasının yanına konulması gereken sayı: tasarımın gücü.
    """
    n = df.groupby("model").size().min()
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return float((z_a + z_b) / np.sqrt(n))


# ---------------------------------------------------------------------------
# Faz 0: tekrarlı çapraz doğrulamada korelasyonu hesaba katan testler
# ---------------------------------------------------------------------------
#
# Tekrarlı k-fold CV'de ölçümler bağımsız değildir: farklı bölünmelerin eğitim
# kümeleri büyük ölçüde örtüşür. Ölçümleri bağımsız sayan testler (Wilcoxon,
# sıradan eşleştirilmiş t) varyansı küçük tahmin eder ve yanlış pozitif oranını
# şişirir (Nadeau ve Bengio 2003; Bouckaert ve Frank 2004).
#
# Düzeltme, farkların örneklem varyansını şu çarpanla büyütür:
#
#     SE^2 = (1/n + n_test/n_train) * s^2
#
# Burada n = k*r ölçüm sayısıdır. Aynı ölçeklendirme, korelasyonlu Bayesçi t
# testinin (Corani ve Benavoli; Benavoli ve ark. 2017) sonsal dağılımında da
# kullanılır; k-fold için n_test/n_train = 1/(k-1).


def corrected_se(d: np.ndarray, test_train_ratio: float) -> float:
    """Tekrarlı CV için düzeltilmiş standart hata."""
    n = len(d)
    s2 = d.var(ddof=1)
    return float(np.sqrt((1.0 / n + test_train_ratio) * s2))


def corrected_ttest(a: np.ndarray, b: np.ndarray, test_train_ratio: float) -> tuple[float, float]:
    """Düzeltilmiş tekrarlı k-fold t testi. (t, iki yönlü p) döner."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    se = corrected_se(d, test_train_ratio)
    if se < 1e-12:
        return (0.0, 1.0) if abs(d.mean()) < 1e-12 else (np.inf, 0.0)
    t = d.mean() / se
    return float(t), float(2 * stats.t.sf(abs(t), n - 1))


def corrected_equivalence_bound(a: np.ndarray, b: np.ndarray, test_train_ratio: float,
                                alpha: float = 0.05) -> float:
    """Düzeltilmiş varyansla denklik sınırı δ_min."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    return float(abs(d.mean()) + stats.t.ppf(1 - alpha, n - 1) * corrected_se(d, test_train_ratio))


def correlated_bayes(a: np.ndarray, b: np.ndarray, test_train_ratio: float,
                     rope: float) -> tuple[float, float, float]:
    """Korelasyonlu Bayesçi t testi.

    Ortalama farkın sonsal dağılımı: serbestlik derecesi n-1, konumu örneklem
    ortalaması, ölçeği düzeltilmiş standart hata olan Student t.

    Döner: (P(a daha kötü), P(pratik olarak denk), P(a daha iyi))
    Yani sırasıyla fark < -rope, |fark| <= rope, fark > +rope olasılıkları.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    mu, se = d.mean(), corrected_se(d, test_train_ratio)
    if se < 1e-12:
        if mu < -rope:
            return 1.0, 0.0, 0.0
        if mu > rope:
            return 0.0, 0.0, 1.0
        return 0.0, 1.0, 0.0
    dist = stats.t(df=n - 1, loc=mu, scale=se)
    p_left = float(dist.cdf(-rope))
    p_right = float(dist.sf(rope))
    return p_left, float(max(0.0, 1.0 - p_left - p_right)), p_right


def sign_test(a: np.ndarray, b: np.ndarray) -> float:
    """İşaret testi. Wilcoxon'un aksine fark dağılımının simetrisini varsaymaz."""
    d = np.asarray(a, float) - np.asarray(b, float)
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    if pos + neg == 0:
        return 1.0
    return float(stats.binomtest(pos, pos + neg, 0.5).pvalue)
