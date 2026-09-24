"""Makale figürleri.

Tasarım kararları (bilinçli):
- Ana karşılaştırma bir *forest plot*: her modelin ortalaması ve %95 GA'sı. Amaç
  büyüklük + belirsizlik; çubuk grafik belirsizliği gizlerdi.
- Tek seri olan figürlerde legend yok; değerler doğrudan etiketlenir.
- Model ailesi renkle değil, konum ve metin etiketiyle kodlanır (renk-tek-başına
  kodlama yapılmaz).
- Anlamlılık matrisi üç durumlu: farklı / denk / belirsiz — hem renk hem metin.
- Renkler renk körlüğü doğrulamasından geçmiş paletten; ızgara ve eksenler geri planda.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .stats import pairwise_table, summarize

# Doğrulanmış kategorik palet (light mode)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b6b0"

FAMILY = {
    "logvar": "Sığ referans", "shallow": "Sığ referans",
    "raw1d": "Tek modalite", "spec2d": "Tek modalite",
    "raw1d_wide": "Tek modalite (eşlenmiş)", "spec2d_wide": "Tek modalite (eşlenmiş)",
    "early": "Füzyon", "late": "Füzyon", "gated": "Füzyon",
    "attention": "Füzyon", "score": "Füzyon",
}
DISPLAY = {
    "logvar": "LogVar (1 öz.)", "shallow": "Shallow-LR (7 öz.)",
    "raw1d": "1D-CNN", "spec2d": "2D-CNN",
    "raw1d_wide": "1D-CNN (geniş)", "spec2d_wide": "2D-CNN (geniş)",
    "early": "Early/Interm.", "late": "Late", "gated": "Gated",
    "attention": "Attention", "score": "Score",
}
FAMILY_ORDER = ["Füzyon", "Tek modalite (eşlenmiş)", "Tek modalite", "Sığ referans"]

# --- Dil ------------------------------------------------------------------
# Makale İngilizce, tez Türkçe olabilir. Figür metinleri tek yerden çevrilir:
#   set_lang("en")  ->  İngilizce figürler
LANG = "tr"
_EN = {
    "Füzyon": "Fusion", "Tek modalite (eşlenmiş)": "Unimodal (matched)",
    "Tek modalite": "Unimodal", "Sığ referans": "Shallow reference",
    "LogVar (1 öz.)": "LogVar (1 feat.)", "Shallow-LR (7 öz.)": "Shallow-LR (7 feat.)",
    "1D-CNN (geniş)": "1D-CNN (wide)", "2D-CNN (geniş)": "2D-CNN (wide)",
    "Macro-F1": "Macro-F1",
    "ortalama ve %95 güven aralığı": "mean and 95% CI",
    "eşleştirilmiş ölçüm": "paired measurements",
    "gölgeli bant: en iyi modelin güven aralığı":
        "shaded band: CI of the best model",
    "eksen dışı": "off-axis",
    "Eşleştirilmiş karşılaştırma": "Pairwise comparison",
    "anlamlı fark": "significant difference", "denk": "equivalent",
    "belirsiz": "inconclusive",
    "Spektrogram normalizasyonunun etkisi":
        "Effect of spectrogram amplitude normalisation",
    "yalnızca early/intermediate füzyon etkileniyor":
        "only early/intermediate fusion is affected",
    "ham/ham": "raw/raw", "z / ham spek.": "z / raw spec",
    "(hatalı kol)": "(the buggy arm)", "z / z spek.": "z / z spec",
    "z / z + spek. std.": "z / z + spec std",
    "Çıkarım gecikmesi (ms/örnek, batch 32, CPU)":
        "Inference latency (ms/sample, batch 32, CPU)",
    "Aynı parametre bütçesi, iki kat maliyet farkı":
        "Same parameter budget, two-fold cost difference",
    "kesikli çizgi: doğruluk/maliyet Pareto sınırı":
        "dashed: accuracy/cost Pareto frontier",
    "füzyon": "fusion", "tek modalite": "unimodal",
    "turuncu = sığ referans": "orange = shallow reference",
    "düzeltilmiş test": "corrected test",
}


def set_lang(lang: str) -> None:
    global LANG
    LANG = lang


def T(s: str) -> str:
    """Figür metni çevirisi. Karşılığı yoksa metin olduğu gibi kalır."""
    return _EN.get(s, s) if LANG == "en" else s


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
    print("kaydedildi:", outdir / f"{name}.png")


def fig_forest(df: pd.DataFrame, outdir: Path, metric: str = "f1_macro",
               xlabel: str = "Macro-F1", name: str = "F1_forest",
               focus_span: float = 0.12):
    """Ana figür: model başına ortalama ± %95 GA, aileye göre gruplanmış.

    Aile adları ayrı bir başlık satırı olarak yazılır (etiket çakışmasını önler).
    Odak aralığının dışında kalan modeller (ör. tek öznitelikli referans) sol kenara
    ok işaretiyle sabitlenir; aksi hâlde tek bir aykırı değer bütün ekseni ezer.
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
    inrange = mu_all[mu_all >= lo_focus]
    x_lo = float((s.loc[s["mean"] >= lo_focus, "mean"] -
                  s.loc[s["mean"] >= lo_focus, "ci95"]).min()) - 0.012
    x_hi = min(1.005, float((s["mean"] + s["ci95"]).max()) + 0.022)

    fig, ax = plt.subplots(figsize=(7.4, 0.36 * y + 1.4))

    best_m = s["mean"].idxmax()
    ax.axvspan(s.loc[best_m, "mean"] - s.loc[best_m, "ci95"],
               s.loc[best_m, "mean"] + s.loc[best_m, "ci95"],
               color=BLUE, alpha=0.08, zorder=0)
    ax.axvline(s.loc[best_m, "mean"], color=BLUE, lw=1, ls="--", alpha=0.55, zorder=1)

    yticks, ylabels = [], []
    for yy, kind, payload in entries:
        yticks.append(yy)
        if kind == "header":
            ylabels.append(T(payload).upper())
            continue
        ylabels.append("   " + T(DISPLAY.get(payload, payload)))
        m, c = s.loc[payload, "mean"], s.loc[payload, "ci95"]
        if m < lo_focus:                              # odak dışı: kenara sabitle
            ax.plot([x_lo + 0.004], [yy], marker="<", ms=8, color=ORANGE, zorder=3)
            ax.text(x_lo + 0.012, yy, f"{m:.3f}  ↩ " + T("eksen dışı"),
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
    ax.set_title(f"{xlabel}: {T('ortalama ve %95 güven aralığı')} "
                 f"({n} {T('eşleştirilmiş ölçüm')})\n"
                 f"{T('gölgeli bant: en iyi modelin güven aralığı')}",
                 loc="left", fontsize=10)
    _save(fig, outdir, name)


def fig_significance(df: pd.DataFrame, outdir: Path, metric: str = "f1_macro",
                     margin: float = 0.01, name: str = "significance_matrix"):
    """Üç durumlu karar matrisi: farklı / denk / belirsiz."""
    _style()
    pw = pairwise_table(df, metric, margin=margin)
    if pw.empty:
        return
    models = sorted(set(pw.model_a) | set(pw.model_b),
                    key=lambda m: -df[df.model == m][metric].mean())
    n = len(models)
    idx = {m: i for i, m in enumerate(models)}

    state = np.full((n, n), np.nan)      # 0 belirsiz, 1 denk, 2 farklı
    for _, r in pw.iterrows():
        i, j = idx[r.model_a], idx[r.model_b]
        v = 2.0 if r.significant else (1.0 if r.equivalent else 0.0)
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

    ax.set_xticks(range(n)); ax.set_xticklabels([T(DISPLAY.get(m, m)) for m in models],
                                                rotation=40, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels([T(DISPLAY.get(m, m)) for m in models])
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title(f"{T('Eşleştirilmiş karşılaştırma')}: {metric}\n"
                 f"≠ {T('anlamlı fark')} (Wilcoxon+Holm, p<0.05)   "
                 f"≡ {T('denk')} (TOST, ±{margin})   ? {T('belirsiz')}",
                 loc="left", fontsize=10)
    _save(fig, outdir, name)


def fig_significance_corrected(stats_df: pd.DataFrame, outdir: Path,
                               equiv_margin: float, models: list[str] | None = None,
                               metric_label: str = "macro F1",
                               name: str = "significance_matrix_corrected"):
    """Üç durumlu karar matrisi, düzeltilmiş istatistikten.

    `fig_significance` (üstteki) `pairwise_table` ile *naif* Wilcoxon+Holm hesaplar;
    Faz 0'ın gösterdiği gibi bu, tekrarlı CV'de yüzde 38 yanlış pozitif üretiyor
    (bkz. src/sim_cv_correlation.py, docs/07). Bu fonksiyon onun yerine geçer:
    hesaplama yapmaz, `src/phase0.py` / `src/chbmit_stats.py`'nin ürettiği ve zaten
    düzeltilmiş testten geçmiş bir tabloyu (model_a, model_b, mean_diff,
    sig_corrected, delta_min_corrected sütunları) doğrudan çizer.

    ≠ : sig_corrected True (düzeltilmiş testte anlamlı)
    ≡ : sig_corrected False VE delta_min_corrected <= equiv_margin (denklik kurulmuş)
    ? : ikisi de değil (ne fark ne denklik gösterilebiliyor, belirsiz)
    """
    _style()
    if stats_df.empty:
        return
    all_models = sorted(set(stats_df.model_a) | set(stats_df.model_b))
    order = [m for m in (models or all_models) if m in all_models]
    n = len(order)
    idx = {m: i for i, m in enumerate(order)}

    state = np.full((n, n), np.nan)
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
    ax.set_xticklabels([T(DISPLAY.get(m, m)) for m in order], rotation=40, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([T(DISPLAY.get(m, m)) for m in order])
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title(f"{T('Eşleştirilmiş karşılaştırma')}: {metric_label} "
                 f"({T('düzeltilmiş test')})\n"
                 f"≠ {T('anlamlı fark')} (corrected+Holm, p<0.05)   "
                 f"≡ {T('denk')} (δmin ≤ {equiv_margin:g})   "
                 f"? {T('belirsiz')}",
                 loc="left", fontsize=10)
    _save(fig, outdir, name)


def fig_ablation(runs: dict[str, pd.DataFrame], outdir: Path,
                 models=("early", "late", "spec2d"), metric: str = "f1_macro",
                 name: str = "normalization_ablation"):
    """Normalizasyon ablasyonu: spektrogram normalizasyonunun etkisi.

    Gösterilen modeller bilerek seçildi: `early` istatistiksel olarak etkilenen tek
    modeldir (+0.062, p<1e-6); `late` ve `spec2d` etkilenmeyenlerin temsilcisi olarak
    karşılaştırma sağlar. Yalnızca etkilenmeyen modelleri göstermek bulguyu gizlerdi.
    """
    _style()
    order = [k for k in ("N0_raw_raw", "N1_z_rawspec", "N2_z_zspec", "N3_z_zspec_specz")
             if k in runs]
    _arm = {"N0_raw_raw": "ham/ham", "N1_z_rawspec": "z / ham spek.",
            "N2_z_zspec": "z / z spek.", "N3_z_zspec_specz": "z / z + spek. std."}
    short = {k: f"{k.split('_')[0]}\n{T(v)}"
                + (f"\n{T('(hatalı kol)')}" if k == "N1_z_rawspec" else "")
             for k, v in _arm.items()}
    colors = [BLUE, ORANGE, AQUA]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(order))
    for k, (m, c) in enumerate(zip(models, colors)):
        mu, ci = [], []
        for nm in order:
            s = summarize(runs[nm], metric).set_index("model")
            mu.append(s.loc[m, "mean"] if m in s.index else np.nan)
            ci.append(s.loc[m, "ci95"] if m in s.index else np.nan)
        ax.errorbar(x, mu, yerr=ci, marker="o", ms=7, lw=2, capsize=3,
                    color=c, label=T(DISPLAY.get(m, m)))
        ax.text(x[-1] + 0.08, mu[-1], T(DISPLAY.get(m, m)), color=c,
                va="center", ha="left", fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels([short[k] for k in order], fontsize=9)
    ax.set_ylabel(T("Macro-F1"))
    ax.set_xlim(-0.35, len(order) - 0.35)
    ax.grid(axis="y", ls="--", alpha=0.3, color=MUTED); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.set_title(T("Spektrogram normalizasyonunun etkisi") + "\n"
                 + T("yalnızca early/intermediate füzyon etkileniyor")
                 + " (+0.062, p<1e-6)", loc="left")
    _save(fig, outdir, name)


def fig_cost(df: pd.DataFrame, outdir: Path, metric: str = "f1_macro",
             name: str = "cost_performance", cost_csv: Path | None = None):
    """Maliyet–başarı: gerçek çıkarım gecikmesi vs macro-F1.

    Maliyet ekseni olarak parametre sayısı değil **ölçülmüş gecikme** kullanılır:
    aynı parametre bütçesindeki modeller iki kattan fazla hız farkı gösteriyor,
    çünkü 1D dal 4097 örnek üzerinde, 2D dal 59x31 spektrogram üzerinde çalışıyor.
    Süreler ayrı, tek süreçli ve boştaki makinede ölçülmüştür.
    """
    _style()
    if cost_csv is None or not Path(cost_csv).exists():
        return
    cm = pd.read_csv(cost_csv).set_index("model")
    f1 = df.groupby("model")[metric].mean()
    common = [m for m in cm.index if m in f1.index]
    if not common:
        return

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    x = cm.loc[common, "latency_ms_b32"].to_numpy()
    y = f1.loc[common].to_numpy()
    is_uni = np.array([m in ("raw1d", "spec2d", "raw1d_wide", "spec2d_wide")
                       for m in common])

    ax.scatter(x[~is_uni], y[~is_uni], s=80, color=BLUE, marker="o",
               edgecolor="white", lw=1.2, zorder=3, label=T("füzyon"))
    ax.scatter(x[is_uni], y[is_uni], s=80, facecolor="white", edgecolor=ORANGE,
               marker="s", lw=1.8, zorder=3, label=T("tek modalite"))
    # Etiket yerleşimi: yakın noktalarda çakışmayı önlemek için dikey ofset
    # sırayla açılır ve gerekirse ince bir bağlantı çizgisi eklenir.
    xr = (x.max() - x.min()) or 1.0
    yr = (y.max() - y.min()) or 1.0
    placed: list[tuple[float, float]] = []
    for m, xi, yi in sorted(zip(common, x, y), key=lambda t: (-t[2], t[1])):
        dx, dy = 9.0, 4.0
        for _ in range(8):
            clash = any(abs((xi - px) / xr) < 0.16 and abs((yi + dy / 260 * yr - py) / yr) < 0.055
                        for px, py in placed)
            if not clash:
                break
            dy += 13.0
        ax.annotate(T(DISPLAY.get(m, m)), (xi, yi), textcoords="offset points",
                    xytext=(dx, dy), fontsize=9, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6,
                                    shrinkA=0, shrinkB=3) if dy > 6 else None)
        placed.append((xi, yi + dy / 260 * yr))

    # en iyi doğruluk/maliyet dengesi: pareto sınırı
    order = np.argsort(x)
    best = -np.inf
    px, py = [], []
    for i in order:
        if y[i] > best:
            best = y[i]
            px.append(x[i]); py.append(y[i])
    ax.step(px, py, where="post", color=MUTED, lw=1.2, ls="--", zorder=1)

    ax.set_xlabel(T("Çıkarım gecikmesi (ms/örnek, batch 32, CPU)"))
    ax.set_ylabel(T("Macro-F1"))
    ax.grid(ls="--", alpha=0.3, color=MUTED); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    # kaydırılmış etiketlerin başlığa değmemesi için üstte pay bırak
    ax.set_ylim(y.min() - yr * 0.10, y.max() + yr * 0.22)
    ax.set_xlim(x.min() - xr * 0.08, x.max() + xr * 0.20)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title(T("Aynı parametre bütçesi, iki kat maliyet farkı") + "\n"
                 + T("kesikli çizgi: doğruluk/maliyet Pareto sınırı"), loc="left")
    _save(fig, outdir, name)


def fig_task_comparison(runs: dict[str, pd.DataFrame], outdir: Path,
                        metric: str = "f1_macro", name: str = "task_comparison"):
    """Görevler arası karşılaştırma; sığ referans aynı eksende gösterilir."""
    _style()
    tasks = list(runs)
    show = ["logvar", "shallow", "raw1d", "spec2d", "late", "gated", "attention"]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 4.2), sharey=True)
    if len(tasks) == 1:
        axes = [axes]
    for ax, t in zip(axes, tasks):
        s = summarize(runs[t], metric).set_index("model")
        ms = [m for m in show if m in s.index]
        yy = np.arange(len(ms))
        cols = [ORANGE if m in ("logvar", "shallow") else BLUE for m in ms]
        ax.errorbar(s.loc[ms, "mean"], yy, xerr=s.loc[ms, "ci95"], fmt="none",
                    ecolor=INK2, capsize=3, lw=1.5)
        ax.scatter(s.loc[ms, "mean"], yy, s=55, c=cols, zorder=3)
        ax.set_yticks(yy); ax.set_yticklabels([T(DISPLAY.get(m, m)) for m in ms])
        ax.invert_yaxis()
        ax.set_xlabel(T("Macro-F1")); ax.set_title(t, loc="left", fontsize=10)
        ax.grid(axis="x", ls="--", alpha=0.3, color=MUTED); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].text(0.02, 0.02, T("turuncu = sığ referans"), transform=axes[0].transAxes,
                 fontsize=8.5, color=ORANGE)
    _save(fig, outdir, name)
