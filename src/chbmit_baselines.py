"""CHB-MIT üzerinde sığ referans modeller: logvar (tek öznitelik) ve shallow.

Bonn'daki `src/baselines.py` tek kanallı sinyal bekliyordu (N, T). CHB-MIT çok
kanallı (N, C, T), bu yüzden aynı özellik matematiği **kanal başına** uygulanıp
kanallar boyunca birleştirilir. logvar'da bu 18 kanal x 1 öznitelik = 18 boyut,
shallow'da 18 kanal x 7 öznitelik = 126 boyut demektir.

Frekans bantları Bonn'daki 40 Hz tavanından farklı: CHB-MIT için 64 Hz tavan ve
gama bandı (30-64 Hz) kullanılır, çünkü `src/chbmit_prep.py`'de bu projede ölçülen
en büyük ictal/interictal güç oranının gama bandında olduğu belgelenmiştir
(bkz. docs/11_CHBMIT_PLANI.md §10.2).

Bölme: aynı LOSO katmanları, ama iç doğrulama ayrımı yok. Erken durdurma
gerekmediği için tüm eğitim deneklerine (tr_all) doğrudan fit edilir; bu, Bonn'daki
`run_benchmark.py`'nin idx_trval kullanımıyla aynı ilkedir.

Amaç: Bonn'daki en önemli bulgulardan biri, tek öznitelikli bir modelin klasik
görevde 0.954 F1 alması, yani ölçütün doygun olmasıydı. CHB-MIT'te aynı testi
yapmadan "bu görev zor/kolay" diye bir şey söylenemez.

Kullanım:  python -m src.chbmit_baselines
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from . import baselines
from .chbmit_corpus import build_corpus, leave_one_subject_out
from .config import RESULTS_ROOT
from .evaluate import metrics

FS_CHB = 256.0
BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 64)]     # delta..gama, 64 Hz tavan
OUT_DIR = RESULTS_ROOT / "chbmit" / "loso_main"
CSV_PATH = OUT_DIR / "perfold.csv"


def _per_channel(fn, X: np.ndarray) -> np.ndarray:
    """fn: (N, T) -> (N, k) şeklinde tek kanallık öznitelik fonksiyonu.
    X: (N, C, T). Dönen: (N, C*k), kanal başına öznitelikler yan yana."""
    n, c, t = X.shape
    out = [fn(X[:, ch, :]) for ch in range(c)]
    return np.concatenate(out, axis=1)


def logvar_features(X: np.ndarray) -> np.ndarray:
    return _per_channel(baselines.logvar_features, X)


def shallow_features(X: np.ndarray) -> np.ndarray:
    def fn(x1: np.ndarray) -> np.ndarray:
        x = x1.astype(np.float64)
        logvar = np.log(x.var(axis=1) + 1e-9)
        line_len = np.abs(np.diff(x, axis=1)).mean(axis=1)
        log_ll = np.log(line_len + 1e-9)
        n = x.shape[1]
        freqs = np.fft.rfftfreq(n, d=1.0 / FS_CHB)
        psd = (np.abs(np.fft.rfft(x, axis=1)) ** 2) / n
        total = psd.sum(axis=1) + 1e-12
        feats = [logvar, log_ll]
        for lo, hi in BANDS:
            m = (freqs >= lo) & (freqs < hi)
            feats.append(np.log(psd[:, m].sum(axis=1) / total + 1e-9))
        return np.stack(feats, axis=1)
    return _per_channel(fn, X)


def main() -> None:
    d = build_corpus(verbose=False)
    X, y, subject = d["X"], d["y"], d["subject"]
    info = d["info"]
    hours_of = {s: info["per_subject"][s]["hours"] for s in info["per_subject"]}
    ncls = 2

    print("öznitelik çıkarımı...")
    t0 = time.time()
    feat_logvar = logvar_features(X)
    feat_shallow = shallow_features(X)
    print(f"  logvar: {feat_logvar.shape}  shallow: {feat_shallow.shape}  "
          f"({time.time() - t0:.1f}s)")

    rows = []
    for fi, (key, tr, te) in enumerate(leave_one_subject_out(subject)):
        te_subs = sorted(set(subject[te]))
        hrs = sum(hours_of.get(s, 0.0) for s in te_subs)
        for name, feat in (("logvar", feat_logvar), ("shallow", feat_shallow)):
            t0 = time.time()
            prob = baselines.fit_predict(feat[tr], y[tr], feat[te])
            mm = metrics(y[te], prob, ncls)
            pred = prob.argmax(1)
            tp = int(((pred == 1) & (y[te] == 1)).sum())
            fn_ = int(((pred == 0) & (y[te] == 1)).sum())
            fp = int(((pred == 1) & (y[te] == 0)).sum())
            tn = int(((pred == 0) & (y[te] == 0)).sum())
            mm.update(sensitivity=tp / max(tp + fn_, 1),
                      specificity=tn / max(tn + fp, 1),
                      false_alarms=fp,
                      fa_per_hour_subsampled=fp / hrs if hrs > 0 else float("nan"),
                      model=name, fold=fi, test_subjects=",".join(te_subs),
                      n_test=len(te), n_test_ictal=int(y[te].sum()),
                      params=feat.shape[1], sec=round(time.time() - t0, 2),
                      best_epoch=-1, epochs_run=0, val_f1=float("nan"))
            rows.append(mm)
        print(f"[{fi + 1}/24] {key if isinstance(key, str) else te_subs}: "
              f"logvar f1={rows[-2]['f1_macro']:.3f}  shallow f1={rows[-1]['f1_macro']:.3f}")

    new_df = pd.DataFrame(rows)

    if CSV_PATH.exists():
        backup = OUT_DIR / "perfold_before_baselines.csv"
        if not backup.exists():
            import shutil
            shutil.copy(CSV_PATH, backup)
            print(f"yedeklendi: {backup}")
        old_df = pd.read_csv(CSV_PATH)
        old_df = old_df[~old_df.model.isin(("logvar", "shallow"))]     # tekrar koşuya karşı
        merged = pd.concat([old_df, new_df], ignore_index=True)
    else:
        merged = new_df

    merged.to_csv(CSV_PATH, index=False)
    print(f"\nkaydedildi: {CSV_PATH} ({len(merged)} satır)")
    print(new_df.groupby("model")[["f1_macro", "auc", "brier"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
