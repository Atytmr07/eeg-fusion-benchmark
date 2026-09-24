"""Sığ (shallow) referans modeller.

Bunlar derin modellerle *aynı* CV bölmeleri üzerinde koşulur, böylece istatistiksel
karşılaştırmaya eşleştirilmiş biçimde girebilirler.

`logvar`: tek öznitelik — sinyalin log-varyansı. Amacı, bildirilen başarının ne kadarının
mutlak genlik farkından (amplitude shortcut) geldiğini ölçmektir. Bonn'da iktal
segmentlerin varyansı diğerlerinden ~30x büyüktür.

`shallow`: küçük ve klasik bir öznitelik kümesi (log-varyans, line-length, bant güçleri).
"Derin model gerçekten gerekli mi" sorusunun referansı.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import FS

BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 40)]


def logvar_features(X_raw: np.ndarray) -> np.ndarray:
    return np.log(X_raw.astype(np.float64).var(axis=1) + 1e-9)[:, None]


def shallow_features(X_raw: np.ndarray) -> np.ndarray:
    x = X_raw.astype(np.float64)
    logvar = np.log(x.var(axis=1) + 1e-9)
    line_len = np.abs(np.diff(x, axis=1)).mean(axis=1)
    log_ll = np.log(line_len + 1e-9)

    n = x.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    psd = (np.abs(np.fft.rfft(x, axis=1)) ** 2) / n
    total = psd.sum(axis=1) + 1e-12

    feats = [logvar, log_ll]
    for lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        feats.append(np.log(psd[:, m].sum(axis=1) / total + 1e-9))  # göreli bant gücü
    return np.stack(feats, axis=1)


def make_clf() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
    )


def fit_predict(feat_tr: np.ndarray, y_tr: np.ndarray, feat_te: np.ndarray) -> np.ndarray:
    clf = make_clf()
    clf.fit(feat_tr, y_tr)
    return clf.predict_proba(feat_te)
