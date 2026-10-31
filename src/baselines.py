"""Classical reference models, run on exactly the same splits as the deep models so
they can enter the paired statistical comparison.

logvar:  a single feature, the log variance of the signal. It measures how much of
         the reported performance comes from absolute amplitude alone (on Bonn the
         ictal segments have about 30x the variance of the others).
shallow: a small classical feature set (log variance, log line length, relative band
         powers). The reference for "is a deep model needed at all?".
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import FS

# delta, theta, alpha, beta, and gamma up to the Bonn acquisition limit of 40 Hz
BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 40)]


def logvar_features(X_raw: np.ndarray) -> np.ndarray:
    """(N, T) -> (N, 1)"""
    return np.log(X_raw.astype(np.float64).var(axis=1) + 1e-9)[:, None]


def shallow_features(X_raw: np.ndarray, fs: float = FS, bands=BANDS) -> np.ndarray:
    """(N, T) -> (N, 2 + len(bands)): log variance, log line length, and the log of
    each band's power relative to total power."""
    x = X_raw.astype(np.float64)
    logvar = np.log(x.var(axis=1) + 1e-9)
    line_len = np.abs(np.diff(x, axis=1)).mean(axis=1)
    log_ll = np.log(line_len + 1e-9)

    n = x.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = (np.abs(np.fft.rfft(x, axis=1)) ** 2) / n
    total = psd.sum(axis=1) + 1e-12

    feats = [logvar, log_ll]
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        feats.append(np.log(psd[:, m].sum(axis=1) / total + 1e-9))
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
