"""CHB-MIT model inputs: normalisation and spectrogram.

Three decisions differ from Bonn:

Frequency ceiling. Bonn was band-limited to 0.53-40 Hz at acquisition, so fmax = 40
lost nothing there. CHB-MIT is not band-limited, and a measurement in this project
(chb01_03, seizure period vs 20 seizure-free periods) found the largest ictal /
interictal power ratio in the gamma band (30-70 Hz): 18.3x, against 7.7x for delta,
6.5x for theta and 4.8x for beta. Cutting at 40 Hz would discard the most
discriminative band, so the default ceiling is 64 Hz.

Mains noise. CHB-MIT was recorded in Boston; 60 Hz mains falls inside the gamma band.
Being constant it cannot explain the ictal/interictal ratio on its own, but it draws a
strong line in the spectrogram. With notch_hz set, that frequency and its harmonics are
suppressed. Off by default; meant to be run as a sensitivity axis.

Normalisation. Amplitude scale varies between subjects, which matters directly for
subject-wise evaluation. Four options, and the choice is recorded:
  none     leave as is
  window   z-score each channel of each window on its own (default)
  robust   like window, with median and interquartile range (pipeline P5)
  channel  z-score each channel with statistics from the training windows only
"""
from __future__ import annotations

import numpy as np
from scipy.signal import get_window, iirnotch, filtfilt, stft

FS = 256.0
FMAX_DEFAULT = 64.0


def notch_filter(x: np.ndarray, fs: float = FS, f0: float = 60.0,
                 q: float = 30.0, harmonics: int = 2) -> np.ndarray:
    """Suppress f0 and its harmonics. x: (..., T)."""
    out = x.astype(np.float64)
    for k in range(1, harmonics + 1):
        f = f0 * k
        if f >= fs / 2:
            break
        b, a = iirnotch(f / (fs / 2), q)
        out = filtfilt(b, a, out, axis=-1)
    return out.astype(np.float32)


def spectrogram(x: np.ndarray, fs: float = FS, win: int = 256, hop: int = 128,
                nfft: int = 256, fmax: float = FMAX_DEFAULT) -> np.ndarray:
    """log(1 + |STFT|), cropped at fmax. x: (N, C, T) -> (N, C, F, T')."""
    f, _, Z = stft(x, fs=fs, nperseg=win, noverlap=win - hop, nfft=nfft,
                   window=get_window("hann", win), boundary=None, padded=False,
                   axis=-1)
    band = f <= fmax
    return np.log1p(np.abs(Z[..., band, :])).astype(np.float32)


def zscore_window(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Standardise each channel of each window on its own. x: (N, C, T)."""
    m = x.mean(axis=-1, keepdims=True)
    s = x.std(axis=-1, keepdims=True)
    return ((x - m) / (s + eps)).astype(np.float32)


def robust_window(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per window and channel: subtract the median, divide by the interquartile range."""
    med = np.median(x, axis=-1, keepdims=True)
    q75, q25 = np.percentile(x, [75, 25], axis=-1, keepdims=True)
    return ((x - med) / (q75 - q25 + eps)).astype(np.float32)


def zscore_channel_from_train(x: np.ndarray, train_mask: np.ndarray,
                              eps: float = 1e-8) -> np.ndarray:
    """Per-channel z-score with statistics from the TRAINING windows only.

    Using test-fold statistics would be leakage, a common form of it in this
    literature.
    """
    if train_mask.sum() == 0:
        raise ValueError("empty training mask, cannot compute normalisation statistics")
    tr = x[train_mask]
    m = tr.mean(axis=(0, 2), keepdims=True)
    s = tr.std(axis=(0, 2), keepdims=True)
    return ((x - m) / (s + eps)).astype(np.float32)


def prepare(X: np.ndarray, norm: str = "window", train_mask: np.ndarray | None = None,
            notch_hz: float | None = None, fmax: float = FMAX_DEFAULT,
            fs: float = FS) -> tuple[np.ndarray, np.ndarray, dict]:
    """Model inputs from raw windows.

    Returns X1d (N, C, T), X2d (N, C, F, T'), and a record of the choices made, which
    should be stored with the results.
    """
    if norm not in ("none", "window", "robust", "channel"):
        raise ValueError(f"unknown normalisation: {norm}")
    x = X
    if notch_hz:
        x = notch_filter(x, fs=fs, f0=notch_hz)
    if norm == "window":
        x = zscore_window(x)
    elif norm == "robust":
        x = robust_window(x)
    elif norm == "channel":
        if train_mask is None:
            raise ValueError("norm='channel' needs train_mask; using test statistics "
                             "would be leakage")
        x = zscore_channel_from_train(x, train_mask)

    spec = spectrogram(x, fs=fs, fmax=fmax)
    meta = {"norm": norm, "notch_hz": notch_hz, "fmax": fmax, "fs": fs,
            "x1d_shape": list(x.shape), "x2d_shape": list(spec.shape)}
    return x.astype(np.float32), spec, meta
