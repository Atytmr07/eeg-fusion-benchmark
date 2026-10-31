"""Bonn EEG loading, preprocessing, and spectrogram computation."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, get_window, stft

from .config import DATA_ROOT, FS, NORM_MODES, PROJECT_ROOT, SET_NAMES, TASKS, Config

# The 500 text files are parsed once and cached as .npy (about 8 MB).
_CACHE_NPY = PROJECT_ROOT / "data" / "_bonn_raw.npy"
_CACHE_KEY = PROJECT_ROOT / "data" / "_bonn_keys.npy"


def load_raw() -> tuple[np.ndarray, np.ndarray]:
    """All Bonn segments.

    Returns:
        X: (500, 4097) float32 raw signals
        sets: (500,) set letter (A..E) of each segment
    """
    if _CACHE_NPY.exists() and _CACHE_KEY.exists():
        return np.load(_CACHE_NPY), np.load(_CACHE_KEY)

    xs, keys = [], []
    for s in SET_NAMES:
        for p in sorted((DATA_ROOT / s).glob("*.txt")):
            xs.append(np.loadtxt(p).astype(np.float32))
            keys.append(s)
    X = np.stack(xs).astype(np.float32)
    sets = np.array(keys)
    np.save(_CACHE_NPY, X)
    np.save(_CACHE_KEY, sets)
    return X, sets


def select_task(X: np.ndarray, sets: np.ndarray, task: str):
    """Subset and labels for a task definition (see config.TASKS)."""
    mapping = TASKS[task]
    mask = np.isin(sets, list(mapping.keys()))
    y = np.array([mapping[s] for s in sets[mask]], dtype=np.int64)
    return X[mask], y, sets[mask]


def lowpass(x: np.ndarray, fs: float, fc: float, order: int) -> np.ndarray:
    b, a = butter(order, fc / (fs / 2), btype="low")
    return filtfilt(b, a, x, axis=-1)


def zscore(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.mean(axis=axis, keepdims=True)
    s = x.std(axis=axis, keepdims=True)
    return (x - m) / (s + 1e-6)


def make_spec(x: np.ndarray, cfg: Config) -> np.ndarray:
    """log(1 + |STFT|) spectrogram, cropped at cfg.fmax. x: (N, T) -> (N, F, T')."""
    f, _, Z = stft(
        x, fs=FS, nperseg=cfg.win, noverlap=cfg.win - cfg.hop, nfft=cfg.nfft,
        window=get_window("hann", cfg.win), boundary=None, padded=False, axis=-1,
    )
    band = f <= cfg.fmax
    return np.log1p(np.abs(Z[..., band, :])).astype(np.float32)


def build_arrays(cfg: Config):
    """Model inputs for a task and normalisation mode.

    Returns X1d (N, 1, T) for the raw branch, X2d (N, 1, F, T') for the spectrogram
    branch, the labels, and the set letter of each segment.
    """
    mode = NORM_MODES[cfg.norm_mode]
    X, sets = load_raw()
    X, y, sets = select_task(X, sets, cfg.task)

    x = X.astype(np.float64)
    if cfg.apply_lowpass:
        x = lowpass(x, FS, cfg.lowpass_hz, cfg.lowpass_order)

    x_z = zscore(x)

    sig = x_z if mode["sig_z"] else x
    spec_src = x_z if mode["spec_from_z"] else x
    spec = make_spec(spec_src.astype(np.float32), cfg)
    if mode["spec_z"]:
        # per example, over the whole time-frequency plane
        m = spec.mean(axis=(1, 2), keepdims=True)
        s = spec.std(axis=(1, 2), keepdims=True)
        spec = (spec - m) / (s + 1e-6)

    X1d = sig.astype(np.float32)[:, None, :]
    X2d = spec.astype(np.float32)[:, None, :, :]
    return X1d, X2d, y, sets
