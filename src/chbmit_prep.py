"""CHB-MIT pencerelerinden model girdisi: spektrogram ve normalizasyon.

Bonn'dan ayrılan iki karar ve gerekçeleri:

**Frekans tavanı.** Bonn kayıt sırasında 0.53-40 Hz'e sınırlandırılmıştı, bu yüzden
orada fmax=40 bilgi kaybı değildi. CHB-MIT band sınırlı değil. Üstelik bu projede
yapılan ölçüm (chb01_03, nöbet dönemi ile 20 nöbetsiz dönem karşılaştırması) gama
bandında (30-70 Hz) ictal/interictal güç oranını **18.3 kat** buldu; delta 7.7,
theta 6.5, beta 4.8 katta kalıyordu. 40 Hz'de kesmek en ayırt edici bandı atmak
olurdu. Varsayılan tavan bu yüzden 64 Hz.

**Şebeke gürültüsü.** CHB-MIT Boston'da kaydedilmiştir, şebeke frekansı 60 Hz ve bu
gama bandının içine düşer. Gürültü sabit olduğu için ictal/interictal oranını tek
başına açıklayamaz, ancak spektrogramda güçlü bir şerit oluşturur. `notch_hz`
verilirse o frekans ve harmonikleri bastırılır. Varsayılan kapalıdır ve bu bir
duyarlılık ekseni olarak koşulmalıdır.

**Normalizasyon.** Denekler arasında genlik ölçeği değişir; denek bazlı
değerlendirmede bu doğrudan genellemeyi etkiler. Üç seçenek sunulur ve seçim
kaydedilir:
  none     : dokunma
  window   : her pencere, her kanal kendi içinde z-score (varsayılan)
  channel  : denek içinde kanal başına z-score, eğitim katmanından hesaplanır

`channel` modu eğitim istatistiğini gerektirdiği için fonksiyon eğitim maskesi alır;
test katmanının istatistiği asla kullanılmaz.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import get_window, iirnotch, filtfilt, stft

FS = 256.0
FMAX_DEFAULT = 64.0


def notch_filter(x: np.ndarray, fs: float = FS, f0: float = 60.0,
                 q: float = 30.0, harmonics: int = 2) -> np.ndarray:
    """f0 ve harmoniklerini bastırır. x: (..., T)."""
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
    """log(1+|STFT|). x: (N, C, T) -> (N, C, F, T')."""
    f, _, Z = stft(x, fs=fs, nperseg=win, noverlap=win - hop, nfft=nfft,
                   window=get_window("hann", win), boundary=None, padded=False,
                   axis=-1)
    band = f <= fmax
    return np.log1p(np.abs(Z[..., band, :])).astype(np.float32)


def zscore_window(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Her pencerenin her kanalını kendi içinde standartlaştırır. x: (N, C, T)."""
    m = x.mean(axis=-1, keepdims=True)
    s = x.std(axis=-1, keepdims=True)
    return ((x - m) / (s + eps)).astype(np.float32)


def zscore_channel_from_train(x: np.ndarray, train_mask: np.ndarray,
                              eps: float = 1e-8) -> np.ndarray:
    """Kanal başına z-score; istatistik **yalnızca eğitim pencerelerinden** gelir.

    Test katmanının istatistiğini kullanmak sızıntıdır ve bu tür normalizasyon
    sızıntısı literatürde sık görülür.
    """
    if train_mask.sum() == 0:
        raise ValueError("eğitim maskesi boş, normalizasyon istatistiği hesaplanamaz")
    tr = x[train_mask]
    m = tr.mean(axis=(0, 2), keepdims=True)
    s = tr.std(axis=(0, 2), keepdims=True)
    return ((x - m) / (s + eps)).astype(np.float32)


def prepare(X: np.ndarray, norm: str = "window", train_mask: np.ndarray | None = None,
            notch_hz: float | None = None, fmax: float = FMAX_DEFAULT,
            fs: float = FS) -> tuple[np.ndarray, np.ndarray, dict]:
    """Ham pencerelerden (X1d, X2d) üretir.

    Dönen X1d: (N, C, T), X2d: (N, C, F, T'). Üçüncü dönen değer, yapılan seçimlerin
    kaydıdır; sonuçla birlikte saklanmalıdır.
    """
    if norm not in ("none", "window", "channel"):
        raise ValueError(f"bilinmeyen normalizasyon: {norm}")
    x = X
    if notch_hz:
        x = notch_filter(x, fs=fs, f0=notch_hz)
    if norm == "window":
        x = zscore_window(x)
    elif norm == "channel":
        if train_mask is None:
            raise ValueError("norm='channel' için train_mask gerekir; test "
                             "istatistiğinin kullanılması sızıntı olur")
        x = zscore_channel_from_train(x, train_mask)

    spec = spectrogram(x, fs=fs, fmax=fmax)
    meta = {"norm": norm, "notch_hz": notch_hz, "fmax": fmax, "fs": fs,
            "x1d_shape": list(x.shape), "x2d_shape": list(spec.shape)}
    return x.astype(np.float32), spec, meta
