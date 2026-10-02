"""Preprocessing multiverse: the pipelines P0-P6.

The question is whether the comparison between fusion operators depends on how the EEG
was preprocessed. Each pipeline changes one hypothesis-driven aspect of preprocessing
and everything else stays fixed: the same windows (same recordings, times, labels and
subsampling), the same spectrogram settings (fmax 64 Hz, so input shapes and parameter
counts are identical), the same models, training and evaluation.

    name  band-pass    notch   artefacts                     normalisation   purpose
    P0    none         none    none                          z-score         original benchmark
    P1    0.5-40 Hz    none    none                          z-score         standard EEG band
    P2    0.5-70 Hz    60 Hz   none                          z-score         information above 40 Hz
    P3    1-40 Hz      none    none                          z-score         low frequency / drift
    P4    0.5-40 Hz    none    amplitude epoch rejection     z-score         artefact rejection
    P5    0.5-40 Hz    none    none                          median / IQR    normalisation
    P6    0.5-40 Hz    none    ICA component removal         z-score         aggressive cleaning

Where each step runs:

  signal level   band-pass, notch and ICA run on the CONTINUOUS recording, before it is
                 cut into windows. Filtering 10 s windows one by one would put filter
                 transients at every window edge (a 0.5 Hz high-pass settles over
                 seconds). These steps change the corpus, so each distinct combination
                 has its own cache; P1, P4 and P5 share one.
  window level   normalisation (src/chbmit_prep.py: prepare) and artefact rejection
                 (artifact_mask) run per window at training time.

Artefact rejection (P4) removes windows from TRAINING and VALIDATION only; the test set
is the same as in every other pipeline, so the pipelines stay paired fold by fold.
Rejecting test windows would also discard seizures selectively, because ictal EEG is
high in amplitude: the rejection rate is recorded separately for ictal and non-ictal
windows.

Open choices, pending the advisor (recorded in every run's meta.json):
  - P4 threshold (REJECT_UV, see the measurement next to it) and whether rejection
    should also apply to test windows.
  - P6 component selection. There is no EOG channel in CHB-MIT, so ocular components
    are identified from the frontal (FP) channels: topography concentrated on them and
    power concentrated below 4 Hz. Muscle components are not removed by default,
    because the gamma band carries the strongest ictal signal in this corpus
    (src/chbmit_prep.py) and removing high-frequency components could remove seizures.
"""
from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from .chbmit import TARGET_CHANNELS
from .chbmit_prep import notch_filter

BUTTER_ORDER = 4            # zero-phase (forward-backward), so effective order 8

# P4: a window is rejected if any channel's peak-to-peak amplitude exceeds this (uV,
# measured after the 0.5-40 Hz band-pass). None until the advisor decides; P4 refuses
# to run without it. `python -m src.preprocess --scan-rejection` measured on CHB-MIT:
# median worst-channel peak-to-peak is 482 uV in non-ictal and 890 uV in ictal windows,
# so every fixed threshold removes ictal windows 1.5 to 4 times as often as non-ictal
# ones (500 uV: 72 % vs 48 %; 1000 uV: 44 % vs 12 %). A per-subject relative rule does
# not escape this (robust z > 10: 21 % vs 4.5 %). Amplitude-based rejection is
# confounded with the class itself on this corpus.
REJECT_UV: float | None = None

# P6: ICA settings and the ocular component rule.
ICA_FIT_STRIDE = 2          # fit on every 2nd sample; the 0.5-40 Hz signal has no
                            # content above 64 Hz, so this loses nothing
ICA_MAX_ITER = 1000
ICA_FRONTAL = ("FP1-F7", "FP1-F3", "FP2-F4", "FP2-F8")
ICA_FRONTAL_FRAC = 0.5      # share of the component's squared topography on FP channels
ICA_LOW_FRAC = 0.6          # share of its 0.5-40 Hz power below 4 Hz
ICA_MAX_REMOVE = 2          # at most this many components removed per recording


@dataclass(frozen=True)
class Pipeline:
    name: str
    band: tuple[float, float] | None = None
    notch_hz: float | None = None
    ica: bool = False
    reject_uv: float | None = None
    norm: str = "window"            # "window" (z-score) or "robust" (median / IQR)

    @property
    def signal_tag(self) -> str:
        """Cache suffix for the signal-level steps; empty when there are none (P0)."""
        parts = []
        if self.band:
            parts.append(f"bp{self.band[0]:g}-{self.band[1]:g}")
        if self.notch_hz:
            parts.append(f"n{self.notch_hz:g}")
        if self.ica:
            parts.append("ica")
        return "_".join(parts)

    @property
    def has_signal_steps(self) -> bool:
        return bool(self.signal_tag)

    def apply_signal(self, x: np.ndarray, fs: float, key: str = "") -> np.ndarray:
        """Signal-level steps on a continuous recording x (channel, time).

        `key` names the recording; it seeds ICA so the result does not depend on the
        order in which recordings are processed.
        """
        if self.band:
            x = bandpass(x, fs, *self.band)
        if self.notch_hz:
            x = notch_filter(x, fs=fs, f0=self.notch_hz, harmonics=1)
        if self.ica:
            x, _ = ica_clean(x, fs, seed=zlib.crc32(key.encode()))
        return x.astype(np.float32)

    def describe(self) -> dict:
        d = asdict(self) | {"signal_tag": self.signal_tag}
        if self.reject_uv is not None:
            d["reject_scope"] = "train and validation"
        if self.ica:
            d["ica_rule"] = {"frontal": list(ICA_FRONTAL), "frontal_frac": ICA_FRONTAL_FRAC,
                             "low_frac": ICA_LOW_FRAC, "max_remove": ICA_MAX_REMOVE,
                             "fit_stride": ICA_FIT_STRIDE}
        return d


PIPELINES = {p.name: p for p in (
    Pipeline("P0"),
    Pipeline("P1", band=(0.5, 40.0)),
    Pipeline("P2", band=(0.5, 70.0), notch_hz=60.0),
    Pipeline("P3", band=(1.0, 40.0)),
    Pipeline("P4", band=(0.5, 40.0), reject_uv=REJECT_UV if REJECT_UV else float("nan")),
    Pipeline("P5", band=(0.5, 40.0), norm="robust"),
    Pipeline("P6", band=(0.5, 40.0), ica=True),
)}


# --- Signal level --------------------------------------------------------------------

def bandpass(x: np.ndarray, fs: float, lo: float, hi: float,
             order: int = BUTTER_ORDER) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last axis."""
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x.astype(np.float64), axis=-1).astype(np.float32)


def ica_clean(x: np.ndarray, fs: float, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Remove ocular ICA components from a band-passed recording x (channel, time).

    FastICA with as many components as channels. A component is ocular if its
    topography is concentrated on the frontal channels AND its power is concentrated
    below 4 Hz; the strongest such components (up to ICA_MAX_REMOVE) are zeroed and
    the signal is reconstructed. Returns the cleaned signal and a record of what was
    removed.
    """
    from sklearn.decomposition import FastICA

    xd = x.astype(np.float64)
    ica = FastICA(n_components=xd.shape[0], whiten="unit-variance", max_iter=ICA_MAX_ITER,
                  random_state=seed % (2 ** 32))
    ica.fit(xd[:, ::ICA_FIT_STRIDE].T)
    s = ica.transform(xd.T)                                  # (time, component)

    topo = ica.mixing_ ** 2                                  # (channel, component)
    front = [TARGET_CHANNELS.index(c) for c in ICA_FRONTAL]
    frontal_frac = topo[front].sum(axis=0) / topo.sum(axis=0)
    f, p = welch(s.T, fs=fs, nperseg=int(4 * fs), axis=-1)
    band = (f >= 0.5) & (f <= 40.0)
    low_frac = p[:, (f >= 0.5) & (f < 4.0)].sum(axis=1) / p[:, band].sum(axis=1)

    ocular = np.flatnonzero((frontal_frac > ICA_FRONTAL_FRAC) & (low_frac > ICA_LOW_FRAC))
    ocular = ocular[np.argsort(-frontal_frac[ocular])][:ICA_MAX_REMOVE]
    s[:, ocular] = 0.0
    out = ica.inverse_transform(s).T.astype(np.float32)
    return out, {"removed": ocular.tolist(), "converged_iter": int(ica.n_iter_),
                 "frontal_frac": frontal_frac.round(3).tolist(),
                 "low_frac": low_frac.round(3).tolist()}


# --- Window level --------------------------------------------------------------------

def artifact_mask(x: np.ndarray, threshold_uv: float) -> np.ndarray:
    """True for windows in which any channel's peak-to-peak amplitude exceeds the
    threshold. x: (window, channel, time) in microvolts, before normalisation."""
    ptp = x.max(axis=-1) - x.min(axis=-1)
    return (ptp > threshold_uv).any(axis=-1)


# --- Building and checking the pipeline corpora -----------------------------------------

def build_all(names: list[str]) -> None:
    """Build the corpus cache of every distinct signal-level step set among `names`,
    and check that each holds exactly the same windows as P0."""
    from .chbmit_corpus import build_corpus

    ref = build_corpus(verbose=False)
    done: set[str] = set()
    for name in names:
        p = PIPELINES[name]
        if not p.has_signal_steps or p.signal_tag in done:
            continue
        done.add(p.signal_tag)
        print(f"\n{name}: building corpus '{p.signal_tag}' (shared by "
              f"{', '.join(q.name for q in PIPELINES.values() if q.signal_tag == p.signal_tag)})")
        d = build_corpus(signal_fn=p.apply_signal, signal_tag=p.signal_tag)
        same = all(np.array_equal(d[k], ref[k]) for k in
                   ("y", "subject", "seizure_id", "record", "t0"))
        if not same or d["X"].shape != ref["X"].shape:
            raise SystemExit(f"{name}: windows differ from P0; the pipelines would not "
                             f"be paired")
        print(f"{name}: same {len(d['y'])} windows as P0, checked")


def scan_rejection(thresholds=(200, 300, 400, 500, 750, 1000)) -> None:
    """Share of ictal and non-ictal windows that P4 would reject, per threshold."""
    from .chbmit_corpus import build_corpus

    p = PIPELINES["P4"]
    d = build_corpus(verbose=False, signal_fn=p.apply_signal, signal_tag=p.signal_tag)
    # per subject and channel robust z of the peak-to-peak amplitude, for comparison
    ptp_c = d["X"].max(axis=-1) - d["X"].min(axis=-1)
    z = np.zeros_like(ptp_c)
    for s in np.unique(d["subject"]):
        m = d["subject"] == s
        med = np.median(ptp_c[m], axis=0)
        z[m] = (ptp_c[m] - med) / (1.4826 * np.median(np.abs(ptp_c[m] - med), axis=0) + 1e-6)
    ptp = (d["X"].max(axis=-1) - d["X"].min(axis=-1)).max(axis=-1)    # worst channel
    y = d["y"]
    print("max peak-to-peak over channels (uV), percentiles 50/90/99:")
    for lab, m in (("ictal", y == 1), ("non-ictal", y == 0)):
        print(f"  {lab:9s} " + " / ".join(f"{v:.0f}" for v in np.percentile(ptp[m], [50, 90, 99])))
    print("threshold  ictal rejected  non-ictal rejected")
    for t in thresholds:
        r = ptp > t
        print(f"  {t:6.0f}  {r[y == 1].mean():13.1%}  {r[y == 0].mean():17.1%}")
    print("relative rule (per subject and channel robust z)  ictal  non-ictal")
    for k in (5, 10, 15):
        r = z.max(axis=1) > k
        print(f"  z > {k:2d}  {r[y == 1].mean():40.1%}  {r[y == 0].mean():9.1%}")


def main() -> None:
    import argparse
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", nargs="*", metavar="P",
                    help="build the corpus caches of these pipelines (default: all)")
    ap.add_argument("--scan-rejection", action="store_true",
                    help="rejection rates of P4 at several thresholds")
    args = ap.parse_args()
    if args.build is not None:
        build_all(args.build or list(PIPELINES))
    if args.scan_rejection:
        scan_rejection()
    if args.build is None and not args.scan_rejection:
        for p in PIPELINES.values():
            print(p.name, p.describe())


if __name__ == "__main__":
    main()
