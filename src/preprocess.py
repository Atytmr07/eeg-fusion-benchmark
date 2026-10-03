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
    P4    0.5-40 Hz    none    technical artefact rejection  z-score         artefact rejection
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

Artefact rejection (P4) targets technical artefacts only (flat channels and signal
dropouts, see FLAT_STD_UV) and not amplitude: on CHB-MIT every amplitude criterion
removes seizures selectively, because high amplitude is part of ictal EEG. Rejected
windows leave TRAINING and VALIDATION only; the test set is the same as in every other
pipeline, so the pipelines stay paired fold by fold. The rejection rate is recorded
separately for ictal and non-ictal windows.

Open choices, pending the advisor (recorded in every run's meta.json):
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

# P4: technical artefacts. A window is rejected if a channel is flat (standard deviation
# below FLAT_STD_UV over the window: a disconnected electrode) or holds one constant
# value for at least FLAT_RUN_S seconds (a signal dropout; in CHB-MIT these sit at 0 uV).
# Detected on the UNFILTERED signal, since filtering smears a flat segment into ringing.
#
# Why not amplitude (python -m src.preprocess --scan-rejection, CHB-MIT): the median
# worst-channel peak-to-peak amplitude is 482 uV in non-ictal and 890 uV in ictal
# windows, so every amplitude threshold removes seizures selectively (500 uV: 72 % of
# ictal vs 48 % of non-ictal windows; 1000 uV: 44 % vs 12 %; per subject robust z > 10:
# 21 % vs 4.5 %). Clipping is not a usable criterion either: the declared 12-bit range
# (e.g. +-800 uV) is exceeded by the stored int16 data, and no recording piles up at a
# limit. The rule below rejects 3.3 % of ictal and 0.8 % of non-ictal windows. The
# higher ictal share is not physiological: the 0.5 s before a dropout have the same
# amplitude in both classes (median about 40 uV); dropouts occur in a few recordings
# that also contain seizures.
FLAT_STD_UV = 1.0
FLAT_RUN_S = 0.5

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
    reject: bool = False            # P4: technical artefact rejection (artifact_mask)
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
        if self.reject:
            d["reject_rule"] = {"flat_std_uv": FLAT_STD_UV, "flat_run_s": FLAT_RUN_S,
                                "detected_on": "unfiltered signal",
                                "scope": "train and validation"}
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
    Pipeline("P4", band=(0.5, 40.0), reject=True),
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

def longest_constant_run(x: np.ndarray) -> np.ndarray:
    """Longest run of identical consecutive samples, per window and channel, counted in
    sample steps. x: (window, channel, time)."""
    same = np.diff(x, axis=-1) == 0
    best = np.zeros(x.shape[:2], np.int32)
    cur = np.zeros(x.shape[:2], np.int32)
    for k in range(same.shape[-1]):
        cur = np.where(same[..., k], cur + 1, 0)
        np.maximum(best, cur, out=best)
    return best


def artifact_mask(x_raw: np.ndarray, fs: float = 256.0) -> np.ndarray:
    """True for windows with a technical artefact: a flat channel or a constant segment
    of at least FLAT_RUN_S seconds on any channel. x_raw: (window, channel, time) in
    microvolts, UNFILTERED (the P0 corpus, which holds the same windows)."""
    flat = (x_raw.std(axis=-1) < FLAT_STD_UV).any(axis=-1)
    dropout = (longest_constant_run(x_raw) >= int(FLAT_RUN_S * fs)).any(axis=-1)
    return flat | dropout


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
    """Share of ictal and non-ictal windows rejected by the P4 rule, and, for comparison,
    by amplitude thresholds on the band-passed signal."""
    from .chbmit_corpus import build_corpus

    raw = build_corpus(verbose=False)
    r = artifact_mask(raw["X"])
    yr = raw["y"]
    print(f"P4 rule (flat channel or constant segment >= {FLAT_RUN_S:g} s, unfiltered): "
          f"ictal {r[yr == 1].mean():.1%} ({int(r[yr == 1].sum())}), "
          f"non-ictal {r[yr == 0].mean():.1%} ({int(r[yr == 0].sum())})")
    del raw
    print("\nfor comparison, amplitude criteria (not used):")
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
                    help="rejection rates of the P4 rule and of amplitude criteria")
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
