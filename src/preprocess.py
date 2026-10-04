"""Preprocessing multiverse: the pipelines P0-P6 (P6 in three variants).

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
    P6a   0.5-40 Hz    none    Extended Infomax ICA          z-score         aggressive cleaning
    P6b   0.5-40 Hz    none    GEDAI                         z-score         aggressive cleaning
    P6c   0.5-40 Hz    none    AMICA                         z-score         aggressive cleaning

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

P6 compares three artefact removal methods on the same 0.5-40 Hz signal, so that the
effect of the algorithm can be separated from the rest of the pipeline:

  P6a, P6c  Extended Infomax and AMICA, decomposed with the same settings (16
            components, fitted on every 8th sample) and cleaned with the SAME automatic
            rule: MNE's ocular component detection (find_bads_eog, |z| > 3) with the four
            frontal FP channels as EOG proxies, since CHB-MIT has no EOG channel.
            Muscle components are not removed: the gamma band carries the strongest
            ictal signal in this corpus (src/chbmit_prep.py).
  P6b       GEDAI, which is not an ICA: it selects what to remove itself, against a
            leadfield reference (SENSAI). The same component rule therefore cannot be
            applied to it; its default threshold is used.

16 components, not 18: the 18 bipolar channels contain two closed electrode loops
(FP1-F7-T7-P7-O1 against FP1-F3-C3-P3-O1, and the right-hand pair), so the data have
rank 16; the two remaining singular values are quantisation noise.

Every recording's decomposition is logged (components removed, power kept, time) under
data/ica_logs/<signal tag>/. ICA and GEDAI need mne, jamica and gedai, which are not
in requirements.txt (see requirements-ica.txt); they are only needed to BUILD the P6
caches, not to train on them.
"""
from __future__ import annotations

import json
import time
import zlib
from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .chbmit import TARGET_CHANNELS
from .chbmit_prep import notch_filter
from .config import DATA_ROOT

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

# P6: decomposition settings, shared by Infomax and AMICA.
ICA_N_COMPONENTS = 16       # rank of the bipolar montage (see the module docstring)
ICA_FIT_DECIM = 8           # fit on every 8th sample. ICA models the instantaneous mixing
                            # across channels, which subsampling leaves intact; at 1 h per
                            # recording this still gives 115,200 samples for 256 weights
ICA_MAX_ITER = {"infomax": 1000, "amica": 300}   # AMICA: same components removed at 300
                                                 # and 1000 iterations in a trial, 3x faster
EOG_PROXY = ("FP1-F7", "FP1-F3", "FP2-F4", "FP2-F8")
EOG_Z = 3.0                 # find_bads_eog threshold (MNE default)
# GEDAI's threshold preset. "auto-" (noise multiplier 6, more conservative than the
# default "auto") was chosen on signal preservation only, before any classification run
# (src/gedai_qc.py, results_v2/qc/gedai_settings, 4 October 2026): on six subjects the
# default removed more than half the power of 39 % of recordings, "auto-" of 17 %. The
# default's cache is kept under the tag bp0.5-40_gedai for the robustness report.
GEDAI_NOISE = "auto-"
GEDAI_TAG = {"auto": "gedai", "auto-": "gedai-conservative"}
ICA_LOG_ROOT = DATA_ROOT / "ica_logs"
ALIAS_1020 = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


@dataclass(frozen=True)
class Pipeline:
    name: str
    band: tuple[float, float] | None = None
    notch_hz: float | None = None
    ica: str | None = None          # P6: "infomax", "gedai" or "amica"
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
            parts.append(GEDAI_TAG[GEDAI_NOISE] if self.ica == "gedai" else self.ica)
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
            x, log = ica_clean(x, fs, self.ica, seed=zlib.crc32(key.encode()))
            if key:
                d = ICA_LOG_ROOT / self.signal_tag
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{key}.json").write_text(json.dumps(log), encoding="utf-8")
        return x.astype(np.float32)

    def describe(self) -> dict:
        d = asdict(self) | {"signal_tag": self.signal_tag}
        if self.reject:
            d["reject_rule"] = {"flat_std_uv": FLAT_STD_UV, "flat_run_s": FLAT_RUN_S,
                                "detected_on": "unfiltered signal",
                                "scope": "train and validation"}
        if self.ica in ("infomax", "amica"):
            d["ica_settings"] = {"n_components": ICA_N_COMPONENTS, "fit_decim": ICA_FIT_DECIM,
                                 "max_iter": ICA_MAX_ITER[self.ica],
                                 "rule": "find_bads_eog", "eog_proxy": list(EOG_PROXY),
                                 "z": EOG_Z}
        elif self.ica == "gedai":
            d["ica_settings"] = {"noise_multiplier": GEDAI_NOISE,
                                 "reference": "fsaverage leadfield covariance mapped to "
                                              "the bipolar montage"}
        return d


PIPELINES = {p.name: p for p in (
    Pipeline("P0"),
    Pipeline("P1", band=(0.5, 40.0)),
    Pipeline("P2", band=(0.5, 70.0), notch_hz=60.0),
    Pipeline("P3", band=(1.0, 40.0)),
    Pipeline("P4", band=(0.5, 40.0), reject=True),
    Pipeline("P5", band=(0.5, 40.0), norm="robust"),
    Pipeline("P6a", band=(0.5, 40.0), ica="infomax"),
    Pipeline("P6b", band=(0.5, 40.0), ica="gedai"),
    Pipeline("P6c", band=(0.5, 40.0), ica="amica"),
)}


# --- Signal level --------------------------------------------------------------------

def bandpass(x: np.ndarray, fs: float, lo: float, hi: float,
             order: int = BUTTER_ORDER) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last axis."""
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x.astype(np.float64), axis=-1).astype(np.float32)


def _mne_raw(x: np.ndarray, fs: float):
    import mne
    info = mne.create_info(list(TARGET_CHANNELS), fs, ch_types="eeg", verbose=False)
    return mne.io.RawArray(x.astype(np.float64) * 1e-6, info, verbose=False)   # volts


def ica_clean(x: np.ndarray, fs: float, method: str, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Artefact removal on a band-passed recording x (channel, time) in microvolts.

    Returns the cleaned signal and a log: what was removed, how much of the signal's
    power was kept, and how long it took.
    """
    t0 = time.time()
    if method == "gedai":
        out, log = _gedai_clean(x, fs)
    else:
        import mne
        mne.set_log_level("ERROR")
        raw = _mne_raw(x, fs)
        if method == "infomax":
            ica = mne.preprocessing.ICA(n_components=ICA_N_COMPONENTS, method="infomax",
                                        fit_params=dict(extended=True),
                                        max_iter=ICA_MAX_ITER["infomax"],
                                        random_state=seed % (2 ** 32))
            ica.fit(raw, decim=ICA_FIT_DECIM)
        elif method == "amica":
            from jamica import AmicaICA
            ica = AmicaICA(n_components=ICA_N_COMPONENTS, decim=ICA_FIT_DECIM,
                           max_iter=ICA_MAX_ITER["amica"],
                           random_state=seed % (2 ** 32)).fit(raw).models_[0]
        else:
            raise ValueError(f"unknown P6 method: {method}")
        bad: set[int] = set()
        for ch in EOG_PROXY:                       # the same rule for both decompositions
            idx, _ = ica.find_bads_eog(raw, ch_name=ch, threshold=EOG_Z)
            bad |= {int(i) for i in idx}
        ica.exclude = sorted(bad)
        out = (ica.apply(raw.copy()).get_data() * 1e6).astype(np.float32)
        log = {"removed": ica.exclude, "n_removed": len(ica.exclude),
               "share_removed": len(ica.exclude) / ICA_N_COMPONENTS}
    xd = x.astype(np.float64)
    log |= {"method": method,
            "power_kept": float((out.astype(np.float64) ** 2).sum() / (xd ** 2).sum()),
            "seconds": round(time.time() - t0, 1)}
    return out, log


def _gedai_bipolar_cov():
    """GEDAI's leadfield reference covariance mapped onto the 18 bipolar channels.

    The bundled covariance C belongs to a referential 10-05 leadfield; a bipolar channel
    is a difference of electrodes, x_bip = D x_ref, so its reference is D C D^T (a
    difference cancels the common reference). GEDAI average-references its input, so the
    same projection A is applied to the reference to keep both in one space.
    """
    import mne
    from gedai.data import get_leadfield_cov_path

    c = mne.read_cov(str(get_leadfield_cov_path()), verbose=False)
    names = [n.lower() for n in c.ch_names]
    elec = sorted({e for ch in TARGET_CHANNELS for e in ch.split("-")})
    idx = [names.index(ALIAS_1020.get(e, e).lower()) for e in elec]
    C = c.data[np.ix_(idx, idx)]
    D = np.zeros((len(TARGET_CHANNELS), len(elec)))
    for i, ch in enumerate(TARGET_CHANNELS):
        a, b = ch.split("-")
        D[i, elec.index(a)], D[i, elec.index(b)] = 1.0, -1.0
    n = len(TARGET_CHANNELS)
    A = np.eye(n) - np.ones((n, n)) / n
    return mne.Covariance(A @ D @ C @ D.T @ A.T, list(TARGET_CHANNELS), bads=[], projs=[],
                          nfree=1, verbose=False)


def _gedai_clean(x: np.ndarray, fs: float,
                 noise_multiplier: float | str = GEDAI_NOISE) -> tuple[np.ndarray, dict]:
    """GEDAI on the bipolar signal. The mean over channels is removed before (GEDAI would
    otherwise average-reference the data itself) and added back after, so the output stays
    in the bipolar montage; that one common component is left uncleaned."""
    import gedai

    common = x.astype(np.float64).mean(axis=0, keepdims=True)
    raw = _mne_raw(x - common, fs)
    g = gedai.Gedai(engine="numpy")
    g.fit_raw(raw, reference_cov=_gedai_bipolar_cov(), noise_multiplier=noise_multiplier,
              verbose=False)
    out = g.transform_raw(raw, verbose=False).get_data() * 1e6 + common
    thr = getattr(g, "threshold", None)
    return out.astype(np.float32), {"threshold": float(thr) if np.isscalar(thr) else None}


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

def build_all(names: list[str], jobs: int = 1, dataset: str = "chbmit") -> None:
    """Build the corpus cache of every distinct signal-level step set among `names`,
    and check that each holds exactly the same windows as P0. jobs > 1 processes the
    recordings of each subject in parallel worker processes."""
    from .datasets import corpus_builder

    build_corpus = corpus_builder(dataset)
    extra = {"n_jobs": jobs} if dataset == "chbmit" else {}    # Siena builds serially
    ref = build_corpus(verbose=False)
    done: set[str] = set()
    for name in names:
        p = PIPELINES[name]
        if not p.has_signal_steps or p.signal_tag in done:
            continue
        done.add(p.signal_tag)
        print(f"\n{name}: building corpus '{p.signal_tag}' (shared by "
              f"{', '.join(q.name for q in PIPELINES.values() if q.signal_tag == p.signal_tag)})")
        d = build_corpus(signal_fn=p.apply_signal, signal_tag=p.signal_tag, **extra)
        same = all(np.array_equal(d[k], ref[k]) for k in
                   ("y", "subject", "seizure_id", "record", "t0"))
        if not same or d["X"].shape != ref["X"].shape:
            raise SystemExit(f"{name}: windows differ from P0; the pipelines would not "
                             f"be paired")
        print(f"{name}: same {len(d['y'])} windows as P0, checked")
        if p.ica:
            summarize_ica_log(p.signal_tag)


def summarize_ica_log(signal_tag: str) -> None:
    """What the artefact removal did across all recordings of one P6 variant."""
    logs = [json.loads(f.read_text(encoding="utf-8"))
            for f in sorted((ICA_LOG_ROOT / signal_tag).glob("*.json"))]
    if not logs:
        print(f"{signal_tag}: no ICA log found")
        return
    kept = np.array([g["power_kept"] for g in logs])
    sec = np.array([g["seconds"] for g in logs])
    line = (f"{signal_tag}: {len(logs)} recordings, power kept median {np.median(kept):.3f} "
            f"(range {kept.min():.3f}-{kept.max():.3f}), {sec.sum() / 3600:.1f} h of compute")
    if "n_removed" in logs[0]:
        n = np.array([g["n_removed"] for g in logs])
        line += (f", components removed per recording mean {n.mean():.2f} "
                 f"(0: {np.mean(n == 0):.0%}, max {n.max()})")
    print(line)


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
    ap.add_argument("--dataset", default="chbmit", choices=["chbmit", "siena"])
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes when building (useful for P6)")
    ap.add_argument("--scan-rejection", action="store_true",
                    help="rejection rates of the P4 rule and of amplitude criteria")
    args = ap.parse_args()
    if args.build is not None:
        build_all(args.build or list(PIPELINES), jobs=args.jobs, dataset=args.dataset)
    if args.scan_rejection:
        scan_rejection()
    if args.build is None and not args.scan_rejection:
        for p in PIPELINES.values():
            print(p.name, p.describe())


if __name__ == "__main__":
    main()
