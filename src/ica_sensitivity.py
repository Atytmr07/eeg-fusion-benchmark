"""Why does artefact removal not reproduce across machines? A perturbation test.

Two machines compute the same floating-point operations in a slightly different order
(different SIMD code paths, thread partitions), so their results differ at the level of
rounding. This script asks how much each method amplifies a difference of that size: it
cleans a recording, then cleans it again after multiplying every sample by
(1 + eps * N(0, 1)) with eps = 1e-7, about the relative precision of the float32 signal,
and compares the two cleanings (correlation, relative RMS difference, power kept, the
components removed, GEDAI's threshold). A method that changes its result under such a
perturbation cannot be expected to reproduce across machines.

GEDAI is also run on a full-rank variant (16 of the 18 bipolar channels, dropping one
channel of each closed electrode loop) to test whether the rank deficiency of the bipolar
montage (rank 16) makes its threshold unstable.

Usage:  python -m src.ica_sensitivity [--records ...] [--repeats 3]
Output: results_v2/ica_repro/sensitivity.csv and sensitivity.md
"""
from __future__ import annotations

import argparse
import sys
import time
import zlib

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .config import RESULTS_ROOT

OUT = RESULTS_ROOT / "ica_repro"
EPS = 1e-7
# recordings where the two machines disagreed (Infomax and GEDAI), and some where they agreed
DEFAULT_RECORDS = ["chb01_09.edf", "chb01_25.edf", "chb02_08.edf", "chb07_05.edf",
                   "chb02_16+.edf", "chb03_02.edf", "chb01_11.edf", "chb05_14.edf",
                   "chb13_03.edf"]
# one channel of each closed loop (FP1-F7-T7-P7-O1 vs FP1-F3-C3-P3-O1, same on the right)
DROP_FOR_FULL_RANK = ("P3-O1", "P4-O2")


def gedai_on(x: np.ndarray, fs: float, channels: list[str]):
    """GEDAI as in src/preprocess.py (_gedai_clean), on any subset of the bipolar channels."""
    import gedai
    import mne
    from .preprocess import ALIAS_1020, GEDAI_NOISE
    from gedai.data import get_leadfield_cov_path
    c = mne.read_cov(str(get_leadfield_cov_path()), verbose=False)
    names = [n.lower() for n in c.ch_names]
    elec = sorted({e for ch in channels for e in ch.split("-")})
    idx = [names.index(ALIAS_1020.get(e, e).lower()) for e in elec]
    C = c.data[np.ix_(idx, idx)]
    D = np.zeros((len(channels), len(elec)))
    for i, ch in enumerate(channels):
        a, b = ch.split("-")
        D[i, elec.index(a)], D[i, elec.index(b)] = 1.0, -1.0
    n = len(channels)
    A = np.eye(n) - np.ones((n, n)) / n
    ref = mne.Covariance(A @ D @ C @ D.T @ A.T, list(channels), bads=[], projs=[], nfree=1,
                         verbose=False)
    common = x.astype(np.float64).mean(axis=0, keepdims=True)
    info = mne.create_info(list(channels), fs, ch_types="eeg", verbose=False)
    raw = mne.io.RawArray((x - common) * 1e-6, info, verbose=False)
    g = gedai.Gedai(engine="numpy")
    g.fit_raw(raw, reference_cov=ref, noise_multiplier=GEDAI_NOISE, verbose=False)
    out = g.transform_raw(raw, verbose=False).get_data() * 1e6 + common
    thr = getattr(g, "threshold", None)
    return out, (float(thr) if np.isscalar(thr) else None)


def clean(x, fs, method, rec, channels):
    from .preprocess import ica_clean
    if method == "gedai-fullrank":
        out, thr = gedai_on(x, fs, channels)
        return out, {"threshold": thr}
    out, log = ica_clean(x, fs, method, seed=zlib.crc32(rec.encode()))
    return out, log


def compare(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> dict:
    a, b = a.astype(np.float64), b.astype(np.float64)
    d = a - b
    corr = [np.corrcoef(a[c], b[c])[0, 1] for c in range(len(a))]
    xx = (x.astype(np.float64) ** 2).sum()
    return {"corr_min": float(np.nanmin(corr)),
            "rms_diff_rel": float(np.sqrt((d ** 2).mean() / (a ** 2).mean())),
            "power_kept_a": float((a ** 2).sum() / xx), "power_kept_b": float((b ** 2).sum() / xx)}


def main() -> None:
    from .chbmit import CHB_ROOT, TARGET_CHANNELS, read_edf
    from .preprocess import bandpass
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="*", default=DEFAULT_RECORDS)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--methods", nargs="*",
                    default=["infomax", "gedai", "gedai-fullrank", "amica"])
    args = ap.parse_args()
    rows = []
    full = [c for c in TARGET_CHANNELS if c not in DROP_FOR_FULL_RANK]
    for rec in args.records:
        path = next(CHB_ROOT.glob(f"*/{rec}"))
        x, _, fs = read_edf(path, list(TARGET_CHANNELS))
        x = bandpass(x, fs, 0.5, 40.0)
        for m in args.methods:
            xs = x if m != "gedai-fullrank" else x[[list(TARGET_CHANNELS).index(c) for c in full]]
            t = time.time()
            base, lb = clean(xs, fs, m, rec, full)
            for k in range(args.repeats):
                rng = np.random.default_rng(1000 + k)
                xp = (xs.astype(np.float64) * (1 + EPS * rng.standard_normal(xs.shape))).astype(np.float32)
                pert, lp = clean(xp, fs, m, rec, full)
                r = {"record": rec, "method": m, "repeat": k, **compare(base, pert, xs),
                     "removed_a": lb.get("removed"), "removed_b": lp.get("removed"),
                     "same_removed": lb.get("removed") == lp.get("removed"),
                     "threshold_a": lb.get("threshold"), "threshold_b": lp.get("threshold")}
                rows.append(r)
            print(f"{rec} {m}: {time.time() - t:.0f}s, removed {lb.get('removed', '-')}, "
                  f"kept {rows[-1]['power_kept_a']:.4f}; perturbed kept "
                  + ", ".join(f"{q['power_kept_b']:.4f}" for q in rows[-args.repeats:])
                  + (f"; removed " + ", ".join(str(q['removed_b']) for q in rows[-args.repeats:])
                     if lb.get("removed") is not None else ""), flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "sensitivity.csv", index=False)
    df["changed"] = (df.rms_diff_rel > 1e-3) | (~df.same_removed & df.removed_a.notna())
    summ = df.groupby("method").agg(
        perturbations=("record", "size"), recordings=("record", "nunique"),
        changed=("changed", "sum"),
        max_rms_diff_rel=("rms_diff_rel", "max"), min_corr=("corr_min", "min"),
        max_abs_dkept=("power_kept_b", lambda v: float(
            np.abs(v.values - df.loc[v.index, "power_kept_a"].values).max())))
    summ.to_csv(OUT / "sensitivity_summary.csv")
    (OUT / "sensitivity.md").write_text(
        "# Sensitivity of artefact removal to rounding-level perturbations\n\n"
        f"Each recording cleaned, then cleaned again {args.repeats} times after multiplying "
        f"every sample by (1 + {EPS:g} N(0, 1)). 'changed': relative RMS difference above "
        "1e-3 or different components removed.\n\n" + summ.to_string() + "\n",
        encoding="utf-8")
    print(summ.to_string())


if __name__ == "__main__":
    main()
