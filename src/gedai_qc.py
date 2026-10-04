"""GEDAI settings compared by signal preservation only (advisor's request, 4 Oct 2026).

Before P6b runs, the default GEDAI threshold ("auto") is compared with the more
conservative preset ("auto-") on a subset of CHB-MIT. The choice is made on signal
preservation and quality-control criteria only, never on classification performance:

  - retained power per recording (cleaned / band-passed signal power)
  - the same per window, separately for ictal and non-ictal windows: a setting that
    removes more from seizures than from background is removing signal of interest
  - the share of recordings and windows that lose more than half of their power

Both settings see exactly the same windows (the P1 band-pass, then GEDAI); P1 itself is
the reference.

Usage:  python -m src.gedai_qc [--subjects chb01 chb03 ...] [--jobs 6]
        (needs requirements-ica.txt)
Output: results_v2/qc/gedai_settings/  (tables, figure, summary.md)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .chbmit_corpus import build_corpus
from .config import RESULTS_ROOT
from .preprocess import ICA_LOG_ROOT, PIPELINES, _gedai_clean, bandpass

SETTINGS = ("auto", "auto-")                 # default; conservative (noise multiplier 6)
DEFAULT_SUBJECTS = ("chb01", "chb03", "chb05", "chb08", "chb12", "chb15")


class GedaiSignal:
    """Picklable signal step: 0.5-40 Hz band-pass, then GEDAI with one setting; logs the
    retained power of every recording."""

    def __init__(self, noise: str):
        self.noise = noise

    def __call__(self, x: np.ndarray, fs: float, key: str = "") -> np.ndarray:
        t0 = time.time()
        xb = bandpass(x, fs, 0.5, 40.0)
        out, _ = _gedai_clean(xb, fs, noise_multiplier=self.noise)
        d = ICA_LOG_ROOT / f"gedai_qc_{self.noise}"
        d.mkdir(parents=True, exist_ok=True)
        kept = float((out.astype(np.float64) ** 2).sum() / (xb.astype(np.float64) ** 2).sum())
        (d / f"{key}.json").write_text(json.dumps({"power_kept": kept,
                                                   "seconds": round(time.time() - t0, 1)}))
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=list(DEFAULT_SUBJECTS))
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    out = RESULTS_ROOT / "qc" / "gedai_settings"
    out.mkdir(parents=True, exist_ok=True)

    p1 = PIPELINES["P1"]
    ref = build_corpus(subjects=args.subjects, cache=False, verbose=False,
                       signal_fn=p1.apply_signal, signal_tag=p1.signal_tag, n_jobs=args.jobs)
    ref_pow = (ref["X"].astype(np.float64) ** 2).mean(axis=(1, 2))
    y = ref["y"]

    rec_rows, win_rows = [], []
    for noise in SETTINGS:
        t0 = time.time()
        d = build_corpus(subjects=args.subjects, cache=False, verbose=False,
                         signal_fn=GedaiSignal(noise), signal_tag=f"qc_{noise}",
                         n_jobs=args.jobs)
        if not all(np.array_equal(d[k], ref[k]) for k in ("y", "record", "t0")):
            raise SystemExit("windows differ from P1; the comparison would not be paired")
        win_kept = (d["X"].astype(np.float64) ** 2).mean(axis=(1, 2)) / ref_pow
        win_rows.append(pd.DataFrame({"setting": noise, "subject": d["subject"],
                                      "record": d["record"], "ictal": y, "power_kept": win_kept}))
        recs = sorted(set(d["record"]))
        for r in recs:
            g = json.loads((ICA_LOG_ROOT / f"gedai_qc_{noise}" / f"{r}.json").read_text())
            rec_rows.append({"setting": noise, "record": r, **g})
        print(f"{noise}: {len(recs)} recordings, {len(y)} windows, {time.time() - t0:.0f} s")

    rec = pd.DataFrame(rec_rows)
    win = pd.concat(win_rows, ignore_index=True)
    rec.to_csv(out / "recordings.csv", index=False)
    win.to_csv(out / "windows.csv", index=False)

    def summ(v: pd.Series) -> dict:
        q = np.percentile(v, [5, 25, 50, 75])
        return {"n": len(v), "p5": q[0], "p25": q[1], "median": q[2], "p75": q[3],
                "share_below_0.5": float((v < 0.5).mean()),
                "share_below_0.1": float((v < 0.1).mean())}

    rows = []
    for noise in SETTINGS:
        rows.append({"setting": noise, "unit": "recording",
                     **summ(rec[rec.setting == noise].power_kept)})
        for lab, name in ((1, "ictal window"), (0, "non-ictal window")):
            w = win[(win.setting == noise) & (win.ictal == lab)].power_kept
            rows.append({"setting": noise, "unit": name, **summ(w)})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "summary.csv", index=False)

    # ictal vs non-ictal, paired within recordings that contain both
    lines = ["# GEDAI settings: signal preservation (no classification results used)", "",
             f"Subjects: {', '.join(args.subjects)}. Retained power = power after GEDAI / "
             "power of the same 0.5-40 Hz signal.", "",
             "```", tab.round(3).to_string(index=False), "```", ""]
    for noise in SETTINGS:
        w = win[win.setting == noise]
        per = w.groupby(["record", "ictal"]).power_kept.median().unstack()
        per = per.dropna()
        diff = (per[1] - per[0])
        lines.append(f"- {noise}: in {len(per)} recordings with both classes, median retained "
                     f"power is {per[1].median():.3f} for ictal and {per[0].median():.3f} for "
                     f"non-ictal windows; ictal minus non-ictal per recording: median "
                     f"{diff.median():+.3f}, ictal lower in {(diff < 0).mean():.0%} of recordings.")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    bins = np.linspace(0, 1.05, 22)
    for noise in SETTINGS:
        ax[0].hist(rec[rec.setting == noise].power_kept, bins=bins, alpha=0.55, label=noise)
    ax[0].set_xlabel("retained power per recording"); ax[0].set_ylabel("recordings")
    ax[0].legend(title="GEDAI setting")
    data, labels = [], []
    for noise in SETTINGS:
        for lab, name in ((0, "non-ictal"), (1, "ictal")):
            data.append(win[(win.setting == noise) & (win.ictal == lab)].power_kept.clip(0, 1.5))
            labels.append(f"{noise}\n{name}")
    ax[1].boxplot(data, showfliers=False)
    ax[1].set_xticks(range(1, len(labels) + 1), labels, fontsize=8)
    ax[1].set_ylabel("retained power per window")
    fig.suptitle("GEDAI default vs conservative: signal preservation only")
    fig.tight_layout()
    fig.savefig(out / "gedai_settings.png", dpi=150)
    print((out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
