"""What each artefact removal method does to the signal, ictal and non-ictal windows apart.

Answers the advisor's item 3 (10 October 2026): compare GEDAI's default and conservative
presets with Extended Infomax and AMICA by the power they keep, where in the spectrum
they remove it, and whether they treat ictal and non-ictal windows differently.

Every corpus holds the same windows (checked when it is built), so each cleaned window
is compared with the same window of P1, the 0.5 to 40 Hz signal all three methods start
from. Per window: total power kept (cleaned / P1, summed over channels) and power kept
in five bands (delta 0.5-4, theta 4-8, alpha 8-13, beta 13-30, low gamma 30-40 Hz).
Ictal and non-ictal windows are compared within each recording (the median of each
class per recording, then their difference), so that differences between recordings
cannot pass for differences between the classes; a Wilcoxon signed-rank test over the
recordings that hold both classes tests whether a method removes more from one class.

Usage:  python -m src.p6_signal_effects
Output: results_v2/qc/p6_signal_effects_<dataset>.csv (per window),
        results_v2/qc/p6_signal_effects_summary.csv, ..._bands.csv, ..._ictal.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .config import RESULTS_ROOT
from .datasets import corpus_builder
from .preprocess import PIPELINES

FS = 256
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30),
         "low gamma": (30, 40)}
METHODS = [("P6a", "Extended Infomax", "bp0.5-40_infomax"),
           ("P6b", "GEDAI conservative (used)", "bp0.5-40_gedai-conservative"),
           ("P6b-default", "GEDAI default (not used)", "bp0.5-40_gedai"),
           ("P6c", "AMICA", "bp0.5-40_amica")]
OUT = RESULTS_ROOT / "qc"


def band_power(x: np.ndarray, chunk: int = 256) -> np.ndarray:
    """(window, band) power summed over channels, from the periodogram of each window."""
    f = np.fft.rfftfreq(x.shape[-1], 1 / FS)
    masks = [(f >= lo) & (f < hi) for lo, hi in BANDS.values()]
    out = np.empty((len(x), len(BANDS)))
    for i in range(0, len(x), chunk):
        p = np.abs(np.fft.rfft(x[i:i + chunk].astype(np.float64), axis=-1)) ** 2
        p = p.sum(axis=1)                                    # over channels
        out[i:i + chunk] = np.stack([p[:, m].sum(axis=1) for m in masks], axis=1)
    return out


def load(ds: str, tag: str) -> dict | None:
    build = corpus_builder(ds)
    try:
        return build(verbose=False, signal_fn=lambda *a: (_ for _ in ()).throw(FileNotFoundError),
                     signal_tag=tag)
    except FileNotFoundError:
        return None


def main() -> None:
    rows_sum, rows_band, rows_ict = [], [], []
    for ds in ("chbmit", "siena"):
        base = corpus_builder(ds)(verbose=False, signal_fn=PIPELINES["P1"].apply_signal,
                                  signal_tag=PIPELINES["P1"].for_dataset(ds).signal_tag)
        x1 = base["X"]
        p1 = (x1.astype(np.float64) ** 2).sum(axis=(1, 2))
        b1 = band_power(x1)
        y, rec = base["y"].astype(bool), np.asarray(base["record"]).astype(str)
        per_window = {"record": rec, "ictal": y}
        for key, label, tag in METHODS:
            d = load(ds, tag)
            if d is None:
                print(f"{ds}: no cache {tag}, skipped")
                continue
            assert np.array_equal(d["t0"], base["t0"]) and np.array_equal(d["record"], base["record"])
            x6 = d["X"]
            kept = (x6.astype(np.float64) ** 2).sum(axis=(1, 2)) / p1
            bk = band_power(x6) / b1
            per_window[f"{key}_kept"] = kept
            for cls, m in (("all", np.ones_like(y)), ("ictal", y), ("non-ictal", ~y)):
                v = kept[m]
                rows_sum.append({"dataset": ds, "pipeline": key, "method": label, "windows": cls,
                                 "n": int(m.sum()), "median": np.median(v),
                                 "q25": np.quantile(v, .25), "q75": np.quantile(v, .75),
                                 "share_below_0.5": float((v < 0.5).mean()),
                                 "share_below_0.9": float((v < 0.9).mean())})
                rows_band.append({"dataset": ds, "pipeline": key, "method": label, "windows": cls,
                                  **{b: float(np.median(bk[m, j])) for j, b in enumerate(BANDS)}})
            # ictal vs non-ictal within recordings
            t = pd.DataFrame({"record": rec, "ictal": y, "kept": kept})
            g = t.groupby(["record", "ictal"]).kept.median().unstack()
            g = g.dropna()
            diff = g[True] - g[False]
            stat = wilcoxon(g[True], g[False]) if len(g) > 5 and (diff != 0).any() else None
            rows_ict.append({"dataset": ds, "pipeline": key, "method": label,
                             "recordings_with_both": len(g),
                             "median_ictal_minus_nonictal": float(diff.median()),
                             "share_ictal_lower": float((diff < 0).mean()),
                             "wilcoxon_p": float(stat.pvalue) if stat else np.nan})
            print(f"{ds} {key}: kept median {np.median(kept):.3f}, ictal-nonictal "
                  f"{diff.median():+.4f} over {len(g)} recordings")
        pd.DataFrame(per_window).to_csv(OUT / f"p6_signal_effects_{ds}.csv", index=False)
    pd.DataFrame(rows_sum).to_csv(OUT / "p6_signal_effects_summary.csv", index=False)
    pd.DataFrame(rows_band).to_csv(OUT / "p6_signal_effects_bands.csv", index=False)
    pd.DataFrame(rows_ict).to_csv(OUT / "p6_signal_effects_ictal.csv", index=False)
    print(pd.DataFrame(rows_ict).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
