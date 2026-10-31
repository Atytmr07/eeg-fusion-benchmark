"""Does P6's artefact removal reproduce across machines and thread settings?

The event metrics showed that the lab PC, with the same code, package versions and
seeds, cleaned the P6 signals differently from the laptop that built the caches
(events.py's window check). This script measures how different the outputs are and
helps find why.

  --dump    Clean a fixed sample of CHB-MIT recordings with Infomax, GEDAI and AMICA and
            store, per recording and method: the cleaned signal of the first 10 minutes,
            the Welch PSD of the whole cleaned recording, the removed components, the
            retained power, and the run time. Also stores the machine
            and software environment (CPU, versions, BLAS, thread pools).
            --threads N limits the BLAS/OpenMP thread pools to N (default: unlimited),
            to test whether the thread count alone changes the result.
  --compare Two dumps (folders under results_v2/ica_repro): per recording and method,
            the correlation and relative RMS difference of the cleaned signals, the PSD
            change, and whether the same components were removed.

Usage:
    python -m src.ica_repro --dump --tag laptop
    python -m src.ica_repro --dump --tag laptop_t1 --threads 1
    python -m src.ica_repro --compare laptop lab
Output: results_v2/ica_repro/<tag>/ and results_v2/ica_repro/compare_<a>_<b>.csv/.md
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import zlib
from contextlib import nullcontext

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .config import RESULTS_ROOT

OUT = RESULTS_ROOT / "ica_repro"
METHODS = ("infomax", "gedai", "amica")
N_RECORDINGS = 8
SAMPLE_SEED = 20261010
SEGMENT_S = 600


def sample_recordings(n: int = N_RECORDINGS) -> list[str]:
    """A fixed sample of recordings with corpus windows, one per subject drawn."""
    from .chbmit_corpus import build_corpus
    rec = build_corpus(verbose=False)["record"]
    names = sorted(set(map(str, rec)))
    by_subject: dict[str, list[str]] = {}
    for name in names:
        by_subject.setdefault(name.split("_")[0], []).append(name)
    rng = np.random.default_rng(SAMPLE_SEED)
    subjects = sorted(rng.choice(sorted(by_subject), N_RECORDINGS, replace=False))[:n]
    return [by_subject[s][int(rng.integers(len(by_subject[s])))] for s in subjects]


def environment(threads: int | None) -> dict:
    import importlib.metadata as md
    from threadpoolctl import threadpool_info
    pk = {}
    for p in ("numpy", "scipy", "mne", "gedai", "jamica", "scikit-learn", "torch"):
        try:
            pk[p] = md.version(p)
        except md.PackageNotFoundError:
            pk[p] = None
    pools = [{k: d.get(k) for k in ("internal_api", "num_threads", "version",
                                     "threading_layer", "architecture")}
             for d in threadpool_info()]
    return {"machine": platform.node(), "cpu": platform.processor(),
            "platform": platform.platform(), "python": platform.python_version(),
            "packages": pk, "thread_limit": threads, "thread_pools": pools}


def dump(tag: str, threads: int | None, n: int = N_RECORDINGS,
         methods: tuple[str, ...] = METHODS, records: list[str] | None = None) -> None:
    from scipy.signal import welch
    from threadpoolctl import threadpool_limits

    from .chbmit import CHB_ROOT, TARGET_CHANNELS, read_edf
    from .preprocess import bandpass, ica_clean

    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    recs = records or sample_recordings(n)
    env = environment(threads) | {"recordings": recs, "segment_s": SEGMENT_S}
    (out / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(f"{tag}: {len(recs)} recordings, threads {threads or 'unlimited'}", flush=True)
    rows = []
    limit = threadpool_limits(limits=threads) if threads else nullcontext()
    with limit:
        for r in recs:
            path = next(CHB_ROOT.glob(f"*/{r}"))    # chb17a/b/c files live in chb17/
            x, _, fs = read_edf(path, list(TARGET_CHANNELS))
            x = bandpass(x, fs, 0.5, 40.0)                    # P1, as in P6
            seg = int(SEGMENT_S * fs)
            for m in methods:
                t = time.time()
                y, log = ica_clean(x, fs, m, seed=zlib.crc32(r.encode()))
                f, psd = welch(y.astype(np.float64), fs=fs, nperseg=int(4 * fs), axis=-1)
                np.savez_compressed(out / f"{m}_{r}.npz", segment=y[:, :seg], psd=psd, f=f)
                rows.append({"record": r, "method": m, **{k: v for k, v in log.items()
                                                          if k != "method"},
                             "wall_s": round(time.time() - t, 1)})
                print(f"  {r} {m}: removed {log.get('removed', '-')}, power kept "
                      f"{log['power_kept']:.4f} ({time.time() - t:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    print(f"written: {out}")


def md(df: pd.DataFrame, index: bool = True) -> str:
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    fmt = lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)
    head = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    body = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(head + body)


def compare(a: str, b: str) -> None:
    sa = pd.read_csv(OUT / a / "summary.csv").set_index(["record", "method"])
    sb = pd.read_csv(OUT / b / "summary.csv").set_index(["record", "method"])
    rows = []
    for (r, m) in sa.index.intersection(sb.index):
        da, db = np.load(OUT / a / f"{m}_{r}.npz"), np.load(OUT / b / f"{m}_{r}.npz")
        xa, xb = da["segment"].astype(np.float64), db["segment"].astype(np.float64)
        diff = xa - xb
        corr = [np.corrcoef(xa[c], xb[c])[0, 1] for c in range(len(xa))]
        psd_change = np.abs(np.log10(db["psd"][:, 1:] / da["psd"][:, 1:]))
        row = {"record": r, "method": m,
               "identical": bool(np.array_equal(xa, xb)),
               "corr_min": float(np.min(corr)), "corr_median": float(np.median(corr)),
               "rms_diff_rel": float(np.sqrt((diff ** 2).mean() / (xa ** 2).mean())),
               "max_diff_rel": float(np.abs(diff).max() / np.abs(xa).max()),
               "psd_change_db_median": float(10 * np.median(psd_change)),
               "psd_change_db_max": float(10 * psd_change.max()),
               "power_kept_a": sa.at[(r, m), "power_kept"],
               "power_kept_b": sb.at[(r, m), "power_kept"]}
        if "removed" in sa.columns and isinstance(sa.at[(r, m), "removed"], str):
            row |= {"removed_a": sa.at[(r, m), "removed"], "removed_b": sb.at[(r, m), "removed"],
                    "same_count": sa.at[(r, m), "n_removed"] == sb.at[(r, m), "n_removed"]}
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["method", "record"])
    name = f"compare_{a}_{b}"
    df.to_csv(OUT / f"{name}.csv", index=False)
    summ = df.groupby("method").agg(
        recordings=("record", "size"), identical=("identical", "sum"),
        corr_min=("corr_min", "min"), corr_median=("corr_median", "median"),
        rms_diff_rel_median=("rms_diff_rel", "median"), rms_diff_rel_max=("rms_diff_rel", "max"),
        psd_change_db_median=("psd_change_db_median", "median"),
        psd_change_db_max=("psd_change_db_max", "max"))
    if "same_count" in df:
        summ["same_component_count"] = df.groupby("method").same_count.sum()
    envs = {t: json.loads((OUT / t / "environment.json").read_text(encoding="utf-8"))
            for t in (a, b)}
    lines = [f"# P6 reproducibility: {a} vs {b}", "",
             f"{len(set(df.record))} CHB-MIT recordings, cleaned signal of the first "
             f"{SEGMENT_S // 60} minutes; PSD over the whole recording.", "",
             md(summ), "", "## Per recording", "",
             md(df, index=False), "", "## Environments", ""]
    for t, e in envs.items():
        lines += [f"**{t}**: {e['machine']}, {e['cpu']}, threads {e['thread_limit'] or 'unlimited'}, "
                  + ", ".join(f"{k} {v}" for k, v in e["packages"].items()), "",
                  "thread pools: " + "; ".join(f"{p['internal_api']} {p['num_threads']} "
                                               f"({p.get('architecture')})"
                                               for p in e["thread_pools"]), ""]
    (OUT / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
    print(summ.round(4).to_string())
    print(f"written: {OUT / name}.md")


def compare_logs(a: str, b: str, name: str) -> None:
    """Per recording, the ICA logs of two runs over the whole corpus (data/ica_logs as
    written when a cache was built or when events.py recomputed the cleaning): whether the
    same components were removed and how much the retained power differs."""
    from pathlib import Path
    rows = []
    for tag, method in (("bp0.5-40_infomax", "infomax"), ("bp0.5-40_gedai-conservative", "gedai"),
                        ("bp0.5-40_amica", "amica")):
        da, db = Path(a) / tag, Path(b) / tag
        common = sorted({f.name for f in da.glob("*.json")} & {f.name for f in db.glob("*.json")})
        for f in common:
            x = json.loads((da / f).read_text(encoding="utf-8"))
            y = json.loads((db / f).read_text(encoding="utf-8"))
            rows.append({"method": method, "record": f[:-5],
                         "removed_a": x.get("removed"), "removed_b": y.get("removed"),
                         "same_removed": x.get("removed") == y.get("removed"),
                         "power_kept_a": x["power_kept"], "power_kept_b": y["power_kept"],
                         "abs_dpower": abs(x["power_kept"] - y["power_kept"]),
                         "threshold_a": x.get("threshold"), "threshold_b": y.get("threshold")})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"logs_{name}.csv", index=False)
    summ = df.groupby("method").agg(
        recordings=("record", "size"),
        removed_differs=("same_removed", lambda v: int((~v).sum())),
        dpower_gt_1e6=("abs_dpower", lambda v: int((v > 1e-6).sum())),
        dpower_gt_1e3=("abs_dpower", lambda v: int((v > 1e-3).sum())),
        dpower_gt_0p1=("abs_dpower", lambda v: int((v > 0.1).sum())),
        dpower_max=("abs_dpower", "max"))
    summ.to_csv(OUT / f"logs_{name}_summary.csv")
    print(summ.to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--tag", default=platform.node())
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--compare-logs", nargs=3, metavar=("DIR_A", "DIR_B", "NAME"),
                    help="two ica_logs folders, per recording over the whole corpus")
    ap.add_argument("--records", nargs="*", default=None,
                    help="these recordings instead of the fixed sample")
    ap.add_argument("--n", type=int, default=N_RECORDINGS, help="first n recordings (tests)")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    args = ap.parse_args()
    if args.compare_logs:
        compare_logs(*args.compare_logs)
    elif args.compare:
        compare(*args.compare)
    elif args.dump:
        dump(args.tag, args.threads, args.n, tuple(args.methods), args.records)
    else:
        ap.error("--dump or --compare")


if __name__ == "__main__":
    main()
