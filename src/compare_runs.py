"""Are two CHB-MIT runs identical? Compares per fold metrics and stored predictions.

Used to check reproducibility: the same fold run twice on the GPU, or a pipeline
refactor against the code it replaced. Wall clock time is reported, not compared.

When two runs are not identical, the per model table says how far apart they are:
the macro F1 difference and the share of test windows given the same class. Comparing
a CPU and a GPU run with the same seeds against two CPU runs with different seeds
(--repeat) tells whether the device matters more than the seed does.

Usage:  python -m src.compare_runs results_v2/chbmit/gpu_a results_v2/chbmit/gpu_b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

IGNORE = {"sec"}                     # wall clock time differs by nature


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()
    a, b = Path(args.a), Path(args.b)
    da, db = pd.read_csv(a / "perfold.csv"), pd.read_csv(b / "perfold.csv")
    key = ["fold", "model"]
    m = da.merge(db, on=key, suffixes=("_a", "_b"))
    cols = [c for c in da.columns if c not in IGNORE and c not in key and c in db.columns]
    print(f"rows: {len(da)} vs {len(db)}, {len(m)} matched on (fold, model)")

    same = True
    for c in cols:
        x, y = m[f"{c}_a"], m[f"{c}_b"]
        if pd.api.types.is_numeric_dtype(x):
            diff = (x - y).abs()
            eq = bool(((diff == 0) | (x.isna() & y.isna())).all())
            if not eq:
                print(f"  differs: {c:14s} max |diff| {diff.max():.4g}")
        else:
            eq = bool((x.fillna("") == y.fillna("")).all())
            if not eq:
                print(f"  differs: {c}")
        same &= eq

    n_pred = n_diff = 0
    agree: dict[str, list[float]] = {}
    for f in sorted((a / "preds").glob("*.npz")):
        g = b / "preds" / f.name
        if not g.exists():
            continue
        za, zb = np.load(f), np.load(g)
        n_pred += 1
        if set(za.files) != set(zb.files) or any(not np.array_equal(za[k], zb[k])
                                                 for k in za.files):
            n_diff += 1
        if not np.array_equal(za["idx_te"], zb["idx_te"]):
            continue
        for k in set(za.files) & set(zb.files) - {"idx_te", "y_te"}:
            agree.setdefault(k, []).append(
                float((za[k].argmax(1) == zb[k].argmax(1)).mean()))
    print(f"prediction files: {n_pred} compared, {n_diff} differ")
    same &= n_diff == 0 and n_pred > 0

    if "f1_macro_a" in m:
        tab = m.groupby("model")[["f1_macro_a", "f1_macro_b"]].mean()
        tab["abs_diff"] = (m.assign(d=(m.f1_macro_a - m.f1_macro_b).abs())
                           .groupby("model")["d"].mean())
        tab["same_class"] = pd.Series({k: np.mean(v) for k, v in agree.items()})
        print("\nper model (mean over matched folds): macro F1 in a and b, mean |difference|,"
              "\nshare of test windows given the same class")
        print(tab.round(4).to_string())
        print(f"mean |F1 difference| over models: {tab['abs_diff'].mean():.4f}")

    for name, d in (("a", da), ("b", db)):
        if "sec" in d:
            print(f"training time {name}: {d['sec'].sum() / 60:.1f} minutes")
    print("\nIDENTICAL" if same else "\nNOT IDENTICAL")


if __name__ == "__main__":
    main()
