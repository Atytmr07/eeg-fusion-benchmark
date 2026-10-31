"""Classical baselines on CHB-MIT: logvar (one feature) and shallow.

The Bonn features in src/baselines.py expect one channel (N, T). CHB-MIT has 18
(N, C, T), so the same features are computed per channel and concatenated: 18 x 1 =
18 dimensions for logvar, 18 x 7 = 126 for shallow.

The frequency bands extend to the 64 Hz ceiling used for CHB-MIT (gamma = 30-64 Hz),
because the largest ictal/interictal power ratio measured in this project is in the
gamma band (see src/chbmit_prep.py).

Splits: the same person-wise LOSO folds as the deep models, but without an inner
validation split. Nothing needs early stopping, so each model is fit on all training
persons of the fold, the same principle as Bonn's use of idx_trval.

Why: on Bonn a single-feature model reached 0.954 macro F1 on the classic task, i.e.
the benchmark was saturated. Whether CHB-MIT is hard or easy cannot be claimed without
the same check.

Results are appended to the perfold.csv of the chbmit_run run given by --run, so the
baselines line up with the deep models fold by fold, and their test probabilities are
added to the run's preds/fold<k>.npz (keys "logvar" and "shallow"), so that AUPRC and
prediction stability can be computed for them as for the deep models. The preprocessing pipeline is read
from that run's meta.json, so the baselines see the same signal-level preprocessing and,
for P4, the same artefact rejection of training windows. Normalisation does not apply:
the features are computed from microvolt signals.

Usage:  python -m src.chbmit_baselines --run loso_grouped
        python -m src.chbmit_baselines --dataset siena --run loso_P0
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from . import baselines
from .chbmit import FS_CHB
from .chbmit_corpus import leave_one_subject_out
from .config import RESULTS_ROOT
from .datasets import DATASETS, corpus_builder
from .evaluate import clinical_metrics, metrics
from .preprocess import PIPELINES, artifact_mask

BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 64)]     # delta .. gamma, 64 Hz ceiling


def _per_channel(fn, X: np.ndarray) -> np.ndarray:
    """Apply a single-channel feature function fn: (N, T) -> (N, k) to every channel
    of X (N, C, T) and concatenate: (N, C*k)."""
    n, c, t = X.shape
    out = [fn(X[:, ch, :]) for ch in range(c)]
    return np.concatenate(out, axis=1)


def logvar_features(X: np.ndarray) -> np.ndarray:
    return _per_channel(baselines.logvar_features, X)


def shallow_features(X: np.ndarray) -> np.ndarray:
    return _per_channel(lambda x: baselines.shallow_features(x, fs=FS_CHB, bands=BANDS), X)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chbmit", choices=list(DATASETS))
    ap.add_argument("--run", required=True,
                    help="chbmit_run run to append to (e.g. loso_grouped)")
    args = ap.parse_args()
    out_dir = RESULTS_ROOT / args.dataset / args.run
    build_corpus = corpus_builder(args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "perfold.csv"

    meta_path = out_dir / "meta.json"
    pipe_name = "P0"
    if meta_path.exists():
        pipe_name = json.loads(meta_path.read_text(encoding="utf-8")).get(
            "pipeline", {}).get("name", "P0")
    pipe = PIPELINES[pipe_name].for_dataset(args.dataset)
    d = build_corpus(verbose=False,
                     signal_fn=pipe.apply_signal if pipe.has_signal_steps else None,
                     signal_tag=pipe.signal_tag)
    X, y, subject, group = d["X"], d["y"], d["subject"], d["group"]
    rejected = (artifact_mask(build_corpus(verbose=False)["X"]) if pipe.reject
                else np.zeros(len(y), bool))
    print(f"pipeline {pipe.name}" + (f", {int(rejected.sum())} training-eligible windows "
                                     f"flagged as artefacts" if rejected.any() else ""))
    info = d["info"]
    hours_of = {s: info["per_subject"][s]["hours"] for s in info["per_subject"]}
    ncls = 2

    print("computing features...")
    t0 = time.time()
    feat_logvar = logvar_features(X)
    feat_shallow = shallow_features(X)
    print(f"  logvar: {feat_logvar.shape}  shallow: {feat_shallow.shape}  "
          f"({time.time() - t0:.1f}s)")

    rows = []
    probs: dict[int, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    splits = list(leave_one_subject_out(group))
    for fi, (key, tr, te) in enumerate(splits):
        tr = tr[~rejected[tr]]
        te_subs = sorted(set(subject[te]))
        hrs = sum(hours_of.get(s, 0.0) for s in te_subs)
        for name, feat in (("logvar", feat_logvar), ("shallow", feat_shallow)):
            t0 = time.time()
            prob = baselines.fit_predict(feat[tr], y[tr], feat[te])
            probs.setdefault(fi, (te, {}))[1][name] = prob.astype(np.float32)
            mm = metrics(y[te], prob, ncls)
            mm.update(clinical_metrics(y[te], prob, hrs))
            mm.update(model=name, fold=fi, test_subjects=",".join(te_subs),
                      n_test=len(te), n_test_ictal=int(y[te].sum()),
                      params=feat.shape[1], sec=round(time.time() - t0, 2),
                      best_epoch=-1, epochs_run=0, val_f1=float("nan"))
            rows.append(mm)
        print(f"[{fi + 1}/{len(splits)}] {','.join(te_subs)}: "
              f"logvar f1={rows[-2]['f1_macro']:.3f}  shallow f1={rows[-1]['f1_macro']:.3f}")

    new_df = pd.DataFrame(rows)

    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        # Refuse to mix fold structures (e.g. appending 23 person-wise folds to a
        # 24-fold case-wise run): fold indices would no longer mean the same test set.
        if old_df.fold.nunique() != len(splits):
            raise SystemExit(f"{csv_path} has {old_df.fold.nunique()} folds but this "
                             f"split has {len(splits)}; not appending to results "
                             f"produced with a different grouping")
        backup = out_dir / "perfold_before_baselines.csv"
        if not backup.exists():
            shutil.copy(csv_path, backup)
            print(f"backup written: {backup}")
        old_df = old_df[~old_df.model.isin(("logvar", "shallow"))]     # safe to re-run
        merged = pd.concat([old_df, new_df], ignore_index=True)
    else:
        merged = new_df

    merged.to_csv(csv_path, index=False)

    # Add the baseline probabilities to the run's stored predictions, fold by fold, only
    # where the stored test indices are exactly this fold's.
    added = 0
    for fi, (te, pr) in probs.items():
        f = out_dir / "preds" / f"fold{fi}.npz"
        if not f.exists():
            continue
        z = dict(np.load(f))
        if not np.array_equal(z["idx_te"], te):
            raise SystemExit(f"{f}: stored test indices differ from this split")
        z.update(pr)
        np.savez(f, **z)
        added += 1
    if added:
        print(f"baseline probabilities added to {added} prediction files")
    print(f"\nsaved: {csv_path} ({len(merged)} rows)")
    print(new_df.groupby("model")[["f1_macro", "auc", "brier"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
