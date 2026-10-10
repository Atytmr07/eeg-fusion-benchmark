"""Cross-dataset experiment: train on CHB-MIT, test on Siena.

Separate from the multiverse (src/multiverse.py), as the advisor asked. Every model is
trained once on all 23 CHB-MIT persons, with the same person-wise inner validation
split as the LOSO runs (20 percent of the persons, here five, held out for early
stopping and the score fusion weight), and tested on every Siena window. Results are reported per Siena patient, so
the 14 patients are the paired units of the statistics.

The training settings, models, seeds and preprocessing pipelines are those of
src/chbmit_run.py; both corpora have the same form (18 bipolar channels, 256 Hz, 10 s
windows, the same labelling and subsampling), so the inputs are directly comparable.
The two corpora also differ in patient age, seizure types, recording structure and
acquisition, so a drop in performance is a domain shift and is not attributed to age.

Statistics differ from LOSO: one training set serves every test patient, so the
training-set overlap that the Nadeau-Bengio correction addresses does not arise here.
Paired comparisons across the 14 test patients are ordinary paired tests; their
weakness is the small number of patients, not dependence between folds.

Usage:  python -m src.cross_dataset --pipeline P0 --threads 16 [--repeat r] [--device cuda]
Output: results_v2/cross/chbmit_to_siena_<pipeline>[_r<k>][_cuda]/
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from . import baselines
from .chbmit_baselines import logvar_features, shallow_features
from .chbmit_prep import prepare
from .chbmit_run import DEFAULT_MODELS, inner_split, train_fold
from .config import RESULTS_ROOT, fold_seed, runtime_env
from .datasets import corpus_builder
from .evaluate import clinical_metrics, metrics
from .preprocess import PIPELINES, artifact_mask
from .train import best_blend_weight

CROSS_FOLD = 10_000        # seed index of the single cross-dataset training run, kept
                           # apart from the LOSO fold indices 0-22


def load(pipe, dataset: str, subjects=None) -> dict:
    pipe = pipe.for_dataset(dataset)          # P2: each dataset's own mains frequency
    return corpus_builder(dataset)(subjects=subjects, verbose=pipe.has_signal_steps,
                                   signal_fn=pipe.apply_signal if pipe.has_signal_steps else None,
                                   signal_tag=pipe.signal_tag)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default="P0", choices=list(PIPELINES))
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--siena-subjects", nargs="*", default=None,
                    help="restrict the test set (for quick checks only)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t_start = time.time()
    pipe = PIPELINES[args.pipeline]

    src, tgt = load(pipe, "chbmit"), load(pipe, "siena", args.siena_subjects)
    n_src = len(src["y"])
    X = np.concatenate([src["X"], tgt["X"]])
    y = np.concatenate([src["y"], tgt["y"]])
    group = np.concatenate([src["group"], np.char.add("siena_", tgt["group"].astype(str))])
    tr_all, te = np.arange(n_src), np.arange(n_src, len(y))

    # Window-level normalisation does not mix windows, so preparing both corpora
    # together is the same as preparing them separately.
    x1, x2, prep_meta = prepare(X, norm=pipe.norm)
    X1, X2, Y = torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(y)
    if args.device == "cuda":
        # The inputs (about 1.8 GB) fit in GPU memory; moving them once instead of per
        # mini-batch removes the transfer that dominates with models this small. The
        # values are unchanged; labels stay on the CPU for the metrics.
        X1, X2 = X1.to("cuda"), X2.to("cuda")
    del x1, x2

    tr, va = inner_split(group, tr_all, y)
    rejected = np.zeros(len(y), bool)
    if pipe.reject:                       # P4: same rule, training and validation only
        rejected[:n_src] = artifact_mask(corpus_builder("chbmit")(verbose=False)["X"])
        tr, va = tr[~rejected[tr]], va[~rejected[va]]

    tag = args.tag or (f"chbmit_to_siena_{pipe.name}" + (f"_r{args.repeat}" if args.repeat else "")
                       + ("_cuda" if args.device == "cuda" else ""))
    outdir = RESULTS_ROOT / "cross" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"train CHB-MIT {len(tr)} windows (validation {len(va)}), test Siena {len(te)} "
          f"windows from {len(set(tgt['subject']))} patients; pipeline {pipe.name}, "
          f"seed set {args.repeat}, {args.device}\n")

    probs, val_logits = {}, {}
    want_score = "score" in args.models
    for m in args.models:
        if m == "score":
            continue
        t0 = time.time()
        seed = fold_seed(20260727, args.repeat, CROSS_FOLD, m)
        out = train_fold(m, X1, X2, Y, tr, va, te, 2, X.shape[1], seed, epochs=args.epochs,
                         return_val_logits=want_score and m in ("raw1d", "spec2d"),
                         device=args.device, save_path=outdir / "models" / f"{m}.pt")
        probs[m] = out[0]
        if len(out) == 3:
            val_logits[m] = out[2]
        print(f"  {m:12s} trained ({time.time() - t0:.0f}s, {out[1]['epochs_run']} epochs)")
    if want_score:
        w = best_blend_weight(val_logits["raw1d"], val_logits["spec2d"], Y[va].numpy(), 2)
        probs["score"] = w * probs["raw1d"] + (1 - w) * probs["spec2d"]

    # classical baselines: fitted on all CHB-MIT training persons, as in LOSO
    fit = np.concatenate([tr, va])
    for name, fn in (("logvar", logvar_features), ("shallow", shallow_features)):
        feat = fn(X)
        probs[name] = baselines.fit_predict(feat[fit], y[fit], feat[te]).astype(np.float32)

    # per Siena patient
    rows = []
    subj_te = tgt["subject"]
    hours_of = {s: tgt["info"]["per_subject"][s]["hours"] for s in tgt["info"]["per_subject"]}
    for k, s in enumerate(sorted(set(subj_te))):
        mask = subj_te == s
        for m, prob in probs.items():
            mm = metrics(y[te][mask], prob[mask], 2)
            mm.update(clinical_metrics(y[te][mask], prob[mask], hours_of.get(s, 0.0)))
            mm.update(model=m, fold=k, test_subjects=s, n_test=int(mask.sum()),
                      n_test_ictal=int(y[te][mask].sum()))
            rows.append(mm)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "perfold.csv", index=False)
    np.savez(outdir / "preds.npz", idx_te=te - n_src, y_te=y[te], subject=subj_te, **probs)
    meta = {"args": vars(args), "pipeline": pipe.describe(), "prep": prep_meta,
            "train": {"corpus": "chbmit", "windows": int(len(tr)), "validation": int(len(va))},
            "test": {"corpus": "siena", "windows": int(len(te)),
                     "patients": int(len(set(subj_te)))},
            "score_blend_w": float(w) if want_score else None,
            "env": runtime_env() | {"device": args.device},
            "elapsed_s": round(time.time() - t_start, 1)}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    print(f"\nsaved: {outdir}")
    cols = ["f1_macro", "auc", "sensitivity", "specificity"]
    print(df.groupby("model")[cols].mean().round(3).sort_values("f1_macro",
                                                                ascending=False).to_string())


if __name__ == "__main__":
    main()
