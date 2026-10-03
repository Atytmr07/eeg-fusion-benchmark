"""CHB-MIT fusion operator comparison with person-wise evaluation.

Three differences from the Bonn runs:

  1. Splits are person-wise. Outer folds are leave-one-subject-out (or grouped
     k-fold) over persons, and the inner validation split also holds out whole
     persons. A window-wise inner split would put windows of the same patient in both
     training and validation and inflate the early-stopping decision. A person is the
     `group` field (chb01 and chb21 are the same person, see
     chbmit_corpus.SAME_SUBJECT), so LOSO has 23 folds, not 24.
  2. Class imbalance is 1:4 after subsampling; the loss is class weighted.
  3. Besides window-level accuracy and F1, probabilistic (Brier, log loss) and
     clinically oriented metrics (sensitivity, specificity, false alarms) are recorded.

`score` is not trained separately: its blend weight is chosen from raw1d's and
spec2d's validation logits in the same fold, so raw1d and spec2d must be in the list.

Preprocessing pipelines (src/preprocess.py): --pipeline P0 is the original benchmark
and reproduces loso_grouped exactly; P1-P6 change one preprocessing step each. --repeat
selects a different set of training seeds, to separate pipeline effects from
run-to-run variation; --repeat 0 is the seed set of all earlier runs.

Usage:
    python -m src.chbmit_run --split loso --models late raw1d spec2d --limit-folds 2
    python -m src.chbmit_run --split loso --threads 16 --tag loso_grouped   # full run
    python -m src.chbmit_run --split loso --threads 16 --pipeline P1 --repeat 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .chbmit_corpus import (build_corpus, grouped_kfold_by_subject,
                            leave_one_subject_out)
from .chbmit_prep import prepare
from .config import RESULTS_ROOT, fold_seed, runtime_env
from .evaluate import clinical_metrics, metrics
from .models import build, count_params
from .preprocess import PIPELINES, artifact_mask
from .train import best_blend_weight, set_seed, softmax_np

OUT_ROOT = RESULTS_ROOT / "chbmit"
DEFAULT_MODELS = ("raw1d", "spec2d", "raw1d_wide", "spec2d_wide",
                  "late", "gated", "attention", "early", "score")


def run_tag(split: str, pipeline: str, repeat: int, device: str) -> str:
    """Default output name of a pipeline run, e.g. loso_P1, loso_P1_r2, loso_P0_r1_cuda."""
    return (f"{split}_{pipeline}" + (f"_r{repeat}" if repeat else "")
            + ("_cuda" if device == "cuda" else ""))


def inner_split(subject: np.ndarray, tr: np.ndarray, y: np.ndarray,
                val_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Split a training fold into training and validation by whole persons.

    Persons are sorted by ictal window count and picked at evenly spaced positions of
    that list, so the validation set covers the range of seizure burdens and always
    contains ictal windows. The choice is deterministic. At least one person is held
    out.
    """
    subs = sorted(set(subject[tr]))
    if len(subs) < 2:
        raise ValueError("an inner validation split needs at least two training persons")
    counts = {s: int(y[tr][subject[tr] == s].sum()) for s in subs}
    order = sorted(subs, key=lambda s: (-counts[s], s))
    n_val = max(1, int(round(val_frac * len(subs))))
    picks = list(np.linspace(0, len(order) - 1, n_val).round().astype(int))
    val_subs = {order[i] for i in dict.fromkeys(picks)}
    if not any(counts[s] > 0 for s in val_subs):          # no ictal windows: fix that
        val_subs = {max(order, key=lambda s: counts[s])}
    m = np.isin(subject[tr], list(val_subs))
    return tr[~m], tr[m]


def train_fold(model_name: str, X1: torch.Tensor, X2: torch.Tensor, Y: torch.Tensor,
               tr: np.ndarray, va: np.ndarray, te: np.ndarray, ncls: int, in_ch: int,
               seed: int, epochs: int = 60, min_epochs: int = 20, patience: int = 12,
               batch: int = 32, lr: float = 1e-3, wd: float = 1e-4,
               return_val_logits: bool = False, device: str = "cpu",
               verbose: bool = False) -> tuple[np.ndarray, dict] | tuple[np.ndarray, dict, np.ndarray]:
    """Train one model on one fold and return its test-set probabilities.

    With return_val_logits=True the raw validation logits (before softmax) are
    returned as a third value; score fusion needs them to choose its blend weight on
    the validation set only.
    """
    set_seed(seed)
    model = build(model_name, ncls, in_ch=in_ch).to(device)
    # AdamW (decoupled weight decay), the same optimiser as the Bonn loop. Plain Adam
    # would add the decay to the gradient, where the adaptive scaling makes its
    # strength differ per parameter. The earlier loso_main run used Adam.
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    ytr = Y[tr].numpy()
    cnt = np.bincount(ytr, minlength=ncls).astype(np.float64)
    w = torch.tensor((cnt.sum() / (ncls * np.maximum(cnt, 1))), dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w.to(device))

    def raw_logits(idx: np.ndarray) -> np.ndarray:
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                b = idx[i:i + 256]
                outs.append(model(X1[b].to(device), X2[b].to(device)).cpu().numpy())
        return np.concatenate(outs)

    def predict(idx: np.ndarray) -> np.ndarray:
        return softmax_np(raw_logits(idx))

    best_f1, best_state, bad, best_ep = -1.0, None, 0, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=g).numpy()]
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(X1[b].to(device), X2[b].to(device)), Y[b].to(device))
            loss.backward()
            opt.step()

        pv = predict(va)
        mv = metrics(Y[va].numpy(), pv, ncls)
        if mv["f1_macro"] > best_f1:
            best_f1, best_ep = mv["f1_macro"], ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        # min_epochs guard: before it, a model can still sit in the trivial
        # single-class solution and early stopping would keep collapsed weights.
        if ep + 1 >= min_epochs and bad >= patience:
            break
        if verbose and ep % 10 == 0:
            print(f"      ep{ep:3d} val_f1={mv['f1_macro']:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    prob = predict(te)
    info = {"best_epoch": best_ep, "epochs_run": ep + 1, "val_f1": best_f1}
    if return_val_logits:
        return prob, info, raw_logits(va)
    return prob, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["loso", "kfold"], default="loso")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--limit-folds", type=int, default=None,
                    help="run only the first N folds (for timing or quick checks)")
    ap.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    ap.add_argument("--norm", default="window", choices=["none", "window", "channel"])
    ap.add_argument("--notch", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--pipeline", default="P0", choices=list(PIPELINES),
                    help="preprocessing pipeline (src/preprocess.py); P0 is the original")
    ap.add_argument("--repeat", type=int, default=0,
                    help="seed set; 0 reproduces the earlier runs")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cuda runs in PyTorch's deterministic mode; results are not "
                         "comparable with CPU runs, so default tags get a _cuda suffix")
    ap.add_argument("--resume", action="store_true",
                    help="skip folds already complete in the outdir's perfold.csv and "
                         "continue from there. A fold counts as complete only if every "
                         "requested model is present. Use the same --threads.")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        # Deterministic GPU kernels; the cuBLAS setting must precede CUDA initialisation.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda: no CUDA device visible to this torch build")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # warn_only: the backward pass of adaptive average pooling (used by every
        # backbone) has no deterministic CUDA kernel, and strict mode would refuse to
        # run. Whether GPU runs still repeat exactly is therefore an empirical question,
        # answered by running the same fold twice (docs/PIPELINE.md).
        torch.use_deterministic_algorithms(True, warn_only=True)
    t_start = time.time()

    want_score = "score" in args.models
    if want_score and not {"raw1d", "spec2d"} <= set(args.models):
        raise SystemExit("score is computed from raw1d and spec2d; add both to --models")

    pipe = PIPELINES[args.pipeline]
    if pipe.reject_uv is not None and not np.isfinite(pipe.reject_uv):
        raise SystemExit("P4: the rejection threshold (src/preprocess.py: REJECT_UV) has "
                         "not been decided yet")
    if pipe.name != "P0" and (args.norm != "window" or args.notch):
        raise SystemExit("--pipeline sets normalisation and filtering itself; do not "
                         "combine it with --norm or --notch")
    norm = pipe.norm if pipe.name != "P0" else args.norm
    d = build_corpus(verbose=pipe.has_signal_steps,
                     signal_fn=pipe.apply_signal if pipe.has_signal_steps else None,
                     signal_tag=pipe.signal_tag)
    X, y, subject, group = d["X"], d["y"], d["subject"], d["group"]
    info = d["info"]
    ncls, in_ch = 2, X.shape[1]

    # Artefact rejection (P4) is decided on the filtered microvolt signal, before
    # normalisation, and applied to training and validation windows only.
    rejected = np.zeros(len(y), bool)
    reject_meta = None
    if pipe.reject_uv is not None:
        rejected = artifact_mask(X, pipe.reject_uv)
        reject_meta = {"threshold_uv": pipe.reject_uv,
                       "ictal_rejected": int(rejected[y == 1].sum()),
                       "ictal_total": int((y == 1).sum()),
                       "nonictal_rejected": int(rejected[y == 0].sum()),
                       "nonictal_total": int((y == 0).sum())}
        print(f"artefact rejection > {pipe.reject_uv:g} uV: "
              f"{reject_meta['ictal_rejected']}/{reject_meta['ictal_total']} ictal and "
              f"{reject_meta['nonictal_rejected']}/{reject_meta['nonictal_total']} "
              f"non-ictal windows flagged (excluded from training and validation)")

    if norm == "channel":
        raise SystemExit("norm='channel' needs per-fold training statistics, which this "
                         "driver does not wire up yet; use 'window'")
    x1, x2, prep_meta = prepare(X, norm=norm, notch_hz=args.notch)
    X1, X2, Y = torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(y)
    del x1, x2

    splits = list(leave_one_subject_out(group) if args.split == "loso"
                  else grouped_kfold_by_subject(group, args.folds))
    if args.limit_folds:
        splits = splits[:args.limit_folds]

    if args.tag:
        tag = args.tag
    elif pipe.name != "P0" or args.repeat or args.device == "cuda":
        tag = run_tag(args.split, pipe.name, args.repeat, args.device)
    else:
        tag = f"{args.split}_{args.norm}" + (f"_notch{args.notch:g}" if args.notch else "")
    outdir = Path(args.outdir) if args.outdir else OUT_ROOT / tag
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "preds").mkdir(exist_ok=True)

    hours_of = {s: info["per_subject"][s]["hours"] for s in info["per_subject"]}
    rows = []
    done_folds: set[int] = set()
    csv_path = outdir / "perfold.csv"
    if args.resume and csv_path.exists():
        prev = pd.read_csv(csv_path)
        rows = prev.to_dict("records")
        have = prev.groupby("fold")["model"].apply(set)
        want = set(args.models)
        done_folds = {int(f) for f, ms in have.items() if want <= ms}
        print(f"--resume: read {csv_path}, {len(rows)} rows, "
              f"{len(done_folds)} folds already complete (skipped): "
              f"{sorted(done_folds)}\n")

    print(f"corpus {X.shape}, {len(splits)} folds, {len(args.models)} models, "
          f"spectrogram {X2.shape[1:]}, threads={args.threads}\n")

    for fi, (key, tr_all, te) in enumerate(splits):
        if fi in done_folds:
            print(f"[{fi+1}/{len(splits)}] skipped (resume, already complete)")
            continue
        tr, va = inner_split(group, tr_all, y)
        # Validation persons are chosen before rejection, so they are the same in
        # every pipeline.
        tr, va = tr[~rejected[tr]], va[~rejected[va]]
        te_subs = sorted(set(subject[te]))              # cases; the chb01 fold has chb01,chb21
        hrs = sum(hours_of.get(s, 0.0) for s in te_subs)
        print(f"[{fi+1}/{len(splits)}] test={','.join(te_subs)} "
              f"train={len(tr)} val={len(va)} test={len(te)} "
              f"(test ictal={int(y[te].sum())})")

        probs, val_logits = {}, {}
        for m in args.models:
            if m == "score":
                continue
            t0 = time.time()
            seed = fold_seed(20260727, args.repeat, fi, m)
            if want_score and m in ("raw1d", "spec2d"):
                prob, tinfo, val_logits[m] = train_fold(
                    m, X1, X2, Y, tr, va, te, ncls, in_ch, seed,
                    epochs=args.epochs, return_val_logits=True, device=args.device)
            else:
                prob, tinfo = train_fold(m, X1, X2, Y, tr, va, te, ncls, in_ch, seed,
                                         epochs=args.epochs, device=args.device)
            mm = metrics(y[te], prob, ncls)
            mm.update(clinical_metrics(y[te], prob, hrs))
            mm.update(model=m, fold=fi, test_subjects=",".join(te_subs),
                      n_test=len(te), n_test_ictal=int(y[te].sum()),
                      params=count_params(build(m, ncls, in_ch=in_ch)),
                      sec=round(time.time() - t0, 1), **tinfo)
            rows.append(mm)
            probs[m] = prob
            print(f"    {m:12s} f1={mm['f1_macro']:.3f} auc={mm['auc']:.3f} "
                  f"brier={mm['brier']:.3f} sens={mm['sensitivity']:.3f} "
                  f"spec={mm['specificity']:.3f} ({mm['sec']:.0f}s, "
                  f"{tinfo['epochs_run']} epochs)")

        if want_score:
            w = best_blend_weight(val_logits["raw1d"], val_logits["spec2d"],
                                  Y[va].numpy(), ncls)
            prob = w * probs["raw1d"] + (1 - w) * probs["spec2d"]
            mm = metrics(y[te], prob, ncls)
            mm.update(clinical_metrics(y[te], prob, hrs))
            mm.update(model="score", fold=fi, test_subjects=",".join(te_subs),
                      n_test=len(te), n_test_ictal=int(y[te].sum()), blend_w=w,
                      params=count_params(build("raw1d", ncls, in_ch=in_ch)) +
                             count_params(build("spec2d", ncls, in_ch=in_ch)),
                      sec=0.0, best_epoch=-1, epochs_run=0, val_f1=float("nan"))
            rows.append(mm)
            probs["score"] = prob
            print(f"    {'score':12s} f1={mm['f1_macro']:.3f} auc={mm['auc']:.3f} "
                  f"brier={mm['brier']:.3f} blend_w={w:.2f}")

        # Saved after every fold, so an interrupted run can be resumed.
        np.savez(outdir / "preds" / f"fold{fi}.npz",
                 idx_te=te, y_te=y[te], **probs)
        pd.DataFrame(rows).to_csv(outdir / "perfold.csv", index=False)

    meta = {"args": vars(args), "optimizer": "AdamW", "pipeline": pipe.describe(),
            "rejection": reject_meta, "prep": prep_meta, "corpus": info,
            "env": runtime_env() | {"device": args.device, "cuda_device":
                                    torch.cuda.get_device_name(0) if args.device == "cuda"
                                    else None},
            "elapsed_s": round(time.time() - t_start, 1)}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    df = pd.DataFrame(rows)
    print(f"\ntotal {meta['elapsed_s']/60:.1f} minutes, saved: {outdir}")
    print("\nmodel means:")
    cols = ["f1_macro", "auc", "brier", "sensitivity", "specificity"]
    print(df.groupby("model")[cols].mean().round(4).sort_values("f1_macro",
                                                               ascending=False).to_string())


if __name__ == "__main__":
    main()
