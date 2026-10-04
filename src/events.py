"""Event-based (seizure-level) metrics on CHB-MIT, as fixed in docs/EVENTS.md.

The window-level results use the subsampled test windows. Here every model of a run is
applied to EVERY 10 s window of every recording of the person held out in each LOSO
fold, and the resulting detections are scored against the annotated seizures with
SzCORE's event scoring (timescoring 0.0.7, default parameters):

  - seizure-level sensitivity: detected / annotated seizure events, per person
  - false alarms per hour: false-positive events / recording hours, per person, on the
    true (unsubsampled) recording duration

Inputs are the weights each run saved per fold (models/fold<k>_<model>.pt), the score
fusion weight of each fold (perfold.csv), and the classical baselines refitted on each
fold's training persons. All seed sets of one pipeline are evaluated in one pass: each
recording is read and preprocessed once (for P6, the ICA or GEDAI decomposition is
recomputed with the same seed as when the corpus was built), and every run's models
are applied to it. P6 pipelines therefore need the packages of requirements-ica.txt.

Window-level macro F1 and AUPRC stay the primary metrics; these are secondary.

Usage:  python -m src.events --pipeline P1 --repeats 0 1 2 --threads 16 [--jobs 8] [--device cuda]
Output: results_v2/chbmit/<run>/events/perperson.csv and summary.csv for every run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from . import baselines
from .chbmit import (CHB_ROOT, TARGET_CHANNELS, parse_summary, plan_windows, read_edf,
                     read_edf_header, subject_files)
from .chbmit_baselines import logvar_features, shallow_features
from .chbmit_corpus import build_corpus, leave_one_subject_out
from .chbmit_prep import prepare
from .chbmit_run import DEFAULT_MODELS
from .config import RESULTS_ROOT
from .models import build
from .preprocess import PIPELINES, artifact_mask
from .run_queue import tag_for
from .train import softmax_np

WIN_S = 10.0
THRESHOLD = 0.5                  # same decision rule as the window-level metrics
DEEP = [m for m in DEFAULT_MODELS if m != "score"]
BASELINES = {"logvar": logvar_features, "shallow": shallow_features}


# --- One recording: signal, windows, labels -------------------------------------------

def recording_windows(path: Path, pipeline: str) -> tuple[np.ndarray, np.ndarray, float]:
    """All 10 s windows of a recording after the pipeline's signal-level steps.
    Returns windows (n, channel, time), their start times (s), and the recording's
    duration in hours. Runs in a worker process."""
    pipe = PIPELINES[pipeline]
    x, _, fs = read_edf(path, list(TARGET_CHANNELS))
    if pipe.has_signal_steps:
        x = pipe.apply_signal(x, fs, path.name)
    fs = int(round(fs))
    n = int(WIN_S * fs)
    starts = np.arange(0, x.shape[1] - n + 1, n)
    windows = np.stack([x[:, s:s + n] for s in starts]).astype(np.float32)
    return windows, starts / fs, read_edf_header(path).duration / 3600.0


def masks_1hz(duration_s: float, seizures, t0: np.ndarray, positive: np.ndarray):
    """Reference (annotated seizures) and hypothesis (positive windows) at 1 Hz."""
    n = int(np.floor(duration_s))
    ref = np.zeros(n, bool)
    for z in seizures:
        ref[int(z.start_s):int(np.ceil(z.end_s))] = True
    hyp = np.zeros(n, bool)
    for t, p in zip(t0, positive):
        if p:
            hyp[int(t):int(t + WIN_S)] = True
    return ref, hyp


def score_events(ref: np.ndarray, hyp: np.ndarray) -> tuple[int, int, int]:
    """(reference events, detected reference events, false-positive events), SzCORE."""
    from timescoring import scoring
    from timescoring.annotations import Annotation

    r = scoring.EventScoring(Annotation(ref, 1), Annotation(hyp, 1))
    return int(r.refTrue), int(r.tp), int(r.fp)


# --- Models of one fold --------------------------------------------------------------

def load_fold_models(run_dir: Path, fold: int, device: str) -> dict:
    out = {}
    for m in DEEP:
        f = run_dir / "models" / f"fold{fold}_{m}.pt"
        if not f.exists():
            raise SystemExit(f"{f} missing: the run did not save its models (run it with the "
                             f"current chbmit_run)")
        model = build(m, 2, in_ch=len(TARGET_CHANNELS))
        model.load_state_dict(torch.load(f, map_location="cpu"))
        out[m] = model.to(device).eval()
    return out


@torch.no_grad()
def predict(model, x1: torch.Tensor, x2: torch.Tensor, device: str, batch: int = 512):
    logits = [model(x1[i:i + batch].to(device), x2[i:i + batch].to(device)).cpu().numpy()
              for i in range(0, len(x1), batch)]
    return softmax_np(np.concatenate(logits))[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default="P0", choices=list(PIPELINES))
    ap.add_argument("--repeats", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--jobs", type=int, default=1,
                    help="worker processes reading and preprocessing recordings")
    ap.add_argument("--limit-folds", type=int, default=None, help="for quick checks")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="explicit run folders under results_v2/chbmit (instead of the "
                         "pipeline's seed sets)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t_start = time.time()
    pipe = PIPELINES[args.pipeline]

    runs = {}
    for tag in args.runs or [tag_for(pipe.name, r, args.device) for r in args.repeats]:
        d = RESULTS_ROOT / "chbmit" / tag
        if not (d / "perfold.csv").exists():
            print(f"skipped {tag}: no results")
            continue
        runs[tag] = d
    if not runs:
        raise SystemExit("no runs to evaluate")

    # The corpus gives the persons and folds (identical in every pipeline) and the training
    # windows for the classical baselines; P4 rejects the same windows as in training.
    corpus = build_corpus(verbose=False,
                          signal_fn=pipe.apply_signal if pipe.has_signal_steps else None,
                          signal_tag=pipe.signal_tag)
    rejected = (artifact_mask(build_corpus(verbose=False)["X"]) if pipe.reject
                else np.zeros(len(corpus["y"]), bool))
    feats = {b: fn(corpus["X"]) for b, fn in BASELINES.items()}
    splits = list(leave_one_subject_out(corpus["group"]))
    if args.limit_folds:
        splits = splits[:args.limit_folds]
    blend = {tag: pd.read_csv(d / "perfold.csv").query("model == 'score'")
             .set_index("fold").blend_w for tag, d in runs.items()}

    rows = []
    pool = ProcessPoolExecutor(args.jobs) if args.jobs > 1 else None
    for fi, (key, tr_all, te) in enumerate(splits):
        t_fold = time.time()
        cases = sorted(set(corpus["subject"][te]))
        tr = tr_all[~rejected[tr_all]]
        clfs = {b: baselines.make_clf().fit(f[tr], corpus["y"][tr]) for b, f in feats.items()}
        models = {tag: load_fold_models(d, fi, args.device) for tag, d in runs.items()}

        recs = [f for c in cases for f in subject_files(c)]
        seiz = {c: parse_summary(CHB_ROOT / c / f"{c}-summary.txt") for c in cases}
        # At most 2 x jobs recordings in flight, so finished windows do not pile up in memory.
        queue, pending = list(recs), []

        def submit_next():
            if queue:
                f = queue.pop(0)
                pending.append((f, pool.submit(recording_windows, f, pipe.name) if pool
                                else None))
        for _ in range(2 * max(args.jobs, 1)):
            submit_next()
        acc: dict[tuple[str, str], np.ndarray] = {}          # (run, model) -> [ref, tp, fp, h]
        while pending:
            f, fut = pending.pop(0)
            submit_next()
            windows, t0, hours = fut.result() if fut else recording_windows(f, pipe.name)
            x1, x2, _ = prepare(windows, norm=pipe.norm)
            x1, x2 = torch.from_numpy(x1), torch.from_numpy(x2)
            case = f.parent.name
            zs = seiz[case].get(f.name, [])
            probs: dict[str, dict[str, np.ndarray]] = {}
            base = {b: clfs[b].predict_proba(BASELINES[b](windows))[:, 1] for b in BASELINES}
            for tag in runs:
                p = {m: predict(mod, x1, x2, args.device) for m, mod in models[tag].items()}
                w = float(blend[tag].get(fi, np.nan))
                if np.isfinite(w):
                    p["score"] = w * p["raw1d"] + (1 - w) * p["spec2d"]
                p |= base
                probs[tag] = p
            for tag, p in probs.items():
                for m, prob in p.items():
                    ref, hyp = masks_1hz(hours * 3600, zs, t0, prob > THRESHOLD)
                    n_ref, tp, fp = score_events(ref, hyp)
                    acc.setdefault((tag, m), np.zeros(4))
                    acc[(tag, m)] += (n_ref, tp, fp, hours)
        person = str(corpus["group"][te][0])
        for (tag, m), (n_ref, tp, fp, hours) in acc.items():
            rows.append({"run": tag, "model": m, "fold": fi, "person": person,
                         "cases": ",".join(cases), "seizure_events": int(n_ref),
                         "detected": int(tp), "false_alarms": int(fp), "hours": hours,
                         "sensitivity": tp / n_ref if n_ref else np.nan,
                         "fa_per_hour": fp / hours if hours else np.nan,
                         "precision": tp / (tp + fp) if tp + fp else np.nan})
        print(f"[{fi + 1}/{len(splits)}] {','.join(cases)}: {len(recs)} recordings, "
              f"{time.time() - t_fold:.0f}s", flush=True)
    if pool:
        pool.shutdown()

    df = pd.DataFrame(rows)
    for tag, d in runs.items():
        out = d / "events"
        out.mkdir(exist_ok=True)
        part = df[df.run == tag].drop(columns="run")
        part.to_csv(out / "perperson.csv", index=False)
        summ = part.groupby("model").agg(
            sensitivity_mean=("sensitivity", "mean"), fa_per_hour_mean=("fa_per_hour", "mean"),
            fa_per_hour_median=("fa_per_hour", "median"), detected=("detected", "sum"),
            seizure_events=("seizure_events", "sum"), false_alarms=("false_alarms", "sum"),
            hours=("hours", "sum"))
        summ["sensitivity_pooled"] = summ.detected / summ.seizure_events
        summ["fa_per_hour_pooled"] = summ.false_alarms / summ.hours
        summ.round(4).to_csv(out / "summary.csv")
        (out / "meta.json").write_text(json.dumps({
            "definition": "docs/EVENTS.md", "threshold": THRESHOLD, "window_s": WIN_S,
            "scoring": "timescoring EventScoring defaults (SzCORE)",
            "pipeline": pipe.describe(), "folds": len(splits),
            "elapsed_s": round(time.time() - t_start, 1)}, indent=2), encoding="utf-8")
        print(f"\n{tag}:")
        print(summ[["sensitivity_pooled", "fa_per_hour_pooled", "sensitivity_mean",
                    "fa_per_hour_median"]].round(3).to_string())


if __name__ == "__main__":
    main()
