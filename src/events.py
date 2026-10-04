"""Event-based (seizure-level) metrics, as fixed in docs/EVENTS.md.

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

Siena (--dataset siena): recordings, seizure lists and their DECISIONS come from
src/siena.py, the signal is read as siena_corpus reads it (read_bipolar: 18 bipolar
channels at 256 Hz, then the pipeline's signal steps), and the 14 LOSO folds from
siena_corpus.build_corpus. Only seizure-level sensitivity is reported for Siena: its
recordings are cut around the seizures, so false alarms per hour are computed but are
not representative and enter no comparison (docs/EVENTS.md).

Cross-dataset (--cross <tag>): the models src/cross_dataset.py trained once on CHB-MIT
(results_v2/cross/<tag>/models/<model>.pt, score fusion weight in meta.json) are
applied to every Siena recording; the classical baselines are refitted on the CHB-MIT
training persons as in cross_dataset. Reported per Siena patient.

--oracle checks the scoring itself: the window labels are scored as if they were the
predictions, which must detect every seizure without a false alarm.

Usage:  python -m src.events --pipeline P1 --repeats 0 1 2 --threads 16 [--jobs 8] [--device cuda]
        python -m src.events --dataset siena --pipeline P1 --repeats 0 1 2
        python -m src.events --cross chbmit_to_siena_P0
        python -m src.events --dataset siena --oracle
Output: results_v2/<dataset>/<run>/events/ (perperson.csv, summary.csv, meta.json) for
        every run; results_v2/cross/<tag>/events/ for --cross
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
from .chbmit import (CHB_ROOT, FS_CHB, TARGET_CHANNELS, parse_summary, plan_windows,
                     read_edf, read_edf_header, subject_files)
from .chbmit_baselines import logvar_features, shallow_features
from .chbmit_corpus import build_corpus, leave_one_subject_out
from .chbmit_prep import prepare
from .chbmit_run import DEFAULT_MODELS, inner_split
from .config import RESULTS_ROOT
from .models import build
from .preprocess import PIPELINES, artifact_mask
from .run_queue import tag_for
from .train import softmax_np

WIN_S = 10.0
THRESHOLD = 0.5                  # same decision rule as the window-level metrics
DEEP = [m for m in DEFAULT_MODELS if m != "score"]
BASELINES = {"logvar": logvar_features, "shallow": shallow_features}
FA_NOTE = "not representative for Siena, not compared"
WINDOW_TOL = 1e-4                # largest window difference from the corpus, relative
FA_COLUMNS = ("fa_per_hour_mean", "fa_per_hour_median", "fa_per_hour_pooled",
              "false_alarms", "hours")


# --- One recording: signal, windows, labels -------------------------------------------

def read_signal(path: Path, pipeline: str, dataset: str = "chbmit") -> tuple[np.ndarray, float]:
    """A recording after the pipeline's signal-level steps, as the corpus builds it:
    CHB-MIT's 18 channels at their own rate, or Siena converted to the CHB-MIT form
    (src/siena.py: read_bipolar, 18 bipolar channels at 256 Hz)."""
    pipe = PIPELINES[pipeline]
    if dataset == "siena":
        from .siena import read_bipolar
        x = read_bipolar(path, signal_fn=pipe.apply_signal if pipe.has_signal_steps else None)
        return x, FS_CHB
    x, _, fs = read_edf(path, list(TARGET_CHANNELS))
    if pipe.has_signal_steps:
        x = pipe.apply_signal(x, fs, path.name)
    return x, fs


def recording_windows(path: Path, pipeline: str, dataset: str = "chbmit"
                      ) -> tuple[np.ndarray, np.ndarray, float]:
    """All 10 s windows of a recording after the pipeline's signal-level steps.
    Returns windows (n, channel, time), their start times (s), and the recording's
    duration in hours. Runs in a worker process."""
    x, fs = read_signal(path, pipeline, dataset)
    fs = int(round(fs))
    n = int(WIN_S * fs)
    starts = np.arange(0, x.shape[1] - n + 1, n)
    windows = np.stack([x[:, s:s + n] for s in starts]).astype(np.float32)
    return windows, starts / fs, read_edf_header(path).duration / 3600.0


def person_recordings(dataset: str, cases: list[str], decisions: dict | None = None
                      ) -> list[tuple[Path, list, float | None]]:
    """(recording, annotated seizures in seconds from its first sample, scored length)
    for every recording of the given cases.

    The scored length is None (the whole recording) except where a Siena DECISION
    excludes a seizure: the corpus drops every window from that seizure's onset to the
    end of the recording, and the event scoring stops there too.
    """
    if dataset == "siena":
        from .siena import (DECISIONS, parse_seizure_list, record_path,
                            seizures_in_record, subject_records)
        decisions = dict(DECISIONS if decisions is None else decisions)
        out = []
        for sub in cases:
            listed = parse_seizure_list(sub, decisions=decisions)
            for name in subject_records(sub):
                path = record_path(sub, name)
                zs, excl, _ = seizures_in_record(path, listed, decisions)
                out.append((path, zs, min(a for a, _ in excl) if excl else None))
        return out
    seiz = {c: parse_summary(CHB_ROOT / c / f"{c}-summary.txt") for c in cases}
    return [(f, seiz[c].get(f.name, []), None) for c in cases for f in subject_files(c)]


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


# --- Models ------------------------------------------------------------------------------

def load_models(model_dir: Path, fold: int | None, device: str) -> dict:
    """The saved deep models of one LOSO fold (fold<k>_<model>.pt) or, with fold None,
    of a single cross-dataset training (<model>.pt)."""
    out = {}
    for m in DEEP:
        f = model_dir / (f"fold{fold}_{m}.pt" if fold is not None else f"{m}.pt")
        if not f.exists():
            raise SystemExit(f"{f} missing: the run did not save its models (run it with the "
                             f"current chbmit_run or cross_dataset)")
        model = build(m, 2, in_ch=len(TARGET_CHANNELS))
        model.load_state_dict(torch.load(f, map_location="cpu"))
        out[m] = model.to(device).eval()
    return out


def load_fold_models(run_dir: Path, fold: int, device: str) -> dict:
    return load_models(run_dir / "models", fold, device)


@torch.no_grad()
def predict(model, x1: torch.Tensor, x2: torch.Tensor, device: str, batch: int = 512):
    logits = [model(x1[i:i + batch].to(device), x2[i:i + batch].to(device)).cpu().numpy()
              for i in range(0, len(x1), batch)]
    return softmax_np(np.concatenate(logits))[:, 1]


# --- Scoring the recordings of one test set ----------------------------------------------

def score_recordings(recs, pipe, dataset: str, models: dict, blend: dict, clfs: dict,
                     device: str, jobs: int, pool, check: dict | None = None) -> dict:
    """Apply every run's models and the baselines to every window of the recordings and
    sum the event counts: (run, model) -> [reference events, detected, false alarms,
    hours]. blend: run -> score fusion weight (NaN: no score model).

    check: recording name -> (start times, corpus windows) of the windows the runs were
    trained and tested on. The same windows recomputed here are compared with them, and
    the largest difference relative to the corpus amplitude is stored under
    acc["_window_diff"]. It is 0 when the signal steps reproduce the corpus exactly; for
    P6 on another machine it shows whether the decomposition came out the same."""
    # At most 2 x jobs recordings in flight, so finished windows do not pile up in memory.
    queue, pending = list(recs), []

    def submit_next():
        if queue:
            f, zs, limit = queue.pop(0)
            pending.append((f, zs, limit, pool.submit(recording_windows, f, pipe.name, dataset)
                            if pool else None))
    for _ in range(2 * max(jobs, 1)):
        submit_next()
    acc: dict[tuple[str, str], np.ndarray] = {}
    while pending:
        f, zs, limit, fut = pending.pop(0)
        submit_next()
        windows, t0, hours = fut.result() if fut else recording_windows(f, pipe.name, dataset)
        if limit is not None:
            hours = limit / 3600.0
        if check and f.name in check:
            ts, ref_x = check[f.name]
            k = np.searchsorted(t0, ts)
            ok = (k < len(t0)) & (np.abs(t0[np.minimum(k, len(t0) - 1)] - ts) < 1e-3)
            diff = (float(np.abs(windows[k[ok]] - ref_x[ok]).max() / np.abs(ref_x[ok]).max())
                    if ok.all() else np.inf)       # a missing window counts as a mismatch
            acc["_window_diff"] = max(acc.get("_window_diff", 0.0), diff)
        x1, x2, _ = prepare(windows, norm=pipe.norm)
        x1, x2 = torch.from_numpy(x1), torch.from_numpy(x2)
        probs: dict[str, dict[str, np.ndarray]] = {}
        base = {b: clfs[b].predict_proba(BASELINES[b](windows))[:, 1] for b in clfs}
        for tag in models:
            p = {m: predict(mod, x1, x2, device) for m, mod in models[tag].items()}
            w = blend[tag]
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
    return acc


def event_rows(acc: dict, **fields) -> list[dict]:
    rows = []
    acc = {k: v for k, v in acc.items() if k != "_window_diff"}
    for (tag, m), (n_ref, tp, fp, hours) in acc.items():
        rows.append({"run": tag, "model": m, **fields, "seizure_events": int(n_ref),
                     "detected": int(tp), "false_alarms": int(fp), "hours": hours,
                     "sensitivity": tp / n_ref if n_ref else np.nan,
                     "fa_per_hour": fp / hours if hours else np.nan,
                     "precision": tp / (tp + fp) if tp + fp else np.nan})
    return rows


def summarize(part: pd.DataFrame, dataset: str) -> pd.DataFrame:
    summ = part.groupby("model").agg(
        sensitivity_mean=("sensitivity", "mean"), fa_per_hour_mean=("fa_per_hour", "mean"),
        fa_per_hour_median=("fa_per_hour", "median"), detected=("detected", "sum"),
        seizure_events=("seizure_events", "sum"), false_alarms=("false_alarms", "sum"),
        hours=("hours", "sum"))
    summ["sensitivity_pooled"] = summ.detected / summ.seizure_events
    summ["fa_per_hour_pooled"] = summ.false_alarms / summ.hours
    if dataset == "siena":
        # sensitivity is the reported metric; false alarms stay, marked in the header
        summ = summ[["sensitivity_pooled", "sensitivity_mean", "detected", "seizure_events",
                     *FA_COLUMNS]]
        summ = summ.rename(columns={c: f"{c} ({FA_NOTE})" for c in FA_COLUMNS})
    return summ


def write_outputs(out: Path, part: pd.DataFrame, dataset: str, meta: dict) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    part.to_csv(out / "perperson.csv", index=False)
    summ = summarize(part, dataset)
    summ.round(4).to_csv(out / "summary.csv")
    if dataset == "siena":
        meta = meta | {"dataset": "siena", "reported": "seizure-level sensitivity only",
                       "false_alarms_per_hour": FA_NOTE + " (recordings are cut around "
                       "the seizures; docs/EVENTS.md)"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return summ


def print_summary(tag: str, summ: pd.DataFrame, dataset: str) -> None:
    print(f"\n{tag}:")
    if dataset == "siena":
        print(summ[["sensitivity_pooled", "sensitivity_mean", "detected",
                    "seizure_events"]].round(3).to_string())
        print(f"(false alarms per hour: {FA_NOTE})")
    else:
        print(summ[["sensitivity_pooled", "fa_per_hour_pooled", "sensitivity_mean",
                    "fa_per_hour_median"]].round(3).to_string())


# --- Modes -------------------------------------------------------------------------------

def run_loso(args, pipe, t_start: float) -> None:
    """Every fold's saved models on every recording of its held-out person."""
    from .datasets import corpus_builder
    runs = {}
    for tag in args.runs or [tag_for(pipe.name, r, args.device) for r in args.repeats]:
        d = RESULTS_ROOT / args.dataset / tag
        if not (d / "perfold.csv").exists():
            print(f"skipped {tag}: no results")
            continue
        runs[tag] = d
    if not runs:
        raise SystemExit("no runs to evaluate")

    # The corpus gives the persons and folds (identical in every pipeline) and the training
    # windows for the classical baselines; P4 rejects the same windows as in training.
    build = build_corpus if args.dataset == "chbmit" else corpus_builder(args.dataset)
    corpus = build(verbose=False,
                   signal_fn=pipe.apply_signal if pipe.has_signal_steps else None,
                   signal_tag=pipe.signal_tag)
    rejected = (artifact_mask(build(verbose=False)["X"]) if pipe.reject
                else np.zeros(len(corpus["y"]), bool))
    feats = {b: fn(corpus["X"]) for b, fn in BASELINES.items()}
    splits = list(leave_one_subject_out(corpus["group"]))
    if args.limit_folds:
        splits = splits[:args.limit_folds]
    perfold = {tag: pd.read_csv(d / "perfold.csv") for tag, d in runs.items()}
    blend = {tag: pf.query("model == 'score'").set_index("fold").blend_w
             for tag, pf in perfold.items()}
    decisions = corpus["info"].get("decisions") if args.dataset == "siena" else None

    rows, window_diff = [], np.nan
    pool = ProcessPoolExecutor(args.jobs) if args.jobs > 1 else None
    for fi, (key, tr_all, te) in enumerate(splits):
        t_fold = time.time()
        cases = sorted(set(corpus["subject"][te]))
        # the fold must hold out the same person(s) as when the run was trained
        for tag, pf in perfold.items():
            held = set(pf.loc[pf.fold == fi, "test_subjects"].astype(str))
            if held and held != {",".join(cases)}:
                raise SystemExit(f"{tag}: fold {fi} tested {held} in training, but the corpus "
                                 f"now gives {','.join(cases)}; the folds do not match")
        tr = tr_all[~rejected[tr_all]]
        clfs = {b: baselines.make_clf().fit(f[tr], corpus["y"][tr]) for b, f in feats.items()}
        models = {tag: load_fold_models(d, fi, args.device) for tag, d in runs.items()}
        recs = person_recordings(args.dataset, cases, decisions)
        w = {tag: float(blend[tag].get(fi, np.nan)) for tag in runs}
        check = {r: (corpus["t0"][te][corpus["record"][te] == r].astype(float),
                     corpus["X"][te][corpus["record"][te] == r])
                 for r in set(corpus["record"][te])}
        acc = score_recordings(recs, pipe, args.dataset, models, w, clfs, args.device,
                               args.jobs, pool, check)
        diff = acc.get("_window_diff", np.nan)
        window_diff = max(window_diff, diff) if np.isfinite(window_diff) else diff
        person = str(corpus["group"][te][0])
        rows += event_rows(acc, fold=fi, person=person, cases=",".join(cases))
        print(f"[{fi + 1}/{len(splits)}] {','.join(cases)}: {len(recs)} recordings, "
              f"{time.time() - t_fold:.0f}s, windows vs corpus {diff:.1e}"
              + ("  WARNING: the recomputed signal differs from the corpus"
                 if diff > WINDOW_TOL else ""), flush=True)
    if pool:
        pool.shutdown()

    df = pd.DataFrame(rows)
    for tag, d in runs.items():
        part = df[df.run == tag].drop(columns="run")
        meta = {"definition": "docs/EVENTS.md", "threshold": THRESHOLD, "window_s": WIN_S,
                "scoring": "timescoring EventScoring defaults (SzCORE)",
                "pipeline": pipe.describe(), "folds": len(splits),
                "elapsed_s": round(time.time() - t_start, 1)}
        meta["window_diff_vs_corpus"] = window_diff
        meta["window_check"] = ("pass" if window_diff <= WINDOW_TOL else
                                f"FAIL: the recomputed signal differs from the corpus by "
                                f"{window_diff:.1e} of its amplitude (tolerance {WINDOW_TOL})")
        if decisions is not None:
            meta["decisions"] = decisions
        summ = write_outputs(d / "events", part, args.dataset, meta)
        print_summary(tag, summ, args.dataset)


def run_cross(args, t_start: float) -> None:
    """Models trained once on CHB-MIT (src/cross_dataset.py) on every Siena recording."""
    from .datasets import corpus_builder
    from .siena_corpus import available_subjects

    dirs = {}
    for tag in args.cross:
        d = RESULTS_ROOT / "cross" / tag
        if not (d / "meta.json").exists():
            raise SystemExit(f"{d}: no cross-dataset run (meta.json missing)")
        dirs[tag] = d
    metas = {tag: json.loads((d / "meta.json").read_text(encoding="utf-8"))
             for tag, d in dirs.items()}
    names = {m["pipeline"]["name"] for m in metas.values()}
    if len(names) != 1:
        raise SystemExit(f"--cross runs of different pipelines ({sorted(names)}) cannot share "
                         f"one pass")
    pipe = PIPELINES[names.pop()]
    models = {tag: load_models(d / "models", None, args.device) for tag, d in dirs.items()}
    blend = {tag: float(m["score_blend_w"]) if m.get("score_blend_w") is not None else np.nan
             for tag, m in metas.items()}

    # Classical baselines: fitted on the CHB-MIT training persons, as in cross_dataset
    # (the windows of the inner training and validation split, without P4's rejects).
    clfs, base_note = {}, "fitted on all CHB-MIT training persons, as in cross_dataset"
    if args.skip_baselines:
        base_note = "skipped (--skip-baselines)"
    else:
        src = corpus_builder("chbmit")(verbose=False,
                                       signal_fn=pipe.apply_signal if pipe.has_signal_steps
                                       else None, signal_tag=pipe.signal_tag)
        tr, va = inner_split(src["group"], np.arange(len(src["y"])), src["y"])
        if pipe.reject:
            rej = artifact_mask(corpus_builder("chbmit")(verbose=False)["X"])
            tr, va = tr[~rej[tr]], va[~rej[va]]
        fit = np.concatenate([tr, va])
        clfs = {b: baselines.make_clf().fit(fn(src["X"])[fit], src["y"][fit])
                for b, fn in BASELINES.items()}
        del src

    patients = args.siena_subjects or available_subjects()
    rows = []
    pool = ProcessPoolExecutor(args.jobs) if args.jobs > 1 else None
    for k, sub in enumerate(patients):
        t_p = time.time()
        recs = person_recordings("siena", [sub])
        acc = score_recordings(recs, pipe, "siena", models, blend, clfs, args.device,
                               args.jobs, pool)
        rows += event_rows(acc, patient=sub)
        print(f"[{k + 1}/{len(patients)}] {sub}: {len(recs)} recordings, "
              f"{time.time() - t_p:.0f}s", flush=True)
    if pool:
        pool.shutdown()
    df = pd.DataFrame(rows)
    for tag, d in dirs.items():
        part = df[df.run == tag].drop(columns="run")
        meta = {"definition": "docs/EVENTS.md", "threshold": THRESHOLD, "window_s": WIN_S,
                "scoring": "timescoring EventScoring defaults (SzCORE)",
                "mode": "cross-dataset: trained on CHB-MIT (one training), tested on every "
                        "Siena recording", "pipeline": pipe.describe(),
                "score_blend_w": None if np.isnan(blend[tag]) else blend[tag],
                "baselines": base_note, "patients": list(patients),
                "elapsed_s": round(time.time() - t_start, 1)}
        summ = write_outputs(d / "events", part, "siena", meta)
        print_summary(tag, summ, "siena")


def run_oracle(args) -> bool:
    """Score the window labels as predictions: every seizure must be detected and no
    false alarm raised, or the reference, the windows or the scoring disagree."""
    if args.dataset == "siena":
        from .siena_corpus import available_subjects
        persons = available_subjects()
    else:
        from .chbmit_corpus import all_subjects
        persons = all_subjects()
    if not persons:
        raise SystemExit(f"no {args.dataset} recordings on disk")
    rows = []
    for p in persons:
        tot = np.zeros(4)
        for path, zs, limit in person_recordings(args.dataset, [p]):
            y, t0 = plan_windows(path, zs, WIN_S)
            dur = limit if limit is not None else read_edf_header(path).duration
            ref, hyp = masks_1hz(dur, zs, t0, y == 1)
            tot += (*score_events(ref, hyp), dur / 3600)
        rows.append({"person": p, "seizure_events": int(tot[0]), "detected": int(tot[1]),
                     "false_alarms": int(tot[2]), "hours": round(tot[3], 2)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    ok = bool((df.detected == df.seizure_events).all() and (df.false_alarms == 0).all())
    print(f"\noracle ({args.dataset}, {len(df)} persons, {int(df.seizure_events.sum())} "
          f"seizures, {df.hours.sum():.1f} h): "
          + ("PASS, every seizure detected, no false alarm" if ok else "FAIL"))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chbmit", choices=["chbmit", "siena"])
    ap.add_argument("--pipeline", default="P0", choices=list(PIPELINES))
    ap.add_argument("--repeats", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--jobs", type=int, default=1,
                    help="worker processes reading and preprocessing recordings")
    ap.add_argument("--limit-folds", type=int, default=None, help="for quick checks")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="explicit run folders under results_v2/<dataset> (instead of the "
                         "pipeline's seed sets)")
    ap.add_argument("--cross", nargs="+", default=None,
                    help="cross-dataset runs under results_v2/cross (e.g. "
                         "chbmit_to_siena_P0): their models on every Siena recording")
    ap.add_argument("--siena-subjects", nargs="*", default=None,
                    help="with --cross: restrict the Siena patients (quick checks only)")
    ap.add_argument("--skip-baselines", action="store_true",
                    help="with --cross: do not refit the classical baselines (needs no "
                         "CHB-MIT data; for tests only)")
    ap.add_argument("--oracle", action="store_true",
                    help="check the scoring: window labels as predictions")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t_start = time.time()
    if args.oracle:
        sys.exit(0 if run_oracle(args) else 1)
    if args.cross:
        run_cross(args, t_start)
        return
    run_loso(args, PIPELINES[args.pipeline], t_start)


if __name__ == "__main__":
    main()
