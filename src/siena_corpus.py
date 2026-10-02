"""Siena corpus building and leakage-safe, person-wise splitting.

The Siena counterpart of src/chbmit_corpus.py, with the same interface and output, so
every downstream step (src/chbmit_prep.py, the models, the splits, the statistics)
runs on it unchanged. Window labelling and within-subject subsampling are not
reimplemented: they are CHB-MIT's own functions (plan_windows, _window_seizure_id,
validate, check_leakage), applied to recordings that src/siena.py has already turned
into the CHB-MIT form (18 bipolar channels, 256 Hz, microvolts).

Siena has one recording session per patient, so the person group is the subject
itself (14 groups, 14 LOSO folds).

Usage:
    python -m src.siena_corpus --build      # build the cache
    python -m src.siena_corpus --check      # counts, leakage and integrity checks
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .chbmit import FS_CHB, TARGET_CHANNELS, plan_windows
from .chbmit_corpus import (NON_ICTAL, _window_seizure_id, check_leakage,
                            leave_one_subject_out, validate)
from .siena import (DECISIONS, FS_SIENA, SIENA_ROOT, SUBJECTS, bipolar_pairs,
                    check_decisions, parse_seizure_list, read_bipolar, record_path,
                    seizures_in_record, subject_records)

CACHE = SIENA_ROOT / "_cache"


def available_subjects() -> list[str]:
    """Subjects with at least one recording on disk. A subject with only some of its
    recordings raises in build_corpus instead of being used silently incomplete."""
    if not (SIENA_ROOT / "RECORDS").exists():
        return []
    return [s for s in SUBJECTS
            if any(record_path(s, r).exists() for r in subject_records(s))]


def _decision_tag(decisions: dict) -> str:
    return "dec-" + "-".join(decisions[k] for k in sorted(decisions))


def build_corpus(subjects: list[str] | None = None, win_s: float = 10.0,
                 stride_s: float | None = None, neg_per_pos: float = 4.0,
                 guard_s: float = 0.0, seed: int = 20260727,
                 cache: bool = True, verbose: bool = True, signal_fn=None,
                 signal_tag: str = "", decisions: dict | None = None) -> dict:
    """The windowed, subsampled Siena corpus with group labels.

    Returns a dict with X (window, 18, 2560) float32, y, subject, group (= subject),
    seizure_id, record, t0 and info, exactly as chbmit_corpus.build_corpus.

    Subsampling happens within each subject, neg_per_pos non-ictal windows per ictal
    window, with one random generator seeded once and used subject by subject in a
    fixed order, as in CHB-MIT.

    signal_fn(x, fs, key) is applied to each continuous 256 Hz bipolar recording before
    windowing (src/preprocess.py: Pipeline.apply_signal); signal_tag names it in the
    cache file. Window selection does not depend on it, so every pipeline gets exactly
    the same windows.

    decisions (default DECISIONS) settles the ambiguities in the seizure lists; the
    choice is part of the cache name and recorded in info. Windows overlapping a
    seizure that the decisions exclude are dropped, not labelled.
    """
    decisions = dict(DECISIONS if decisions is None else decisions)
    check_decisions(decisions)
    subs = subjects or available_subjects()
    if not subs:
        raise RuntimeError(f"no Siena recordings under {SIENA_ROOT}; download with "
                           f"python -m src.siena_download")
    subset = "" if sorted(subs) == sorted(SUBJECTS) else \
        f"-{zlib.crc32(' '.join(sorted(subs)).encode()):08x}"
    tag = (f"w{win_s:g}_s{(stride_s or win_s):g}_n{neg_per_pos:g}"
           f"_g{guard_s:g}_seed{seed}_{len(subs)}subj{subset}_{_decision_tag(decisions)}"
           + (f"_{signal_tag}" if signal_tag else ""))
    npz = CACHE / f"corpus_{tag}.npz"
    if cache and npz.exists():
        if verbose:
            print(f"reading from cache: {npz.name}")
        d = np.load(npz, allow_pickle=False)
        info = json.loads((CACHE / f"corpus_{tag}.json").read_text(encoding="utf-8"))
        out = {k: d[k] for k in d.files}
        return out | {"group": out["subject"].copy(), "info": info}

    rng = np.random.default_rng(seed)
    Xs, ys, subj, sids, fnames, t0s = [], [], [], [], [], []
    per_subject = {}
    n = int(win_s * FS_CHB)

    for si, sub in enumerate(subs):
        names = subject_records(sub)
        missing = [r for r in names if not record_path(sub, r).exists()]
        if missing:
            raise RuntimeError(f"{sub}: recordings missing on disk: {missing}; run "
                               f"python -m src.siena_download --subjects {sub}")
        listed = parse_seizure_list(sub, decisions=decisions)

        # Pass 1: labels and seizure ids from headers and lists only.
        plan = []                                    # (file, t0, y, seizure_id)
        n_excl, ictal_s = 0, 0.0
        for name in names:
            path = record_path(sub, name)
            zs, excl, ids = seizures_in_record(path, listed, decisions)
            ictal_s += sum(z.duration for z in zs)
            y_f, t_f = plan_windows(path, zs, win_s, stride_s, guard_s)
            zid = [(z.start_s, z.end_s, i) for z, i in zip(zs, ids)]
            for yy, tt in zip(y_f, t_f):
                tt = float(tt)
                if any(min(tt + win_s, b) > max(tt, a) for a, b in excl):
                    n_excl += 1
                    continue
                sid = _window_seizure_id(tt, win_s, zid) if yy == 1 else NON_ICTAL
                plan.append((name, tt, int(yy), sid))
        if not plan:
            raise RuntimeError(f"{sub}: no windows produced")

        y_all = np.array([p[2] for p in plan])
        pos = np.flatnonzero(y_all == 1)
        neg = np.flatnonzero(y_all == 0)
        if len(pos) == 0:
            if verbose:
                print(f"{sub}: no ictal windows, skipped")
            continue
        keep_neg = neg
        if neg_per_pos > 0 and len(neg) > neg_per_pos * len(pos):
            keep_neg = rng.choice(neg, int(neg_per_pos * len(pos)), replace=False)
        idx = np.sort(np.concatenate([pos, keep_neg]))

        # Pass 2: read each recording once, convert, cut the selected windows.
        by_file: dict[str, list[tuple[int, float]]] = {}
        for j, i in enumerate(idx):
            by_file.setdefault(plan[i][0], []).append((j, plan[i][1]))
        X = np.empty((len(idx), len(TARGET_CHANNELS), n), np.float32)
        for name, items in by_file.items():
            x = read_bipolar(record_path(sub, name), signal_fn=signal_fn)
            for j, t in items:
                s = int(round(t * FS_CHB))
                X[j] = x[:, s:s + n]
            del x

        Xs.append(X)
        ys.append(y_all[idx])
        subj += [sub] * len(idx)
        sids += [plan[i][3] for i in idx]
        fnames += [plan[i][0] for i in idx]
        t0s += [plan[i][1] for i in idx]
        per_subject[sub] = {
            "n_files": len(names),
            "n_windows_total": int(len(plan)), "n_ictal": int(len(pos)),
            "n_nonictal_kept": int(len(keep_neg)),
            "n_seizures": int(len({p[3] for p in plan if p[3]})),
            "seizure_seconds": float(ictal_s),
            "n_windows_excluded": int(n_excl),
            # true (unsubsampled) recording time, needed for real false-alarm rates
            "hours": float((len(plan) + n_excl) * (stride_s or win_s) / 3600.0),
        }
        if verbose:
            print(f"[{si+1:2d}/{len(subs)}] {sub}: {len(idx):5d} windows "
                  f"({len(pos):4d} ictal, {per_subject[sub]['n_seizures']:2d} seizures, "
                  f"{per_subject[sub]['hours']:6.1f} hours)")

    out = {
        "X": np.concatenate(Xs),
        "y": np.concatenate(ys).astype(np.int64),
        "subject": np.array(subj),
        "seizure_id": np.array(sids),
        "record": np.array(fnames),
        "t0": np.array(t0s, np.float32),
    }
    info = {
        "corpus": "siena", "subjects": subs, "n_subjects": len(per_subject),
        "channels": list(TARGET_CHANNELS),
        "derivation": {ch: f"{a} - {b}" for ch, (a, b) in
                       zip(TARGET_CHANNELS, bipolar_pairs())},
        "fs": FS_CHB, "fs_source": FS_SIENA,
        "resampling": "scipy.signal.resample_poly(x, 1, 2), continuous recording",
        "win_s": win_s, "stride_s": stride_s or win_s, "guard_s": guard_s,
        "neg_per_pos": neg_per_pos, "seed": seed, "signal_tag": signal_tag,
        "decisions": decisions,
        "n_windows": int(len(out["y"])), "n_ictal": int(out["y"].sum()),
        "n_seizures": int(len({s for s in sids if s})),
        "per_subject": per_subject,
    }
    validate(out, info)

    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(npz, **out)
        (CACHE / f"corpus_{tag}.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        if verbose:
            print(f"cache written: {npz} ({npz.stat().st_size/1e6:.0f} MB)")
    return out | {"group": out["subject"].copy(), "info": info}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--win", type=float, default=10.0)
    ap.add_argument("--stride", type=float, default=None)
    ap.add_argument("--neg-per-pos", type=float, default=4.0)
    ap.add_argument("--guard", type=float, default=0.0)
    args = ap.parse_args()

    d = build_corpus(args.subjects, win_s=args.win, stride_s=args.stride,
                     neg_per_pos=args.neg_per_pos, guard_s=args.guard)
    info = d["info"]
    print(f"\ncorpus: {info['n_windows']} windows, {info['n_ictal']} ictal, "
          f"{info['n_seizures']} seizures, {info['n_subjects']} subjects, "
          f"{d['X'].shape[1]} channels x {d['X'].shape[2]} samples")
    print(f"decisions: {info['decisions']}")
    print(f"memory: {d['X'].nbytes/1e9:.2f} GB")

    if args.check:
        print("\n--- per subject ---")
        print(f"  {'subject':8s} {'files':>5s} {'hours':>6s} {'seizures':>8s} "
              f"{'seizure s':>9s} {'ictal win':>9s} {'kept':>6s} {'excluded':>8s}")
        for s, p in info["per_subject"].items():
            print(f"  {s:8s} {p['n_files']:5d} {p['hours']:6.2f} {p['n_seizures']:8d} "
                  f"{p['seizure_seconds']:9.0f} {p['n_ictal']:9d} "
                  f"{p['n_ictal'] + p['n_nonictal_kept']:6d} "
                  f"{p['n_windows_excluded']:8d}")

        n_subj = len(set(d["group"]))
        print("\n--- leakage check ---")
        if n_subj < 2:
            print(f"  only {n_subj} subject in the corpus, LOSO needs at least two; "
                  f"skipped")
        else:
            p1 = check_leakage(d, leave_one_subject_out(d["group"]), "LOSO")
            for p in p1:
                print("  PROBLEM:", p)
            if not p1:
                print(f"  LOSO ({n_subj} folds): no leakage")

        print("\n--- negative control: a deliberately leaky split must be caught ---")
        def bad_splitter():
            """Deliberately leaky split: windows shuffled at random into two halves."""
            rng = np.random.default_rng(0)
            idx = rng.permutation(len(d["y"]))
            half = len(idx) // 2
            yield "random", idx[:half], idx[half:]
        caught = check_leakage(d, bad_splitter(), "LEAKY")
        if caught:
            print(f"  check caught the deliberate leak ({len(caught)} findings):")
            for c in caught[:3]:
                print("   ", c[:110])
        else:
            print("  WARNING: the check did NOT catch the deliberate leak; it is broken")


if __name__ == "__main__":
    main()
