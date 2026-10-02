"""CHB-MIT corpus building and leakage-safe, person-wise splitting.

The point of this corpus is what Bonn cannot offer: person-wise evaluation. Bonn
publishes no subject identifiers, so there one cannot ask whether a model generalises
to a new patient.

Two leakage rules:

  1. No window of the same PERSON may be in both training and test. A person is the
     `group` field, not the case folder: chb01 and chb21 are the same person
     (SAME_SUBJECT), so LOSO has 23 folds, not 24.
  2. No single seizure's windows may be split across training and test. With
     overlapping windows, windows of one seizure are near-identical; if they are split
     the model's memorisation looks like generalisation. Rule 1 already implies this,
     but it must be checked separately if overlapping windows or within-person splits
     are ever tried, so seizure ids travel with the data and check_leakage tests both.

The loader never returns a silently empty set: counts of subjects, recordings and
channels are validated and mismatches raise. This project hit that failure mode
twice (chb17 skipped entirely because of its file-name pattern, and chb12's montage
making the channel intersection empty); both returned zero without an error.

Usage:
    python -m src.chbmit_corpus --build     # build the cache
    python -m src.chbmit_corpus --check     # leakage and integrity checks
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .chbmit import (CHB_ROOT, FS_CHB, TARGET_CHANNELS, extract_windows,
                     parse_summary, plan_windows, seizure_ids, subject_files)

CACHE = CHB_ROOT / "_cache"
NON_ICTAL = ""                      # seizure id of non-ictal windows
CASES = tuple(f"chb{i:02d}" for i in range(1, 25))

# PhysioNet: "Case chb21 was obtained 1.5 years after case chb01, from the same
# female subject." Splits and leakage checks use the person, not the case folder.
SAME_SUBJECT = {"chb21": "chb01"}


def subject_groups(cases: np.ndarray) -> np.ndarray:
    """Map case names to person groups: chb21 -> chb01, all others unchanged."""
    return np.array([SAME_SUBJECT.get(c, c) for c in cases])


def all_subjects() -> list[str]:
    return sorted(d.name for d in CHB_ROOT.glob("chb*") if d.is_dir())


def _window_seizure_id(t0: float, win_s: float,
                       zs: list[tuple[float, float, str]]) -> str:
    """Id of the seizure the window overlaps most; empty string if none."""
    best, best_ov = NON_ICTAL, 0.0
    for b0, b1, sid in zs:
        ov = max(0.0, min(t0 + win_s, b1) - max(t0, b0))
        if ov > best_ov:
            best, best_ov = sid, ov
    return best


def build_corpus(subjects: list[str] | None = None, win_s: float = 10.0,
                 stride_s: float | None = None, neg_per_pos: float = 4.0,
                 guard_s: float = 0.0, seed: int = 20260727,
                 cache: bool = True, verbose: bool = True, signal_fn=None,
                 signal_tag: str = "") -> dict:
    """The windowed, subsampled corpus with group labels.

    Returns a dict with X (window, channel, sample), y, subject (case folder), group
    (person, see SAME_SUBJECT), seizure_id, record, t0 and info.

    Subsampling happens WITHIN each case: every case keeps neg_per_pos non-ictal windows
    per ictal window of its own. One corpus-wide ratio would let cases with many
    seizures dominate the non-ictal class.

    The result is cached under data/chbmit/_cache/. With the cache present the raw EDF
    files are not needed at all.

    signal_fn(x, fs, key) is applied to each continuous recording before windowing (the
    signal-level steps of a preprocessing pipeline, src/preprocess.py); signal_tag names
    it in the cache file. Window selection does not depend on it, so every pipeline
    gets exactly the same windows.
    """
    # Without raw EDFs (cache-only use) the folder scan is empty; fall back to all cases.
    subs = subjects or all_subjects() or list(CASES)
    tag = (f"w{win_s:g}_s{(stride_s or win_s):g}_n{neg_per_pos:g}"
           f"_g{guard_s:g}_seed{seed}_{len(subs)}subj"
           + (f"_{signal_tag}" if signal_tag else ""))
    npz = CACHE / f"corpus_{tag}.npz"
    if cache and npz.exists():
        if verbose:
            print(f"reading from cache: {npz.name}")
        d = np.load(npz, allow_pickle=False)
        info = json.loads((CACHE / f"corpus_{tag}.json").read_text(encoding="utf-8"))
        out = {k: d[k] for k in d.files}
        return out | {"group": subject_groups(out["subject"]), "info": info}

    rng = np.random.default_rng(seed)
    Xs, ys, subj, sids, fnames, t0s = [], [], [], [], [], []
    per_subject = {}

    for si, sub in enumerate(subs):
        files_all = subject_files(sub, require_channels=False)
        files = subject_files(sub)
        if not files:
            raise RuntimeError(f"{sub}: no recording contains the target montage "
                               f"({len(files_all)} recordings scanned)")
        summ = parse_summary(CHB_ROOT / sub / f"{sub}-summary.txt")
        zmap = seizure_ids(sub)

        # Pass 1: labels and seizure ids from the headers only; no signal is read.
        plan = []                                    # (file, t0, y, seizure_id)
        for f in files:
            y_f, t_f = plan_windows(f, summ.get(f.name, []), win_s, stride_s, guard_s)
            zs = zmap.get(f.name, [])
            for yy, tt in zip(y_f, t_f):
                sid = _window_seizure_id(float(tt), win_s, zs) if yy == 1 else NON_ICTAL
                plan.append((f.name, float(tt), int(yy), sid))
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

        # Pass 2: read only the selected windows, one read per file.
        by_file: dict[str, list[tuple[int, float]]] = {}
        for j, i in enumerate(idx):
            by_file.setdefault(plan[i][0], []).append((j, plan[i][1]))
        X = np.empty((len(idx), len(TARGET_CHANNELS), int(win_s * FS_CHB)), np.float32)
        path_of = {f.name: f for f in files}
        for name, items in by_file.items():
            js = np.array([a for a, _ in items])
            ts = np.array([b for _, b in items], np.float32)
            X[js] = extract_windows(path_of[name], list(TARGET_CHANNELS), ts, win_s,
                                    signal_fn=signal_fn)

        Xs.append(X)
        ys.append(y_all[idx])
        subj += [sub] * len(idx)
        sids += [plan[i][3] for i in idx]
        fnames += [plan[i][0] for i in idx]
        t0s += [plan[i][1] for i in idx]
        per_subject[sub] = {
            "n_files": len(files), "n_files_skipped_montage": len(files_all) - len(files),
            "n_windows_total": int(len(plan)), "n_ictal": int(len(pos)),
            "n_nonictal_kept": int(len(keep_neg)),
            "n_seizures": int(len({p[3] for p in plan if p[3]})),
            # true (unsubsampled) recording time, needed for real false-alarm rates
            "hours": float(len(plan) * (stride_s or win_s) / 3600.0),
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
        "subjects": subs, "n_subjects": len(per_subject),
        "channels": list(TARGET_CHANNELS), "fs": FS_CHB,
        "win_s": win_s, "stride_s": stride_s or win_s, "guard_s": guard_s,
        "neg_per_pos": neg_per_pos, "seed": seed, "signal_tag": signal_tag,
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
    return out | {"group": subject_groups(out["subject"]), "info": info}


def validate(d: dict, info: dict) -> None:
    """Checks against silently empty or malformed data. Raises on failure."""
    n = len(d["y"])
    if n == 0:
        raise RuntimeError("corpus is empty")
    for k in ("X", "subject", "seizure_id", "record", "t0"):
        if len(d[k]) != n:
            raise RuntimeError(f"{k} has length {len(d[k])}, y has length {n}")
    if d["X"].shape[1] != len(TARGET_CHANNELS):
        raise RuntimeError(f"{d['X'].shape[1]} channels, "
                           f"expected {len(TARGET_CHANNELS)}")
    if d["X"].shape[2] != int(info["win_s"] * FS_CHB):
        raise RuntimeError(f"window length {d['X'].shape[2]}, "
                           f"expected {int(info['win_s'] * FS_CHB)}")
    if not np.isfinite(d["X"]).all():
        bad = int((~np.isfinite(d["X"])).sum())
        raise RuntimeError(f"{bad} non-finite values in the signal")
    if d["y"].sum() == 0:
        raise RuntimeError("no ictal windows")
    # every ictal window must carry a seizure id, and no non-ictal window may
    ict = d["y"] == 1
    if (d["seizure_id"][ict] == NON_ICTAL).any():
        raise RuntimeError("ictal window without a seizure id")
    if (d["seizure_id"][~ict] != NON_ICTAL).any():
        raise RuntimeError("non-ictal window with a seizure id")
    if len(set(d["subject"])) < 2:
        print("warning: fewer than two cases in the corpus, person-wise splits impossible")


# --- Splits ---------------------------------------------------------------------------

def leave_one_subject_out(subject: np.ndarray):
    """Each group in turn is the test set. Pass the `group` array (persons)."""
    for s in sorted(set(subject)):
        te = np.flatnonzero(subject == s)
        tr = np.flatnonzero(subject != s)
        yield s, tr, te


def grouped_kfold_by_subject(subject: np.ndarray, n_folds: int = 5,
                             seed: int = 20260727):
    """Distribute groups over k folds, for when LOSO is too expensive.

    Groups are sorted by window count and each is assigned to the currently least
    loaded fold, which balances fold sizes and keeps every fold's ictal count non-zero.
    """
    subs = sorted(set(subject))
    if n_folds > len(subs):
        raise ValueError(
            f"{n_folds} folds requested but only {len(subs)} groups exist; at least "
            f"one fold would be empty. Use fewer folds or leave_one_subject_out.")
    rng = np.random.default_rng(seed)
    counts = {s: int((subject == s).sum()) for s in subs}
    order = sorted(subs, key=lambda s: (-counts[s], s))
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    load = [0] * n_folds
    for s in order:
        k = int(np.argmin(load))
        folds[k].append(s)
        load[k] += counts[s]
    for k in range(n_folds):
        rng.shuffle(folds[k])
        te_subs = set(folds[k])
        te = np.flatnonzero(np.isin(subject, list(te_subs)))
        tr = np.flatnonzero(~np.isin(subject, list(te_subs)))
        yield sorted(te_subs), tr, te


def check_leakage(d: dict, splitter, name: str = "") -> list[str]:
    """Check that every split satisfies both leakage rules. Returns a list of problems."""
    problems = []
    group, sid, y = d["group"], d["seizure_id"], d["y"]
    for key, tr, te in splitter:
        if len(tr) == 0 or len(te) == 0:
            problems.append(f"{name} {key}: empty fold (train {len(tr)}, test {len(te)})")
            continue
        shared_s = set(group[tr]) & set(group[te])
        if shared_s:
            problems.append(f"{name} {key}: same person in train and test: "
                            f"{sorted(map(str, shared_s))}")
        a = {s for s in sid[tr] if s != NON_ICTAL}
        b = {s for s in sid[te] if s != NON_ICTAL}
        if a & b:
            problems.append(f"{name} {key}: seizure split across train and test: "
                            f"{sorted(map(str, a & b))[:5]}")
        if y[te].sum() == 0:
            problems.append(f"{name} {key}: no ictal window in the test fold")
        if y[tr].sum() == 0:
            problems.append(f"{name} {key}: no ictal window in the training fold")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--win", type=float, default=10.0)
    ap.add_argument("--stride", type=float, default=None)
    ap.add_argument("--neg-per-pos", type=float, default=4.0)
    ap.add_argument("--guard", type=float, default=0.0)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    d = build_corpus(args.subjects, win_s=args.win, stride_s=args.stride,
                     neg_per_pos=args.neg_per_pos, guard_s=args.guard)
    info = d["info"]
    print(f"\ncorpus: {info['n_windows']} windows, {info['n_ictal']} ictal, "
          f"{info['n_seizures']} seizures, {info['n_subjects']} cases "
          f"({len(set(d['group']))} persons), "
          f"{d['X'].shape[1]} channels x {d['X'].shape[2]} samples")
    print(f"memory: {d['X'].nbytes/1e9:.2f} GB")

    if args.check:
        n_subj = len(set(d["group"]))
        folds = min(args.folds, n_subj)
        if folds != args.folds:
            print(f"\nnote: {args.folds} folds requested but only {n_subj} persons, "
                  f"reduced to {folds}")

        print("\n--- leakage check ---")
        p1 = check_leakage(d, leave_one_subject_out(d["group"]), "LOSO")
        p2 = check_leakage(d, grouped_kfold_by_subject(d["group"], folds),
                           f"{folds}-fold")
        for p in p1 + p2:
            print("  PROBLEM:", p)
        if not (p1 or p2):
            print(f"  LOSO ({n_subj} folds) and grouped {folds}-fold: no leakage")

        print("\n--- fold balance (grouped k-fold) ---")
        for te_subs, tr, te in grouped_kfold_by_subject(d["group"], folds):
            print(f"  test persons {len(te_subs):2d}: {len(te):5d} windows, "
                  f"{int(d['y'][te].sum()):4d} ictal  |  train {len(tr):5d}")

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

        print("\n--- old case-based LOSO (chb01 and chb21 as separate subjects) ---")
        old = check_leakage(d, leave_one_subject_out(d["subject"]), "case-LOSO")
        if old:
            print(f"  check caught the same-person leak of the old split "
                  f"({len(old)} folds):")
            for c in old:
                print("   ", c[:110])
        else:
            print("  WARNING: the chb01/chb21 leak of the old split was NOT caught")


if __name__ == "__main__":
    main()
