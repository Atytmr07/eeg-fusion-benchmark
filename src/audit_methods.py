"""Methods audit: do the Methods describe what the code and the stored outputs do?

Written for the advisor's first-priority checks of 22 October 2026. Every check reads the
cached corpora, the stored run results and the code's own functions; nothing is
retrained. Each check prints PASS or FAIL and the report collects the numbers the
Methods quote.

  1. counts       persons, cases, recordings, seizures and windows per dataset; the
                  non-ictal to ictal ratio within each case; 198 annotated -> 185 in the
                  corpus -> reference events of the seizure-level scoring
  2. windows      every pipeline's corpus holds the same windows, labels and seizure ids
  3. partitions   every LOSO fold: training, validation and test persons disjoint, no
                  seizure on two sides, chb01/chb21 together; the held-out persons of
                  every stored run match; the validation persons are listed
  4. statistics   the corrected t test, the TOST bound, Holm and Kendall's W against
                  independent re-implementations; the ratios used; the operator
                  comparisons again with the conservative ratio
  5. events       the SzCORE parameters of timescoring as used
  6. reproducibility  the cross-machine numbers quoted in the manuscript against the
                  stored comparison tables

Usage:  python -m src.audit_methods          (needs the corpus caches; raw EDF headers
                                              for the seizure-level counts)
Output: results_v2/audit/methods_audit.md and CSV tables
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .chbmit_corpus import NON_ICTAL, check_leakage, leave_one_subject_out, subject_groups
from .chbmit_run import inner_split
from .chbmit_stats import FUSION, ratios
from .config import DATA_ROOT, RESULTS_ROOT
from .preprocess import PIPELINES
from .stats import corrected_equivalence_bound, corrected_ttest, holm

OUT = RESULTS_ROOT / "audit"
NAMES = {"chbmit": "CHB-MIT", "siena": "Siena"}
REPORT: list[str] = []
FAILS: list[str] = []


def say(line: str = "") -> None:
    print(line)
    REPORT.append(line)


def check(ok: bool, what: str) -> None:
    say(f"- {'PASS' if ok else 'FAIL'}: {what}")
    if not ok:
        FAILS.append(what)


CACHE_STEM = {"chbmit": "corpus_w10_s10_n4_g0_seed20260727_24subj",
              "siena": "corpus_w10_s10_n4_g0_seed20260727_14subj_dec-typo-first-electric"}


def corpus(ds: str, tag: str) -> dict:
    """A cached corpus without its signals (X is not loaded); a missing cache raises."""
    stem = DATA_ROOT / ds / "_cache" / (CACHE_STEM[ds] + (f"_{tag}" if tag else ""))
    npz = stem.with_name(stem.name + ".npz")
    if not npz.exists():
        raise FileNotFoundError(npz)
    with np.load(npz, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "X"}
    out["group"] = subject_groups(out["subject"]) if ds == "chbmit" else out["subject"].copy()
    out["info"] = json.loads(stem.with_name(stem.name + ".json").read_text(encoding="utf-8"))
    return out


# --- 1. counts --------------------------------------------------------------------------------

def counts(ds: str) -> dict:
    say(f"\n### {NAMES[ds]}: counts\n")
    d = corpus(ds, "")
    y, subj, grp = d["y"].astype(bool), np.asarray(d["subject"]), np.asarray(d["group"])
    sids = {s for s in d["seizure_id"] if s != NON_ICTAL}
    per = pd.DataFrame({"case": subj, "y": y}).groupby("case").y.agg(["sum", "size"])
    per["ratio"] = (per["size"] - per["sum"]) / per["sum"]
    r = {"persons": len(set(grp)), "cases": len(set(subj)),
         "recordings_in_corpus": len(set(map(str, d["record"]))), "seizures_in_corpus":
         len(sids), "windows": int(len(y)), "ictal": int(y.sum()),
         "nonictal": int((~y).sum()), "ratio_min": float(per.ratio.min()),
         "ratio_max": float(per.ratio.max())}
    for k, v in r.items():
        say(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    check(abs(r["nonictal"] / r["ictal"] - 4) < 1e-9, "four non-ictal windows per ictal window "
          "overall")
    check(r["ratio_min"] <= 4 + 1e-9 and r["ratio_max"] <= 4 + 1e-9,
          f"no case has more than four non-ictal windows per ictal window "
          f"(per case {r['ratio_min']:.2f} to {r['ratio_max']:.2f}; lower only where a case "
          f"has too few non-ictal windows)")
    check(d["info"].get("seed") == 20260727, "non-ictal windows drawn with the fixed seed "
          "20260727 (corpus metadata)")
    if ds == "chbmit":
        check(set(grp[subj == "chb21"]) == {"chb01"}, "chb21 is grouped with chb01")
    return r | {"_corpus": d}


def event_counts(ds: str) -> dict:
    """Annotated seizures in the scored recordings and the reference events that SzCORE
    forms from them (merge closer than 90 s, split longer than 5 min)."""
    from .events import WIN_S, masks_1hz, person_recordings, score_events
    from .chbmit import plan_windows, read_edf_header
    if ds == "siena":
        from .siena_corpus import available_subjects
        cases = available_subjects()
    else:
        from .chbmit_corpus import all_subjects
        cases = all_subjects()
    n_rec = n_ann = n_ref = 0
    hours = 0.0
    for c in cases:
        for path, zs, limit in person_recordings(ds, [c]):
            y, t0 = plan_windows(path, zs, WIN_S)
            dur = limit if limit is not None else read_edf_header(path).duration
            ref, _ = masks_1hz(dur, zs, t0, y == 1)
            nr, _, _ = score_events(ref, ref)
            n_rec += 1
            n_ann += len(zs)
            n_ref += nr
            hours += dur / 3600
    say(f"\n### {NAMES[ds]}: seizure-level scoring\n")
    say(f"  scored recordings: {n_rec}, recording time {hours:.1f} h")
    say(f"  annotated seizures in them: {n_ann}; reference events after SzCORE merging "
        f"and splitting: {n_ref}")
    return {"scored_recordings": n_rec, "annotated": n_ann, "reference_events": n_ref,
            "hours": hours}


def chbmit_source_counts() -> dict:
    from .chbmit import CHB_ROOT, parse_summary, read_edf_header, subject_files
    from .chbmit_corpus import all_subjects
    rec_all = rec_mont = sz_all = sz_mont = 0
    hours, dropped = 0.0, []
    for c in all_subjects():
        summ = parse_summary(CHB_ROOT / c / f"{c}-summary.txt")
        paths = subject_files(c, require_channels=False)
        files_all = {f.name for f in paths}
        files = {f.name for f in subject_files(c)}
        rec_all += len(files_all)
        rec_mont += len(files)
        sz_all += sum(len(v) for k, v in summ.items() if k in files_all)
        sz_mont += sum(len(v) for k, v in summ.items() if k in files)
        hours += sum(read_edf_header(f).duration for f in paths) / 3600
        dropped += [f"{n} ({len(summ.get(n, []))} seizures)" for n in sorted(files_all - files)]
    say("\n### CHB-MIT: source\n")
    say(f"  recordings: {rec_all} ({hours:.1f} h by the EDF headers), with the 18-channel "
        f"target montage: {rec_mont}")
    say(f"  annotated seizures: {sz_all}, in recordings with the target montage: {sz_mont}")
    say(f"  recordings without the target montage: {', '.join(dropped)}")
    return {"recordings": rec_all, "recordings_montage": rec_mont, "seizures": sz_all,
            "seizures_montage": sz_mont}


# --- 2. windows -------------------------------------------------------------------------------

def window_identity(ds: str, base: dict) -> None:
    say(f"\n### {NAMES[ds]}: the same windows in every pipeline\n")
    for p, pipe in PIPELINES.items():
        tag = pipe.for_dataset(ds).signal_tag
        try:
            d = base if not tag else corpus(ds, tag)
        except FileNotFoundError:
            say(f"- (no cache for {p}, tag {tag})")
            continue
        same = all(np.array_equal(np.asarray(d[k]), np.asarray(base[k]))
                   for k in ("t0", "record", "y", "seizure_id"))
        check(same, f"{p} ({tag or 'raw'}): windows, labels and seizure ids identical to P0")


# --- 3. partitions ----------------------------------------------------------------------------

def partitions(ds: str, d: dict) -> pd.DataFrame:
    say(f"\n### {NAMES[ds]}: LOSO partitions\n")
    grp, subj = np.asarray(d["group"]), np.asarray(d["subject"])
    sid, y = np.asarray(d["seizure_id"]), d["y"]
    rows, bad = [], []
    for fi, (key, tr_all, te) in enumerate(leave_one_subject_out(grp)):
        tr, va = inner_split(grp, tr_all, y)
        P = {k: set(grp[ix]) for k, ix in (("train", tr), ("val", va), ("test", te))}
        S = {k: {s for s in sid[ix] if s != NON_ICTAL} for k, ix in
             (("train", tr), ("val", va), ("test", te))}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if P[a] & P[b]:
                bad.append(f"fold {fi}: person in {a} and {b}: {P[a] & P[b]}")
            if S[a] & S[b]:
                bad.append(f"fold {fi}: seizure in {a} and {b}")
        if len(tr) + len(va) + len(te) != len(grp):
            bad.append(f"fold {fi}: windows lost or duplicated")
        rows.append({"fold": fi, "test": ",".join(sorted(set(subj[te]))),
                     "validation": ",".join(sorted(P["val"])), "n_train_persons":
                     len(P["train"]), "n_val_persons": len(P["val"]),
                     "train_windows": len(tr), "val_windows": len(va),
                     "test_windows": len(te), "val_ictal": int(y[va].sum())})
    t = pd.DataFrame(rows)
    check(not bad, f"{len(t)} folds: training, validation and test persons disjoint, no "
          f"seizure on two sides, every window in exactly one partition"
          + (f" ({bad[:3]})" if bad else ""))
    check((t.val_ictal > 0).all(), "every validation set holds ictal windows")
    rng = np.random.default_rng(0)                    # negative control, as in chbmit_corpus
    idx = rng.permutation(len(y))
    leak = check_leakage(d, iter([("random", idx[:len(idx) // 2], idx[len(idx) // 2:])]))
    check(bool(leak), "the leakage checker catches a deliberately leaky split (windows "
          "shuffled at random into two halves)")
    if ds == "chbmit":
        check(any(r == "chb01,chb21" for r in t.test), "chb01 and chb21 are held out together")
    # the stored runs held out the same persons in the same folds
    runs = sorted((RESULTS_ROOT / ds).glob("loso_P*_cuda")) + \
        sorted((RESULTS_ROOT / ds).glob("loso_P*_r*_cuda"))
    runs = sorted(set(runs))
    mism = []
    for r in runs:
        pf = pd.read_csv(r / "perfold.csv")
        got = pf.groupby("fold").test_subjects.first().astype(str)
        want = t.set_index("fold").test
        if not got.reindex(want.index).equals(want):
            mism.append(r.name)
    check(not mism, f"all {len(runs)} stored runs held out the same persons in the same "
          f"folds" + (f" (mismatch: {mism[:3]})" if mism else ""))
    say(f"  validation persons per fold: {t.n_val_persons.min()} to {t.n_val_persons.max()};"
        f" training persons {t.n_train_persons.min()} to {t.n_train_persons.max()}")
    return t


# --- 4. statistics ----------------------------------------------------------------------------

def statistics() -> None:
    say("\n### Statistical procedures against independent re-implementations\n")
    rng = np.random.default_rng(7)
    ok_t = ok_b = True
    for _ in range(200):
        n = int(rng.integers(10, 30))
        a, b = rng.normal(0.68, 0.1, n), rng.normal(0.68, 0.1, n) + rng.normal(0, 0.03, n)
        r = 1 / (n - 1)
        d = a - b
        se = np.sqrt((1 / n + r) * d.var(ddof=1))         # Nadeau and Bengio (2003)
        t = d.mean() / se
        p = 2 * sps.t.sf(abs(t), n - 1)
        t2, p2 = corrected_ttest(a, b, r)
        ok_t &= np.isclose(t, t2) and np.isclose(p, p2)
        # TOST: the smallest margin at which both one-sided tests reject at 0.05
        m = abs(d.mean()) + sps.t.ppf(0.95, n - 1) * se
        p_lo = sps.t.sf((d.mean() + m) / se, n - 1)       # H0: diff <= -m
        p_hi = sps.t.cdf((d.mean() - m) / se, n - 1)      # H0: diff >= +m
        ok_b &= np.isclose(corrected_equivalence_bound(a, b, r), m) and \
            np.isclose(max(p_lo, p_hi), 0.05)
    check(ok_t, "corrected t test: t = mean(d) / sqrt((1/n + n_test/n_train) var(d)), "
          "two-sided p on n-1 df (200 random cases)")
    check(ok_b, "equivalence bound delta_min = |mean(d)| + t(0.95, n-1) SE, i.e. both "
          "one-sided tests reject at alpha 0.05 exactly at delta_min (TOST)")
    p = np.array([0.01, 0.04, 0.03, 0.2])
    order = np.argsort(p)
    adj = np.empty(4)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (4 - i) * p[k]))
        adj[k] = run
    check(np.allclose(holm(p), adj), "Holm step-down adjustment")
    from .multiverse import kendall_w
    rk = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], float)
    rk2 = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], float)
    check(np.isclose(kendall_w(rk), 1.0) and np.isclose(kendall_w(rk2), 0.0),
          "Kendall's W: 1 for identical rankings, 0 for two opposite ones")
    for ds, k in (("chbmit", 23), ("siena", 14)):
        r1, r2 = ratios(k)
        say(f"  {NAMES[ds]}: {k} folds, test/train ratio 1/{round(1 / r1)} (primary), "
            f"1/{round(1 / r2)} with the validation persons removed (conservative)")


def conservative_ratio(ds: str) -> dict:
    """The operator comparisons of the multiverse again with the conservative ratio."""
    from .make_results import complete_runs
    from .multiverse import _pipeline_key, to_array
    inc, long, _ = complete_runs(RESULTS_ROOT / ds, ds, "cuda")
    pipes = sorted(long.pipeline.unique(), key=_pipeline_key)
    seeds, folds = sorted(long.seed.unique()), sorted(long.fold.unique())
    y = to_array(long, "f1_macro", list(FUSION), pipes, seeds, folds).mean(axis=3)
    out = {}
    for lab, r in zip(("primary", "conservative"), ratios(len(folds))):
        sig = eq5 = eq2 = n = 0
        for pi in range(len(pipes)):
            ps, bs = [], []
            for ia in range(5):
                for ib in range(ia + 1, 5):
                    a, b = y[:, pi, ia], y[:, pi, ib]
                    ps.append(corrected_ttest(a, b, r)[1])
                    bs.append(corrected_equivalence_bound(a, b, r))
            sig += int((holm(np.array(ps)) < 0.05).sum())
            eq5 += sum(x <= 0.05 for x in bs)
            eq2 += sum(x <= 0.02 for x in bs)
            n += len(ps)
        out[lab] = (sig, eq5, eq2, n)
        say(f"  {NAMES[ds]}, {lab} ratio: significant {sig}/{n}, equivalent within 0.05 "
            f"{eq5}/{n}, within 0.02 {eq2}/{n}")
    return out


# --- 5. events ---------------------------------------------------------------------------------

def event_parameters() -> None:
    say("\n### Seizure-level scoring parameters (timescoring, as used)\n")
    from timescoring import scoring
    p = scoring.EventScoring.Parameters()
    vals = {k: getattr(p, k) for k in ("toleranceStart", "toleranceEnd", "minOverlap",
                                       "maxEventDuration", "minDurationBetweenEvents")}
    say("  " + ", ".join(f"{k} = {v}" for k, v in vals.items()))
    check(vals["toleranceStart"] == 30 and vals["toleranceEnd"] == 60 and
          vals["minOverlap"] == 0 and vals["maxEventDuration"] == 300 and
          vals["minDurationBetweenEvents"] == 90,
          "30 s before onset, 60 s after offset, any overlap, events longer than 5 min "
          "split, events closer than 90 s merged (as in Section 2.5)")
    from .events import THRESHOLD
    check(THRESHOLD == 0.5, "seizure-level threshold 0.5 for every model")


# --- 6. reproducibility -----------------------------------------------------------------------

def device_check() -> None:
    """The CPU-GPU check of docs/PIPELINE.md: training time per fold and the macro F1
    difference between devices against that between two CPU seed sets."""
    say("\n### Device check (CHB-MIT, first folds of P0)\n")
    root = RESULTS_ROOT / "chbmit"
    runs = ("gpu_a", "gpu_b", "cpu_r1", "loso_grouped", "loso_P0_cuda")
    if not all((root / r / "perfold.csv").exists() for r in runs):
        say("- (gpu_a, gpu_b, cpu_r1 or loso_grouped not present)")
        return
    pf = {r: pd.read_csv(root / r / "perfold.csv") for r in runs}

    def diff(a: str, b: str) -> float:
        m = pf[a].merge(pf[b], on=["fold", "model"], suffixes=("_a", "_b"))
        return float((m.f1_macro_a - m.f1_macro_b).abs().groupby(m.model).mean().mean())

    gg, cg = diff("gpu_a", "gpu_b"), diff("gpu_a", "loso_grouped")
    cc = diff("cpu_r1", "loso_grouped")
    say(f"  mean |macro F1 difference|: two GPU runs {gg:.4f}; GPU against CPU, same seeds "
        f"{cg:.4f}; two CPU seed sets {cc:.4f}")
    check(gg == 0, "two GPU runs of the same fold are identical")
    say(f"  criterion (CPU-GPU difference no larger than the seed difference): "
        f"{'met' if cg <= cc else 'narrowly missed'} ({cg:.3f} against {cc:.3f})")
    for r, d in pf.items():
        say(f"  {r}: {d.fold.nunique()} folds, training time per fold "
            f"{d.groupby('fold').sec.sum().mean() / 60:.1f} min (sum over its models)")


def reproducibility() -> None:
    say("\n### Cross-machine numbers quoted in the manuscript\n")
    rp = RESULTS_ROOT / "ica_repro"
    t = pd.read_csv(rp / "logs_cache_vs_lab.csv")
    t = t[t.record.str.startswith("chb")]
    g = t[t.method == "gedai"]
    i = t[t.method == "infomax"]
    a = t[t.method == "amica"]
    share_g = (g.abs_dpower > 1e-3).mean()
    say(f"  CHB-MIT, laptop vs lab: GEDAI {len(g)} recordings, retained power differs by "
        f"> 0.001 in {(g.abs_dpower > 1e-3).sum()} ({100 * share_g:.1f} %); Infomax "
        f"{len(i)} recordings, different components in {(~i.same_removed).sum()} "
        f"({100 * (~i.same_removed).mean():.1f} %); AMICA different components in "
        f"{(~a.same_removed).sum()}, largest power difference {a.abs_dpower.max():.4f}")
    check(abs(100 * share_g - 27.5) < 0.6, "GEDAI: about 27 percent of recordings cleaned "
          "differently on the second machine")
    check((~a.same_removed).sum() == 0 and a.abs_dpower.max() < 0.003 + 1e-9,
          "AMICA: same components everywhere, retained power within 0.003")
    rr = pd.read_csv(rp / "logs_cache_vs_laptop_rerun.csv")
    r = rr[rr.record.str.startswith("chb") & (rr.method == "infomax")]
    say(f"  CHB-MIT, same laptop, rerun: Infomax different components in "
        f"{(~r.same_removed).sum()} of {len(r)} ({100 * (~r.same_removed).mean():.1f} %)")
    a2 = rr[rr.method == "amica"]
    say(f"  same laptop, rerun (CHB-MIT and Siena): AMICA different components in "
        f"{(~a2.same_removed).sum()} of {len(a2)}, largest power difference "
        f"{a2.abs_dpower.max():.4f}")
    imp = pd.read_csv(rp / "impact_events_summary.csv")
    say("  seizure-level impact (impact_events_summary.csv):")
    for line in imp.to_string(index=False).splitlines():
        say("    " + line)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    say("# Methods audit")
    say("")
    say("Generated by `python -m src.audit_methods`; checks the Methods against the code "
        "and the stored outputs (advisor's first-priority list, 22 October 2026).")
    summary = {}
    src = chbmit_source_counts()
    for ds in ("chbmit", "siena"):
        c = counts(ds)
        d = c.pop("_corpus")
        window_identity(ds, d)
        part = partitions(ds, d)
        part.to_csv(OUT / f"partitions_{ds}.csv", index=False)
        ev = event_counts(ds)
        summary[ds] = c | ev
    say("\n### CHB-MIT: why 198, 185 and 187\n")
    say(f"  {src['seizures']} annotated seizures; {src['seizures_montage']} lie in the "
        f"{src['recordings_montage']} recordings with the 18-channel target montage and "
        f"form the windowed corpus ({summary['chbmit']['seizures_in_corpus']}). The "
        f"seizure-level scoring uses the same recordings; SzCORE splits events longer than "
        f"5 min and merges events closer than 90 s, which turns the "
        f"{summary['chbmit']['annotated']} annotated seizures into "
        f"{summary['chbmit']['reference_events']} reference events.")
    check(src["seizures_montage"] == summary["chbmit"]["seizures_in_corpus"],
          "seizures in target-montage recordings = seizures in the corpus")
    statistics()
    say("\n### Operator comparisons with the conservative ratio\n")
    for ds in ("chbmit", "siena"):
        conservative_ratio(ds)
    event_parameters()
    device_check()
    reproducibility()
    pd.DataFrame(summary).T.to_csv(OUT / "counts.csv")
    say("")
    say(f"**{len(FAILS)} check(s) failed.**" if FAILS else "**All checks passed.**")
    (OUT / "methods_audit.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    print(f"written: {OUT / 'methods_audit.md'}")


if __name__ == "__main__":
    main()
