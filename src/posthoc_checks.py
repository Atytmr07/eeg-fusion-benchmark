"""Statistical checks of the post hoc sensitivity analyses (advisor's review of 26 October 2026).

Reads what src/sensitivity_preds.py wrote and the stored test predictions; nothing is
trained or run again.

  1. holm        threshold sensitivity (window level): every operator pair in every
                 pipeline at 0.3, 0.5 and 0.7, with its raw corrected p, the Holm
                 adjustment within the ten pairs of one threshold (the family of
                 Section 2.6) and, as a more conservative sensitivity check, across the
                 thirty comparisons of one dataset and pipeline (3 thresholds x 10 pairs)
  2. bootstrap   the corrected paired t test next to the subject-level bootstrap, per
                 operator pair: mean difference, both 95% intervals, corrected p,
                 resampling unit, iterations, interval method, direction consistency
  3. prevalence  CHB-MIT prevalence-adjusted macro F1: the weight of every case, the
                 weighted confusion matrix, subject-level (each held-out person's macro F1,
                 then the mean over persons) against pooled aggregation (the confusion
                 matrices of all persons of a run summed first), and the decline of each
                 model in absolute and relative terms
  4. pairs       CHB-MIT prevalence-adjusted macro F1 and AUPRC: every operator pair in
                 every pipeline with the procedure of the primary analysis (raw corrected
                 p, Holm within the ten pairs of one pipeline)

Usage:  python -m src.posthoc_checks          (after python -m src.sensitivity_preds)
Output: results_v2/sensitivity_preds/posthoc_checks.md and CSV tables
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import f1_score

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .audit_methods import corpus
from .chbmit_stats import FUSION, ratios
from .config import RESULTS_ROOT
from .make_results import complete_runs
from .multiverse import _pipeline_key, load_preds, md_table, to_array
from .sensitivity_preds import MODELS, N_BOOT, THRESHOLDS, arr, prevalence_weights
from .stats import corrected_se, corrected_ttest, holm

OUT = RESULTS_ROOT / "sensitivity_preds"
NAMES = {"chbmit": "CHB-MIT", "siena": "Siena"}
PAIRS = list(itertools.combinations(range(len(FUSION)), 2))
REPORT: list[str] = []


def say(line: str = "") -> None:
    print(line)
    REPORT.append(line)


def md(df: pd.DataFrame, digits: int = 3) -> str:
    return md_table(df, digits=digits)


def fmt_p(p: float) -> str:
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


# --- 1. Holm families at the three thresholds ------------------------------------------------

def holm_thresholds(ds: str) -> pd.DataFrame:
    fm = pd.read_csv(OUT / f"fold_metrics_{ds}.csv")
    r, _ = ratios(fm.fold.nunique())
    rows = []
    for t in THRESHOLDS:
        y, pipes = arr(fm, f"f1_macro@{t}", FUSION)
        for pi, p in enumerate(pipes):
            ps = [corrected_ttest(y[:, pi, a], y[:, pi, b], r)[1] for a, b in PAIRS]
            for (a, b), pr, ph in zip(PAIRS, ps, holm(np.array(ps))):
                rows.append({"dataset": ds, "pipeline": p, "threshold": t,
                             "pair": f"{FUSION[a]}-{FUSION[b]}",
                             "mean_diff": float((y[:, pi, a] - y[:, pi, b]).mean()),
                             "p_raw": pr, "p_holm_within": float(ph)})
    d = pd.DataFrame(rows)
    d["p_holm_across"] = np.nan
    for _, g in d.groupby("pipeline"):                    # 3 thresholds x 10 pairs
        d.loc[g.index, "p_holm_across"] = holm(g.p_raw.to_numpy())
    return d


def report_holm(tabs: list[pd.DataFrame]) -> None:
    say("## 1. Threshold sensitivity: two Holm families\n")
    say("Corrected resampled t test (Nadeau and Bengio; test/train ratio 1/22 on CHB-MIT, "
        "1/13 on Siena) on the fold values of macro F1 at each threshold, seed sets "
        "averaged within a fold, as in the primary analysis. **Within**: Holm over the ten "
        "operator pairs of one dataset, pipeline and threshold (the family of Section 2.6, "
        "applied to each threshold separately). **Across**: Holm over the thirty "
        "comparisons of one dataset and pipeline (three thresholds x ten pairs), an "
        "additional, more conservative sensitivity check.\n")
    d = pd.concat(tabs)
    d.to_csv(OUT / "posthoc_holm_thresholds.csv", index=False)
    cnt = d.groupby(["dataset", "threshold"]).agg(
        within=("p_holm_within", lambda s: f"{(s < 0.05).sum()}/{len(s)}"),
        across=("p_holm_across", lambda s: f"{(s < 0.05).sum()}/{len(s)}"),
        smallest_raw_p=("p_raw", "min")).reset_index()
    cnt["dataset"] = cnt.dataset.map(NAMES)
    cnt["threshold"] = cnt.threshold.map("{:.1f}".format)
    say("Pairs significant at 0.05:\n")
    say(md(cnt.rename(columns={"within": "Holm within threshold",
                               "across": "Holm across thresholds",
                               "smallest_raw_p": "smallest raw p"}), 4))
    show = d[(d.p_holm_within < 0.05) | (d.p_holm_across < 0.05)].copy()
    say("\nEvery comparison significant under either correction:\n")
    if show.empty:
        say("(none)")
    else:
        show["dataset"] = show.dataset.map(NAMES)
        show["threshold"] = show.threshold.map("{:.1f}".format)
        for c in ("p_raw", "p_holm_within", "p_holm_across"):
            show[c] = show[c].map(fmt_p)
        say(md(show.rename(columns={"mean_diff": "mean difference (first minus second)",
                                    "p_raw": "raw p", "p_holm_within": "Holm within",
                                    "p_holm_across": "Holm across"})))
    say("")


# --- 2. Corrected t test against the subject bootstrap ----------------------------------------

def bootstrap_compare(ds: str) -> pd.DataFrame:
    """The bootstrap of sensitivity_preds.bootstrap, drawn again with the same generator, so
    the intervals are those of bootstrap_<ds>.csv; checked below."""
    _, long, _ = complete_runs(RESULTS_ROOT / ds, ds, "cuda")
    pipes = sorted(long.pipeline.unique(), key=_pipeline_key)
    seeds, folds = sorted(long.seed.unique()), sorted(long.fold.unique())
    y = to_array(long, "f1_macro", list(MODELS), pipes, seeds, folds)   # fold, pipe, model, seed
    n, k = len(folds), len(seeds)
    rng = np.random.default_rng(20261022)
    bi = rng.integers(0, n, (N_BOOT, n))
    bs = rng.integers(0, k, (N_BOOT, n, k))
    r, _ = ratios(n)
    tq = sps.t.ppf(0.975, n - 1)
    stored = pd.read_csv(OUT / f"bootstrap_{ds}.csv").set_index(["pipeline", "pair"])
    rows = []
    for pi, p in enumerate(pipes + ["all"]):
        for a, b in PAIRS:
            # d4: (subject, seed set) differences of the two operators; a resample draws
            # subjects, and with them both operators' results (paired resampling)
            d4 = ((y[:, :, a] - y[:, :, b]).mean(1) if p == "all" else y[:, pi, a] - y[:, pi, b])
            d = d4.mean(1)                                           # per subject
            one = d[bi].mean(1)
            two = np.take_along_axis(d4[bi], bs, axis=2).mean((1, 2))
            se = corrected_se(d, r)
            m = float(d.mean())
            lo1, hi1 = np.percentile(one, [2.5, 97.5])
            lo2, hi2 = np.percentile(two, [2.5, 97.5])
            pair = f"{FUSION[a]}-{FUSION[b]}"
            ok = stored.loc[(p, pair), "subject 95% CI"] == f"[{lo1:+.3f}, {hi1:+.3f}]"
            rows.append({
                "dataset": ds, "pipeline": p, "pair": pair, "mean difference": m,
                "corrected 95% CI": f"[{m - tq * se:+.3f}, {m + tq * se:+.3f}]",
                "corrected p": corrected_ttest(d, np.zeros_like(d), r)[1],
                "bootstrap 95% CI (subject)": f"[{lo1:+.3f}, {hi1:+.3f}]",
                "bootstrap 95% CI (subject, then seed set)": f"[{lo2:+.3f}, {hi2:+.3f}]",
                "direction consistency": float(np.mean(np.sign(one) == np.sign(m))),
                "bootstrap (subject) excludes 0": bool(lo1 > 0 or hi1 < 0),
                "bootstrap (subject, then seed set) excludes 0": bool(lo2 > 0 or hi2 < 0),
                "matches bootstrap_<ds>.csv": bool(ok)})
    return pd.DataFrame(rows)


def report_bootstrap(tabs: list[pd.DataFrame]) -> None:
    say("## 2. Corrected paired t test and subject-level bootstrap\n")
    say(f"- **Resampling unit**: the held-out subject (person; chb01 and chb21 together). A "
        f"resample draws subjects with replacement and takes, for each drawn subject, the "
        f"paired difference of the two operators, so both operators' results of a subject "
        f"always enter together; operators are never resampled separately.\n"
        f"- **Second variant**: within each drawn subject, the five seed sets are also drawn "
        f"with replacement (the same seed sets for both operators).\n"
        f"- **Iterations**: {N_BOOT:,} per dataset (generator seed 20261022); the same draws "
        f"are used for every pair.\n"
        f"- **Interval**: percentile 95% interval (2.5th and 97.5th percentiles).\n"
        f"- **Corrected test**: mean difference +- t(0.975, n-1) x SE, with SE^2 = (1/n + "
        f"n_test/n_train) var(d) (Nadeau and Bengio), the same fold values (seed sets "
        f"averaged).\n"
        f"- **Direction consistency**: share of bootstrap means with the sign of the observed "
        f"mean difference.\n"
        f"- The bootstrap treats subjects as independent and does not model the overlap of "
        f"the LOSO training sets, which the corrected test accounts for. The two rest on "
        f"different dependence assumptions; the corrected test stays the primary analysis "
        f"and the bootstrap is reported as inference sensitivity.\n")
    d = pd.concat(tabs)
    d.to_csv(OUT / "posthoc_bootstrap_compare.csv", index=False)
    say(f"Check: the bootstrap intervals drawn again equal those of bootstrap_<ds>.csv in "
        f"{int(d['matches bootstrap_<ds>.csv'].sum())} of {len(d)} rows.\n")
    # nominal against nominal: neither interval is adjusted for multiplicity, so only
    # the Holm column is comparable with the primary analysis
    per = d[d.pipeline != "all"].reset_index(drop=True)
    per["p_holm"] = np.nan
    for _, g in per.groupby(["dataset", "pipeline"]):
        per.loc[g.index, "p_holm"] = holm(g["corrected p"].to_numpy())
    cnt = per.groupby("dataset").agg(
        corrected_nominal=("corrected p", lambda s: f"{(s < 0.05).sum()}/{len(s)}"),
        corrected_holm=("p_holm", lambda s: f"{(s < 0.05).sum()}/{len(s)}"),
        boot_subject=("bootstrap (subject) excludes 0", lambda s: f"{s.sum()}/{len(s)}"),
        boot_seed=("bootstrap (subject, then seed set) excludes 0",
                   lambda s: f"{s.sum()}/{len(s)}")).reset_index()
    cnt["dataset"] = cnt.dataset.map(NAMES)
    say("Operator pairs per pipeline whose 95% interval excludes zero. The corrected interval "
        "and both bootstrap intervals are nominal (not adjusted for multiplicity); only the "
        "Holm column, the primary analysis, is adjusted (ten pairs of one pipeline):\n")
    say(md(cnt.rename(columns={
        "corrected_nominal": "corrected test, nominal (raw p < 0.05)",
        "corrected_holm": "corrected test, Holm-adjusted p < 0.05",
        "boot_subject": "bootstrap (subject), nominal",
        "boot_seed": "bootstrap (subject, then seed set), nominal"})))
    say("")
    show = d[d.pipeline == "all"].drop(columns=[
        "pipeline", "matches bootstrap_<ds>.csv", "bootstrap (subject) excludes 0",
        "bootstrap (subject, then seed set) excludes 0"])
    show["dataset"] = show.dataset.map(NAMES)
    show["corrected p"] = show["corrected p"].map(fmt_p)
    say("Pooled over the nine pipelines (each subject's difference averaged over pipelines "
        "and seed sets; positive: first operator higher):\n")
    say(md(show))
    say("\nPer pipeline (90 pairs per dataset) in posthoc_bootstrap_compare.csv.\n")


# --- 3. Prevalence weighting --------------------------------------------------------------------

def prevalence_audit() -> None:
    ds = "chbmit"
    weights, info = prevalence_weights(ds)
    d = corpus(ds, "")
    subj = np.asarray(d["subject"])
    say("## 3. CHB-MIT prevalence-adjusted macro F1\n")
    per = pd.DataFrame(d["info"]["per_subject"]).T
    tab = pd.DataFrame({
        "case": per.index, "windows (all, 10 s)": per.n_windows_total.astype(int),
        "ictal": per.n_ictal.astype(int),
        "non-ictal, full": (per.n_windows_total - per.n_ictal).astype(int),
        "non-ictal, sampled": per.n_nonictal_kept.astype(int)})
    tab["weight w"] = tab["non-ictal, full"] / tab["non-ictal, sampled"]
    tab.to_csv(OUT / "posthoc_prevalence_weights.csv", index=False)
    say("**A. Definition.** For each case (CHB-MIT case folder; chb21 belongs to the same "
        "person as chb01 and keeps its own weight):\n\n"
        "    w = N_nonictal,full / N_nonictal,sampled        (ictal windows: weight 1)\n\n"
        "N_nonictal,full counts every non-overlapping 10 s window of the case's 683 "
        "target-montage recordings that does not overlap a seizure; N_nonictal,sampled "
        "the non-ictal windows drawn for the corpus (four per ictal window). Every ictal "
        "window is in the corpus, so the ictal class needs no weight. With prediction at "
        "the 0.5 threshold, each test window adds its weight to one cell of the confusion "
        "matrix:\n\n"
        "    TP = sum over ictal windows predicted ictal (weight 1)\n"
        "    FN = sum over ictal windows predicted non-ictal (weight 1)\n"
        "    FP = sum of w over non-ictal windows predicted ictal\n"
        "    TN = sum of w over non-ictal windows predicted non-ictal\n"
        "    F1_ictal = 2 TP / (2 TP + FP + FN),  F1_nonictal = 2 TN / (2 TN + FN + FP)\n"
        "    macro F1 = (F1_ictal + F1_nonictal) / 2\n\n"
        "So TP and FN are unchanged, FP and TN are scaled up to the full non-ictal count. "
        "This assumes that the sampled non-ictal windows have the error rate of the "
        "non-ictal windows that were not sampled (they are a random sample within each "
        "case, seed 20260727); the result is a prevalence-adjusted estimate, not a "
        "measurement on all windows.\n")
    say(md(tab.drop(columns=[]), 1))
    say(f"\nAll cases: {int(tab['non-ictal, full'].sum()):,} non-ictal windows, of which "
        f"{int(tab['non-ictal, sampled'].sum()):,} sampled; weights {tab['weight w'].min():.1f} "
        f"to {tab['weight w'].max():.1f}.\n")

    runs, _, _ = complete_runs(RESULTS_ROOT / ds, ds, "cuda")
    rows, max_dev = [], 0.0
    for r in runs:
        for fold, p in load_preds(r).items():
            y, idx = p["y_te"].astype(int), p["idx_te"]
            w = weights[idx]
            for m in MODELS:
                pred = (p[m][:, 1] > 0.5).astype(int)
                tp = float(((pred == 1) & (y == 1)).sum())
                fn = float(((pred == 0) & (y == 1)).sum())
                fp_s = float(((pred == 1) & (y == 0)).sum())
                tn_s = float(((pred == 0) & (y == 0)).sum())
                fp = float(w[(pred == 1) & (y == 0)].sum())
                tn = float(w[(pred == 0) & (y == 0)].sum())
                rows.append({"pipeline": r.pipeline, "seed": r.seed, "fold": fold, "model": m,
                             "tp": tp, "fn": fn, "fp_sampled": fp_s, "tn_sampled": tn_s,
                             "fp_weighted": fp, "tn_weighted": tn})
                if fold == 0:                                   # formula against sklearn
                    ref = f1_score(y, pred, average="macro", sample_weight=w, zero_division=0)
                    max_dev = max(max_dev, abs(macro_f1(tp, fn, fp, tn) - ref))
    cm = pd.DataFrame(rows)
    cm.to_csv(OUT / "posthoc_prevalence_confusion.csv", index=False)
    say(f"Check: the formula above reproduces scikit-learn's sample-weighted macro F1 to "
        f"{max_dev:.1e} (fold 0 of every run and model).\n")

    cm["f1_sampled"] = [macro_f1(*v) for v in cm[["tp", "fn", "fp_sampled", "tn_sampled"]].values]
    cm["f1_adjusted"] = [macro_f1(*v) for v in cm[["tp", "fn", "fp_weighted", "tn_weighted"]].values]
    cm["fpr"] = cm.fp_sampled / (cm.fp_sampled + cm.tn_sampled)
    cm["precision_adjusted"] = (cm.tp / (cm.tp + cm.fp_weighted)).fillna(0.0)
    subj_lvl = cm.groupby(["pipeline", "seed", "model"])[
        ["f1_sampled", "f1_adjusted", "fpr", "precision_adjusted"]].mean()
    pooled = cm.groupby(["pipeline", "seed", "model"])[
        ["tp", "fn", "fp_sampled", "tn_sampled", "fp_weighted", "tn_weighted"]].sum()
    pooled["f1_sampled"] = [macro_f1(*v) for v in pooled[["tp", "fn", "fp_sampled",
                                                         "tn_sampled"]].values]
    pooled["f1_adjusted"] = [macro_f1(*v) for v in pooled[["tp", "fn", "fp_weighted",
                                                          "tn_weighted"]].values]
    pooled["fpr"] = pooled.fp_sampled / (pooled.fp_sampled + pooled.tn_sampled)
    pooled["precision_adjusted"] = pooled.tp / (pooled.tp + pooled.fp_weighted)
    out = []
    for lab, t in (("subject-level", subj_lvl), ("pooled", pooled)):
        g = t.groupby("model")[["f1_sampled", "f1_adjusted", "fpr", "precision_adjusted"]]
        mean, sd = g.mean(), g.std()
        for mdl in MODELS:
            s, a = mean.loc[mdl, "f1_sampled"], mean.loc[mdl, "f1_adjusted"]
            out.append({"aggregation": lab, "model": mdl, "sampled": s,
                        "sampled SD": sd.loc[mdl, "f1_sampled"], "adjusted": a,
                        "adjusted SD": sd.loc[mdl, "f1_adjusted"], "decline": s - a,
                        "relative decline": (s - a) / s,
                        "false positive rate at 0.5": mean.loc[mdl, "fpr"],
                        "precision at 0.5 (adjusted)": mean.loc[mdl, "precision_adjusted"]})
    res = pd.DataFrame(out)
    res.to_csv(OUT / "posthoc_prevalence_summary.csv", index=False)
    say("**B. Aggregation.** Subject-level: macro F1 from each held-out person's weighted "
        "confusion matrix, the mean over the 23 persons with equal weight, then the mean "
        "and SD over the 45 runs (nine pipelines x five seed sets); this is how "
        "sensitivity_preds computed it. Pooled: the weighted confusion matrices of all "
        "persons of a run summed first, then one macro F1 per run.\n")
    say(md(res))
    say("")
    for lab in ("subject-level", "pooled"):
        x = res[res.aggregation == lab].set_index("model")
        ops = x.loc[list(FUSION)]
        say(f"- {lab}: operators {ops.sampled.min():.3f}-{ops.sampled.max():.3f} sampled, "
            f"{ops.adjusted.min():.3f}-{ops.adjusted.max():.3f} adjusted; largest absolute "
            f"decline {x.decline.idxmax()} ({x.decline.max():.3f}), largest relative decline "
            f"{x['relative decline'].idxmax()} ({100 * x['relative decline'].max():.1f} %)")
    say("")


def prevalence_pairs() -> None:
    """The operator pairs under prevalence adjustment, tested as in the primary analysis."""
    fm = pd.read_csv(OUT / "fold_metrics_chbmit.csv")
    n = fm.fold.nunique()
    r, _ = ratios(n)
    tq = sps.t.ppf(0.975, n - 1)
    say("## 4. CHB-MIT prevalence-adjusted operator comparisons\n")
    say(f"The procedure of the primary analysis (Section 2.6) applied to the prevalence-"
        f"adjusted fold values: per held-out subject the fold value of each operator, seed "
        f"sets averaged; the corrected resampled t test (Nadeau and Bengio, test/train ratio "
        f"1/22, {n - 1} degrees of freedom) on the {n} paired differences; Holm adjustment "
        f"over the ten operator pairs of one pipeline (nine families). Post hoc and "
        f"exploratory.\n")
    rows = []
    for col in ("nat_f1_macro", "nat_auprc"):
        y, pipes = arr(fm, col, FUSION)
        for pi, p in enumerate(pipes):
            res = []
            for a, b in PAIRS:
                d = y[:, pi, a] - y[:, pi, b]
                t, pr = corrected_ttest(y[:, pi, a], y[:, pi, b], r)
                se = corrected_se(d, r)
                res.append({"metric": col, "pipeline": p, "pair": f"{FUSION[a]}-{FUSION[b]}",
                            "mean difference": float(d.mean()),
                            "corrected 95% CI": f"[{d.mean() - tq * se:+.3f}, "
                                                f"{d.mean() + tq * se:+.3f}]",
                            "t": t, "p_raw": pr})
            for row, ph in zip(res, holm(np.array([x["p_raw"] for x in res]))):
                row["p_holm"] = float(ph)
            rows += res
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "posthoc_prevalence_pairs.csv", index=False)
    cnt = d.groupby("metric").agg(
        nominal=("p_raw", lambda s: f"{(s < 0.05).sum()}/{len(s)}"),
        holm=("p_holm", lambda s: f"{(s < 0.05).sum()}/{len(s)}")).reset_index()
    say(md(cnt.rename(columns={"nominal": "raw p < 0.05", "holm": "Holm-adjusted p < 0.05"})))
    show = d[d.p_holm < 0.05].copy()
    say("\nHolm-significant comparisons:\n")
    for c in ("p_raw", "p_holm"):
        show[c] = show[c].map(fmt_p)
    say(md(show.rename(columns={"p_raw": "raw p", "p_holm": "Holm p"})) if len(show)
        else "(none)")
    say("")


def macro_f1(tp: float, fn: float, fp: float, tn: float) -> float:
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    f0 = 2 * tn / (2 * tn + fn + fp) if tn + fn + fp else 0.0
    return (f1 + f0) / 2


def main() -> None:
    say("# Checks of the post hoc sensitivity analyses\n")
    say("Generated by `python -m src.posthoc_checks` (advisor's review of 26 October 2026) "
        "from the outputs of `src/sensitivity_preds.py` and the stored predictions; "
        "nothing was trained or run again. All analyses here are post hoc and exploratory.\n")
    report_holm([holm_thresholds(ds) for ds in ("chbmit", "siena")])
    report_bootstrap([bootstrap_compare(ds) for ds in ("chbmit", "siena")])
    prevalence_audit()
    prevalence_pairs()
    (OUT / "posthoc_checks.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    print(f"written: {OUT / 'posthoc_checks.md'}")


if __name__ == "__main__":
    main()
