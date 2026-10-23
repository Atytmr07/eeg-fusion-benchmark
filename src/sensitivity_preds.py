"""Sensitivity analyses from the stored test predictions (no training; new inference only
for the seizure-level thresholds, item 4).

Written for the advisor's third-priority list of 22 October 2026. Every run stores the
predicted probabilities of all its models for the test windows of every fold
(`preds/fold<k>.npz`); everything here is computed from those and the per-fold metrics.

  1. threshold   macro F1 at decision thresholds 0.3, 0.5 and 0.7: operator means,
                 whether the best operator of a run and the operator ranking change,
                 and the corrected tests again at each threshold
  2. prevalence  CHB-MIT window-level metrics at the natural prevalence of seizures.
                 The non-ictal test windows are a random sample, within each case, of
                 all its non-ictal windows; weighting each by (non-ictal windows of the
                 case) / (non-ictal windows sampled) gives the metrics that all windows
                 of the recordings would give, in expectation, without new inference.
                 Siena is left out: its recordings are cut around seizures, so their
                 prevalence is not natural either.
  3. bootstrap   subject-level bootstrap of the operator differences (seed sets
                 averaged), and a two-level version that also resamples the seed sets
                 within each subject. The bootstrap treats the held-out subjects as
                 independent and so ignores the overlap of the LOSO training sets, which
                 the corrected test accounts for; it is an alternative uncertainty
                 estimate, expected to be narrower, not a replacement.
  4. events      seizure-level sensitivity and false alarms per hour at 0.3, 0.5 and 0.7
                 for P0 and P1, from python -m src.events --thresholds 0.3 0.5 0.7
                 (events_thresholds/ of each run; every recording scored again)

Usage:  python -m src.sensitivity_preds
Output: results_v2/sensitivity_preds/ (CSV tables and summary.md)
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .audit_methods import corpus
from .chbmit_stats import FUSION, ratios
from .config import RESULTS_ROOT
from .make_results import complete_runs
from .multiverse import _pipeline_key, load_preds, md_table, to_array
from .stats import corrected_equivalence_bound, corrected_ttest, holm

OUT = RESULTS_ROOT / "sensitivity_preds"
NAMES = {"chbmit": "CHB-MIT", "siena": "Siena"}
THRESHOLDS = (0.3, 0.5, 0.7)
CTRL = ("spec2d_wide", "shallow")
MODELS = (*FUSION, *CTRL)
N_BOOT = 10_000
REPORT: list[str] = []


def say(line: str = "") -> None:
    print(line)
    REPORT.append(line)


def md(df: pd.DataFrame) -> str:
    return md_table(df, digits=3)


def prevalence_weights(ds: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Per corpus window: 1 for ictal windows, (non-ictal windows of the case) /
    (non-ictal windows sampled from it) for non-ictal windows."""
    d = corpus(ds, "")
    per = pd.DataFrame(d["info"]["per_subject"]).T
    total_non = per.n_windows_total - per.n_ictal - per.get("n_windows_excluded", 0)
    factor = (total_non / per.n_nonictal_kept).astype(float)
    subj = np.asarray(d["subject"])
    w = np.where(d["y"] == 1, 1.0, factor.reindex(subj).to_numpy())
    info = pd.DataFrame({"case": per.index, "windows": per.n_windows_total,
                         "ictal": per.n_ictal, "factor": factor})
    return w, info


def fold_metrics(ds: str, runs, weights: np.ndarray | None) -> pd.DataFrame:
    rows = []
    for r in runs:
        for fold, p in load_preds(r).items():
            y, idx = p["y_te"].astype(int), p["idx_te"]
            sw = weights[idx] if weights is not None else None
            for m in MODELS:
                prob = p[m][:, 1]
                row = {"pipeline": r.pipeline, "seed": r.seed, "fold": fold, "model": m}
                for t in THRESHOLDS:
                    row[f"f1_macro@{t}"] = f1_score(y, (prob > t).astype(int), average="macro",
                                                    zero_division=0)
                if sw is not None:
                    pred = (prob > 0.5).astype(int)
                    row["nat_f1_macro"] = f1_score(y, pred, average="macro", sample_weight=sw,
                                                   zero_division=0)
                    row["nat_f1_ictal"] = f1_score(y, pred, sample_weight=sw, zero_division=0)
                    row["nat_precision"] = precision_score(y, pred, sample_weight=sw,
                                                           zero_division=0)
                    row["sensitivity"] = recall_score(y, pred, zero_division=0)
                    row["nat_auprc"] = average_precision_score(y, prob, sample_weight=sw)
                    row["auprc"] = average_precision_score(y, prob)
                rows.append(row)
    return pd.DataFrame(rows)


def run_means(fm: pd.DataFrame, col: str) -> pd.DataFrame:
    """Fold mean per run (pipeline, seed) and model."""
    return fm.groupby(["pipeline", "seed", "model"])[col].mean().reset_index()


def arr(fm: pd.DataFrame, col: str, models) -> tuple[np.ndarray, list]:
    """(fold, pipeline, model) with seed sets averaged."""
    pipes = sorted(fm.pipeline.unique(), key=_pipeline_key)
    t = fm[fm.model.isin(models)].groupby(["fold", "pipeline", "model"])[col].mean()
    y = t.unstack("model")[list(models)]
    folds = sorted(fm.fold.unique())
    out = np.stack([y.xs(p, level="pipeline").reindex(folds).to_numpy() for p in pipes], 1)
    return out, pipes


def pair_tests(y: np.ndarray, r: float) -> tuple[int, int, int]:
    """Holm-significant, equivalent within 0.05 and within 0.02, over the operator pairs."""
    sig = eq5 = eq2 = 0
    for pi in range(y.shape[1]):
        ps, bs = [], []
        for ia, ib in itertools.combinations(range(len(FUSION)), 2):
            a, b = y[:, pi, ia], y[:, pi, ib]
            ps.append(corrected_ttest(a, b, r)[1])
            bs.append(corrected_equivalence_bound(a, b, r))
        sig += int((holm(np.array(ps)) < 0.05).sum())
        eq5 += sum(x <= 0.05 for x in bs)
        eq2 += sum(x <= 0.02 for x in bs)
    return sig, eq5, eq2


def ranking_agreement(rm: pd.DataFrame, col_a: str, col_b: str) -> tuple[float, float]:
    """Share of runs whose best operator is the same under both columns, and the mean
    Kendall tau between the two operator rankings of a run."""
    same, taus = [], []
    for _, g in rm[rm.model.isin(FUSION)].groupby(["pipeline", "seed"]):
        g = g.set_index("model").loc[list(FUSION)]
        same.append(g[col_a].idxmax() == g[col_b].idxmax())
        taus.append(sps.kendalltau(g[col_a], g[col_b]).statistic)
    return float(np.mean(same)), float(np.nanmean(taus))


# --- 1. threshold -----------------------------------------------------------------------------

def threshold(ds: str, fm: pd.DataFrame, stored: pd.DataFrame) -> None:
    say(f"\n### {NAMES[ds]}: decision threshold\n")
    chk = fm.merge(stored, on=["pipeline", "seed", "fold", "model"])
    dev = float((chk["f1_macro@0.5"] - chk.f1_macro).abs().max())
    say(f"Check: macro F1 at 0.5 recomputed from the stored predictions differs from the "
        f"stored per-fold value by at most {dev:.1e} ({len(chk)} fold results).")
    say("")
    cols = [f"f1_macro@{t}" for t in THRESHOLDS]
    rm = fm.groupby(["pipeline", "seed", "model"])[cols].mean().reset_index()
    tab = rm.groupby("model")[cols].agg(["mean", "std"])
    tab = pd.DataFrame({f"macro F1 at {t}": [f"{a:.3f} ± {b:.3f}" for a, b in
                                             zip(tab[(c, "mean")], tab[(c, "std")])]
                        for t, c in zip(THRESHOLDS, cols)}, index=tab.index)
    say(md(tab.loc[list(MODELS)].reset_index()))
    say("")
    r, _ = ratios(fm.fold.nunique())
    rows = []
    for t, c in zip(THRESHOLDS, cols):
        same, tau = ranking_agreement(rm, "f1_macro@0.5", c)
        y, _ = arr(fm, c, FUSION)
        sig, eq5, eq2 = pair_tests(y, r)
        best = rm[rm.model.isin(FUSION)].loc[
            lambda d: d.groupby(["pipeline", "seed"])[c].idxmax()].model.value_counts()
        rows.append({"threshold": t, "same best operator as at 0.5": f"{100 * same:.0f}%",
                     "mean tau with ranking at 0.5": tau, "Holm-significant pairs":
                     f"{sig}/90", "equivalent within 0.05": f"{eq5}/90",
                     "equivalent within 0.02": f"{eq2}/90",
                     "runs won": ", ".join(f"{k} {v}" for k, v in best.items())})
    say(md(pd.DataFrame(rows)))
    rm.to_csv(OUT / f"threshold_{ds}.csv", index=False)


# --- 2. natural prevalence -------------------------------------------------------------------

def prevalence(fm: pd.DataFrame, info: pd.DataFrame) -> None:
    say("\n### CHB-MIT: window-level metrics at the natural prevalence\n")
    n_win, n_ict = int(info.windows.sum()), int(info.ictal.sum())
    say(f"All 10 s windows of the 683 recordings: {n_win:,}, of which {n_ict:,} ictal "
        f"({100 * n_ict / n_win:.2f} percent, against 20 percent in the sampled corpus). "
        f"Non-ictal test windows are weighted by {info.factor.min():.0f} to "
        f"{info.factor.max():.0f} (per case).")
    say("")
    cols = {"f1_macro@0.5": "macro F1 (sampled)", "nat_f1_macro": "macro F1 (natural)",
            "nat_f1_ictal": "ictal F1 (natural)", "sensitivity": "sensitivity",
            "nat_precision": "precision (natural)", "auprc": "AUPRC (sampled)",
            "nat_auprc": "AUPRC (natural)"}
    rm = fm.groupby(["pipeline", "seed", "model"])[list(cols)].mean().reset_index()
    tab = rm.groupby("model")[list(cols)].mean().loc[list(MODELS)].rename(columns=cols)
    say(md(tab.reset_index()))
    say("")
    r, _ = ratios(fm.fold.nunique())
    rows = []
    for c in ("nat_f1_macro", "nat_auprc"):
        same, tau = ranking_agreement(rm, "f1_macro@0.5" if c == "nat_f1_macro" else "auprc",
                                      c)
        y, _ = arr(fm, c, FUSION)
        sig, _, _ = pair_tests(y, r)
        rows.append({"metric": cols[c], "same best operator as sampled": f"{100 * same:.0f}%",
                     "mean tau with sampled ranking": tau,
                     "Holm-significant pairs": f"{sig}/90"})
    say(md(pd.DataFrame(rows)))
    rm.to_csv(OUT / "prevalence_chbmit.csv", index=False)


# --- 3. bootstrap ------------------------------------------------------------------------------

def bootstrap(ds: str, long: pd.DataFrame) -> None:
    say(f"\n### {NAMES[ds]}: subject-level bootstrap of the operator differences\n")
    pipes = sorted(long.pipeline.unique(), key=_pipeline_key)
    seeds, folds = sorted(long.seed.unique()), sorted(long.fold.unique())
    y = to_array(long, "f1_macro", list(MODELS), pipes, seeds, folds)   # fold, pipe, model, seed
    n, k = len(folds), len(seeds)
    rng = np.random.default_rng(20261022)
    bi = rng.integers(0, n, (N_BOOT, n))                       # subjects
    bs = rng.integers(0, k, (N_BOOT, n, k))                    # seed sets within subject
    eq = pd.read_csv(RESULTS_ROOT / "multiverse" / f"{ds}_cuda" / "equivalence.csv")
    rows = []

    def boot(d4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """d4: (fold, seed) differences. Means over resampled subjects; one-level and
        two-level (seed sets also resampled within each subject)."""
        one = d4.mean(1)[bi].mean(1)
        two = np.take_along_axis(d4[bi], bs, axis=2).mean((1, 2))
        return one, two

    for pi, p in enumerate(pipes + ["all"]):
        for ia, ib in itertools.combinations(range(len(FUSION)), 2):
            if p == "all":
                d4 = (y[:, :, ia] - y[:, :, ib]).mean(1)
            else:
                d4 = y[:, pi, ia] - y[:, pi, ib]
            one, two = boot(d4)
            mean = float(d4.mean())
            row = {"pipeline": p, "pair": f"{FUSION[ia]}-{FUSION[ib]}", "difference": mean}
            for lab, b in (("subject", one), ("subject+seed", two)):
                lo95, hi95 = np.percentile(b, [2.5, 97.5])
                lo90, hi90 = np.percentile(b, [5, 95])
                row[f"{lab} 95% CI"] = f"[{lo95:+.3f}, {hi95:+.3f}]"
                row[f"{lab} excludes 0"] = bool(lo95 > 0 or hi95 < 0)
                row[f"{lab} 90% CI within 0.05"] = bool(lo90 >= -0.05 and hi90 <= 0.05)
            if p != "all":
                e = eq[(eq.pipeline == p) & (eq.pair == row["pair"])]
                if len(e):
                    hw = float(e.delta_min.iloc[0] - abs(e.mean_diff.iloc[0]))
                    row["corrected 90% CI"] = f"[{mean - hw:+.3f}, {mean + hw:+.3f}]"
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / f"bootstrap_{ds}.csv", index=False)
    per = t[t.pipeline != "all"]
    say(f"{N_BOOT:,} resamples of the {n} held-out subjects (seed sets averaged), and of "
        f"subjects and then the {k} seed sets within each subject.")
    say("")
    summ = pd.DataFrame([{
        "uncertainty estimate": lab,
        "pairs whose 95% CI excludes 0": f"{int(per[f'{c} excludes 0'].sum())}/90",
        "pairs whose 90% CI lies within ±0.05": f"{int(per[f'{c} 90% CI within 0.05'].sum())}/90"}
        for lab, c in (("subject bootstrap", "subject"), ("subject and seed bootstrap",
                                                          "subject+seed"))])
    say(md(summ))
    say("")
    say("Pooled over the nine pipelines (fold values averaged over pipelines and seed sets):")
    say("")
    say(md(t[t.pipeline == "all"][["pair", "difference", "subject 95% CI",
                                     "subject+seed 95% CI"]]))
    sig = per[per["subject excludes 0"]]
    if len(sig):
        say("")
        say("Pairs whose subject-bootstrap 95% CI excludes zero (uncorrected for the 90 "
            "comparisons):")
        say("")
        say(md(sig[["pipeline", "pair", "difference", "subject 95% CI", "subject+seed 95% CI",
                    "corrected 90% CI"]]))


# --- 4. seizure-level threshold -----------------------------------------------------------------

def event_threshold(ds: str) -> None:
    """Seizure-level metrics at 0.3, 0.5 and 0.7 for P0 and P1 (python -m src.events
    --thresholds 0.3 0.5 0.7: every recording scored again, the one step here that needed
    new inference). Pooled over persons per run, as in the main event tables."""
    files = sorted((RESULTS_ROOT / ds).glob("loso_P*_cuda/events_thresholds/perperson.csv"))
    if not files:
        return
    d = pd.concat([pd.read_csv(f).assign(run=f.parts[-3]) for f in files])
    runs = {r.name: r for r in complete_runs(RESULTS_ROOT / ds, ds, "cuda")[0]}
    d["pipeline"] = d.run.map(lambda r: runs[r].pipeline)
    say(f"\n### {NAMES[ds]}: seizure-level threshold ({', '.join(sorted(d.pipeline.unique()))}"
        f", {d.run.nunique()} runs)\n")
    old = pd.concat([pd.read_csv(f.parent.parent / "events" / "perperson.csv")
                     .assign(run=f.parts[-3]) for f in files])
    m = d[d.threshold == 0.5].merge(old, on=["run", "fold", "model"], suffixes=("", "_old"))
    nd = int(((m.detected != m.detected_old) | (m.false_alarms != m.false_alarms_old)).sum())
    say(f"Check against the stored results at 0.5: {nd} of {len(m)} person results differ, by "
        f"at most {int((m.detected - m.detected_old).abs().max())} detected seizure and "
        f"{int((m.false_alarms - m.false_alarms_old).abs().max())} false alarms (most likely "
        f"windows whose probability lies at the threshold and flips with rounding); pooled "
        f"false alarms per hour {m.false_alarms.sum() / m.hours.sum():.3f} against "
        f"{m.false_alarms_old.sum() / m.hours_old.sum():.3f}.")
    say("")
    g = d.groupby(["pipeline", "run", "threshold", "model"])[
        ["detected", "seizure_events", "false_alarms", "hours"]].sum().reset_index()
    g["sens"] = g.detected / g.seizure_events
    g["fa"] = g.false_alarms / g.hours
    metrics = [("sens", "sensitivity")] + ([("fa", "false alarms/h")] if ds == "chbmit" else [])
    mean = g.groupby(["model", "threshold"])[[c for c, _ in metrics]].mean().unstack("threshold")
    tab = pd.DataFrame({f"{lab} at {t}": mean[(c, t)] for c, lab in metrics for t in THRESHOLDS})
    say("Mean over the runs:")
    say("")
    say(md(tab.loc[list(MODELS)].reset_index()))
    say("")
    rows = []
    for c, lab in metrics:
        pick = (lambda s: s.idxmax()) if c == "sens" else (lambda s: s.idxmin())
        for t in (0.3, 0.7):
            same, taus = [], []
            for _, r in g[g.model.isin(FUSION)].groupby("run"):
                a = r[r.threshold == 0.5].set_index("model").loc[list(FUSION), c]
                b = r[r.threshold == t].set_index("model").loc[list(FUSION), c]
                same.append(pick(a) == pick(b))
                taus.append(sps.kendalltau(a, b).statistic)
            rows.append({"metric": lab, "threshold": t,
                         "same best operator as at 0.5": f"{sum(same)}/{len(same)}",
                         "mean tau with ranking at 0.5": float(np.nanmean(taus))})
    say(md(pd.DataFrame(rows)))
    g.to_csv(OUT / f"event_threshold_{ds}.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    say("# Sensitivity analyses from the stored predictions")
    say("")
    say("Generated by `python -m src.sensitivity_preds` (advisor's third-priority list, "
        "22 October 2026); no model was trained; only the seizure-level thresholds needed new inference.")
    for ds in ("chbmit", "siena"):
        runs, long, _ = complete_runs(RESULTS_ROOT / ds, ds, "cuda")
        weights, info = prevalence_weights(ds) if ds == "chbmit" else (None, None)
        fm = fold_metrics(ds, runs, weights)
        fm.to_csv(OUT / f"fold_metrics_{ds}.csv", index=False)
        threshold(ds, fm, long[["pipeline", "seed", "fold", "model", "f1_macro"]])
        if ds == "chbmit":
            prevalence(fm, info)
        bootstrap(ds, long)
        event_threshold(ds)
    (OUT / "summary.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    print(f"written: {OUT / 'summary.md'}")


if __name__ == "__main__":
    main()
