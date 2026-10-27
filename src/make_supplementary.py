"""Supplementary tables from stored results: paper/supplementary.md.

Every number comes from a file in results_v2/ (or, for the P4 table, from the corpus,
which it then stores in results_v2/qc/ so later builds need no raw data):

  S1  artefact removal (P6a/b/c): signal retention and compute per recording
  S2  GEDAI preset choice, by signal preservation only
  S3  P4 technical artefact rejection per person, ictal and non-ictal separately

Usage:  python -m src.make_supplementary
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .config import PROJECT_ROOT, RESULTS_ROOT
from .stats import holm

QC = RESULTS_ROOT / "qc"
OUT = PROJECT_ROOT / "paper" / "supplementary.md"


def md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                 else ("" if isinstance(v, float) and not np.isfinite(v) else str(v))
                 for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_s1() -> str:
    d = pd.read_csv(QC / "p6_recording_logs.csv")
    d = d[d.record.str.startswith("chb")]                   # CHB-MIT only
    names = {"infomax": "P6a Extended Infomax", "gedai-conservative": "P6b GEDAI, conservative (used)",
             "amica": "P6c AMICA", "gedai": "GEDAI, default (not used)"}
    rows = []
    for m in ("infomax", "gedai-conservative", "amica", "gedai"):
        g = d[d.method == m]
        v = g.power_kept
        rows.append({"method": names[m], "recordings": len(g),
                     "power kept, median": v.median(),
                     "IQR": f"{v.quantile(.25):.3f}-{v.quantile(.75):.3f}",
                     "min": v.min(), "share < 0.5": (v < 0.5).mean(),
                     "components removed, mean": g.n_removed.mean() if g.n_removed.notna().any()
                     else np.nan,
                     "seconds per recording, median": f"{g.seconds.median():.0f}",
                     "CPU hours, total": f"{g.seconds.sum() / 3600:.1f}"})
    return md(pd.DataFrame(rows))


def table_s2() -> str:
    t = pd.read_csv(QC / "gedai_settings" / "summary.csv")
    t = t.rename(columns={"setting": "preset", "unit": "unit", "share_below_0.5": "share < 0.5",
                          "share_below_0.1": "share < 0.1"})
    t["preset"] = t.preset.map({"auto": "default (auto)",
                                "auto-": "conservative (auto-), used"}).fillna(t.preset)
    return md(t[["preset", "unit", "n", "p5", "p25", "median", "p75", "share < 0.5", "share < 0.1"]])


def table_s3() -> str:
    f = QC / "p4_rejection_per_person.csv"
    if not f.exists():
        from .chbmit_corpus import build_corpus
        from .preprocess import artifact_mask
        d = build_corpus(verbose=False)
        rej = artifact_mask(d["X"])
        g = pd.DataFrame({"person": d["group"], "ictal": d["y"] == 1, "rejected": rej})
        t = g.groupby("person")[["ictal", "rejected"]].apply(lambda x: pd.Series({
            "windows": len(x), "ictal": int(x.ictal.sum()),
            "ictal rejected": int((x.ictal & x.rejected).sum()),
            "non-ictal": int((~x.ictal).sum()),
            "non-ictal rejected": int((~x.ictal & x.rejected).sum())}))
        t = t.reset_index()
        t.to_csv(f, index=False)
    t = pd.read_csv(f)
    tot = t.drop(columns="person").sum()
    t.loc[len(t)] = ["all", *tot.values]
    t["ictal rejected %"] = 100 * t["ictal rejected"] / t.ictal
    t["non-ictal rejected %"] = 100 * t["non-ictal rejected"] / t["non-ictal"]
    t[["windows", "ictal", "ictal rejected", "non-ictal", "non-ictal rejected"]] = \
        t[["windows", "ictal", "ictal rejected", "non-ictal", "non-ictal rejected"]].astype(int)
    return md(t, "{:.1f}")


# Siena metadata decisions (docs/SIENA.md, src/siena.py: DECISIONS and CORRECTIONS). The
# window counts were computed with siena_corpus.build_corpus under each alternative
# (10 October 2026); every ictal window is kept in the corpus, so the ictal counts compare
# directly.
SIENA_DECISIONS = [
    ("PN00", "PN00-3, seizure 3", "listed end 19.29.29 lies 1916 s after the recording ends (18.57.33)",
     "end read as 18.29.29 (a 60 s seizure)", "verified correction",
     "signal: the right frontotemporal 3-20 Hz ictal rhythm (FP2-F8, F8-T8) falls below a quarter of "
     "its peak 14 s after 18.29.29; the patient's other four seizures end 0 to 13 s after their "
     "listed ends by the same rule",
     "offset from an automatic rule on one channel pair",
     "alternative (exclude the seizure and the rest of the recording): 30 instead of 37 ictal "
     "windows for PN00"),
    ("PN05", "PN05-3, seizure 3", "registration start: list 06.01.23, EDF header 06.01.13",
     "EDF header start time, as for every recording", "decision rule (not a verified correction)",
     "none possible: the header defines the first sample and 39 of 41 recordings agree with their "
     "list; the list's value fits the 20 s end offset of 18 other recordings, the header's does not",
     "10 s in the seizure's position",
     "4 ictal windows either way, shifted by one: 2 window labels differ"),
    ("PN10", "PN10-3, seizure 3", "two onsets: 15.43.53 (clinical), 15.43.59 (electrical)",
     "electrical onset (advisor)", "decision",
     "CHB-MIT annotates electrographic onsets; the electrical onset keeps the two corpora consistent",
     "none (both listed)",
     "clinical onset instead: 48 instead of 47 ictal windows for PN10 (1 label)"),
    ("PN10", "PN10-4.5.6, seizure 6", "only a clinical onset (15.18.26)",
     "used as listed, flagged in the corpus's seizure notes", "decision",
     "-", "electrical onset unknown; in seizure 3 it followed the clinical one by 6 s",
     "at most about one window, by analogy with seizure 3"),
    ("PN10", "PN10-2, seizure 2", "end \"11.41.04 opure 11.40.43\" (Italian \"or\")",
     "first value, 11.41.04", "decision rule", "-", "21 s in the seizure's end",
     "second value instead: 45 instead of 47 ictal windows for PN10"),
    ("PN14", "PN14-3", "registration start: list 16.17.45, EDF header 19.17.45",
     "EDF header start time", "verified correction",
     "header start + duration (41,995 s) gives the listed end time 06.57.40 exactly; the list's "
     "start would make the recording three hours longer than the file",
     "none", "seizure at 1 h 52 min into the recording, not 4 h 52 min"),
]


def table_s4() -> str:
    """Two narrower tables (the decisions, then how each was checked), one row per item."""
    t = pd.DataFrame(SIENA_DECISIONS, columns=[
        "patient", "recording / seizure", "problem in the metadata", "used", "type",
        "how it was checked", "remaining uncertainty", "effect on the windows"])
    t.insert(0, "item", range(1, len(t) + 1))
    a = t[["item", "recording / seizure", "problem in the metadata", "used", "type"]]
    c = t[["item", "how it was checked", "remaining uncertainty", "effect on the windows"]]
    return md(a) + "\n\nHow each decision was checked, and its effect:\n\n" + md(c)


def table_s5() -> str:
    s = pd.read_csv(QC / "p6_signal_effects_summary.csv")
    b = pd.read_csv(QC / "p6_signal_effects_bands.csv")
    i = pd.read_csv(QC / "p6_signal_effects_ictal.csv")
    t = s[s.windows != "all"].merge(b, on=["dataset", "pipeline", "method", "windows"])
    t = t[["dataset", "method", "windows", "n", "median", "share_below_0.5", "delta", "theta",
           "alpha", "beta", "low gamma"]].rename(columns={"median": "power kept, median",
                                                          "share_below_0.5": "share < 0.5"})
    i = i[["dataset", "method", "recordings_with_both", "median_ictal_minus_nonictal",
           "share_ictal_lower", "wilcoxon_p"]].rename(columns={
               "recordings_with_both": "recordings", "median_ictal_minus_nonictal":
               "ictal - non-ictal, median", "share_ictal_lower": "share with ictal lower",
               "wilcoxon_p": "Wilcoxon p"})
    return md(t) + "\n\nWithin recordings:\n\n" + md(i, "{:.4f}")


def table_s6() -> str:
    """Per pipeline and operator, mean and SD over seed sets, three window-level metrics."""
    res = RESULTS_ROOT / "results"
    ops = ["early", "late", "gated", "attention", "score"]
    order = ["P0", "P1", "P2", "P3", "P4", "P5", "P6a", "P6b", "P6c"]
    out = []
    for ds, name in (("chbmit", "CHB-MIT"), ("siena", "Siena")):
        for metric, label in (("f1_macro", "macro F1"), ("auprc", "AUPRC"),
                              ("balanced_accuracy", "balanced accuracy")):
            f = res / f"main_{ds}_{metric}.csv"
            if not f.exists():
                continue
            t = pd.read_csv(f)
            n = int(t.n_seeds.min())
            cell = t.assign(c=[f"{m:.3f} ± {s:.3f}" for m, s in zip(t["mean"], t["sd"])])
            w = cell.pivot(index="pipeline", columns="model", values="c").reindex(order)[ops]
            best = t.pivot(index="pipeline", columns="model", values="mean").reindex(order)[ops]
            for p in w.index:
                b = best.loc[p].idxmax()
                w.at[p, b] = f"**{w.at[p, b]}**"
            out += [f"**{name}, {label}** ({n} seed sets; best operator of each pipeline in bold)",
                    "", md(w.reset_index()), ""]
    return "\n".join(out)


PARAMS = {"raw1d": 28850, "spec2d": 34546, "raw1d_wide": 103712, "spec2d_wide": 102236,
          "logvar": 19, "shallow": 127}   # python -m src.models (logistic regressions with intercept)


def table_s7() -> str:
    """All six controls next to each other: mean and SD over the 45 runs per dataset."""
    res = RESULTS_ROOT / "results"
    out = None
    for ds, name in (("chbmit", "CHB-MIT"), ("siena", "Siena")):
        f = res / f"controls_{ds}.csv"
        if not f.exists():
            return "(controls tables missing; run python -m src.make_results)"
        t = pd.read_csv(f)
        t = t[t.kind == "control"].set_index("model")
        cols = {}
        for m, lab in (("f1_macro", "macro F1"), ("auprc", "AUPRC"),
                       ("balanced_accuracy", "bal. acc."), ("brier", "Brier")):
            cols[f"{name} {lab}"] = [f"{a:.3f} ± {b:.3f}" for a, b in
                                     zip(t[f"{m}_mean"], t[f"{m}_sd"])]
        part = pd.DataFrame(cols, index=t.index)
        out = part if out is None else out.join(part)
    out.insert(0, "parameters", [f"{PARAMS[m]:,}" for m in out.index])
    return md(out.reset_index().rename(columns={"model": "control"}))


def table_s8() -> str:
    """Every operator pair in every pipeline: paired difference, 90% CI, p, delta_min."""
    mv = RESULTS_ROOT / "multiverse"
    out = []
    for ds, name in (("chbmit", "CHB-MIT"), ("siena", "Siena")):
        f, fs = mv / f"{ds}_cuda" / "equivalence.csv", mv / f"{ds}_cuda" / "equivalence_strict.csv"
        if not f.exists():
            continue
        e = pd.read_csv(f)
        strict = pd.read_csv(fs).set_index(["pipeline", "pair"]).equivalent if fs.exists() else None
        hw = e.delta_min - e.mean_diff.abs()           # half-width of the 90% CI
        t = pd.DataFrame({
            "pipeline": e.pipeline, "pair": e.pair,
            "difference": [f"{d:+.3f}" for d in e.mean_diff],
            "90% CI": [f"[{d - h:+.3f}, {d + h:+.3f}]" for d, h in zip(e.mean_diff, hw)],
            "p corrected": [f"{p:.3f}" for p in e.p_corrected],
            "p Holm": [f"{p:.3f}" for p in e.p_holm],
            "delta_min": [f"{x:.3f}" for x in e.delta_min],
            "equivalent ±0.05": ["yes" if x else "no" for x in e.equivalent],
            "equivalent ±0.02": (["yes" if strict.loc[(p, q)] else "no"
                                  for p, q in zip(e.pipeline, e.pair)]
                                 if strict is not None else ""),
        })
        out += [f"**{name}** (difference: first minus second operator, macro F1; fold "
                f"values averaged over the seed sets)", "", md(t), ""]
    return "\n".join(out)


SP = RESULTS_ROOT / "sensitivity_preds"           # post hoc analyses (S9, S10)
OPS = ["early", "late", "gated", "attention", "score"]
THR = (0.3, 0.5, 0.7)


def table_s9a() -> str:
    """Macro F1 per dataset, pipeline and threshold (mean over seed sets of the fold means)."""
    out = []
    for ds, name in (("chbmit", "CHB-MIT"), ("siena", "Siena")):
        f = SP / f"threshold_{ds}.csv"
        if not f.exists():
            return "(run python -m src.sensitivity_preds)"
        rm = pd.read_csv(f)
        pipes = [p for p in ["P0", "P1", "P2", "P3", "P4", "P5", "P6a", "P6b", "P6c"]
                 if p in set(rm.pipeline)]
        rows = []
        for p in pipes:
            for t in THR:
                m = rm[rm.pipeline == p].groupby("model")[f"f1_macro@{t}"].mean()
                rows.append({"pipeline": p, "threshold": f"{t:.1f}",
                             **{k: f"{m[k]:.3f}" for k in OPS + ["spec2d_wide", "shallow"]}})
        out += [f"**{name}**", "", md(pd.DataFrame(rows)), ""]
    return "\n".join(out)


def table_s9b() -> str:
    """Per dataset and threshold: each operator's mean rank and share of runs ranked first."""
    rows = []
    for ds, name in (("chbmit", "CHB-MIT"), ("siena", "Siena")):
        rm = pd.read_csv(SP / f"threshold_{ds}.csv")
        rm = rm[rm.model.isin(OPS)]
        win05 = None
        for t in THR:
            w = rm.pivot_table(index=["pipeline", "seed"], columns="model",
                               values=f"f1_macro@{t}")[OPS]
            rk = w.rank(axis=1, ascending=False, method="average")
            best = w.idxmax(axis=1)
            if t == 0.5:
                win05 = best
            rows.append({"dataset": name, "threshold": f"{t:.1f}",
                         **{f"{k}: mean rank / first": f"{rk[k].mean():.2f} / "
                            f"{100 * (rk[k] == 1).mean():.0f}%" for k in OPS},
                         "_best": best})
        for r in rows[-3:]:
            r["winner differs from 0.5"] = f"{100 * (r.pop('_best') != win05).mean():.0f}%"
    return md(pd.DataFrame(rows))


def table_s9c() -> str:
    f = SP / "posthoc_holm_thresholds.csv"
    if not f.exists():
        return "(run python -m src.posthoc_checks)"
    d = pd.read_csv(f)
    show = d[d.p_raw < 0.05].copy()
    show["dataset"] = show.dataset.map({"chbmit": "CHB-MIT", "siena": "Siena"})
    show["threshold"] = show.threshold.map("{:.1f}".format)
    show["difference"] = show.mean_diff.map("{:+.3f}".format)
    for c in ("p_raw", "p_holm_within", "p_holm_across"):
        show[c] = show[c].map("{:.4f}".format)
    show = show[["dataset", "pipeline", "threshold", "pair", "difference", "p_raw",
                 "p_holm_within", "p_holm_across"]].rename(columns={
                     "p_raw": "raw p", "p_holm_within": "Holm within threshold",
                     "p_holm_across": "Holm across thresholds"})
    sig_w = int((d.p_holm_within < 0.05).sum())
    sig_a = int((d.p_holm_across < 0.05).sum())
    return (md(show) + f"\n\nAll {len(d)} comparisons (2 datasets x 9 pipelines x 3 thresholds "
            f"x 10 pairs) are in `results_v2/sensitivity_preds/posthoc_holm_thresholds.csv`; "
            f"{sig_w} are significant after the Holm adjustment within a threshold and "
            f"{sig_a} after the adjustment across thresholds.")


def table_s10a() -> str:
    """Seizure-level metrics of P0 and P1 at the three thresholds."""
    out = []
    for ds, name in (("chbmit", "CHB-MIT"), ("siena", "Siena")):
        f = SP / f"event_threshold_{ds}.csv"
        if not f.exists():
            return "(run python -m src.events --thresholds 0.3 0.5 0.7 and src.sensitivity_preds)"
        g = pd.read_csv(f)
        metrics = [("sens", "sensitivity")] + ([("fa", "false alarms/h")] if ds == "chbmit"
                                               else [])
        rows = []
        for p in sorted(g.pipeline.unique()):
            for m in OPS + ["spec2d_wide", "shallow"]:
                x = g[(g.pipeline == p) & (g.model == m)].groupby("threshold")
                row = {"pipeline": p, "model": m}
                for c, lab in metrics:
                    st = x[c].agg(["mean", "std"])
                    for t in THR:
                        row[f"{lab} {t:.1f}"] = f"{st.at[t, 'mean']:.2f} ± {st.at[t, 'std']:.2f}"
                rows.append(row)
        out += [f"**{name}**", "", md(pd.DataFrame(rows)), ""]
    return "\n".join(out)


def table_s10b() -> str:
    f, w = SP / "posthoc_prevalence_summary.csv", SP / "posthoc_prevalence_weights.csv"
    if not f.exists():
        return "(run python -m src.posthoc_checks)"
    s = pd.read_csv(f)
    s["relative decline"] = s["relative decline"].map(lambda v: f"{100 * v:.1f}%")
    ww = pd.read_csv(w).rename(columns={"windows (all, 10 s)": "all windows"})
    ww["weight w"] = ww["weight w"].map("{:.1f}".format)
    return (md(s) + "\n\nWeights per case (all windows: every 10 s window of the case's "
            "target-montage recordings):\n\n" + md(ww))


def table_s10c() -> str:
    f = SP / "posthoc_bootstrap_compare.csv"
    if not f.exists():
        return "(run python -m src.posthoc_checks)"
    d = pd.read_csv(f)
    names = {"chbmit": "CHB-MIT", "siena": "Siena"}
    b1, b2 = "bootstrap (subject) excludes 0", "bootstrap (subject, then seed set) excludes 0"
    per = d[d.pipeline != "all"].reset_index(drop=True)
    per["holm"] = np.nan
    for _, g in per.groupby(["dataset", "pipeline"]):
        per.loc[g.index, "holm"] = holm(g["corrected p"].to_numpy())
    cnt = per.groupby("dataset").agg(**{
        "corrected test, nominal (raw p < 0.05)": ("corrected p", lambda v: f"{(v < 0.05).sum()}/{len(v)}"),
        "corrected test, Holm-adjusted p < 0.05": ("holm", lambda v: f"{(v < 0.05).sum()}/{len(v)}"),
        "bootstrap (subject), nominal": (b1, lambda v: f"{v.sum()}/{len(v)}"),
        "bootstrap (subject, then seed set), nominal": (b2, lambda v: f"{v.sum()}/{len(v)}"),
    }).reset_index()
    cnt["dataset"] = cnt.dataset.map(names)
    d = d[d.pipeline == "all"].drop(columns=["pipeline", "matches bootstrap_<ds>.csv", b1, b2])
    d["dataset"] = d.dataset.map(names)
    d["corrected p"] = d["corrected p"].map("{:.4f}".format)
    d["mean difference"] = d["mean difference"].map("{:+.3f}".format)
    d["direction consistency"] = d["direction consistency"].map(lambda v: f"{100 * v:.1f}%")
    return (md(d) + "\n\nPairs per pipeline (90 per dataset) whose 95% interval excludes "
            "zero; only the Holm column is adjusted for multiplicity:\n\n" + md(cnt))


def table_s10d() -> str:
    f = SP / "posthoc_prevalence_pairs.csv"
    if not f.exists():
        return "(run python -m src.posthoc_checks)"
    d = pd.read_csv(f)
    d["metric"] = d.metric.map({"nat_f1_macro": "macro F1 (adjusted)",
                                "nat_auprc": "AUPRC (adjusted)"})
    cnt = d.groupby("metric").agg(**{
        "raw p < 0.05": ("p_raw", lambda v: f"{(v < 0.05).sum()}/{len(v)}"),
        "Holm-adjusted p < 0.05": ("p_holm", lambda v: f"{(v < 0.05).sum()}/{len(v)}"),
    }).reset_index()
    sig = d[d.p_holm < 0.05].copy()
    sig["mean difference"] = sig["mean difference"].map("{:+.3f}".format)
    sig["t"] = sig.t.map("{:.2f}".format)
    for c in ("p_raw", "p_holm"):
        sig[c] = sig[c].map("{:.4f}".format)
    sig = sig.rename(columns={"p_raw": "raw p", "p_holm": "Holm-adjusted p"})
    return md(cnt) + "\n\nHolm-significant comparisons:\n\n" + md(sig)


def main() -> None:
    text = f"""# Supplementary Material

*Generated by `python -m src.make_supplementary` from `results_v2/qc/`; do not edit by hand.*

## Table S1. Artefact removal in P6: signal retention and compute

Per recording of CHB-MIT with selected windows (670), on the 0.5 to 40 Hz signal. Power
kept = signal power after removal / before. Extended Infomax and AMICA share 16
components, fitting on every 8th sample and the same automatic ocular component rule;
GEDAI selects what to remove by its own criterion. Compute on a laptop CPU (Intel Core
Ultra 7 155U), six recordings in parallel.

{table_s1()}

## Table S2. GEDAI preset, chosen on signal preservation only

Six CHB-MIT subjects (chb01, chb03, chb05, chb08, chb12, chb15), the same windows under
both presets. Power kept per recording and per window (relative to the same 0.5 to 40 Hz
signal); no classification result was used for the choice (`src/gedai_qc.py`).

{table_s2()}

## Table S3. P4: technical artefact rejection per person

A window is rejected if a channel is flat (standard deviation below 1 uV) or holds one
constant value for at least 0.5 s, on the unfiltered signal. Rejected windows are removed
from training and validation only; test windows are unchanged. Rejection rates are given
for ictal and non-ictal windows separately. chb01 includes chb21, the same person.

{table_s3()}

## Table S4. Siena: metadata problems and the decisions taken

Every departure from the published seizure lists, with the decision used in the main
analysis, its kind (a correction verified against the recording, or a decision rule
applied without such a check), how it was checked, and how much the alternative would
change the windows. Other, purely textual fixes (a space inside a time in PN10's list,
misspelt file names, PN12's missing registration times taken from the same file's
first block, recordings past midnight read modulo 24 h) are listed in `docs/SIENA.md`,
Section 4. Recording time is taken from the EDF headers, 141.0 h; the dataset's
description gives 128.4 h, a difference explained for PN14 (the start time above) but
not for PN03, PN10 and PN12, whose files contain no padding (`docs/SIENA.md`, Section 8).

{table_s4()}

## Table S5. Artefact removal: power kept per window, by band and class

What each method removes from the 0.5 to 40 Hz signal, per 10 s window (cleaned power
divided by the power of the same window before cleaning, summed over channels), for
ictal and non-ictal windows, overall and in five bands (delta 0.5-4, theta 4-8, alpha
8-13, beta 13-30, low gamma 30-40 Hz; medians). The second part compares the classes
within recordings: per recording the median of each class, their difference, and a
Wilcoxon signed-rank test over the recordings that hold both. A difference between the
classes over the whole corpus that is absent within recordings reflects which
recordings a method cleans heavily, not a preference for one class
(`src/p6_signal_effects.py`). GEDAI's default preset was applied to CHB-MIT only.

{table_s5()}

## Table S6. Window-level performance per pipeline and operator

Mean and SD over the five seed sets of each run's fold mean, for every pipeline and
fusion operator (`python -m src.make_results`, `results_v2/results/main_*.csv`).

{table_s6()}

## Table S7. All controls

The six controls over the same complete runs as Table 4 (mean and SD over the 45 runs of
each dataset, fold means within a run; Brier: lower is better). Parameters with 18 input
channels; the classical models are logistic regressions on 18 log variances (logvar) and
on seven features per channel (shallow), with an intercept
(`results_v2/results/controls_*.csv`).

{table_s7()}

## Table S8. Paired differences between operators, with confidence intervals and equivalence

For every operator pair in every pipeline: the mean paired difference in macro F1, its
90 percent confidence interval with the corrected variance (the interval used by the two
one-sided tests), the corrected and the Holm-adjusted p of a difference, the smallest
equivalence margin the data can establish (delta_min), and the outcome at the margins of
±0.05 and ±0.02 (`results_v2/multiverse/*_cuda/equivalence*.csv`; Figure 3).

{table_s8()}

## Table S9. Post hoc: sensitivity to the decision threshold (window level)

Exploratory, specified after the primary analysis (Section 2.6). Every model's stored test
predictions were thresholded at 0.3, 0.5 and 0.7; 0.5 is the primary analysis
(`python -m src.sensitivity_preds`, `src.posthoc_checks`; Figure 9a).

**S9a.** Macro F1 per pipeline and threshold, mean over the five seed sets of each run's
fold mean.

{table_s9a()}

**S9b.** Each operator's mean rank (1 = best) over the 45 runs of a dataset and the share
of runs in which it ranked first (ties share the rank), and the share of runs whose best
operator differs from the best at 0.5.

{table_s9b()}

**S9c.** Operator pairs with a raw p below 0.05 at any threshold: corrected resampled t
test on the fold values (seed sets averaged), Holm-adjusted within the ten pairs of one
dataset, pipeline and threshold (the family of the primary analysis), and across the
thirty comparisons of one dataset and pipeline (three thresholds x ten pairs), a more
conservative sensitivity check. Difference: first minus second operator.

{table_s9c()}

## Table S10. Post hoc: seizure-level thresholds, class prevalence and inference

Exploratory, specified after the primary analysis (Section 2.6).

**S10a. Seizure-level metrics at three thresholds**, P0 and P1 only: every recording of
every held-out person scored again with the saved models (`python -m src.events
--thresholds 0.3 0.5 0.7`), no post-processing; pooled over the persons of a run, mean ±
SD over the five seed sets. False alarms per hour are given for CHB-MIT only (Siena's
recordings are cut around seizures).

{table_s10a()}

**S10b. CHB-MIT prevalence-adjusted macro F1.** Each non-ictal test window is weighted by
the number of non-ictal windows of its case in the complete recordings divided by the
number sampled from that case (ictal windows: weight 1), so true
positives and false negatives keep their counts and false positives and true negatives
are scaled to the full non-ictal count; macro F1 is computed from this weighted
confusion matrix at the 0.5 threshold. Subject-level: macro F1 per held-out person, mean
over the 23 persons, then mean and SD over the 45 runs; pooled: the weighted confusion
matrices of a run's persons summed first. False positive rate: share of the sampled
non-ictal test windows predicted ictal at 0.5 (unchanged by the weighting); precision
at 0.5 (adjusted): TP / (TP + weighted FP). The estimate assumes that the sampled
non-ictal windows have the error rate of those not sampled; it is not a measurement on
all windows (`src/posthoc_checks.py`).

{table_s10b()}

**S10c. Corrected paired t test and subject-level bootstrap**, operator pairs pooled over
the nine pipelines (each subject's difference averaged over pipelines and seed sets).
Corrected 95% interval: mean ± t(0.975, n-1) SE with the Nadeau-Bengio variance.
Bootstrap: the held-out subject is the resampling unit, and both operators' results of a
drawn subject enter together (paired); 10,000 resamples, percentile intervals; the second
interval also resamples the five seed sets within each drawn subject. Direction
consistency: share of resampled means with the sign of the observed difference. The
bootstrap does not model the overlap of the LOSO training sets; the corrected test
remains the primary analysis. Assumptions: the corrected test treats the per-subject
differences as approximately normal and approximates the dependence between folds by the
Nadeau-Bengio variance inflation (test/train ratio 1/22 and 1/13); the percentile
bootstrap treats the held-out subjects as independent draws from a population of
subjects, and with 23 and 14 subjects its intervals may be too narrow. Neither interval
is adjusted for multiplicity: they are comparable with each other, whereas the primary
analysis applies Holm adjustment within each pipeline. All 90 pairs per dataset in
`results_v2/sensitivity_preds/posthoc_bootstrap_compare.csv`.

{table_s10c()}

**S10d. CHB-MIT prevalence-adjusted operator comparisons.** The procedure of the primary
analysis applied to the prevalence-adjusted fold values of S10b: per held-out subject
the fold value of each operator, seed sets averaged; corrected resampled t test
(Nadeau-Bengio, test/train ratio 1/22, 22 degrees of freedom) on the 23 paired
differences; Holm adjustment within the ten operator pairs of one pipeline (nine
families per metric). Difference: first minus second operator. Post hoc and exploratory
(`results_v2/sensitivity_preds/posthoc_prevalence_pairs.csv`).

{table_s10d()}

## Table S11. Reproducibility map

Where each analysis of the manuscript is implemented and where its results are stored
in the public repository. Every `src/...` module runs as `python -m src.<module>`; the
README lists the full commands and their order.

| Manuscript | Code | Results and records |
|---|---|---|
| 2.1 CHB-MIT annotations (seizures assigned by file name; `chb09_08.edf`), counts of recordings and seizures | `src/chbmit.py` (`parse_summary`), `src/verify_chbmit.py`, `src/audit_methods.py` | `results_v2/audit/methods_audit.md`, `counts.csv` |
| 2.1 Siena seizure lists, the six decisions, resampling to 256 Hz and the bipolar montage | `src/siena.py` (`read_bipolar`), `src/siena_qc.py`, `src/verify_siena.py`; `docs/SIENA.md` | Table S4 |
| 2.2 Preprocessing pipelines P0 to P6c; spectrogram; GEDAI preset (Table S2) | `src/preprocess.py`, `src/chbmit_prep.py`, `src/gedai_qc.py` | per run `meta.json`, `results_v2/qc/gedai_settings/` |
| 2.2 Artefact removal records per recording | `src/preprocess.py` | `results_v2/qc/p6_recording_logs.csv`, `p6_recording_logs_events.csv` |
| 2.3 Models, parameter matching | `src/models.py` (`python -m src.models` prints the counts) | |
| 2.4 Windows, LOSO partitions, inner validation, leakage check and its negative control | `src/chbmit_corpus.py`, `src/datasets.py`, `src/audit_methods.py` | `results_v2/audit/methods_audit.md`, `partitions_<ds>.csv` |
| 2.4 Seeds per seed set, fold and model | `src/config.py` (`fold_seed`) | per run `meta.json` |
| 2.4, 3.7 Training on CHB-MIT, testing on Siena | `src/cross_dataset.py` | `results_v2/cross/`, `results_v2/results/cross_*.csv` |
| 2.5, 3.6 Seizure-level scoring (rules fixed before the runs), the 187 reference events | `src/events.py`; `docs/EVENTS.md`; `src/audit_methods.py` | per run `events/`, `results_v2/results/events_<ds>.csv` |
| 2.6 Simulation of the naive paired test | `src/sim_cv_correlation.py` | |
| 2.6, 3.1 to 3.4 Corrected tests, equivalence, rankings, RM-ANOVA, variance components, conservative test/train ratio | `src/multiverse.py`, `src/stats.py`, `src/audit_methods.py` | `results_v2/multiverse/chbmit_cuda/`, `siena_cuda/`, `results_v2/audit/methods_audit.md` |
| 3.1 Controls against the operators | `src/controls_stats.py` | `results_v2/results/controls_*.csv`, `controls_tests.md` |
| 3.1 to 3.7 Tables and figures | `src/make_results.py`, `src/make_figures.py`, `src/make_figure1.py`, `src/make_supplementary.py` | `results_v2/results/`, `paper/figures_manuscript/` |
| 2.6, 3.8 Post hoc threshold, prevalence and bootstrap analyses (Tables S9, S10) | `src/sensitivity_preds.py`, `src/posthoc_checks.py`; seizure-level thresholds: `src/events.py --thresholds` | `results_v2/sensitivity_preds/` |
| 2.7 CPU against GPU, repeated runs | `src/compare_runs.py`, `src/audit_methods.py`; `docs/PIPELINE.md` | `results_v2/audit/methods_audit.md` |
| 2.7, 3.5 Reproducibility of artefact removal across machines, perturbation test, impact on the results, signal effects (Tables S1, S5) | `src/ica_repro.py`, `src/ica_sensitivity.py`, `src/ica_repro_impact.py`, `src/p6_signal_effects.py`, `src/gedai_qc.py`; `docs/ICA_REPRO.md` | `results_v2/ica_repro/`, `results_v2/qc/` |

## Supplementary Note 1. The Bonn corpus (earlier stage)

An earlier stage of this project compared the same operators and controls on the Bonn
corpus (Andrzejak et al. 2001): five sets of 100 single-channel segments (23.6 s,
173.61 Hz); A and B scalp recordings of healthy volunteers, C and D interictal and E
ictal intracranial recordings. Three label groupings were used: T1, three classes
(A,B / C,D / E); T2, intracranial only (C,D vs E); T3, the common binary split
(A,B,C,D vs E). Evaluation was 5 x 5 repeated stratified cross-validation over
segments: Bonn has no subject identifiers, so a subject-wise split, as in the main
study, is not possible. One preprocessing pipeline, CPU, earlier code version; these
results are not part of the multiverse.

| Model | T1 F1 | T1 AUC | T2 F1 | T2 AUC | T3 F1 | T3 AUC | Params |
|---|---|---|---|---|---|---|---|
| late | 0.974 | 0.999 | 0.987 | 1.000 | 0.986 | 1.000 | 98,571 |
| gated | 0.973 | 0.999 | 0.987 | 0.999 | 0.980 | 1.000 | 98,698 |
| attention | 0.969 | 0.997 | 0.985 | 0.999 | 0.977 | 0.998 | 98,544 |
| spec2d_wide | 0.966 | 0.998 | 0.984 | 0.999 | 0.982 | 1.000 | 97,775 |
| score | 0.964 | 0.998 | 0.981 | 0.999 | 0.976 | 0.999 | 59,302 |
| raw1d_wide | 0.963 | 0.997 | 0.973 | 0.998 | 0.976 | 0.992 | 99,795 |
| spec2d | 0.960 | 0.998 | 0.979 | 0.999 | 0.972 | 0.999 | 32,227 |
| shallow | 0.954 | 0.996 | 0.964 | 0.995 | 0.964 | 0.995 | 8 |
| raw1d | 0.951 | 0.996 | 0.964 | 0.997 | 0.969 | 0.995 | 27,075 |
| early | 0.896 | 0.977 | 0.923 | 0.981 | 0.899 | 0.965 | 97,507 |
| logvar | 0.646 | 0.755 | 0.933 | 0.986 | 0.954 | 0.992 | 1 |

Macro F1 and AUC, mean over the 25 paired measurements (`results_v2/T{{1,2,3}}_*/perfold.csv`;
parameter counts of that code version, single-channel input).

Findings: (1) the naive paired Wilcoxon test found 22, 20 and 13 of 55 pairs significant
on T1, T2 and T3; the corrected test found 10, 0 and 0, all ten on T1 involving logvar
and none between two deep models (`results_v2/phase0/summary.csv`); (2) late, gated,
attention and score fusion were equivalent to each other within about 0.02 to 0.03
macro F1 on T1; (3) early fusion was the one operator that separated, and only
downward; (4) T3 is close to saturation: a single-feature logistic regression (logvar)
reached 0.954 macro F1 against 0.986 for the best model.
"""
    OUT.write_text(text, encoding="utf-8")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
