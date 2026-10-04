"""Results section generator: the manuscript's result tables and figures in one command.

From the stored runs (the multiverse runs of each dataset, their event metrics and the
cross-dataset runs) it writes:

  Main table     pipeline x fusion operator: window-level macro F1 and AUPRC, mean and SD
                 over the seed sets (per dataset)
  Event table    seizure-level sensitivity and false alarms per hour on CHB-MIT; on Siena
                 sensitivity only, since its recordings are cut around the seizures
                 (docs/EVENTS.md)
  Rank figure    rank distribution of the fusion operators over all pipeline x seed runs
                 (src/multiverse.py: rank_distribution)
  Cross table    CHB-MIT -> Siena, per Siena patient: macro F1, AUPRC and detected
                 seizures per fusion operator

Only complete runs (every LOSO fold for all five fusion operators) enter the tables;
incomplete and missing ones are listed in the report. Tables are Markdown, ready to
paste into paper/manuscript.md, with their sources; every table is also a CSV.

Usage:  python -m src.make_results [--datasets chbmit siena] [--device cpu]
        python -m src.make_results --synthetic     # self-test on synthetic runs
Output: results_v2/results/results.md and *.csv;
        paper/figures_manuscript/results_rank_distribution_<dataset>.png/.pdf
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .chbmit_stats import FUSION
from .config import PROJECT_ROOT, RESULTS_ROOT
from .datasets import N_FOLDS
from .multiverse import (PIPELINE_ORDER, _pipeline_key, discover_runs, fig_rank_distribution,
                         load_long, md_table, rank_distribution)

NAMES = {"chbmit": "CHB-MIT", "siena": "Siena"}
CROSS_RUN = re.compile(r"^chbmit_to_siena_(P\d[a-z]?)(?:_r(\d+))?(_cuda)?$")
FIG_DIR = PROJECT_ROOT / "paper" / "figures_manuscript"


def pm(mean: float, sd: float, digits: int = 3) -> str:
    """mean ± SD as printed in the tables; only the mean when there is a single seed."""
    if not np.isfinite(mean):
        return ""
    return f"{mean:.{digits}f}" + (f" ± {sd:.{digits}f}" if np.isfinite(sd) else "")


def seed_stats(df: pd.DataFrame, keys: list[str], value: str) -> pd.DataFrame:
    """Mean, SD (ddof 1) and number of seed sets of `value` per `keys`."""
    g = df.groupby(keys, sort=False)[value]
    return pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1), "n_seeds": g.size()}).reset_index()


def wide_pm(stats: pd.DataFrame, row: str, models, bold_max: bool = True) -> pd.DataFrame:
    """Rows x fusion operators with "mean ± SD" cells; the best mean of a row in bold."""
    out = []
    for r, g in stats.groupby(row, sort=False):
        g = g.set_index("model")
        best = g["mean"].idxmax() if bold_max and g["mean"].notna().any() else None
        d = {row: r, "seeds": int(g.n_seeds.max())}
        for m in models:
            if m in g.index:
                cell = pm(g.at[m, "mean"], g.at[m, "sd"])
                d[m] = f"**{cell}**" if m == best else cell
            else:
                d[m] = ""
        out.append(d)
    return pd.DataFrame(out)


# --- Multiverse runs ---------------------------------------------------------------------

def complete_runs(root: Path, dataset: str, device: str):
    """Runs with every LOSO fold for all five fusion operators, the long table of their
    fold metrics, and the incomplete runs (name, folds present)."""
    runs = discover_runs(root, device=device)
    if not runs:
        return [], pd.DataFrame(), []
    long = load_long(runs)
    n = N_FOLDS[dataset]
    inc, exc = [], []
    for r in runs:
        sub = long[(long.run == r.name) & long.model.isin(FUSION)]
        folds = [set(sub.fold[sub.model == m]) for m in FUSION]
        k = len(set.intersection(*folds)) if all(folds) else 0
        (inc if k == n else exc).append(r if k == n else (r.name, k))
    return inc, long[long.run.isin([r.name for r in inc])], exc


def run_means(long: pd.DataFrame) -> pd.DataFrame:
    """Fold means per (pipeline, seed, model) of macro F1 and AUPRC, fusion operators."""
    sub = long[long.model.isin(FUSION)]
    return (sub.groupby(["pipeline", "seed", "model"], sort=False)[["f1_macro", "auprc"]]
            .mean().reset_index())


def order_pipelines(df: pd.DataFrame, col: str = "pipeline") -> pd.DataFrame:
    return df.iloc[sorted(range(len(df)), key=lambda i: _pipeline_key(df[col].iat[i]))]


def event_table(runs, dataset: str) -> tuple[pd.DataFrame, list[str]]:
    """Pooled seizure-level sensitivity (and, for CHB-MIT, false alarms per hour) of
    every run with event metrics, from its events/perperson.csv (summary.csv is rounded
    to four decimals): detected / annotated seizures and false alarms / hours, summed
    over the persons."""
    rows, missing = [], []
    for r in runs:
        f = r.path / "events" / "perperson.csv"
        if not f.exists():
            missing.append(r.name)
            continue
        pp = pd.read_csv(f)
        g = pp[pp.model.isin(FUSION)].groupby("model", sort=False)[
            ["detected", "seizure_events", "false_alarms", "hours"]].sum()
        for m, x in g.iterrows():
            rows.append({"pipeline": r.pipeline, "seed": r.seed, "model": m,
                         "sensitivity": x.detected / x.seizure_events,
                         "detected": int(x.detected), "seizures": int(x.seizure_events),
                         "fa_per_hour": x.false_alarms / x.hours if dataset == "chbmit"
                         else np.nan})
    return pd.DataFrame(rows), missing


# --- Cross-dataset runs ------------------------------------------------------------------

def cross_runs(root: Path, device: str) -> list[tuple[str, str, int, Path]]:
    out = []
    for d in sorted(root.iterdir()) if root.exists() else []:
        m = CROSS_RUN.match(d.name)
        if not m or bool(m.group(3)) != (device == "cuda") or not (d / "perfold.csv").exists():
            continue
        out.append((d.name, m.group(1), int(m.group(2) or 0), d))
    return out


def cross_patient_metrics(path: Path) -> pd.DataFrame:
    """Per Siena patient and fusion operator: macro F1 (perfold.csv), AUPRC (preds.npz)
    and detected / annotated seizures (events/perperson.csv, if evaluated)."""
    from sklearn.metrics import average_precision_score

    pf = pd.read_csv(path / "perfold.csv")
    pf = pf[pf.model.isin(FUSION)].rename(columns={"test_subjects": "patient"})
    out = pf[["patient", "model", "f1_macro"]].copy()
    p = np.load(path / "preds.npz")
    subj = p["subject"].astype(str)
    ap = {}
    for s in np.unique(subj):
        m = subj == s
        for mod in FUSION:
            if mod in p.files and p["y_te"][m].sum() > 0:
                ap[(s, mod)] = average_precision_score(p["y_te"][m], p[mod][m][:, 1])
    out["auprc"] = [ap.get((s, mod), np.nan) for s, mod in zip(out.patient, out.model)]
    ev = path / "events" / "perperson.csv"
    if ev.exists():
        e = pd.read_csv(ev)[["patient", "model", "detected", "seizure_events"]]
        out = out.merge(e, on=["patient", "model"], how="left")
    return out


# --- Report ------------------------------------------------------------------------------

def build_results(results_root: Path, out: Path, fig_dir: Path, datasets=("chbmit", "siena"),
                  device: str = "cpu", verbose: bool = True) -> dict:
    """Write every table and figure; return them (used by the self-test)."""
    def say(msg=""):
        if verbose:
            print(msg)

    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    models = list(FUSION)
    lines = ["# Results tables and figures",
             "",
             "Generated by `python -m src.make_results` from the stored runs; do not edit by "
             "hand. Table numbers are placeholders (R1, R2, ...) to be matched to the "
             "manuscript's numbering. Every table is also written as CSV next to this file.",
             ""]
    t = 0

    for ds in datasets:
        name = NAMES[ds]
        inc, long, exc = complete_runs(results_root / ds, ds, device)
        lines += [f"## {name}", ""]
        if not inc:
            lines += [f"No complete {name} run yet"
                      + (f" (incomplete: {', '.join(f'{n} {k}/{N_FOLDS[ds]} folds' for n, k in exc)})"
                         if exc else "") + ".", ""]
            say(f"{name}: no complete run")
            res[ds] = None
            continue
        rm = run_means(long)
        r = {"runs": [x.name for x in inc], "excluded": exc, "run_means": rm}
        lines += [f"Complete runs ({len(inc)}, {N_FOLDS[ds]} folds each): "
                  + ", ".join(f"{x.pipeline}/seed {x.seed}" for x in inc) + "."
                  + (" Not included (incomplete): "
                     + ", ".join(f"{n} ({k}/{N_FOLDS[ds]} folds)" for n, k in exc) + "."
                     if exc else ""), ""]

        # main table
        for metric, label in (("f1_macro", "window-level macro F1"), ("auprc", "AUPRC")):
            st = order_pipelines(seed_stats(rm, ["pipeline", "model"], metric))
            st.to_csv(out / f"main_{ds}_{metric}.csv", index=False)
            w = wide_pm(st, "pipeline", models)
            r[f"main_{metric}"] = st
            t += 1
            lines += [f"**Table R{t}. {name}, {label} by preprocessing pipeline and fusion "
                      f"operator, mean ± SD over seed sets.** Mean over the {N_FOLDS[ds]} "
                      f"LOSO folds within each run; best operator of each pipeline in bold. "
                      f"Source: `results_v2/{ds}/loso_*/perfold.csv`"
                      + (" and `preds/`" if metric == "auprc" else "") + ".", "",
                      md_table(w), ""]

        # event table
        ev, missing = event_table(inc, ds)
        if len(ev):
            ev.to_csv(out / f"events_{ds}_per_run.csv", index=False)
            se = order_pipelines(seed_stats(ev, ["pipeline", "model"], "sensitivity"))
            r["events_sensitivity"] = se
            t += 1
            lines += [f"**Table R{t}. {name}, seizure-level sensitivity (pooled over persons) "
                      f"by pipeline and fusion operator, mean ± SD over seed sets.** SzCORE "
                      f"event scoring of every 10 s window (docs/EVENTS.md). Source: "
                      f"`results_v2/{ds}/<run>/events/perperson.csv`.", "",
                      md_table(wide_pm(se, "pipeline", models)), ""]
            if ds == "chbmit":
                fa = order_pipelines(seed_stats(ev, ["pipeline", "model"], "fa_per_hour"))
                r["events_fa"] = fa
                t += 1
                lines += [f"**Table R{t}. CHB-MIT, false alarms per hour (pooled over "
                          f"persons, true recording time) by pipeline and fusion operator, "
                          f"mean ± SD over seed sets.** Lower is better; no bold. Source: as "
                          f"Table R{t - 1}.", "",
                          md_table(wide_pm(fa, "pipeline", models, bold_max=False)), ""]
                pd.concat([se.assign(metric="sensitivity"), fa.assign(metric="fa_per_hour")]
                          ).to_csv(out / f"events_{ds}.csv", index=False)
            else:
                se.to_csv(out / f"events_{ds}.csv", index=False)
                lines += ["Siena: only seizure-level sensitivity is reported. Its recordings "
                          "are cut around the seizures, so a false alarm rate per hour would "
                          "not represent clinical use; it is computed by src/events.py but "
                          "enters no table or comparison (a limitation, docs/EVENTS.md).", ""]
        if missing:
            lines += [f"Event metrics not yet computed for: {', '.join(missing)}.", ""]

        # rank distribution figure
        rmw = rm.pivot_table(index=["pipeline", "seed"], columns="model", values="f1_macro")
        rmw = order_pipelines(rmw[models].reset_index())
        rd = rank_distribution(rmw, models)
        rd["distribution"].to_csv(out / f"rank_distribution_{ds}.csv", index=False)
        rd["summary"].to_csv(out / f"rank_summary_{ds}.csv", index=False)
        for ext in ("png", "pdf"):
            fig_rank_distribution(rd, models, fig_dir / f"results_rank_distribution_{ds}.{ext}",
                                  f"{name}: rank of each fusion operator by macro F1")
        r["rank"] = rd
        lines += [f"**Figure R ({name}).** Rank of each fusion operator (1 = best macro F1) "
                  f"in each of the {len(rmw)} pipeline x seed runs: counts per rank (left) "
                  f"and the rank of every run (right). "
                  f"`paper/figures_manuscript/results_rank_distribution_{ds}.png`", "",
                  f"![rank distribution {name}](../../paper/figures_manuscript/"
                  f"results_rank_distribution_{ds}.png)", "",
                  md_table(rd["summary"], digits=2), ""]
        res[ds] = r
        say(f"{name}: {len(inc)} complete runs, {len(exc)} incomplete, "
            f"events for {len(set(zip(ev.pipeline, ev.seed))) if len(ev) else 0}")

    # cross-dataset table
    lines += ["## Cross-dataset: CHB-MIT -> Siena", ""]
    cr = cross_runs(results_root / "cross", device)
    res["cross"] = {}
    if not cr:
        lines += ["No cross-dataset run yet (`python -m src.cross_dataset`).", ""]
    for pipe in sorted({c[1] for c in cr}, key=_pipeline_key):
        sel = [c for c in cr if c[1] == pipe]
        per = pd.concat([cross_patient_metrics(p).assign(seed=s) for _, _, s, p in sel])
        per.to_csv(out / f"cross_{pipe}_per_run.csv", index=False)
        tabs = {}
        for metric, label in (("f1_macro", "macro F1"), ("auprc", "AUPRC")):
            st = seed_stats(per, ["patient", "model"], metric)
            w = wide_pm(st.sort_values("patient", kind="stable"), "patient", models)
            allrow = seed_stats(per.groupby(["seed", "model"])[metric].mean().reset_index(),
                                ["model"], metric).assign(patient="mean over patients")
            w = pd.concat([w, wide_pm(allrow, "patient", models)], ignore_index=True)
            tabs[metric] = st
            t += 1
            lines += [f"**Table R{t}. Trained on CHB-MIT, tested on Siena ({pipe}): {label} "
                      f"per Siena patient and fusion operator, mean ± SD over "
                      f"{len(sel)} seed set(s).** Window level, all Siena windows of the "
                      f"patient. Source: `results_v2/cross/chbmit_to_siena_{pipe}*/`.", "",
                      md_table(w), ""]
        if "detected" in per and per.detected.notna().any():
            d = per.groupby(["patient", "model"]).agg(detected=("detected", "mean"),
                                                     seizures=("seizure_events", "first"))
            wd = d.detected.unstack("model")[models]
            wd.insert(0, "seizures", d.seizures.groupby("patient").first().astype(int))
            bad = per.groupby("patient").seizure_events.nunique()
            bad = list(bad[bad > 1].index)
            tot = wd.sum().to_frame().T
            tot.index = ["all patients"]
            wd = pd.concat([wd, tot])
            wd["seizures"] = wd["seizures"].astype(int)
            wd.index.name = "patient"
            tabs["events"] = wd
            wd.to_csv(out / f"cross_{pipe}_events.csv")
            sens = (wd.loc["all patients", models] / wd.loc["all patients", "seizures"])
            t += 1
            lines += [f"**Table R{t}. Trained on CHB-MIT, tested on Siena ({pipe}): detected "
                      f"seizures per patient (mean over seed sets) out of the annotated "
                      f"ones.** SzCORE event scoring; sensitivity over all patients: "
                      + ", ".join(f"{m} {sens[m]:.2f}" for m in models) + ". False alarms "
                      "are not reported for Siena. Source: `events/perperson.csv`.", "",
                      md_table(wd.reset_index(), digits=2), ""]
            if bad:
                lines += ["WARNING: the number of annotated seizures differs between models "
                          "or seed sets for " + ", ".join(bad) + "; check the event runs.", ""]
        for metric, st in tabs.items():
            if metric != "events":
                st.to_csv(out / f"cross_{pipe}_{metric}.csv", index=False)
        res["cross"][pipe] = tabs
        say(f"cross {pipe}: {len(sel)} seed set(s), {per.patient.nunique()} patients")

    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    say(f"written: {out / 'results.md'}")
    return res


# --- Synthetic runs in the real formats ---------------------------------------------------

def _persons(dataset: str, n: int) -> list[str]:
    if dataset == "siena":
        from .siena import SUBJECTS
        return list(SUBJECTS)[:n]
    cases = [f"chb{i:02d}" for i in range(1, 25) if i != 21]
    return ["chb01,chb21" if c == "chb01" else c for c in cases][:n]


def make_synthetic_runs(root: Path, dataset: str, seed: int = 20261004,
                        pipelines=PIPELINE_ORDER, n_seeds: int = 3) -> dict:
    """LOSO runs in the layout chbmit_run writes (perfold.csv with all its columns, from
    src/evaluate.py; preds/fold<k>.npz) and their events/ (written by src/events.py's own
    writer), with known model quality. Returns the exact fold-mean F1 and AUPRC of every
    run and the pooled event counts, computed from the generated arrays."""
    from sklearn.metrics import average_precision_score

    from .evaluate import clinical_metrics, metrics
    from .events import write_outputs

    rng = np.random.default_rng(seed + (7 if dataset == "siena" else 0))
    n = N_FOLDS[dataset]
    persons = _persons(dataset, n)
    deep = ["raw1d", "spec2d", "raw1d_wide", "spec2d_wide", "late", "gated", "attention",
            "early", "score"]
    quality = {"raw1d": 1.0, "spec2d": 1.1, "raw1d_wide": 1.05, "spec2d_wide": 1.15,
               "late": 1.25, "gated": 1.3, "attention": 1.2, "early": 0.8, "score": 1.15,
               "logvar": 0.6, "shallow": 0.9}
    pipe_eff = dict(zip(pipelines, rng.normal(0, 0.1, len(pipelines))))
    n_win = rng.integers(60, 160, n)
    n_seiz = rng.integers(1, 8, n)              # annotated seizures: fixed per person
    truth = {}
    for p in pipelines:
        for s in range(n_seeds):
            name = ("loso_grouped" if (dataset == "chbmit" and p == "P0" and s == 0)
                    else f"loso_{p}" + (f"_r{s}" if s else ""))
            d = root / dataset / name
            (d / "preds").mkdir(parents=True, exist_ok=True)
            rows, ev = [], []
            for f in range(n):
                y = (rng.random(n_win[f]) < 0.2).astype(np.int64)
                y[:3] = 1
                arrs = {"idx_te": np.arange(n_win[f]) + 1000 * f, "y_te": y}
                for m, q in quality.items():
                    a = q + pipe_eff[p] + rng.normal(0, 0.05)
                    z = a * (2 * y - 1) + rng.normal(0, 1.0, len(y))
                    p1 = 1 / (1 + np.exp(-z))
                    prob = np.column_stack([1 - p1, p1]).astype(np.float32)
                    if m in deep:
                        arrs[m] = prob
                    row = metrics(y, prob, 2) | clinical_metrics(y, prob, len(y) * 10 / 3600)
                    row.update(model=m, fold=f, test_subjects=persons[f], n_test=len(y),
                               n_test_ictal=int(y.sum()), params=1000, sec=1.0, best_epoch=5,
                               epochs_run=20, val_f1=np.nan,
                               blend_w=0.5 if m == "score" else np.nan)
                    rows.append(row)
                    if m in FUSION:
                        truth[(p, s, m, f)] = (row["f1_macro"],
                                               average_precision_score(y, prob[:, 1]))
                    nz = int(n_seiz[f])
                    det = int(rng.binomial(nz, min(0.95, 0.4 + 0.4 * (q - 0.6))))
                    hours = float(rng.uniform(5, 40))
                    fp = int(rng.poisson(hours * 0.5 / q))
                    ev.append({"run": name, "model": m, "fold": f,
                               "person": persons[f].split(",")[0], "cases": persons[f],
                               "seizure_events": nz, "detected": det, "false_alarms": fp,
                               "hours": hours, "sensitivity": det / nz,
                               "fa_per_hour": fp / hours, "precision": det / (det + fp)
                               if det + fp else np.nan})
                np.savez(d / "preds" / f"fold{f}.npz", **arrs)
            pd.DataFrame(rows).to_csv(d / "perfold.csv", index=False)
            (d / "meta.json").write_text(json.dumps({"synthetic": True, "dataset": dataset}))
            e = pd.DataFrame(ev)
            write_outputs(d / "events", e.drop(columns="run"), dataset,
                          {"synthetic": True, "folds": n})
            for m in FUSION:
                em = e[e.model == m]
                truth[(p, s, m, "events")] = (em.detected.sum() / em.seizure_events.sum(),
                                              em.false_alarms.sum() / em.hours.sum())
    return truth


def make_synthetic_cross(root: Path, seed: int = 20261004, n_seeds: int = 3) -> dict:
    """Cross-dataset runs in cross_dataset's layout (perfold.csv per Siena patient,
    preds.npz with the subject array, meta.json) with events/ from src/events.py."""
    from sklearn.metrics import average_precision_score

    from .evaluate import clinical_metrics, metrics
    from .events import write_outputs

    rng = np.random.default_rng(seed + 99)
    patients = _persons("siena", 14)
    models = ["raw1d", "spec2d", "raw1d_wide", "spec2d_wide", "late", "gated", "attention",
              "early", "score", "logvar", "shallow"]
    nw = {s: int(rng.integers(80, 300)) for s in patients}
    n_seiz = {s: int(rng.integers(1, 6)) for s in patients}    # fixed per patient
    ys = {s: np.r_[np.ones(5, np.int64), (rng.random(nw[s] - 5) < 0.2).astype(np.int64)]
          for s in patients}
    truth = {}
    for r in range(n_seeds):
        tag = "chbmit_to_siena_P0" + (f"_r{r}" if r else "")
        d = root / "cross" / tag
        d.mkdir(parents=True, exist_ok=True)
        probs = {m: [] for m in models}
        rows, ev = [], []
        for k, s in enumerate(patients):
            y = ys[s]
            for m in models:
                q = 0.5 + 0.1 * models.index(m) % 7 + rng.normal(0, 0.05)
                z = q * (2 * y - 1) + rng.normal(0, 1.0, len(y))
                p1 = 1 / (1 + np.exp(-z))
                prob = np.column_stack([1 - p1, p1]).astype(np.float32)
                probs[m].append(prob)
                row = metrics(y, prob, 2) | clinical_metrics(y, prob, 1.0)
                row.update(model=m, fold=k, test_subjects=s, n_test=len(y),
                           n_test_ictal=int(y.sum()))
                rows.append(row)
                nz = n_seiz[s]
                det = int(rng.binomial(nz, 0.5))
                ev.append({"run": tag, "model": m, "patient": s, "seizure_events": nz,
                           "detected": det, "false_alarms": 3, "hours": 2.0,
                           "sensitivity": det / nz, "fa_per_hour": 1.5,
                           "precision": det / (det + 3)})
                if m in FUSION:
                    truth[(r, s, m)] = (row["f1_macro"],
                                        average_precision_score(y, prob[:, 1]), det, nz)
        pd.DataFrame(rows).to_csv(d / "perfold.csv", index=False)
        subj = np.concatenate([[s] * nw[s] for s in patients])
        y_all = np.concatenate([ys[s] for s in patients])
        np.savez(d / "preds.npz", idx_te=np.arange(len(y_all)), y_te=y_all, subject=subj,
                 **{m: np.concatenate(v) for m, v in probs.items()})
        (d / "meta.json").write_text(json.dumps({"synthetic": True, "score_blend_w": 0.5,
                                                 "pipeline": {"name": "P0"}}))
        write_outputs(d / "events", pd.DataFrame(ev).drop(columns="run"), "siena",
                      {"synthetic": True, "mode": "cross-dataset"})
    return truth


def self_test(out: Path) -> bool:
    """Synthetic runs of both datasets and the cross experiment, in the real formats;
    checks every table against values computed from the generated arrays, and runs the
    multiverse analysis on the 14-fold Siena runs."""
    from .multiverse import analyze, first_report

    checks, report = [], ["# Self-test of src/make_results.py", "",
                          "Synthetic runs in the formats the real runs write (chbmit_run: "
                          "perfold.csv with all columns from src/evaluate.py and preds/; "
                          "src/events.py: events/ through its own writer; cross_dataset: "
                          "perfold.csv per patient, preds.npz, meta.json), 9 pipelines x 3 "
                          "seeds on CHB-MIT (23 folds) and Siena (14 folds), 3 cross runs. "
                          "CHB-MIT P6c seed 2 is truncated to 10 folds, so it must be left "
                          "out.", "", "| check | result | |", "|---|---|---|"]

    def check(name, ok, detail):
        checks.append(bool(ok))
        report.append(f"| {name} | {detail} | {'PASS' if ok else 'FAIL'} |")

    tmp = Path(tempfile.mkdtemp(prefix="make_results_"))
    try:
        truth = {ds: make_synthetic_runs(tmp, ds) for ds in ("chbmit", "siena")}
        tc = make_synthetic_cross(tmp)
        pf = tmp / "chbmit" / "loso_P6c_r2" / "perfold.csv"
        df = pd.read_csv(pf)
        df[df.fold < 10].to_csv(pf, index=False)
        res = build_results(tmp, out / "results", out / "figures", verbose=False)

        for ds in ("chbmit", "siena"):
            r, tr = res[ds], truth[ds]
            for metric, j in (("f1_macro", 0), ("auprc", 1)):
                st = r[f"main_{metric}"].set_index(["pipeline", "model"])
                err, n = 0.0, 0
                for (p, m), row in st.iterrows():
                    seeds = [s for s in range(3) if not (ds == "chbmit" and p == "P6c" and s == 2)]
                    v = [np.mean([tr[(p, s, m, f)][j] for f in range(N_FOLDS[ds])]) for s in seeds]
                    err = max(err, abs(row["mean"] - np.mean(v)), abs(row["sd"] - np.std(v, ddof=1)))
                    n += 1
                check(f"{ds}: main table {metric} = values from the generated arrays",
                      err < 1e-9 and n == 45, f"{n} cells, max error {err:.1e}")
            ev = r["events_sensitivity"].set_index(["pipeline", "model"])
            err = max(abs(row["mean"] - np.mean([tr[(p, s, m, "events")][0] for s in range(3)
                                                if not (ds == "chbmit" and p == "P6c" and s == 2)]))
                      for (p, m), row in ev.iterrows())
            check(f"{ds}: event sensitivity table = pooled sensitivity", err < 1e-9,
                  f"max error {err:.1e}")
        st = res["chbmit"]["main_f1_macro"]
        check("chbmit: incomplete run left out and listed",
              res["chbmit"]["excluded"] == [("loso_P6c_r2", 10)]
              and int(st[st.pipeline == "P6c"].n_seeds.max()) == 2
              and len(res["chbmit"]["runs"]) == 26,
              f"excluded {res['chbmit']['excluded']}; P6c with 2 seeds; 26 runs")
        fa = res["chbmit"]["events_fa"].set_index(["pipeline", "model"])
        err = max(abs(row["mean"] - np.mean([truth["chbmit"][(p, s, m, "events")][1]
                                             for s in range(3) if not (p == "P6c" and s == 2)]))
                  for (p, m), row in fa.iterrows())
        check("chbmit: false alarms per hour table = pooled rate", err < 1e-9,
              f"max error {err:.1e}")
        md = (out / "results" / "results.md").read_text(encoding="utf-8")
        siena_md = md.split("## Siena")[1].split("## Cross-dataset")[0]
        check("siena: no false alarm table or column",
              "events_fa" not in res["siena"] and "false alarms per hour (pooled" not in siena_md,
              "sensitivity only; limitation note present"
              if "only seizure-level sensitivity" in siena_md else "note missing")
        for ds in ("chbmit", "siena"):
            rd = res[ds]["rank"]
            nr = len(res[ds]["runs"])
            ok = (int(rd["distribution"].query("scope == 'all'")["count"].sum()) == 5 * nr
                  and all((out / "figures" / f"results_rank_distribution_{ds}.{e}").exists()
                          for e in ("png", "pdf")))
            check(f"{ds}: rank distribution figure and counts", ok,
                  f"{5 * nr} ranks = {nr} runs x 5; PNG and PDF written")
        cx = res["cross"]["P0"]
        st = cx["f1_macro"].set_index(["patient", "model"])
        err = max(abs(row["mean"] - np.mean([tc[(r, p, m)][0] for r in range(3)]))
                  for (p, m), row in st.iterrows())
        evs = cx["events"]
        det_ok = all(abs(evs.at[p, m] - np.mean([tc[(r, p, m)][2] for r in range(3)])) < 1e-9
                     for p in _persons("siena", 14) for m in FUSION)
        det_ok &= bool((evs[list(FUSION)].le(evs["seizures"], axis=0)).all().all())
        check("cross: per-patient table, 14 Siena patients",
              st.index.get_level_values(0).nunique() == 14 and err < 1e-9 and det_ok,
              f"14 patients; macro F1 max error {err:.1e}; detected seizures exact and "
              f"never above the annotated ones")

        # multiverse on the 14-fold Siena runs
        mv = analyze(tmp / "siena", out / "multiverse_siena", "synthetic siena", n_perm=500,
                     verbose=False, dataset="siena")
        summ = (out / "multiverse_siena" / "summary.md").read_text(encoding="utf-8")
        check("multiverse --dataset siena: 14 folds",
              mv["n_folds"] == 14 and len(mv["pipelines"]) == 9 and len(mv["seeds"]) == 3
              and "ratio 1/13" in summ and np.isfinite(mv["interaction_F"])
              and np.isfinite(mv["tau_between"]),
              f"balanced block {len(mv['pipelines'])} pipelines x {len(mv['seeds'])} seeds x "
              f"{mv['n_folds']} folds; corrected tests with test/train ratio 1/13; "
              f"interaction F = {mv['interaction_F']:.2f}")
        fr = first_report(tmp / "siena", out / "multiverse_siena" / "first_report",
                          "synthetic siena", dataset="siena", verbose=False)
        check("multiverse --first-report on Siena", len(fr["included"]) == 27
              and fr["n_folds"] == 14, f"{len(fr['included'])} runs, {fr['n_folds']} folds")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    report += ["", f"**{sum(checks)}/{len(checks)} checks passed.** Generated tables: "
               f"`results/results.md`; figures: `figures/`.", ""]
    out.mkdir(parents=True, exist_ok=True)
    (out / "self_test.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return all(checks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+", default=["chbmit", "siena"],
                    choices=["chbmit", "siena"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", default=None, help="tables (default results_v2/results)")
    ap.add_argument("--fig-dir", default=None,
                    help="figures (default paper/figures_manuscript)")
    ap.add_argument("--synthetic", action="store_true",
                    help="self-test on synthetic runs in the real file formats")
    args = ap.parse_args()
    if args.synthetic:
        out = Path(args.out) if args.out else RESULTS_ROOT / "results_selftest"
        sys.exit(0 if self_test(out) else 1)
    build_results(RESULTS_ROOT, Path(args.out) if args.out else RESULTS_ROOT / "results",
                  Path(args.fig_dir) if args.fig_dir else FIG_DIR, args.datasets, args.device)


if __name__ == "__main__":
    main()
