"""How much does P6's irreproducibility change the results? (advisor's item 2)

The same trained models were evaluated twice on CHB-MIT's full recordings: on the lab PC,
whose artefact removal differed from the training-time cleaning (events_lab_FAIL/), and
on the laptop that built the caches (events/, the valid evaluation). Window-level results
use the cached windows and are unaffected; this compares the seizure-level metrics, the
only results that depend on cleaning the recordings again.

Per P6 pipeline and seed set, for every fusion operator: pooled sensitivity and false
alarms per hour under both cleanings, the persons whose detected-seizure count changed,
and the Kendall tau between the operators' rankings under the two cleanings.

Usage:  python -m src.ica_repro_impact [--dataset chbmit]
Output: results_v2/ica_repro/impact_events.csv, impact_events_summary.csv, impact_events.md
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .chbmit_stats import FUSION
from .config import RESULTS_ROOT
from .multiverse import RUN_NAME

OUT = RESULTS_ROOT / "ica_repro"


def pooled(pp: pd.DataFrame) -> pd.DataFrame:
    g = pp[pp.model.isin(FUSION)].groupby("model")[
        ["detected", "seizure_events", "false_alarms", "hours"]].sum()
    return pd.DataFrame({"sensitivity": g.detected / g.seizure_events,
                         "fa_per_hour": g.false_alarms / g.hours}).reindex(list(FUSION))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chbmit")
    args = ap.parse_args()
    rows, runs = [], []
    for d in sorted((RESULTS_ROOT / args.dataset).iterdir()):
        m = RUN_NAME["cuda"].match(d.name)
        a, b = d / "events" / "perperson.csv", d / "events_lab_FAIL" / "perperson.csv"
        if not m or not (a.exists() and b.exists()):
            continue
        pipe, seed = m.group(1), int(m.group(2) or 0)
        va, vb = pd.read_csv(a), pd.read_csv(b)
        pa, pb = pooled(va), pooled(vb)
        key = ["model", "fold"]
        j = va.merge(vb, on=key, suffixes=("_valid", "_lab"))
        j = j[j.model.isin(FUSION)]
        persons_changed = (j.detected_valid != j.detected_lab).groupby(j.model).sum()
        for mdl in FUSION:
            rows.append({"pipeline": pipe, "seed": seed, "model": mdl,
                         "sens_valid": pa.at[mdl, "sensitivity"], "sens_lab": pb.at[mdl, "sensitivity"],
                         "fa_valid": pa.at[mdl, "fa_per_hour"], "fa_lab": pb.at[mdl, "fa_per_hour"],
                         "persons_detected_changed": int(persons_changed.get(mdl, 0)),
                         "persons": int(j[j.model == mdl].fold.nunique())})
        runs.append({"pipeline": pipe, "seed": seed,
                     "tau_sensitivity": kendalltau(pa.sensitivity, pb.sensitivity).statistic,
                     "tau_fa": kendalltau(pa.fa_per_hour, pb.fa_per_hour).statistic,
                     "same_best_sensitivity": pa.sensitivity.idxmax() == pb.sensitivity.idxmax(),
                     "same_lowest_fa": pa.fa_per_hour.idxmin() == pb.fa_per_hour.idxmin()})
    if not rows:
        raise SystemExit("no run has both events/ and events_lab_FAIL/")
    df = pd.DataFrame(rows)
    df["d_sens"] = df.sens_lab - df.sens_valid
    df["d_fa"] = df.fa_lab - df.fa_valid
    rk = pd.DataFrame(runs)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "impact_events.csv", index=False)
    summ = df.groupby("pipeline").agg(
        runs=("seed", "nunique"),
        abs_d_sens_mean=("d_sens", lambda v: float(np.abs(v).mean())),
        abs_d_sens_max=("d_sens", lambda v: float(np.abs(v).max())),
        abs_d_fa_mean=("d_fa", lambda v: float(np.abs(v).mean())),
        abs_d_fa_max=("d_fa", lambda v: float(np.abs(v).max())),
        persons_changed_mean=("persons_detected_changed", "mean"))
    summ = summ.join(rk.groupby("pipeline").agg(
        tau_sensitivity_mean=("tau_sensitivity", "mean"), tau_fa_mean=("tau_fa", "mean"),
        same_best_sensitivity=("same_best_sensitivity", "mean"),
        same_lowest_fa=("same_lowest_fa", "mean")))
    summ.to_csv(OUT / "impact_events_summary.csv")
    rk.to_csv(OUT / "impact_events_rankings.csv", index=False)

    def md(t: pd.DataFrame) -> str:
        t = t.reset_index() if t.index.name else t
        cols = list(t.columns)
        f = lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)
        return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                         + ["| " + " | ".join(f(v) for v in r) + " |" for r in t.itertuples(index=False)])
    lines = ["# Effect of P6's irreproducibility on the event metrics", "",
             f"{args.dataset}: the same models, evaluated on recordings cleaned on the lab PC "
             "(different from training-time cleaning) and on the laptop (identical to it, "
             "or within rounding for AMICA). Differences are lab minus valid.", "",
             "## Per pipeline (over operators and seed sets)", "", md(summ), "",
             "## Per run: ranking of the five operators under both cleanings", "", md(rk), "",
             "## Per operator", "", md(df.round(4))]
    (OUT / "impact_events.md").write_text("\n".join(lines), encoding="utf-8")
    print(summ.round(4).to_string())
    print(rk.to_string(index=False))


if __name__ == "__main__":
    main()
