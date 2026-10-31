"""Bonn statistics on probabilistic metrics, plus the thread-count comparison.

Macro F1 could not separate the operators. Log loss and Brier score carry more
information, because they also penalise how confident a wrong prediction was. Do the
equivalence bounds narrow, or does a difference appear, with these metrics?

Uses the 12-thread rerun in results_v2/rerun_probs/, which stored predicted
probabilities. The same rerun is also compared, model by model, with the original
2-thread run: the two are the same experiment, so any difference comes from the
thread count alone.

Usage:   python -m src.phase0_probscores
Output:  results_v2/phase0/probscores_*.csv, thread_realisation.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from . import phase0
from .config import RESULTS_ROOT, Config

OUT = RESULTS_ROOT / "phase0"
RERUN = RESULTS_ROOT / "rerun_probs" / "T1_3class__N2"
FUSION = ("early", "late", "gated", "attention", "score")

# Region of practical equivalence per metric. Probabilistic metrics live on a different
# scale than F1, so each has its own ROPEs; tables report them next to the observed
# spread of the metric.
ROPE_BY_METRIC = {
    "f1_macro": (0.01, 0.02),
    "log_loss": (0.02, 0.05),
    "brier": (0.01, 0.02),
}
LOWER_IS_BETTER = {"log_loss", "brier"}


def per_model_summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = df.groupby("model")[metric]
    out = pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1), "n": g.size()})
    return out.sort_values("mean", ascending=metric in LOWER_IS_BETTER)


def analyse_metric(df: pd.DataFrame, metric: str, n_folds: int) -> pd.DataFrame:
    res = phase0.analyze(df, n_folds, metric=metric, ropes=ROPE_BY_METRIC[metric])
    res.insert(0, "metric", metric)
    return res


def thread_realisation(orig: pd.DataFrame, rerun: pd.DataFrame,
                       metric: str = "f1_macro") -> pd.DataFrame:
    """Per-model difference between two numerical realisations of the same run."""
    key = ["model", "repeat", "fold"]
    m = orig[key + [metric]].merge(rerun[key + [metric]], on=key,
                                   suffixes=("_t2", "_t12"))
    m["diff"] = m[f"{metric}_t12"] - m[f"{metric}_t2"]
    rows = []
    for model, g in m.groupby("model"):
        d = g["diff"].to_numpy(float)
        rows.append({
            "model": model, "n_folds": len(d),
            "mean_t2": g[f"{metric}_t2"].mean(), "mean_t12": g[f"{metric}_t12"].mean(),
            "mean_diff": d.mean(), "mean_abs_diff": np.abs(d).mean(),
            "max_abs_diff": np.abs(d).max(),
            "n_identical": int(np.isclose(d, 0).sum()),
        })
    return pd.DataFrame(rows).sort_values("max_abs_diff", ascending=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    f = RERUN / "perfold.csv"
    if not f.exists():
        print(f"not found: {f}")
        return
    df = pd.read_csv(f)
    print(f"rerun: {len(df)} rows, {df.model.nunique()} models, "
          f"{df.groupby(['repeat', 'fold']).ngroups} splits\n")

    all_res = []
    for metric in ("f1_macro", "log_loss", "brier"):
        s = per_model_summary(df, metric)
        print(f"=== {metric} (lower is better: {metric in LOWER_IS_BETTER}) ===")
        print(s.round(4).to_string())
        spread = s["mean"].max() - s["mean"].min()
        fus = s.loc[[m for m in FUSION if m in s.index], "mean"]
        print(f"spread over all models: {spread:.4f}   "
              f"spread over fusion operators: {fus.max() - fus.min():.4f}")

        res = analyse_metric(df, metric, cfg.n_folds)
        res.to_csv(OUT / f"probscores_{metric}.csv", index=False)
        all_res.append(res)

        sig_n = int(res.sig_naive.sum())
        sig_c = int(res.sig_corrected.sum())
        print(f"pairs={len(res)}  significant: naive={sig_n}  corrected={sig_c}")

        sig = res[res.sig_corrected]
        if len(sig):
            print("pairs significant under the corrected test:")
            print(sig[["model_a", "model_b", "mean_diff", "p_corrected_holm"]]
                  .round(4).to_string(index=False))

        fus_pairs = res[res.model_a.isin(FUSION) & res.model_b.isin(FUSION)]
        rope_lo = ROPE_BY_METRIC[metric][0]
        tag = f"{int(rope_lo * 100):02d}"
        print(f"fusion pairs: n={len(fus_pairs)}  "
              f"corrected significant={int(fus_pairs.sig_corrected.sum())}  "
              f"delta_min range="
              f"[{fus_pairs.delta_min_corrected.min():.4f}, "
              f"{fus_pairs.delta_min_corrected.max():.4f}]  "
              f"P(equivalent within ROPE {rope_lo}) range="
              f"[{fus_pairs[f'bayes_p_equiv_r{tag}'].min():.2f}, "
              f"{fus_pairs[f'bayes_p_equiv_r{tag}'].max():.2f}]\n")

    pd.concat(all_res).to_csv(OUT / "probscores_all.csv", index=False)

    # --- two numerical realisations of the same experiment ---
    orig_dirs = [d for d in sorted(RESULTS_ROOT.glob("T1_3class__N2_z_zspec__*"))
                 if (d / "perfold.csv").exists()]
    if orig_dirs:
        orig = pd.read_csv(orig_dirs[0] / "perfold.csv")
        tr = thread_realisation(orig, df)
        tr.to_csv(OUT / "thread_realisation.csv", index=False)
        print("=== same experiment, two thread counts (12 minus 2), f1_macro ===")
        print(tr.round(4).to_string(index=False))
        print(f"\nsource: {orig_dirs[0].name}")
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
