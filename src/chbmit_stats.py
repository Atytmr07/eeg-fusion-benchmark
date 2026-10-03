"""CHB-MIT statistics: the corrected framework of src/phase0.py applied to LOSO.

phase0.py pairs measurements by (repeat, fold) for 5x5 CV. CHB-MIT LOSO has one
measurement per fold, each fold holding out a different person, and no repeats, so it
has its own module.

Test/train ratio. In LOSO each fold tests one person and trains on the rest, so the
ratio is 1/(n_folds - 1): 1/22 with the 23 person-wise folds. This is the direct
extension of the Nadeau-Bengio correction, whose source is the overlap between the
folds' training sets, not the k-fold form as such. A conservative ratio that also
removes the inner validation persons from training is reported alongside.

The ROPE constants match the Bonn analysis so the two corpora stay comparable.

Usage:   python -m src.chbmit_stats                     # loso_grouped: chb01+chb21 merged, 23 folds
         python -m src.chbmit_stats --run loso_main     # earlier case based run, 24 folds
Output:  results_v2/chbmit/phase0_<run>/*.csv, except loso_main: results_v2/chbmit/phase0/
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .config import RESULTS_ROOT
from .datasets import DATASETS
from .stats import (corrected_equivalence_bound, corrected_ttest, correlated_bayes,
                    holm, sign_test)

FUSION = ("early", "late", "gated", "attention", "score")
ROPE_BY_METRIC = {
    "f1_macro": (0.01, 0.02),
    "log_loss": (0.02, 0.05),
    "brier": (0.01, 0.02),
}
LOWER_IS_BETTER = {"log_loss", "brier"}
# Must match val_frac in chbmit_run.inner_split.
VAL_FRAC = 0.2


def ratios(n_folds: int, val_frac: float = VAL_FRAC) -> tuple[float, float]:
    """Primary 1/(k-1); the conservative ratio drops the validation persons from training."""
    n_train = n_folds - 1
    n_val = max(1, int(round(val_frac * n_train)))
    return 1.0 / n_train, 1.0 / (n_train - n_val)


def analyze_metric(df: pd.DataFrame, metric: str, ratio: float,
                   ratio_con: float) -> pd.DataFrame:
    """All pairwise comparisons for one metric. Rows are paired by fold (person)."""
    piv = df.pivot_table(index="fold", columns="model", values=metric).sort_index()
    ropes = ROPE_BY_METRIC[metric]
    rows = []
    for m1, m2 in combinations(piv.columns, 2):
        a, b = piv[m1].to_numpy(float), piv[m2].to_numpy(float)
        ok = ~(np.isnan(a) | np.isnan(b))
        a, b = a[ok], b[ok]
        d = a - b
        if len(d) < 3:
            continue
        identical = bool(np.allclose(d, 0))
        try:
            p_w = 1.0 if identical else stats.wilcoxon(a, b).pvalue
        except ValueError:
            p_w = 1.0
        _, p_c = corrected_ttest(a, b, ratio)
        _, p_c2 = corrected_ttest(a, b, ratio_con)
        row = {
            "metric": metric, "model_a": m1, "model_b": m2, "n": len(d),
            "mean_a": a.mean(), "mean_b": b.mean(), "mean_diff": d.mean(),
            "p_naive_wilcoxon": p_w,
            "p_corrected": p_c,
            "p_corrected_conservative": p_c2,
            "p_sign": sign_test(a, b),
            "delta_min_naive": abs(d.mean()) + stats.t.ppf(0.95, len(d) - 1)
                               * d.std(ddof=1) / np.sqrt(len(d)),
            "delta_min_corrected": corrected_equivalence_bound(a, b, ratio),
        }
        for rope in ropes:
            pl, pr, pg = correlated_bayes(a, b, ratio, rope)
            tag = f"{int(rope * 100):02d}"
            row[f"bayes_p_b_better_r{tag}"] = pl
            row[f"bayes_p_equiv_r{tag}"] = pr
            row[f"bayes_p_a_better_r{tag}"] = pg
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_naive_wilcoxon_holm"] = holm(out["p_naive_wilcoxon"].to_numpy())
    out["p_corrected_holm"] = holm(out["p_corrected"].to_numpy())
    out["p_corrected_conservative_holm"] = holm(out["p_corrected_conservative"].to_numpy())
    out["p_sign_holm"] = holm(out["p_sign"].to_numpy())
    out["sig_naive"] = out["p_naive_wilcoxon_holm"] < 0.05
    out["sig_corrected"] = out["p_corrected_holm"] < 0.05
    return out


def per_model_summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = df.groupby("model")[metric]
    s = pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1), "n": g.size()})
    return s.sort_values("mean", ascending=metric in LOWER_IS_BETTER)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chbmit", choices=list(DATASETS))
    ap.add_argument("--run", default="loso_grouped")
    args = ap.parse_args()
    in_csv = RESULTS_ROOT / args.dataset / args.run / "perfold.csv"
    out = RESULTS_ROOT / args.dataset / ("phase0" if args.run == "loso_main"
                                     else f"phase0_{args.run}")
    if not in_csv.exists():
        print(f"not found: {in_csv}")
        return
    df = pd.read_csv(in_csv)
    n_folds = df.fold.nunique()
    ratio, ratio_con = ratios(n_folds)
    print(f"{args.dataset} LOSO: {len(df)} rows, {n_folds} folds, "
          f"{df.model.nunique()} models, test/train ratio = 1/{n_folds - 1} "
          f"(conservative: 1/{round(1 / ratio_con)})\n")

    out.mkdir(parents=True, exist_ok=True)
    all_res, summary_rows = [], []

    for metric in ("f1_macro", "log_loss", "brier"):
        s = per_model_summary(df, metric)
        print(f"=== {metric} (lower is better: {metric in LOWER_IS_BETTER}) ===")
        print(s.round(4).to_string())
        spread_all = s["mean"].max() - s["mean"].min()
        fus = s.loc[[m for m in FUSION if m in s.index], "mean"]
        spread_fus = fus.max() - fus.min() if len(fus) else float("nan")
        print(f"spread over all models: {spread_all:.4f}   "
              f"spread over fusion operators: {spread_fus:.4f}")

        res = analyze_metric(df, metric, ratio, ratio_con)
        if res.empty:
            print("(no pairs, skipped)\n")
            continue
        res.to_csv(out / f"chbmit_{metric}.csv", index=False)
        all_res.append(res)

        sig = res[res.sig_corrected]
        print(f"pairs={len(res)}  significant: naive={int(res.sig_naive.sum())}  "
              f"corrected={int(res.sig_corrected.sum())}")
        if len(sig):
            print("pairs significant under the corrected test:")
            print(sig[["model_a", "model_b", "mean_diff", "p_corrected_holm"]]
                  .round(4).to_string(index=False))

        fus_pairs = res[res.model_a.isin(FUSION) & res.model_b.isin(FUSION)]
        rope_lo = ROPE_BY_METRIC[metric][0]
        tag = f"{int(rope_lo * 100):02d}"
        if len(fus_pairs):
            print(f"fusion pairs: n={len(fus_pairs)}  "
                  f"corrected significant={int(fus_pairs.sig_corrected.sum())}  "
                  f"delta_min range=[{fus_pairs.delta_min_corrected.min():.4f}, "
                  f"{fus_pairs.delta_min_corrected.max():.4f}]  "
                  f"P(equivalent within ROPE {rope_lo}) range="
                  f"[{fus_pairs[f'bayes_p_equiv_r{tag}'].min():.2f}, "
                  f"{fus_pairs[f'bayes_p_equiv_r{tag}'].max():.2f}]")
        print()

        summary_rows.append({
            "metric": metric, "pairs": len(res),
            "sig_naive": int(res.sig_naive.sum()),
            "sig_corrected": int(res.sig_corrected.sum()),
            "fusion_pairs": len(fus_pairs),
            "fusion_sig_corrected": int(fus_pairs.sig_corrected.sum()) if len(fus_pairs) else 0,
        })

    if all_res:
        pd.concat(all_res).to_csv(out / "chbmit_all.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out / "summary.csv", index=False)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
