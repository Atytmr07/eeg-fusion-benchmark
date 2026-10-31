"""Controls against the fusion operators with the same corrected test as the operator pairs.

Table 3 of the manuscript compares the single-branch and classical controls with the
fusion operators by their means only. This module tests each control against each
operator in every pipeline as src/multiverse.py tests the operator pairs (Section 5,
`equivalence`): fold values averaged over the seed sets, the Nadeau-Bengio corrected
t test with test/train ratio 1/(k-1), Holm across the comparisons of one pipeline
(6 controls x 5 operators), and the equivalence bound delta_min.

Usage:  python -m src.controls_stats [--device cuda] [--margin 0.05]
Output: results_v2/results/controls_tests_<dataset>_<metric>.csv (per pipeline and pair),
        results_v2/results/controls_tests.md (summary per control and operator)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .chbmit_stats import FUSION, ratios
from .config import RESULTS_ROOT
from .make_results import CONTROLS, complete_runs
from .multiverse import _pipeline_key, md_table, to_array
from .stats import corrected_equivalence_bound, corrected_ttest, holm

OUT = RESULTS_ROOT / "results"
NAMES = {"chbmit": "CHB-MIT", "siena": "Siena"}
METRICS = {"f1_macro": "macro F1", "auprc": "AUPRC"}


def tests(y: np.ndarray, pipes, models, margin: float) -> pd.DataFrame:
    """y: (fold, pipeline, model), seeds averaged; models = [*FUSION, *CONTROLS]."""
    ratio, _ = ratios(y.shape[0])
    idx = {m: i for i, m in enumerate(models)}
    rows = []
    for pi, p in enumerate(pipes):
        block = []
        for c in CONTROLS:
            for f in FUSION:
                xc, xf = y[:, pi, idx[c]], y[:, pi, idx[f]]
                _, pv = corrected_ttest(xc, xf, ratio)
                block.append({"pipeline": p, "control": c, "operator": f,
                              "mean_diff": float((xc - xf).mean()), "p_corrected": pv,
                              "delta_min": corrected_equivalence_bound(xc, xf, ratio)})
        for r, h in zip(block, holm(np.array([r["p_corrected"] for r in block]))):
            r["p_holm"] = float(h)
            r["significant"] = bool(h < 0.05)
            r["equivalent"] = bool(r["delta_min"] <= margin)
        rows += block
    return pd.DataFrame(rows)


def summary(t: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby(["control", "operator"], sort=False)
    return pd.DataFrame({
        "pipelines": g.size(),
        "mean_diff": g.mean_diff.mean(),
        "control_higher_sig": g.apply(lambda d: int((d.significant & (d.mean_diff > 0)).sum()),
                                      include_groups=False),
        "control_lower_sig": g.apply(lambda d: int((d.significant & (d.mean_diff < 0)).sum()),
                                     include_groups=False),
        "equivalent": g.equivalent.sum(),
        "min_p_holm": g.p_holm.min(),
    }).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--margin", type=float, default=0.05)
    args = ap.parse_args()
    models = [*FUSION, *CONTROLS]
    lines = ["# Controls against the fusion operators (corrected test)", "",
             f"Per pipeline: fold values averaged over seed sets, Nadeau-Bengio corrected t "
             f"test, Holm across the {len(CONTROLS) * len(FUSION)} control-operator "
             f"comparisons of the pipeline; equivalent = delta_min <= {args.margin}. "
             f"Counts are pipelines (of 9). mean_diff = control minus operator.", ""]
    for ds in ("chbmit", "siena"):
        inc, long, _ = complete_runs(RESULTS_ROOT / ds, ds, args.device)
        pipes = sorted(long.pipeline.unique(), key=_pipeline_key)
        seeds = sorted(long.seed.unique())
        folds = sorted(long.fold.unique())
        for metric, label in METRICS.items():
            y = to_array(long, metric, models, pipes, seeds, folds).mean(axis=3)
            t = tests(y, pipes, models, args.margin)
            t.to_csv(OUT / f"controls_tests_{ds}_{metric}.csv", index=False)
            s = summary(t)
            lines += [f"## {NAMES[ds]}, {label} ({len(inc)} runs, {len(folds)} folds)", "",
                      md_table(s, digits=3), ""]
            print(f"{NAMES[ds]} {label}: significant {int(t.significant.sum())} of {len(t)} "
                  f"(control higher {int((t.significant & (t.mean_diff > 0)).sum())}), "
                  f"equivalent {int(t.equivalent.sum())}")
    (OUT / "controls_tests.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"written: {OUT / 'controls_tests.md'}")


if __name__ == "__main__":
    main()
