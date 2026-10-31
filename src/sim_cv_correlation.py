"""How much does assuming independence inflate false positives under repeated CV?

A simulation that mirrors this project's Bonn design: n = 500, 5 folds x 5 repeats,
stratified splits, test/train ratio 100/400.

The null hypothesis is exactly true: two classifiers use two different but
statistically identical features, so their expected performance is equal by
construction and every "significant difference" is a false positive.

Four tests are applied to each simulated dataset and their rejection rates at
alpha = 0.05 are measured:
  - Wilcoxon signed-rank (assumes independence)
  - paired t test (assumes independence)
  - sign test (assumes independence, not symmetry)
  - corrected resampled t test (Nadeau and Bengio / Bouckaert and Frank)

A well-calibrated test rejects about 5 percent of the time.

Usage:  python -m src.sim_cv_correlation --sims 1000
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .stats import corrected_ttest, sign_test


def make_dataset(rng: np.random.Generator, n: int = 500, signal: float = 0.9):
    y = np.r_[np.zeros(n // 2, int), np.ones(n - n // 2, int)]
    rng.shuffle(y)
    s = (2 * y - 1) * signal
    # Both features carry the same signal with independent, identically distributed noise.
    x1 = s + rng.normal(0, 1, n)
    x2 = s + rng.normal(0, 1, n)
    return x1[:, None], x2[:, None], y


def repeated_cv_scores(x1, x2, y, k: int, r: int, seed: int):
    a, b = [], []
    for rep in range(r):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed + rep)
        for tr, te in skf.split(x1, y):
            m1 = LogisticRegression().fit(x1[tr], y[tr])
            m2 = LogisticRegression().fit(x2[tr], y[tr])
            a.append(f1_score(y[te], m1.predict(x1[te]), average="macro"))
            b.append(f1_score(y[te], m2.predict(x2[te]), average="macro"))
    return np.array(a), np.array(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--r", type=int, default=5)
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    rng = np.random.default_rng(20260917)
    ratio = 1.0 / (args.k - 1)
    rejects = {"wilcoxon": 0, "paired_t": 0, "sign": 0, "corrected_t": 0}
    t0 = time.perf_counter()

    for i in range(args.sims):
        x1, x2, y = make_dataset(rng, args.n)
        a, b = repeated_cv_scores(x1, x2, y, args.k, args.r, seed=10_000 + i)
        d = a - b
        if np.allclose(d, 0):
            continue
        try:
            p_w = stats.wilcoxon(a, b).pvalue
        except ValueError:
            p_w = 1.0
        rejects["wilcoxon"] += p_w < 0.05
        rejects["paired_t"] += stats.ttest_rel(a, b).pvalue < 0.05
        rejects["sign"] += sign_test(a, b) < 0.05
        rejects["corrected_t"] += corrected_ttest(a, b, ratio)[1] < 0.05

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{args.sims}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    print(f"\nDesign: n={args.n}, {args.k} folds x {args.r} repeats, "
          f"test/train = 1/{args.k-1}, {args.sims} simulations")
    print("True difference: exactly zero. A well-calibrated test rejects about 5%.\n")
    print(f"{'test':<34s} {'false positive rate':>20s}")
    print("-" * 56)
    labels = {"wilcoxon": "Wilcoxon (assumes independence)",
              "paired_t": "Paired t (assumes independence)",
              "sign": "Sign test (assumes independence)",
              "corrected_t": "Corrected resampled t"}
    se = np.sqrt(0.05 * 0.95 / args.sims)
    for k, lab in labels.items():
        rate = rejects[k] / args.sims
        print(f"{lab:<34s} {rate:>19.1%}")
    print(f"\n(simulation error for a nominal 5%: about +-{1.96*se:.1%})")


if __name__ == "__main__":
    main()
