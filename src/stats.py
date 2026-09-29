"""Statistics for comparing models under repeated cross-validation.

Why not the usual paired Wilcoxon or t test: in repeated k-fold CV the training sets
of different folds and repeats overlap heavily, so the per-fold scores are not
independent. Tests that assume independence underestimate the variance and inflate
false positives (38 percent instead of 5 in our simulation, src/sim_cv_correlation.py).

The corrected resampled t test (Nadeau and Bengio 2003; Bouckaert and Frank 2004)
inflates the variance of the mean difference to account for this:

    SE^2 = (1/n + n_test/n_train) * s^2        (instead of s^2 / n)

where n is the number of paired measurements and s^2 the sample variance of the
differences. For k-fold CV the standard choice is n_test/n_train = 1/(k-1). The same
scaling is used in the posterior of the correlated Bayesian t test (Corani and
Benavoli 2015).

"No significant difference" is not evidence of equivalence, so null results are also
reported as an equivalence bound (the narrowest margin within which the two models
are shown equivalent) and as the posterior probability of practical equivalence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (multiple comparison correction)."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def summarize(df: pd.DataFrame, metric: str = "f1_macro") -> pd.DataFrame:
    """Per-model mean, standard deviation and 95% confidence interval half-width."""
    rows = []
    for m, g in df.groupby("model"):
        v = g[metric].dropna().to_numpy()
        n = len(v)
        ci = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows.append({"model": m, "n": n, "mean": v.mean(), "std": v.std(ddof=1),
                     "ci95": ci, "lo": v.mean() - ci, "hi": v.mean() + ci})
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


def corrected_se(d: np.ndarray, test_train_ratio: float) -> float:
    """Corrected standard error of the mean of paired differences d."""
    n = len(d)
    s2 = d.var(ddof=1)
    return float(np.sqrt((1.0 / n + test_train_ratio) * s2))


def corrected_ttest(a: np.ndarray, b: np.ndarray, test_train_ratio: float) -> tuple[float, float]:
    """Corrected resampled t test of a vs b. Returns (t, two-sided p)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    se = corrected_se(d, test_train_ratio)
    if se < 1e-12:
        return (0.0, 1.0) if abs(d.mean()) < 1e-12 else (np.inf, 0.0)
    t = d.mean() / se
    return float(t), float(2 * stats.t.sf(abs(t), n - 1))


def corrected_equivalence_bound(a: np.ndarray, b: np.ndarray, test_train_ratio: float,
                                alpha: float = 0.05) -> float:
    """Equivalence bound delta_min with the corrected variance.

    Two one-sided tests (TOST) establish equivalence within +-delta exactly when the
    (1 - 2*alpha) confidence interval of the difference lies inside +-delta, so the
    narrowest margin that can be established is

        delta_min = |mean difference| + t_{1-alpha, n-1} * SE

    Reporting delta_min instead of a yes/no decision at a chosen margin lets the
    reader apply their own threshold of practical relevance.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    return float(abs(d.mean()) + stats.t.ppf(1 - alpha, n - 1) * corrected_se(d, test_train_ratio))


def correlated_bayes(a: np.ndarray, b: np.ndarray, test_train_ratio: float,
                     rope: float) -> tuple[float, float, float]:
    """Correlated Bayesian t test with a region of practical equivalence (ROPE).

    The posterior of the mean difference is a Student t with n-1 degrees of freedom,
    located at the sample mean, scaled by the corrected standard error.

    Returns (P(a worse), P(practically equivalent), P(a better)), i.e. the posterior
    mass of difference < -rope, |difference| <= rope, and difference > +rope.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    mu, se = d.mean(), corrected_se(d, test_train_ratio)
    if se < 1e-12:
        if mu < -rope:
            return 1.0, 0.0, 0.0
        if mu > rope:
            return 0.0, 0.0, 1.0
        return 0.0, 1.0, 0.0
    dist = stats.t(df=n - 1, loc=mu, scale=se)
    p_left = float(dist.cdf(-rope))
    p_right = float(dist.sf(rope))
    return p_left, float(max(0.0, 1.0 - p_left - p_right)), p_right


def sign_test(a: np.ndarray, b: np.ndarray) -> float:
    """Sign test. Unlike Wilcoxon, it does not assume a symmetric distribution of
    differences (it does still assume independence)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    if pos + neg == 0:
        return 1.0
    return float(stats.binomtest(pos, pos + neg, 0.5).pvalue)
