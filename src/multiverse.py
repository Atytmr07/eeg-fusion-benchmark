"""Multiverse analysis: how robust are the fusion comparisons to preprocessing and seeds?

The project's question (advisor, Sevgi Hoca): how robust are the results between fusion
operators to preprocessing and to stochastic training decisions? The main output is
therefore the stability of the fusion RANKING, next to mean performance. This module
reads every pipeline x seed run of one dataset and produces all analyses and figures:

  1. performance     pipeline x seed x model means of macro F1, AUROC, AUPRC,
                     sensitivity, specificity, Brier, log loss (AUPRC from the stored
                     test probabilities, per fold, then averaged)
  2. rank stability  (main analysis) ranking of the five fusion operators per run;
                     Kendall's W and pairwise tau between seeds of one pipeline; tau
                     between the seed-averaged rankings of the pipelines (heatmap);
                     within-pipeline versus between-pipeline tau on the same scale,
                     with a permutation test; the same restricted to P6a/P6b/P6c
  3. interaction     Fusion x Preprocessing: interaction plot and repeated-measures
                     ANOVA (folds as subjects, Greenhouse-Geisser corrected)
  4. variance        variance components of the fully crossed fold x pipeline x fusion
                     x seed design (method of moments)
  5. equivalence     corrected test and equivalence bound for every fusion pair in
                     every pipeline; "robust equivalence" = equivalent in all
  6. robustness      mean F1 against the SD (and CV) of the pipeline means
  7. prediction      window-level agreement of predicted classes between pipelines
                     and, as the noise reference, between seeds

Runs are found by name in results_v2/<dataset>/: loso_<P> (seed 0), loso_<P>_r<k>
(seed k), and loso_grouped for P0 seed 0 (src/run_queue.py: tag_for). With --device
cuda, only GPU runs are read (the same names with a _cuda suffix, P0 seed 0 being
loso_P0_cuda): CPU and GPU runs are never mixed in one analysis. Missing runs are
skipped: every analysis uses the largest complete (balanced) subset that exists and
says which runs it used, so the module can be run while results arrive.

Usage:
    python -m src.multiverse --dataset chbmit
    python -m src.multiverse --synthetic          # validation on synthetic runs

Outputs (CSV + PNG + summary.md) go to results_v2/multiverse/<dataset>/.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .chbmit_stats import FUSION, ratios
from .config import RESULTS_ROOT
from .stats import corrected_equivalence_bound, corrected_ttest, holm

PIPELINE_ORDER = ("P0", "P1", "P2", "P3", "P4", "P5", "P6a", "P6b", "P6c")
P6_FAMILY = ("P6a", "P6b", "P6c")
OTHER_MODELS = ("raw1d", "spec2d", "raw1d_wide", "spec2d_wide", "logvar", "shallow")
METRICS = ("f1_macro", "auc", "auprc", "sensitivity", "specificity", "brier", "log_loss")
# Equivalence margins in macro F1, fixed before any multiverse result existed (advisor's
# decision, 4 October 2026; see the git history): +-0.05 for the main analysis, +-0.02
# as a strict sensitivity analysis showing how much the conclusion depends on the margin.
EQUIV_MARGIN = 0.05
STRICT_MARGIN = 0.02
N_PERM = 10000
SEED = 20260727

RUN_NAME = {"cpu": re.compile(r"^loso_(P\d[a-z]?)(?:_r(\d+))?$"),
            "cuda": re.compile(r"^loso_(P\d[a-z]?)(?:_r(\d+))?_cuda$")}


# --- Loading ------------------------------------------------------------------------------

@dataclass
class Run:
    name: str
    pipeline: str
    seed: int
    path: Path


def parse_run_name(name: str, device: str = "cpu") -> tuple[str, int] | None:
    """(pipeline, seed) of a run folder name on the given device, or None for anything
    else (loso_main, runs of the other device, timing runs)."""
    if name == "loso_grouped":
        return ("P0", 0) if device == "cpu" else None
    m = RUN_NAME[device].match(name)
    return (m.group(1), int(m.group(2) or 0)) if m else None


def _pipeline_key(p: str) -> tuple[int, str]:
    return (PIPELINE_ORDER.index(p), p) if p in PIPELINE_ORDER else (len(PIPELINE_ORDER), p)


def discover_runs(root: Path, pipelines=None, seeds=None, device: str = "cpu") -> list[Run]:
    runs = []
    for d in sorted(root.iterdir()) if root.exists() else []:
        key = parse_run_name(d.name, device)
        if key is None or not (d / "perfold.csv").exists():
            continue
        p, s = key
        if (pipelines and p not in pipelines) or (seeds is not None and s not in seeds):
            continue
        runs.append(Run(d.name, p, s, d))
    if len({(r.pipeline, r.seed) for r in runs}) != len(runs):
        raise RuntimeError(f"two runs map to the same (pipeline, seed) in {root}")
    return sorted(runs, key=lambda r: (_pipeline_key(r.pipeline), r.seed))


def load_preds(run: Run) -> dict[int, dict[str, np.ndarray]]:
    """Stored test predictions per fold: idx_te, y_te and (N, 2) probabilities per model."""
    out = {}
    for f in sorted((run.path / "preds").glob("fold*.npz")):
        d = np.load(f)
        out[int(f.stem[4:])] = {k: d[k] for k in d.files}
    return out


def load_long(runs: list[Run]) -> pd.DataFrame:
    """One row per (pipeline, seed, model, fold) with all METRICS; AUPRC from preds."""
    from sklearn.metrics import average_precision_score

    frames = []
    for r in runs:
        df = pd.read_csv(r.path / "perfold.csv")
        preds = load_preds(r)
        ap = {}
        for fold, d in preds.items():
            for m in d:
                if m in ("idx_te", "y_te") or d["y_te"].sum() == 0:
                    continue
                ap[(m, fold)] = average_precision_score(d["y_te"], d[m][:, 1])
        df["auprc"] = [ap.get((m, f), np.nan) for m, f in zip(df.model, df.fold)]
        df["pipeline"], df["seed"], df["run"] = r.pipeline, r.seed, r.name
        frames.append(df[["pipeline", "seed", "run", "model", "fold", *METRICS]])
    return pd.concat(frames, ignore_index=True)


def balanced_subset(long: pd.DataFrame, models, pipelines=None
                    ) -> tuple[list[str], list[int], list[int]]:
    """The largest fully crossed (pipelines, seeds, folds) block for `models`.

    A seed set is chosen among the subsets of available seeds to maximise
    pipelines x seeds, with every included pipeline having all of those seeds and every
    model on the common folds.
    """
    sub = long[long.model.isin(models)]
    if pipelines is not None:
        sub = sub[sub.pipeline.isin(pipelines)]
    cells = sub.groupby(["pipeline", "seed"])
    folds_of = {k: set(g.fold[g.model.isin(models)].unique()) for k, g in cells
                if set(models) <= set(g.model)}
    all_seeds = sorted({s for _, s in folds_of})
    best = ([], [], [])
    for k in range(len(all_seeds), 0, -1):
        for seeds in itertools.combinations(all_seeds, k):
            pipes = sorted({p for p, _ in folds_of
                            if all((p, s) in folds_of for s in seeds)}, key=_pipeline_key)
            if not pipes:
                continue
            folds = set.intersection(*[folds_of[(p, s)] for p in pipes for s in seeds])
            if len(pipes) * len(seeds) > len(best[0]) * len(best[1]) and folds:
                best = (pipes, list(seeds), sorted(folds))
    return best


def to_array(long: pd.DataFrame, metric: str, models, pipes, seeds, folds) -> np.ndarray:
    """Array (fold, pipeline, model, seed) of one metric."""
    sub = long[long.model.isin(models) & long.pipeline.isin(pipes) & long.seed.isin(seeds)
               & long.fold.isin(folds)]
    piv = sub.set_index(["fold", "pipeline", "model", "seed"])[metric]
    idx = pd.MultiIndex.from_product([folds, pipes, list(models), seeds])
    a = piv.reindex(idx).to_numpy().reshape(len(folds), len(pipes), len(models), len(seeds))
    if np.isnan(a).any():
        raise RuntimeError(f"{metric}: missing values in the balanced block")
    return a


def md_table(df: pd.DataFrame, index: bool = False, digits: int = 4) -> str:
    """A DataFrame as a Markdown table (pandas' to_markdown needs tabulate, which is not
    a dependency of this project)."""
    d = df.reset_index() if index else df
    def fmt(v) -> str:
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, sep, *body])


# --- Balanced ANOVA machinery -----------------------------------------------------------------

def effect_table(y: np.ndarray, names: tuple[str, ...]) -> dict[tuple[str, ...], dict]:
    """Sums of squares, df and mean squares of every main effect and interaction of a
    fully crossed design with one observation per cell (balanced, so the decomposition
    is orthogonal). Effects are estimated by inclusion-exclusion over marginal means."""
    k = y.ndim
    n = y.shape
    means = {}
    for r in range(k + 1):
        for sub in itertools.combinations(range(k), r):
            other = tuple(i for i in range(k) if i not in sub)
            means[sub] = y.mean(axis=other, keepdims=True) if other else y
    out = {}
    for r in range(1, k + 1):
        for sub in itertools.combinations(range(k), r):
            eff = 0.0
            for rr in range(r + 1):
                for z in itertools.combinations(sub, rr):
                    eff = eff + (-1) ** (r - rr) * means[z]
            ss = float((np.broadcast_to(eff, y.shape) ** 2).sum())
            df = int(np.prod([n[i] - 1 for i in sub]))
            out[tuple(names[i] for i in sub)] = {"ss": ss, "df": df,
                                                 "ms": ss / df if df else np.nan}
    return out


def _contrasts(k: int) -> np.ndarray:
    """Orthonormal contrasts (k, k-1): orthogonal to the constant vector."""
    q, _ = np.linalg.qr(np.column_stack([np.ones(k), np.eye(k)[:, :k - 1]]))
    return q[:, 1:k]


def gg_epsilon(cells: np.ndarray, contrast: np.ndarray) -> float:
    """Greenhouse-Geisser epsilon from subject x cell data and an orthonormal contrast."""
    z = cells @ contrast
    s = np.atleast_2d(np.cov(z, rowvar=False))
    df1 = contrast.shape[1]
    eps = np.trace(s) ** 2 / (df1 * np.trace(s @ s))
    return float(np.clip(eps, 1.0 / df1, 1.0))


def rm_anova(y: np.ndarray) -> pd.DataFrame:
    """Repeated-measures ANOVA of y (fold, pipeline, model), seeds already averaged.

    Folds (held-out persons) are the subjects; pipeline and model are within-subject
    factors. Each effect is tested against its interaction with fold, the correct error
    term when folds are random and both factors fixed. Seed noise enters numerator and
    denominator alike, so averaging seeds first keeps the F tests valid. Greenhouse-
    Geisser epsilon corrects for non-sphericity.
    """
    nf, npip, nm = y.shape
    t = effect_table(y, ("fold", "pipeline", "model"))
    cp, cm = _contrasts(npip), _contrasts(nm)
    designs = {
        ("pipeline",): (y.mean(axis=2), cp),
        ("model",): (y.mean(axis=1), cm),
        ("pipeline", "model"): (y.reshape(nf, npip * nm), np.kron(cp, cm)),
    }
    rows = []
    for eff, (cells, con) in designs.items():
        err = ("fold", *eff)
        e, r = t[eff], t[err]
        f = e["ms"] / r["ms"]
        eps = gg_epsilon(cells, con)
        rows.append({
            "effect": " x ".join(eff), "ss": e["ss"], "df": e["df"], "ms": e["ms"],
            "error": " x ".join(err), "ss_error": r["ss"], "df_error": r["df"],
            "ms_error": r["ms"], "F": f, "p": float(sps.f.sf(f, e["df"], r["df"])),
            "gg_epsilon": eps,
            "p_gg": float(sps.f.sf(f, eps * e["df"], eps * r["df"])),
            "partial_eta2": e["ss"] / (e["ss"] + r["ss"]),
        })
    return pd.DataFrame(rows)


def variance_components(y: np.ndarray, names: tuple[str, ...]) -> pd.DataFrame:
    """Variance components of a fully crossed design by the method of moments.

    All factors are treated as random facets (generalizability theory), so a fixed
    factor's component is the variance of its effects. With one observation per cell
    the highest interaction is the residual. For an effect X,
        sigma2_X = sum over effects Y containing X of (-1)^|Y - X| MS_Y / prod_{f not in X} n_f
    Negative estimates are set to zero for the shares (and reported).
    """
    t = effect_table(y, names)
    n = dict(zip(names, y.shape))
    rows = []
    for x in t:
        acc = 0.0
        for yk, v in t.items():
            if set(x) <= set(yk):
                acc += (-1) ** (len(yk) - len(x)) * v["ms"]
        k = np.prod([n[f] for f in names if f not in x])
        rows.append({"component": " x ".join(x), "sigma2_raw": acc / k})
    df = pd.DataFrame(rows)
    full = " x ".join(names)
    df.loc[df.component == full, "component"] = "residual (" + full + ")"
    df["sigma2"] = df.sigma2_raw.clip(lower=0.0)
    df["share"] = df.sigma2 / df.sigma2.sum()
    return df


def group_components(vc: pd.DataFrame) -> pd.DataFrame:
    """The components the advisor asked for, with the rest grouped."""
    def label(c: str) -> str:
        if c.startswith("residual"):
            return "residual"
        parts = set(c.split(" x "))
        if parts == {"fold"}:
            return "fold (subject)"
        if parts == {"pipeline"}:
            return "pipeline"
        if parts == {"model"}:
            return "fusion"
        if parts == {"pipeline", "model"}:
            return "fusion x pipeline"
        if "seed" in parts:
            return "seed (all seed terms)"
        return "fold x (pipeline, fusion)"
    g = vc.assign(group=vc.component.map(label)).groupby("group", sort=False)[
        ["sigma2", "share"]].sum()
    order = ["fold (subject)", "pipeline", "fusion", "fusion x pipeline",
             "seed (all seed terms)", "fold x (pipeline, fusion)", "residual"]
    return g.reindex([o for o in order if o in g.index]).reset_index()


# --- Rank stability -----------------------------------------------------------------------------

def ranks_of(means: np.ndarray) -> np.ndarray:
    """Ranks, 1 = best (highest mean); ties get the average rank."""
    return sps.rankdata(-np.asarray(means), method="average")


def kendall_w(rank_matrix: np.ndarray) -> float:
    """Kendall's W for m raters (rows) ranking n items (columns), tie-corrected."""
    m, n = rank_matrix.shape
    if m < 2:
        return np.nan
    s = ((rank_matrix.sum(axis=0) - m * (n + 1) / 2) ** 2).sum()
    ties = 0.0
    for row in rank_matrix:
        _, c = np.unique(row, return_counts=True)
        ties += (c ** 3 - c).sum()
    return float(12 * s / (m ** 2 * (n ** 3 - n) - m * ties))


def tau(a: np.ndarray, b: np.ndarray) -> float:
    return float(sps.kendalltau(a, b).statistic)


def rank_stability(run_means: pd.DataFrame, models, n_perm: int = N_PERM,
                   seed: int = SEED, family: tuple[str, ...] | None = None) -> dict:
    """Within-seed and between-pipeline stability of the fusion ranking.

    run_means: one row per (pipeline, seed) with a column per model (mean F1 over the
    common folds). Within: pairs of runs of the same pipeline (different seeds).
    Between: pairs of runs of different pipelines. Both use single-run rankings, so the
    two mean taus are on the same scale. D = mean within tau - mean between tau.

    Permutation test of H0 "the pipeline does not change the ranking beyond seed noise":
    pipeline labels are permuted among the runs of each seed (which keeps the seed
    structure and any pairing by seed, and breaks only the pipeline structure), and D is
    recomputed. Shuffling seed labels within a pipeline would leave every run in its
    pipeline and so could not produce a null for the between-pipeline tau.
    """
    rm = run_means
    if family is not None:
        rm = rm[rm.pipeline.isin(family)]
    rows = rm.reset_index(drop=True)
    r = np.array([ranks_of(rows.loc[i, list(models)].to_numpy(float)) for i in range(len(rows))])
    pip = rows.pipeline.to_numpy()
    sd = rows.seed.to_numpy()
    n = len(rows)
    tmat = np.full((n, n), np.nan)
    for i, j in itertools.combinations(range(n), 2):
        tmat[i, j] = tmat[j, i] = tau(r[i], r[j])
    iu = np.triu_indices(n, 1)

    def d_stat(p: np.ndarray) -> tuple[float, float, float]:
        same = (p[:, None] == p[None, :])[iu]
        tv = tmat[iu]
        w = tv[same].mean() if same.any() else np.nan
        b = tv[~same].mean() if (~same).any() else np.nan
        return w, b, w - b

    w_obs, b_obs, d_obs = d_stat(pip)
    per_pipe = []
    for p in dict.fromkeys(pip):
        m = pip == p
        pairs = [tmat[i, j] for i, j in itertools.combinations(np.flatnonzero(m), 2)]
        per_pipe.append({"pipeline": p, "n_seeds": int(m.sum()),
                         "kendall_w": kendall_w(r[m]),
                         "mean_pairwise_tau": float(np.mean(pairs)) if pairs else np.nan})

    p_value, null = np.nan, np.array([])
    if np.isfinite(d_obs):
        rng = np.random.default_rng(seed)
        null = np.empty(n_perm)
        strata = [np.flatnonzero(sd == s) for s in np.unique(sd)]
        for k in range(n_perm):
            perm = pip.copy()
            for idx in strata:
                perm[idx] = pip[rng.permutation(idx)]
            null[k] = d_stat(perm)[2]
        p_value = float((1 + (null >= d_obs - 1e-12).sum()) / (1 + n_perm))

    return {"ranks": pd.concat([rows[["pipeline", "seed"]],
                                pd.DataFrame(r, columns=list(models))], axis=1),
            "within": per_pipe, "tau_within": w_obs, "tau_between": b_obs,
            "D": d_obs, "p_perm": p_value, "null": null, "n_runs": n}


def pipeline_tau_matrix(run_means: pd.DataFrame, models) -> pd.DataFrame:
    """Kendall tau between the seed-averaged fusion rankings of every pair of pipelines."""
    avg = run_means.groupby("pipeline", sort=False)[list(models)].mean()
    pipes = list(avg.index)
    rk = {p: ranks_of(avg.loc[p].to_numpy(float)) for p in pipes}
    return pd.DataFrame([[tau(rk[a], rk[b]) for b in pipes] for a in pipes],
                        index=pipes, columns=pipes)


# --- Equivalence -----------------------------------------------------------------------------------

def equivalence(y: np.ndarray, pipes, models, margin: float) -> pd.DataFrame:
    """Corrected test and equivalence bound for every fusion pair in every pipeline.

    y: (fold, pipeline, model) with seeds averaged. Test/train ratio 1/(k-1) for LOSO
    with k folds (src/chbmit_stats.py: ratios). Holm across the pairs of one pipeline.
    """
    nf = y.shape[0]
    ratio, _ = ratios(nf)
    rows = []
    for pi, p in enumerate(pipes):
        block = []
        for (ia, a), (ib, b) in itertools.combinations(enumerate(models), 2):
            xa, xb = y[:, pi, ia], y[:, pi, ib]
            _, pv = corrected_ttest(xa, xb, ratio)
            dmin = corrected_equivalence_bound(xa, xb, ratio)
            block.append({"pipeline": p, "pair": f"{a}-{b}", "model_a": a, "model_b": b,
                          "mean_diff": float((xa - xb).mean()), "p_corrected": pv,
                          "delta_min": dmin, "equivalent": dmin <= margin})
        ph = holm(np.array([r["p_corrected"] for r in block]))
        for r, h in zip(block, ph):
            r["p_holm"] = float(h)
            r["significant"] = bool(h < 0.05)
        rows += block
    return pd.DataFrame(rows)


def robust_equivalence(eq: pd.DataFrame) -> pd.DataFrame:
    g = eq.groupby("pair", sort=False)
    return pd.DataFrame({
        "n_pipelines": g.size(),
        "equivalent_in": g.equivalent.sum(),
        "robust_equivalent": g.equivalent.all(),
        "never_significant": ~g.significant.any(),
        "max_delta_min": g.delta_min.max(),
    }).reset_index()


# --- Prediction stability ------------------------------------------------------------------------

def _aligned_classes(preds: dict, runs: list[tuple[str, int]], model: str
                     ) -> tuple[np.ndarray, list[int]] | None:
    """Predicted classes (run, window) on the windows common to all runs, folds pooled."""
    folds = set.intersection(*[set(preds[k]) for k in runs])
    cols = []
    for fold in sorted(folds):
        ds = [preds[k][fold] for k in runs]
        if any(model not in d for d in ds):
            return None
        common = ds[0]["idx_te"]
        for d in ds[1:]:
            common = np.intersect1d(common, d["idx_te"])
        block = []
        for d in ds:
            pos = {v: i for i, v in enumerate(d["idx_te"])}
            sel = np.array([pos[v] for v in common])
            block.append((d[model][sel, 1] > 0.5).astype(np.int8))
        cols.append(np.vstack(block))
    if not cols:
        return None
    return np.hstack(cols), sorted(folds)


def prediction_stability(preds: dict, pipes, seeds, models) -> tuple[pd.DataFrame, dict]:
    """Agreement of window-level predicted classes between pipelines and between seeds.

    Between pipelines: pairs of runs with the same seed and different pipelines.
    Between seeds: pairs with the same pipeline and different seeds (the noise
    reference). Pairwise agreement and Cohen's kappa are on the same scale for both;
    the share of windows on which ALL runs agree is reported too, but depends on the
    number of runs compared.
    """
    from sklearn.metrics import cohen_kappa_score

    rows, kappa_mats = [], {}
    for m in models:
        for kind in ("pipelines", "seeds"):
            groups = ([[(p, s) for p in pipes] for s in seeds] if kind == "pipelines"
                      else [[(p, s) for s in seeds] for p in pipes])
            for grp in groups:
                grp = [k for k in grp if k in preds]
                if len(grp) < 2:
                    continue
                al = _aligned_classes(preds, grp, m)
                if al is None:
                    continue
                cls, _ = al
                same_all = float((cls == cls[0]).all(axis=0).mean())
                for (i, a), (j, b) in itertools.combinations(enumerate(grp), 2):
                    with np.errstate(all="ignore"):
                        k = cohen_kappa_score(cls[i], cls[j])
                    rows.append({"model": m, "between": kind, "run_a": f"{a[0]}/s{a[1]}",
                                 "run_b": f"{b[0]}/s{b[1]}", "n_windows": cls.shape[1],
                                 "agreement": float((cls[i] == cls[j]).mean()),
                                 "kappa": float(k), "all_agree_in_group": same_all})
                if kind == "pipelines":
                    km = np.eye(len(grp))
                    for (i, _), (j, _) in itertools.combinations(enumerate(grp), 2):
                        with np.errstate(all="ignore"):
                            km[i, j] = km[j, i] = cohen_kappa_score(cls[i], cls[j])
                    kappa_mats.setdefault(m, []).append(
                        pd.DataFrame(km, index=[g[0] for g in grp], columns=[g[0] for g in grp]))
    return pd.DataFrame(rows), kappa_mats


# --- Figures --------------------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


COLORS = {"early": "#8172b2", "late": "#4c72b0", "gated": "#55a868",
          "attention": "#c44e52", "score": "#dd8452"}


def fig_interaction(y: np.ndarray, pipes, models, out: Path, title: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(1.1 * len(pipes) + 3, 4.5))
    x = np.arange(len(pipes))
    for mi, m in enumerate(models):
        v = y[:, :, mi]                                      # (fold, pipeline)
        mean = v.mean(axis=0)
        se = v.std(axis=0, ddof=1) / np.sqrt(v.shape[0])
        ax.errorbar(x + (mi - 2) * 0.06, mean, yerr=se, marker="o", ms=4, capsize=2,
                    lw=1.5, label=m, color=COLORS.get(m))
    ax.set_xticks(x)
    ax.set_xticklabels(pipes)
    ax.set_xlabel("preprocessing pipeline")
    ax.set_ylabel("macro F1 (seed mean; bars: SE over folds)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.08))
    ax.grid(alpha=0.3)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_tau_heatmap(tm: pd.DataFrame, out: Path, title: str) -> None:
    plt = _plt()
    k = len(tm)
    fig, ax = plt.subplots(figsize=(0.6 * k + 2.5, 0.55 * k + 1.8))
    im = ax.imshow(tm.to_numpy(), cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(range(k))
    ax.set_xticklabels(tm.columns)
    ax.set_yticks(range(k))
    ax.set_yticklabels(tm.index)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{tm.iat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(tm.iat[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Kendall tau")
    ax.set_title(title, fontsize=10)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_rank_null(rs: dict, out: Path, title: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(rs["null"], bins=40, color="0.7", label="permutation null of D")
    ax.axvline(rs["D"], color="#c44e52", lw=2,
               label=f"observed D = {rs['D']:.3f} (p = {rs['p_perm']:.4f})")
    ax.set_xlabel("D = mean within-pipeline tau - mean between-pipeline tau")
    ax.set_ylabel("permutations")
    ax.set_title(title + f"\nwithin tau {rs['tau_within']:.3f}, between tau "
                 f"{rs['tau_between']:.3f}", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_variance(groups: pd.DataFrame, out: Path, title: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh(groups.group[::-1], 100 * groups.share[::-1], color="#4c72b0")
    for i, v in enumerate(100 * groups.share[::-1]):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("share of variance (%)")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, max(100 * groups.share.max() * 1.18, 5))
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_robustness(rp: pd.DataFrame, out: Path, title: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for _, r in rp.iterrows():
        ax.scatter(r.mean_f1, r.sd_across_pipelines, s=60, color=COLORS.get(r.model, "0.4"))
        ax.annotate(r.model, (r.mean_f1, r.sd_across_pipelines), xytext=(5, 4),
                    textcoords="offset points", fontsize=9)
    ax.set_xlabel("mean macro F1 over pipelines and seeds")
    ax.set_ylabel("SD of pipeline means (preprocessing sensitivity)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.margins(0.25)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_prediction(ps: pd.DataFrame, out: Path, title: str) -> None:
    plt = _plt()
    g = ps.groupby(["model", "between"]).kappa.mean().unstack()
    g = g.reindex([m for m in FUSION if m in g.index] +
                  [m for m in g.index if m not in FUSION])
    fig, ax = plt.subplots(figsize=(1.0 * len(g) + 2.5, 3.8))
    x = np.arange(len(g))
    for k, (col, c) in enumerate((("seeds", "0.6"), ("pipelines", "#c44e52"))):
        if col in g:
            ax.bar(x + (k - 0.5) * 0.38, g[col], width=0.38, color=c,
                   label=f"between {col}" + (" (noise reference)" if col == "seeds" else ""))
    ax.set_xticks(x)
    ax.set_xticklabels(g.index, rotation=30, ha="right")
    ax.set_ylabel("mean pairwise Cohen's kappa")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_equivalence(eq: pd.DataFrame, margin: float, out: Path, title: str) -> None:
    plt = _plt()
    piv = eq.pivot(index="pair", columns="pipeline", values="delta_min")
    piv = piv[[p for p in eq.pipeline.unique()]].reindex(eq.pair.unique())
    fig, ax = plt.subplots(figsize=(0.75 * piv.shape[1] + 3, 0.4 * piv.shape[0] + 1.6))
    im = ax.imshow(piv.to_numpy(), cmap="viridis_r", vmin=0,
                   vmax=max(2 * margin, float(np.nanmax(piv.to_numpy()))))
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            ax.text(j, i, f"{v:.3f}" + ("*" if v <= margin else ""), ha="center",
                    va="center", fontsize=7, color="white" if v > 1.2 * margin else "black")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    fig.colorbar(im, ax=ax, fraction=0.046, label="equivalence bound delta_min (F1)")
    ax.set_title(title + f"\n* equivalent within {margin}", fontsize=10)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --- Driver ---------------------------------------------------------------------------------------

def analyze(root: Path, out: Path, label: str, pipelines=None, seeds=None,
            margin: float = EQUIV_MARGIN, n_perm: int = N_PERM, verbose: bool = True,
            device: str = "cpu", strict_margin: float | None = STRICT_MARGIN) -> dict:
    """Run every analysis on the runs under `root`; write CSV, PNG and summary.md to
    `out`. Returns the key numbers (used by the synthetic validation)."""
    def say(msg=""):
        if verbose:
            print(msg)

    runs = discover_runs(root, pipelines, seeds, device)
    if not runs:
        raise SystemExit(f"no pipeline runs under {root}")
    out.mkdir(parents=True, exist_ok=True)
    long = load_long(runs)
    models = [m for m in FUSION if m in set(long.model)]
    res: dict = {"runs": [r.name for r in runs]}
    lines = [f"# Multiverse analysis: {label}", "",
             f"Runs found ({len(runs)}): " + ", ".join(f"{r.name} ({r.pipeline}, seed {r.seed})"
                                                    for r in runs), ""]
    say(f"{len(runs)} runs: " + ", ".join(r.name for r in runs))

    # 1. performance
    perf = (long.groupby(["pipeline", "seed", "model"], sort=False)[list(METRICS)]
            .mean().reset_index())
    perf["n_folds"] = long.groupby(["pipeline", "seed", "model"], sort=False).fold.nunique().values
    perf.to_csv(out / "performance.csv", index=False)
    by_pipe = (long.groupby(["pipeline", "model", "seed"], sort=False)[list(METRICS)].mean()
               .groupby(["pipeline", "model"], sort=False).agg(["mean", "std"]))
    by_pipe.columns = [f"{a}_{b}" for a, b in by_pipe.columns]
    by_pipe.reset_index().to_csv(out / "performance_by_pipeline.csv", index=False)
    lines += ["## 1. Performance", "",
              "Means over folds per pipeline, seed and model: `performance.csv`; seed mean "
              "and SD per pipeline and model: `performance_by_pipeline.csv`. AUPRC is "
              "computed from the stored test probabilities (logvar and shallow store none).",
              ""]
    say("1. performance.csv, performance_by_pipeline.csv")

    pipes, seeds_used, folds = balanced_subset(long, models)
    res.update(pipelines=pipes, seeds=seeds_used, n_folds=len(folds))
    lines += [f"Balanced block for the analyses below: pipelines {', '.join(pipes)}; "
              f"seeds {', '.join(map(str, seeds_used))}; {len(folds)} folds; fusion models "
              f"{', '.join(models)}.", ""]
    say(f"balanced block: pipelines {pipes}, seeds {seeds_used}, {len(folds)} folds")
    a4 = to_array(long, "f1_macro", models, pipes, seeds_used, folds)   # fold,pipe,model,seed
    y3 = a4.mean(axis=3)

    # 2. rank stability
    run_means = pd.DataFrame(
        [{"pipeline": p, "seed": s, **{m: a4[:, pi, mi, si].mean() for mi, m in enumerate(models)}}
         for pi, p in enumerate(pipes) for si, s in enumerate(seeds_used)])
    rs = rank_stability(run_means, models, n_perm=n_perm)
    rs["ranks"].to_csv(out / "rank_per_run.csv", index=False)
    pd.DataFrame(rs["within"]).to_csv(out / "rank_within_pipeline.csv", index=False)
    tm = pipeline_tau_matrix(run_means, models)
    tm.to_csv(out / "rank_tau_between_pipelines.csv")
    res.update(tau_within=rs["tau_within"], tau_between=rs["tau_between"], D=rs["D"],
               p_perm=rs["p_perm"])
    lines += ["## 2. Rank stability (main analysis)", "",
              "Ranking of the fusion operators by mean macro F1 per run: `rank_per_run.csv`.",
              "",
              "| | value |", "|---|---|",
              f"| mean Kendall tau, same pipeline, different seeds | {rs['tau_within']:.3f} |",
              f"| mean Kendall tau, different pipelines | {rs['tau_between']:.3f} |",
              f"| D = within - between | {rs['D']:.3f} |",
              f"| permutation p (pipeline labels permuted within seeds, {n_perm} permutations) "
              f"| {rs['p_perm']:.4f} |", ""]
    if len(tm) > 1:
        fig_tau_heatmap(tm, out / "rank_stability_heatmap.png",
                        f"{label}: Kendall tau between seed-averaged fusion rankings")
    if np.isfinite(rs["D"]):
        fig_rank_null(rs, out / "rank_stability_permutation.png", f"{label}: rank stability")
    for w in rs["within"]:
        lines.append(f"- {w['pipeline']}: Kendall's W over {w['n_seeds']} seeds = "
                     f"{w['kendall_w']:.3f}, mean pairwise tau = {w['mean_pairwise_tau']:.3f}")
    lines.append("")
    fam = [p for p in P6_FAMILY if p in pipes]
    if len(fam) >= 2:
        rs6 = rank_stability(run_means, models, n_perm=n_perm, family=tuple(fam))
        tm6 = tm.loc[fam, fam]
        res.update(p6_D=rs6["D"], p6_p_perm=rs6["p_perm"])
        lines += [f"ICA algorithm ({', '.join(fam)}): within tau {rs6['tau_within']:.3f}, "
                  f"between tau {rs6['tau_between']:.3f}, D {rs6['D']:.3f}, permutation p "
                  f"{rs6['p_perm']:.4f}; seed-averaged tau matrix:", "",
                  md_table(tm6, index=True, digits=3), ""]
    say(f"2. rank stability: within tau {rs['tau_within']:.3f}, between tau "
        f"{rs['tau_between']:.3f}, D {rs['D']:.3f}, p {rs['p_perm']:.4f}")

    # 3. interaction
    fig_interaction(y3, pipes, models, out / "interaction.png",
                    f"{label}: Fusion x Preprocessing")
    lines += ["## 3. Fusion x Preprocessing", ""]
    if len(pipes) >= 2:
        aov = rm_anova(y3)
        aov.to_csv(out / "anova.csv", index=False)
        inter = aov[aov.effect == "pipeline x model"].iloc[0]
        res.update(interaction_F=float(inter.F), interaction_p=float(inter.p_gg))
        lines += ["Repeated-measures ANOVA on macro F1 (seed means; folds as subjects; each "
                  "effect tested against its interaction with fold; Greenhouse-Geisser):", "",
                  md_table(aov[["effect", "df", "df_error", "F", "p", "gg_epsilon", "p_gg",
                                "partial_eta2"]]), ""]
        say(f"3. interaction: F = {inter.F:.2f}, p_GG = {inter.p_gg:.4g}")
    else:
        lines += ["Only one pipeline: no ANOVA.", ""]

    # 4. variance components
    lines += ["## 4. Variance components", ""]
    if len(pipes) >= 2:
        names = ("fold", "pipeline", "model") + (("seed",) if len(seeds_used) > 1 else ())
        yv = a4 if len(seeds_used) > 1 else y3
        vc = variance_components(yv, names)
        vc.to_csv(out / "variance_components.csv", index=False)
        gv = group_components(vc)
        gv.to_csv(out / "variance_components_grouped.csv", index=False)
        fig_variance(gv, out / "variance_components.png", f"{label}: variance of macro F1")
        res["variance"] = dict(zip(gv.group, gv.share))
        res["variance_raw"] = dict(zip(vc.component, vc.sigma2_raw))
        lines += [md_table(gv.assign(share_pct=100 * gv.share).drop(columns="share"), digits=6), "",
                  "Negative raw estimates (set to 0): " +
                  (", ".join(vc.component[vc.sigma2_raw < 0]) or "none"), ""]
        say("4. variance: " + ", ".join(f"{g} {100 * s:.1f}%" for g, s in zip(gv.group, gv.share)))
    else:
        lines += ["Only one pipeline: no decomposition.", ""]

    # 5. equivalence, at the main margin and at the strict one
    eq = equivalence(y3, pipes, models, margin)
    eq.to_csv(out / "equivalence.csv", index=False)
    req = robust_equivalence(eq)
    req.to_csv(out / "equivalence_robust.csv", index=False)
    fig_equivalence(eq, margin, out / "equivalence.png",
                    f"{label}: fusion pairs, equivalence bound per pipeline")
    res["robust_equivalent"] = req.loc[req.robust_equivalent, "pair"].tolist()
    lines += ["## 5. Equivalence and robust equivalence", "",
              f"Equivalent = corrected equivalence bound delta_min <= {margin} macro F1 "
              f"(test/train ratio 1/{len(folds) - 1}). Robust equivalence = equivalent in "
              f"every pipeline.", "", md_table(req), ""]
    say(f"5. robust equivalent pairs (margin {margin}): {res['robust_equivalent'] or 'none'}")
    if strict_margin is not None and strict_margin != margin:
        eqs = equivalence(y3, pipes, models, strict_margin)
        eqs.to_csv(out / "equivalence_strict.csv", index=False)
        reqs = robust_equivalence(eqs)
        reqs.to_csv(out / "equivalence_robust_strict.csv", index=False)
        res["robust_equivalent_strict"] = reqs.loc[reqs.robust_equivalent, "pair"].tolist()
        lines += [f"Strict sensitivity analysis, margin {strict_margin}:", "", md_table(reqs), ""]
        say(f"   strict (margin {strict_margin}): "
            f"{res['robust_equivalent_strict'] or 'none'}")

    # 6. robustness-performance
    pm = y3.mean(axis=0)                                     # (pipeline, model)
    rp = pd.DataFrame({"model": models, "mean_f1": pm.mean(axis=0),
                       "sd_across_pipelines": pm.std(axis=0, ddof=1) if len(pipes) > 1
                       else np.nan})
    rp["cv_across_pipelines"] = rp.sd_across_pipelines / rp.mean_f1
    rp["robustness"] = 1 - rp.sd_across_pipelines
    rp.to_csv(out / "robustness_performance.csv", index=False)
    if len(pipes) > 1:
        fig_robustness(rp, out / "robustness_performance.png",
                       f"{label}: performance vs preprocessing sensitivity")
    lines += ["## 6. Robustness and performance", "", md_table(rp), ""]
    say("6. robustness_performance.csv")

    # 7. prediction stability
    preds = {}
    for r in runs:
        if r.pipeline in pipes and r.seed in seeds_used:
            preds[(r.pipeline, r.seed)] = load_preds(r)
    pmodels = models + [m for m in OTHER_MODELS if m not in ("logvar", "shallow")]
    ps, kmats = prediction_stability(preds, pipes, seeds_used, pmodels)
    lines += ["## 7. Prediction stability", ""]
    if len(ps):
        ps.to_csv(out / "prediction_stability_pairs.csv", index=False)
        summ = (ps.groupby(["model", "between"]).agg(
            agreement=("agreement", "mean"), kappa=("kappa", "mean"),
            all_agree=("all_agree_in_group", "mean"), pairs=("kappa", "size"))
            .reset_index())
        summ.to_csv(out / "prediction_stability.csv", index=False)
        fig_prediction(ps, out / "prediction_stability.png",
                       f"{label}: window-level agreement of predicted classes")
        for m, mats in kmats.items():
            if m in models:
                avg = sum(mats) / len(mats)
                avg.to_csv(out / f"prediction_kappa_pipelines_{m}.csv")
        fk = summ[summ.model.isin(models)].groupby("between").kappa.mean()
        res.update(kappa_pipelines=float(fk.get("pipelines", np.nan)),
                   kappa_seeds=float(fk.get("seeds", np.nan)))
        lines += ["Pairwise agreement and Cohen's kappa of predicted classes (p > 0.5), "
                  "between pipelines (same seed) and between seeds (same pipeline):", "",
                  md_table(summ), ""]
        say(f"7. prediction stability: kappa between pipelines "
            f"{res['kappa_pipelines']:.3f}, between seeds {res['kappa_seeds']:.3f}")
    else:
        lines += ["Fewer than two runs with predictions: no comparison.", ""]

    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(
        {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}, indent=2,
        default=float), encoding="utf-8")
    say(f"\nwritten: {out}")
    return res


# --- Synthetic validation -------------------------------------------------------------------------

def make_synthetic(root: Path, effect: bool, seed: int = SEED, n_folds: int = 23,
                   pipelines=PIPELINE_ORDER, n_seeds: int = 3) -> dict:
    """Synthetic runs in the layout of results_v2/chbmit/loso_grouped, with known effects.

    Per-fold macro F1 of fusion model m, pipeline p, fold f, seed s:
        F1 = 0.68 + fold_f + pipe_p + model_m + inter_pm + (fold x model)_fm + seed noise
    with fold SD 0.12 (CHB-MIT's between-subject spread), pipeline SD 0.02, fixed model
    effects, fold x model SD 0.015 and seed noise SD 0.02 per (fold, pipeline, model,
    seed). With effect=True an interaction reverses the ranking of early/late/score in
    P2-P4 and changes it differently in each of P6a, P6b and P6c (an ICA-algorithm
    effect); late and attention get identical effects everywhere
    (a truly equivalent pair). Predictions: window logits share a per-window difficulty;
    each pipeline adds its own window-level perturbation (SD 1.0 with effect=True, 0
    without) and each run an independent seed perturbation (SD 0.4). The perfold metrics
    and the predictions are generated independently; each serves its own analyses.

    Returns the true values the analyses should recover.
    """
    rng = np.random.default_rng(seed + (1 if effect else 0))
    models = list(FUSION) + list(OTHER_MODELS)
    pips = list(pipelines)
    npip, nfus = len(pips), len(FUSION)
    fold_eff = rng.normal(0, 0.12, n_folds)
    pipe_eff = rng.normal(0, 0.02, npip)
    model_eff = dict(zip(FUSION, (-0.030, 0.010, 0.020, 0.010, -0.010)))
    inter = np.zeros((npip, nfus))
    if effect:
        for p in ("P2", "P3", "P4"):
            if p in pips:
                inter[pips.index(p)] = (0.050, -0.030, 0.0, -0.030, 0.040)
        # the three ICA algorithms change the ranking in different ways
        for p, row in (("P6a", (0.0, 0.0, -0.045, 0.0, 0.0)),
                       ("P6b", (0.0, -0.035, 0.0, -0.035, 0.045)),
                       ("P6c", (0.050, 0.0, 0.0, 0.0, -0.030))):
            if p in pips:
                inter[pips.index(p)] = row
    inter[:, FUSION.index("attention")] = inter[:, FUSION.index("late")]
    inter -= inter.mean(axis=0, keepdims=True)          # pure interaction: zero margins
    inter -= inter.mean(axis=1, keepdims=True)
    fm = rng.normal(0, 0.015, (n_folds, nfus))
    fm[:, FUSION.index("attention")] = fm[:, FUSION.index("late")]
    seed_sd = 0.02

    n_win = rng.integers(150, 500, n_folds)
    difficulty = [rng.normal(0, 1, n) for n in n_win]
    labels = [(rng.random(n) < 0.2).astype(np.int64) for n in n_win]
    pipe_pert = {p: [rng.normal(0, 1.0 if effect else 0.0, n) for n in n_win] for p in pips}
    seed_noise = {}

    for pi, p in enumerate(pips):
        for s in range(n_seeds):
            name = "loso_grouped" if (p == "P0" and s == 0) else (
                f"loso_{p}" + (f"_r{s}" if s else ""))
            d = root / name
            (d / "preds").mkdir(parents=True, exist_ok=True)
            rows = []
            for f in range(n_folds):
                for m in models:
                    if m in FUSION:
                        mi = FUSION.index(m)
                        e = rng.normal(0, seed_sd)
                        seed_noise[(f, pi, mi, s)] = e
                        f1 = (0.68 + fold_eff[f] + pipe_eff[pi] + model_eff[m] + inter[pi, mi]
                              + fm[f, mi] + e)
                    else:
                        f1 = 0.64 + fold_eff[f] + rng.normal(0, 0.03)
                    rows.append({"f1_macro": f1, "auc": min(0.99, f1 + 0.15),
                                 "brier": 0.3 - 0.2 * f1, "log_loss": 1.2 - f1,
                                 "sensitivity": f1 - 0.1, "specificity": min(1, f1 + 0.2),
                                 "model": m, "fold": f})
            pd.DataFrame(rows).to_csv(d / "perfold.csv", index=False)
            for f in range(n_folds):
                arrs = {"idx_te": np.arange(n_win[f]) + 1000 * f, "y_te": labels[f]}
                for m in models:
                    if m in ("logvar", "shallow"):
                        continue
                    z = (2.0 * labels[f] - 1.0 + 0.8 * difficulty[f] + pipe_pert[p][f]
                         + rng.normal(0, 0.4, n_win[f]))
                    p1 = 1 / (1 + np.exp(-z))
                    arrs[m] = np.column_stack([1 - p1, p1]).astype(np.float32)
                np.savez(d / "preds" / f"fold{f}.npz", **arrs)
            (d / "meta.json").write_text(json.dumps({"synthetic": True, "effect": effect}))

    # true values on the scale of the method-of-moments estimators: sample variances of
    # the realised effects (ddof = 1 for main effects; interaction df for the rest)
    noise = np.array(list(seed_noise.values()))
    true = {
        "fold": float(np.var(fold_eff, ddof=1)),
        "pipeline": float(np.var(pipe_eff, ddof=1)),
        "model": float(np.var(list(model_eff.values()), ddof=1)),
        "pipeline x model": float((inter ** 2).sum() / ((npip - 1) * (nfus - 1))),
        "fold x model": float(((fm - fm.mean(0) - fm.mean(1, keepdims=True) + fm.mean()) ** 2
                               ).sum() / ((n_folds - 1) * (nfus - 1))),
        "residual": float(np.var(noise, ddof=1)),
        "interaction_present": effect,
        "equivalent_pair": "late-attention",
        "nonequivalent_pair": "early-late",
    }
    return true


def synthetic_validation(out: Path, n_perm: int = 2000) -> bool:
    """Generate both scenarios, run every analysis, check the known effects are found."""
    checks, report = [], ["# Synthetic validation of src/multiverse.py", "",
                          "Synthetic runs with the structure of `loso_grouped` (23 folds, "
                          "9 pipelines, 3 seeds, 11 models) are generated twice: with an "
                          "injected Fusion x Preprocessing interaction and pipeline-specific "
                          "prediction changes (effect), and without them (null). Every "
                          "analysis is then run unchanged on the files. "
                          "(`src/multiverse.py: make_synthetic` documents the generator.)", ""]

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append(ok)
        report.append(f"| {name} | {detail} | {'PASS' if ok else 'FAIL'} |")

    results = {}
    for scenario in ("effect", "null"):
        tmp = Path(tempfile.mkdtemp(prefix=f"multiverse_{scenario}_"))
        try:
            true = make_synthetic(tmp, effect=scenario == "effect")
            res = analyze(tmp, out / scenario, f"synthetic ({scenario})", n_perm=n_perm,
                          verbose=False, margin=STRICT_MARGIN, strict_margin=None)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        results[scenario] = (true, res)

    report += ["## Checks", "", "| check | result | |", "|---|---|---|"]
    te, re_ = results["effect"]
    tn, rn = results["null"]
    check("effect: interaction detected", re_["interaction_p"] < 0.05,
          f"F = {re_['interaction_F']:.2f}, p_GG = {re_['interaction_p']:.2g}")
    check("null: no false interaction", rn["interaction_p"] >= 0.05,
          f"F = {rn['interaction_F']:.2f}, p_GG = {rn['interaction_p']:.3f}")
    check("effect: rankings differ more between pipelines than between seeds",
          re_["D"] > 0 and re_["p_perm"] < 0.05,
          f"tau within {re_['tau_within']:.3f}, between {re_['tau_between']:.3f}, "
          f"p_perm = {re_['p_perm']:.4f}")
    check("null: no pipeline effect on rankings", rn["p_perm"] >= 0.05,
          f"tau within {rn['tau_within']:.3f}, between {rn['tau_between']:.3f}, "
          f"p_perm = {rn['p_perm']:.3f}")
    check("effect: ICA-algorithm effect on rankings (P6a-c)",
          re_["p6_D"] > 0 and re_["p6_p_perm"] < 0.05,
          f"D = {re_['p6_D']:.3f}, p_perm = {re_['p6_p_perm']:.4f} (3 pipelines x 3 seeds)")
    check("null: no ICA-algorithm effect on rankings", rn["p6_p_perm"] >= 0.05,
          f"D = {rn['p6_D']:.3f}, p_perm = {rn['p6_p_perm']:.3f}")
    check("effect: predictions change more between pipelines than seeds",
          re_["kappa_pipelines"] < re_["kappa_seeds"] - 0.05,
          f"kappa pipelines {re_['kappa_pipelines']:.3f}, seeds {re_['kappa_seeds']:.3f}")
    check("null: pipeline and seed agreement alike",
          abs(rn["kappa_pipelines"] - rn["kappa_seeds"]) < 0.02,
          f"kappa pipelines {rn['kappa_pipelines']:.3f}, seeds {rn['kappa_seeds']:.3f}")
    check("effect: identical pair is robustly equivalent",
          te["equivalent_pair"] in re_["robust_equivalent"],
          f"robust equivalent: {', '.join(re_['robust_equivalent']) or 'none'}")
    check("effect: pair with interaction is not robustly equivalent",
          te["nonequivalent_pair"] not in re_["robust_equivalent"],
          f"{te['nonequivalent_pair']} not in the list")

    report += ["", "## Variance components: injected against estimated", "",
               "| component | scenario | injected | estimated | |", "|---|---|---|---|---|"]
    for scenario, (t, r) in results.items():
        raw = r["variance_raw"]
        total = sum(max(v, 0) for v in raw.values())
        for comp, key in (("fold", "fold"), ("pipeline", "pipeline"), ("model", "model"),
                          ("pipeline x model", "pipeline x model"),
                          ("fold x model", "fold x model"),
                          ("residual", "residual (fold x pipeline x model x seed)")):
            est, tv = raw[key], t[comp]
            ok = abs(est - tv) <= max(0.25 * tv, 0.01 * total)
            checks.append(ok)
            report.append(f"| {comp} | {scenario} | {tv:.6f} | {est:.6f} | "
                          f"{'PASS' if ok else 'FAIL'} |")
        for key in ("seed", "fold x seed", "pipeline x seed", "model x seed"):
            est = raw[key]
            ok = abs(est) <= 0.01 * total
            checks.append(ok)
            report.append(f"| {key} | {scenario} | 0 | {est:.6f} | {'PASS' if ok else 'FAIL'} |")

    report += ["", f"**{sum(checks)}/{len(checks)} checks passed.** Figures and tables of "
               f"each scenario are in `effect/` and `null/` next to this report.", ""]
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nwritten: {out}")
    return all(checks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="chbmit", help="results_v2/<dataset>")
    ap.add_argument("--pipelines", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--margin", type=float, default=EQUIV_MARGIN,
                    help="main equivalence margin in macro F1 (fixed: 0.05)")
    ap.add_argument("--perms", type=int, default=N_PERM)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="analyse the CPU runs or the GPU (_cuda) runs")
    ap.add_argument("--out", default=None, help="output folder")
    ap.add_argument("--synthetic", action="store_true",
                    help="validate the analyses on synthetic runs with known effects")
    args = ap.parse_args()
    if args.synthetic:
        out = Path(args.out) if args.out else RESULTS_ROOT / "multiverse" / "synthetic"
        ok = synthetic_validation(out, n_perm=min(args.perms, 2000))
        sys.exit(0 if ok else 1)
    name = args.dataset + ("_cuda" if args.device == "cuda" else "")
    out = Path(args.out) if args.out else RESULTS_ROOT / "multiverse" / name
    analyze(RESULTS_ROOT / args.dataset, out, name, args.pipelines, args.seeds,
            args.margin, args.perms, device=args.device)


if __name__ == "__main__":
    main()
