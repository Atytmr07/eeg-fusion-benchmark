"""Results figures 2 to 9 of the manuscript (paper/manuscript_v2.md), from stored runs.

The order follows the Results (advisor's structure of 21 October 2026):

  Figure 2  fusion operators against the controls: macro F1 and AUPRC per pipeline with
            the spectrogram network of the fusion budget and the shallow classifier, and
            the paired difference of every operator from each control
  Figure 3  equivalence matrix: every operator pair in every pipeline, mean paired
            difference, 90 percent confidence interval and test outcome (+-0.05, +-0.02)
  Figure 4  ranking stability: rank of every operator in every run, share of runs won,
            and Kendall's W across seed sets with three and with five seed sets
  Figure 5  variance decomposition, and macro F1 per held-out subject and model
  Figure 6  artefact removal (P6a/b/c): signal preservation, compute, performance change
            against P1, and reproducibility across machines
  Figure 7  seizure-level operating points: sensitivity against false alarms per hour
            (CHB-MIT), and seizure-level sensitivity on Siena
  Figure 8  cross-dataset: macro F1 in three settings, share of runs each operator wins
            in each setting, and the agreement of the rankings
  Figure 9  post hoc sensitivity: share of runs each operator wins at thresholds 0.3,
            0.5 and 0.7; seizure-level operating points of P0 and P1 at the same
            thresholds (CHB-MIT); macro F1 sampled against prevalence-adjusted (CHB-MIT).
            Needs src.sensitivity_preds and src.posthoc_checks first

Only complete runs enter (every LOSO fold of all five operators; make_results). Every
figure says how many seed sets it uses. Run it at any time; it uses what exists.

Usage:  python -m src.make_figures [--device cuda] [--figures 2 3 ...]
Output: paper/figures_manuscript/figure{2..9}_*.png and .pdf
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .chbmit_stats import FUSION
from .config import PROJECT_ROOT, RESULTS_ROOT
from .make_results import complete_runs, cross_runs, run_means
from .multiverse import (PIPELINE_ORDER, VC_METRICS, balanced_subset, group_components,
                         to_array, variance_components)

OUT = PROJECT_ROOT / "paper" / "figures_manuscript"
NAMES = {"chbmit": "CHB-MIT", "siena": "Siena"}
OPS = list(FUSION)                                  # early, late, gated, attention, score
# Categorical slots in fixed order (validated: CVD and normal-vision separation pass;
# below 3:1 contrast for three hues, so every operator also has its own marker and
# every figure a legend).
COLOR = dict(zip(OPS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MARKER = dict(zip(OPS, ["o", "s", "^", "D", "v"]))
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e3dd"
# ordinal blue ramp for ranks 1 (best, dark) to 5 (light), steps 650..250
RANK_RAMP = ["#104281", "#256abf", "#3987e5", "#6da7ec", "#86b6ef"]
# controls drawn next to the operators: greys and their own markers (not operator hues)
CTRL = ["spec2d_wide", "shallow"]
CTRL_STYLE = {"spec2d_wide": ("#5f5e5a", "P"), "shallow": ("#a39f94", "X")}
CTRL_LABEL = {"spec2d_wide": "spec2d_wide (control)", "shallow": "shallow (control)"}
MV = RESULTS_ROOT / "multiverse"
VC_LABEL = {"fold (subject)": "subject", "pipeline": "preprocessing", "fusion": "operator",
            "fusion x pipeline": "operator x preprocessing", "seed (all seed terms)": "seed",
            "fold x (pipeline, fusion)": "subject x (prepr., operator)",
            "residual": "residual"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
        "savefig.dpi": 300, "figure.dpi": 100})
    return plt


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        # no creation date in the PDF, so a rerun gives identical files
        meta = {"CreationDate": None} if ext == "pdf" else None
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", metadata=meta)
    print(f"written: {OUT / name}.png/.pdf")


def ordered(pipes) -> list[str]:
    return [p for p in PIPELINE_ORDER if p in set(pipes)]


def op_legend(fig, y: float = 1.02, controls: bool = False) -> None:
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COLOR[m], marker=MARKER[m], ls="", ms=6, label=m) for m in OPS]
    if controls:
        h += [Line2D([], [], color=CTRL_STYLE[c][0], marker=CTRL_STYLE[c][1], ls="", ms=6,
                     label=CTRL_LABEL[c]) for c in CTRL]
    fig.legend(handles=h, loc="upper center", ncol=len(h), bbox_to_anchor=(0.5, y),
               handletextpad=0.3, columnspacing=1.0)


# --- data -----------------------------------------------------------------------------------

def dataset_runs(ds: str, device: str):
    inc, long, _ = complete_runs(RESULTS_ROOT / ds, ds, device)
    if not inc:
        return None, None
    return long, run_means(long)


# --- Figure 2 -------------------------------------------------------------------------------

def figure2(data: dict) -> None:
    """Operators and two controls per pipeline (macro F1, AUPRC), and the paired
    difference of each operator from each control in every pipeline."""
    plt = _plt()
    dss = [d for d in ("chbmit", "siena") if data.get(d)]
    fig, axes = plt.subplots(len(dss), 3, figsize=(7.4, 2.75 * len(dss)), squeeze=False,
                             gridspec_kw={"width_ratios": [3, 3, 1.9]})
    allp = ordered(set().union(*(data[d][1].pipeline.unique() for d in dss)))
    pos = {p: i for i, p in enumerate(allp)}
    models = OPS + CTRL
    for r, ds in enumerate(dss):
        long = data[ds][0]
        rm = (long[long.model.isin(models)]
              .groupby(["pipeline", "seed", "model"])[["f1_macro", "auprc"]].mean()
              .reset_index())
        pipes = ordered(rm.pipeline.unique())
        nseed = rm.groupby("pipeline").seed.nunique().min()
        for c, (metric, label) in enumerate((("f1_macro", "macro F1"), ("auprc", "AUPRC"))):
            ax = axes[r, c]
            st = rm.groupby(["pipeline", "model"])[metric].agg(["mean", "std"])
            for k, m in enumerate(models):
                x = np.array([pos[p] for p in pipes]) + (k - 3) * 0.11
                sm = st.xs(m, level="model").reindex(pipes)
                col, mk = (COLOR[m], MARKER[m]) if m in OPS else CTRL_STYLE[m]
                ax.errorbar(x, sm["mean"], yerr=sm["std"], fmt=mk, color=col, ms=4,
                            elinewidth=0.9, capsize=0, mec="white", mew=0.5)
            ax.set_xticks(range(len(allp)), allp, fontsize=7)
            ax.set_xlim(-0.5, len(allp) - 0.5)
            ax.set_ylabel(label)
            ax.set_title(f"{'abcdef'[3 * r + c]}  {NAMES[ds]}: {label} ({nseed} seed sets)", loc="left")
        # paired differences, operator minus control, one point per pipeline
        ax = axes[r, 2]
        f = RESULTS_ROOT / "results" / f"controls_tests_{ds}_f1_macro.csv"
        if f.exists():
            t = pd.read_csv(f)
            for j, cn in enumerate(CTRL):
                for k, m in enumerate(OPS):
                    d = -t[(t.control == cn) & (t.operator == m)].mean_diff.values
                    x = k + (j - 0.5) * 0.36
                    ax.scatter(np.full(len(d), x), d, s=7, color=CTRL_STYLE[cn][0],
                               marker=CTRL_STYLE[cn][1], linewidths=0, alpha=0.8, zorder=3)
                    ax.plot([x - 0.13, x + 0.13], [d.mean()] * 2, color=COLOR[m], lw=1.6,
                            zorder=4)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_xticks(range(len(OPS)), OPS, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("operator minus control, macro F1")
            ax.set_title(f"{'abcdef'[3 * r + 2]}  Paired difference", loc="left")
            ax.grid(axis="x", visible=False)
            sig = int(t.significant.sum())
            ax.text(0.02, 0.02, f"significant (Holm): {sig} of {len(t)}",
                    transform=ax.transAxes, fontsize=6.5, color=INK2)
    op_legend(fig, y=1.03, controls=True)
    fig.tight_layout()
    save(fig, "figure2_performance")
    plt.close(fig)


# --- Figure 3 -------------------------------------------------------------------------------

def figure3(data: dict) -> None:
    """Equivalence matrix: operator pairs x pipelines, per dataset."""
    plt = _plt()
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    dss = [d for d in ("chbmit", "siena") if (MV / f"{d}_cuda" / "equivalence.csv").exists()]
    if not dss:
        print("figure 3: no equivalence tables, skipped")
        return
    # 0 inconclusive, 1 equivalent within 0.05, 2 within 0.02, 3 significant difference
    cols = ["#ecebe6", "#a6dcc3", "#1b9e6b", "#eb6834"]
    labels = ["inconclusive", "equivalent within 0.05", "equivalent within 0.02",
              "significant difference (Holm)"]
    fig, axes = plt.subplots(len(dss), 1, figsize=(7.2, 3.9 * len(dss)), squeeze=False)
    for r, ds in enumerate(dss):
        eq = pd.read_csv(MV / f"{ds}_cuda" / "equivalence.csv")
        strict_f = MV / f"{ds}_cuda" / "equivalence_strict.csv"
        st = pd.read_csv(strict_f) if strict_f.exists() else None
        pipes = ordered(eq.pipeline.unique())
        pairs = list(dict.fromkeys(eq.pair))
        z = np.zeros((len(pairs), len(pipes)))
        ax = axes[r, 0]
        for _, row in eq.iterrows():
            i, j = pairs.index(row.pair), pipes.index(row.pipeline)
            strict = (st is not None and bool(st[(st.pair == row.pair)
                                                 & (st.pipeline == row.pipeline)].equivalent.iloc[0]))
            z[i, j] = 3 if row.significant else (2 if strict else (1 if row.equivalent else 0))
            hw = row.delta_min - abs(row.mean_diff)          # half-width of the 90% CI
            ax.text(j, i, f"{row.mean_diff:+.3f}\n±{hw:.3f}", ha="center", va="center",
                    fontsize=7, color="white" if z[i, j] >= 2 else INK)
        ax.imshow(z, cmap=ListedColormap(cols), vmin=-0.5, vmax=3.5, aspect="auto",
                  interpolation="nearest")
        ax.set_xticks(range(len(pipes)), pipes)
        ax.set_yticks(range(len(pairs)), [p.replace("-", " - ") for p in pairs], fontsize=7)
        ax.tick_params(length=0)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(len(pipes)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(pairs)) - 0.5, minor=True)
        ax.grid(which="minor", color="white", lw=1.5)
        ax.tick_params(which="minor", length=0)
        ax.set_title(f"{'ab'[r]}  {NAMES[ds]}: mean paired difference in macro F1 (first "
                     f"minus second operator) and half-width of its 90% CI", loc="left")
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(cols, labels)],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "figure3_equivalence")
    plt.close(fig)


# --- Figure 4 -------------------------------------------------------------------------------

def seed_w(ds: str, folder: str) -> dict:
    """Kendall's W across seed sets per pipeline, from a multiverse summary.md."""
    f = MV / folder / "summary.md"
    if not f.exists():
        return {}
    pat = re.compile(r"- (P\w+): Kendall's W over (\d+) seeds = ([0-9.]+)")
    return {m.group(1): float(m.group(3)) for m in pat.finditer(f.read_text(encoding="utf-8"))}


def interaction_p(folder: str):
    f = MV / folder / "summary.md"
    if not f.exists():
        return None
    m = re.search(r"Fusion x Preprocessing: F\([^)]*\) = [0-9.]+, p_GG = ([0-9.]+)",
                  f.read_text(encoding="utf-8"))
    return float(m.group(1)) if m else None


def run_ranks(rm: pd.DataFrame) -> pd.DataFrame:
    w = rm.pivot_table(index=["pipeline", "seed"], columns="model", values="f1_macro")[OPS]
    r = w.rank(axis=1, ascending=False, method="average")
    r = r.reset_index()
    r["key"] = [PIPELINE_ORDER.index(p) for p in r.pipeline]
    return r.sort_values(["key", "seed"]).drop(columns="key")


def figure4(data: dict) -> None:
    plt = _plt()
    from matplotlib.colors import BoundaryNorm, ListedColormap
    dss = [d for d in ("chbmit", "siena") if data.get(d)]
    fig = plt.figure(figsize=(7.2, 2.3 * len(dss) + 2.6))
    gs = fig.add_gridspec(len(dss) + 1, 2, width_ratios=[5, 1.4], wspace=0.08, hspace=0.62,
                          height_ratios=[1] * len(dss) + [1.05])
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(len(dss))])
    cmap = ListedColormap(RANK_RAMP)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
    for r, ds in enumerate(dss):
        rk = run_ranks(data[ds][1])
        ax = axes[r, 0]
        ax.grid(False)
        im = ax.imshow(rk[OPS].T.values, aspect="auto", cmap=cmap, norm=norm,
                       interpolation="nearest")
        ax.set_yticks(range(len(OPS)), OPS)
        pipes = ordered(rk.pipeline.unique())
        centers, edges = [], []
        for p in pipes:
            idx = np.where(rk.pipeline.values == p)[0]
            centers.append(idx.mean())
            edges.append(idx.max() + 0.5)
        ax.set_xticks(centers, pipes)
        ax.tick_params(axis="x", length=0)
        for e in edges[:-1]:
            ax.axvline(e, color="white", lw=2)
        nseed = rk.groupby("pipeline").seed.nunique().min()
        ax.set_title(f"{'ab'[r]}  {NAMES[ds]}: rank in every run ({nseed} seed sets per "
                     f"pipeline)", loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        # share of runs won
        bx = axes[r, 1]
        won = (rk[OPS] == 1).mean()
        bx.barh(range(len(OPS)), won.values, color=[COLOR[m] for m in OPS], height=0.6)
        bx.set_yticks(range(len(OPS)), [""] * len(OPS))
        bx.invert_yaxis()
        bx.set_xlim(0, 1)
        bx.set_title("runs won", loc="left")
        bx.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
        for i, v in enumerate(won.values):
            bx.text(v + 0.02, i, f"{v:.0%}", va="center", fontsize=7.5, color=INK)
        bx.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    axes[-1, 0].legend(handles=[Patch(color=c, label=str(i + 1))
                                for i, c in enumerate(RANK_RAMP)],
                       title="rank (1 = best macro F1)", loc="upper center", ncol=5,
                       bbox_to_anchor=(0.5, -0.14), handlelength=1.2, columnspacing=1.0,
                       title_fontsize=8, fontsize=7.5)
    # Kendall's W across seed sets per pipeline: three against five seed sets
    sub = gs[len(dss), :].subgridspec(1, len(dss), wspace=0.25)
    for c, ds in enumerate(dss):
        ax = fig.add_subplot(sub[0, c])
        w5, w3 = seed_w(ds, f"{ds}_cuda"), seed_w(ds, f"{ds}_cuda_3seeds")
        pipes = ordered(set(w5) | set(w3))
        x = np.arange(len(pipes))
        if w3:
            ax.scatter(x - 0.12, [w3.get(p, np.nan) for p in pipes], s=22, facecolor="white",
                       edgecolor=INK2, lw=1, label="3 seed sets", zorder=3)
        ax.scatter(x + 0.12, [w5.get(p, np.nan) for p in pipes], s=22, color="#256abf",
                   label="5 seed sets", zorder=3)
        ax.set_xticks(x, pipes, fontsize=7)
        ax.set_ylim(0, 1.2)                     # headroom for the note above the points
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylabel("Kendall's W across seeds")
        p3, p5 = interaction_p(f"{ds}_cuda_3seeds"), interaction_p(f"{ds}_cuda")
        note = (f"Fusion x Preprocessing p_GG:\n{p3:.3f} (3 seeds), {p5:.3f} (5 seeds)"
                if p3 is not None and p5 is not None else "")
        ax.set_title(f"{'cd'[c]}  {NAMES[ds]}: agreement of seed rankings", loc="left")
        ax.text(0.02, 0.95, note, transform=ax.transAxes, fontsize=6.3, va="top", color=INK2)
        ax.grid(axis="x", visible=False)
        if c == 0:
            ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.84))
    save(fig, "figure4_rank_stability")
    plt.close(fig)


# --- Figure 5 -------------------------------------------------------------------------------

def variance_shares(long: pd.DataFrame) -> pd.DataFrame:
    models = [m for m in OPS if m in set(long.model)]
    pipes, seeds, folds = balanced_subset(long, models)
    rows = []
    names = ("fold", "pipeline", "model") + (("seed",) if len(seeds) > 1 else ())
    for metric in VC_METRICS:
        try:
            a = to_array(long, metric, models, pipes, seeds, folds)
        except RuntimeError:
            continue
        y = a if len(seeds) > 1 else a.mean(axis=3)
        g = group_components(variance_components(y, names))
        for grp, share in zip(g.group, g.share):
            rows.append({"metric": metric, "component": VC_LABEL[grp], "share": share,
                         "n_seeds": len(seeds)})
    return pd.DataFrame(rows)


def subject_names(ds: str) -> dict:
    """fold index -> held-out subject(s), from one run's perfold.csv."""
    runs = sorted((RESULTS_ROOT / ds).glob("loso_P1*_cuda/perfold.csv"))
    if not runs:
        return {}
    pf = pd.read_csv(runs[0])
    return {int(f): str(s).replace(",", "+") for f, s in
            pf.groupby("fold").test_subjects.first().items()}


def figure5(data: dict) -> None:
    plt = _plt()
    dss = [d for d in ("chbmit", "siena") if data.get(d)]
    fig = plt.figure(figsize=(3.6 * len(dss), 10.2))
    gs = fig.add_gridspec(2, len(dss), height_ratios=[1, 2.25], hspace=0.28, wspace=0.42)
    axes = np.array([[fig.add_subplot(gs[0, c], sharey=None) for c in range(len(dss))]])
    mcol = {"f1_macro": "#2a78d6", "auprc": "#eb6834", "balanced_accuracy": "#1baf7a"}
    mlab = {"f1_macro": "macro F1", "auprc": "AUPRC", "balanced_accuracy": "balanced accuracy"}
    order = list(VC_LABEL.values())
    for c, ds in enumerate(dss):
        vs = variance_shares(data[ds][0])
        ax = axes[0, c]
        mets = [m for m in VC_METRICS if m in set(vs.metric)]
        h = 0.8 / len(mets)
        for k, met in enumerate(mets):
            s = vs[vs.metric == met].set_index("component").share.reindex(order).fillna(0)
            y = np.arange(len(order)) - (k - (len(mets) - 1) / 2) * h   # first metric drawn on top
            ax.barh(y, 100 * s.values, height=h * 0.9, color=mcol[met], label=mlab[met])
            for yi, v in zip(y, s.values):
                ax.text(100 * v + 1, yi, f"{100 * v:.1f}", va="center", fontsize=7,
                        color=INK2)
        ax.set_yticks(range(len(order)), order)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("share of variance (%)")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{'ab'[c]}  {NAMES[ds]}: sources of variance "
                     f"({int(vs.n_seeds.iloc[0])} seed sets)", loc="left")
        if c > 0:
            ax.set_yticklabels([])
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 0.935))
    # macro F1 per held-out subject and model, mean over pipelines and seed sets
    models = OPS + CTRL
    for c, ds in enumerate(dss):
        long = data[ds][0]
        m = (long[long.model.isin(models)].groupby(["fold", "model"]).f1_macro.mean()
             .unstack()[models])
        m.index = [subject_names(ds).get(int(f), str(f)) for f in m.index]
        m = m.loc[m.mean(axis=1).sort_values(ascending=False).index]
        ax = fig.add_subplot(gs[1, c])
        ax.grid(False)
        im = ax.imshow(m.values, cmap="Blues", vmin=0.4, vmax=1.0, aspect="auto",
                       interpolation="nearest")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.2,
                        color="white" if v > 0.78 else INK)
        ax.set_xticks(range(len(models)), models, fontsize=6.5, rotation=45, ha="right")
        ax.set_yticks(range(m.shape[0]), m.index, fontsize=6.5)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{'cd'[c]}  {NAMES[ds]}: macro F1 per held-out subject", loc="left")
    cax = fig.add_axes([0.93, 0.08, 0.012, 0.3])
    fig.colorbar(im, cax=cax, label="macro F1 (mean over pipelines and seeds)")
    save(fig, "figure5_variance")
    plt.close(fig)


# --- Figure 6 -------------------------------------------------------------------------------

def figure6(data: dict) -> None:
    plt = _plt()
    meth = [("P6a", "infomax", "Infomax"), ("P6b", "gedai-conservative", "GEDAI"),
            ("P6c", "amica", "AMICA")]
    logs_f = RESULTS_ROOT / "qc" / "p6_recording_logs.csv"
    rep_f = RESULTS_ROOT / "ica_repro" / "logs_cache_vs_lab.csv"
    rerun_f = RESULTS_ROOT / "ica_repro" / "logs_cache_vs_laptop_rerun.csv"
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    mc = "#4a3aa7"                                          # one hue: methods are not operators
    if logs_f.exists():
        logs = pd.read_csv(logs_f)
        logs = logs[logs.record.str.startswith("chb")]          # CHB-MIT only
        ax = axes[0, 0]
        vals = [logs[logs.method == m].power_kept.values for _, m, _ in meth]
        bp = ax.boxplot(vals, widths=0.45, patch_artist=True, showfliers=True,
                        medianprops={"color": INK, "lw": 1.2},
                        flierprops={"marker": "o", "ms": 2, "mfc": INK2, "mec": "none",
                                    "alpha": 0.4})
        for b in bp["boxes"]:
            b.set(facecolor="#b7d3f6", edgecolor=INK2)
        ax.set_xticks([1, 2, 3], [f"{lab}\n({p})" for p, _, lab in meth])
        ax.set_ylabel("power kept after cleaning")
        ax.set_title("a  Signal preservation per recording (CHB-MIT)", loc="left")
        ax = axes[0, 1]
        hrs = [logs[logs.method == m].seconds.sum() / 3600 for _, m, _ in meth]
        ax.bar(range(3), hrs, color=mc, width=0.55)
        for i, v in enumerate(hrs):
            ax.text(i, v + 1, f"{v:.1f} h", ha="center", fontsize=7.5)
        ax.set_xticks(range(3), [lab for _, _, lab in meth])
        ax.set_ylabel("CPU hours, 670 recordings")
        ax.set_title("b  Compute (laptop CPU)", loc="left")
        ax.grid(axis="x", visible=False)
    ax = axes[1, 0]
    if data.get("chbmit"):
        rm = data["chbmit"][1]
        base = rm[rm.pipeline == "P1"].set_index(["seed", "model"]).f1_macro
        for k, m in enumerate(OPS):
            for j, (p, _, lab) in enumerate(meth):
                s = rm[(rm.pipeline == p) & (rm.model == m)].set_index("seed").f1_macro
                d = (s - base.xs(m, level="model")).dropna()
                if len(d):
                    ax.errorbar(j + (k - 2) * 0.13, d.mean(), yerr=d.std(), fmt=MARKER[m],
                                color=COLOR[m], ms=4.5, elinewidth=1, mec="white", mew=0.6)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(3), [f"{lab}\n({p})" for p, _, lab in meth])
        ax.set_ylabel("macro F1 minus P1 (same seed)")
        ax.set_title("c  Performance change against P1 (CHB-MIT)", loc="left")
    ax = axes[1, 1]
    if rep_f.exists():
        rows = []
        for f, lab in ((rep_f, "between machines"), (rerun_f, "same machine, rerun")):
            if not f.exists():
                continue
            d = pd.read_csv(f)
            d = d[d.record.str.startswith("chb")]
            for _, m, mlab in meth:
                mm = m.split("-")[0]
                g = d[d.method == mm]
                if len(g):
                    changed = ((~g.same_removed) | (g.abs_dpower > 1e-3)).mean()
                    rows.append((mlab, lab, changed))
        t = pd.DataFrame(rows, columns=["method", "context", "share"])
        ctx = ["between machines", "same machine, rerun"]
        for k, cx in enumerate(ctx):
            s = t[t.context == cx].set_index("method").share.reindex([l for _, _, l in meth])
            x = np.arange(3) + (k - 0.5) * 0.3
            ax.bar(x, 100 * s.values, width=0.28, color=["#4a3aa7", "#9085e9"][k], label=cx)
            for xi, v in zip(x, s.values):
                if np.isfinite(v):
                    ax.text(xi, 100 * v + 0.6, f"{100 * v:.1f}", ha="center", fontsize=7)
        ax.set_xticks(range(3), [l for _, _, l in meth])
        ax.set_ylabel("recordings cleaned differently (%)")
        ax.set_title("d  Reproducibility (CHB-MIT, fixed seeds)", loc="left")
        ax.set_ylim(0, 1.3 * 100 * t.share.max())
        # upper right sits above AMICA's near-zero bars, clear of GEDAI's tall one
        ax.legend(fontsize=7.5, loc="upper right", frameon=False)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COLOR[m], marker=MARKER[m], ls="", ms=5, label=m) for m in OPS]
    axes[1, 0].legend(handles=h, fontsize=7, ncol=5, loc="lower center",
                      bbox_to_anchor=(0.5, 1.07), handletextpad=0.2, columnspacing=0.6)
    save(fig, "figure6_artefact_removal")
    plt.close(fig)


# --- Figure 7 -------------------------------------------------------------------------------

def figure7(data: dict) -> None:
    """Seizure-level operating points: CHB-MIT sensitivity against false alarms per hour,
    one point per run and the operator's mean and SD; Siena sensitivity per operator."""
    fc = RESULTS_ROOT / "results" / "events_chbmit_per_run.csv"
    fs = RESULTS_ROOT / "results" / "events_siena_per_run.csv"
    if not fc.exists():
        print("figure 7: no event tables, skipped")
        return
    plt = _plt()
    ev = pd.read_csv(fc)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    for m in OPS:
        g = ev[ev.model == m]
        ax.scatter(g.fa_per_hour, g.sensitivity, s=9, color=COLOR[m], marker=MARKER[m],
                   alpha=0.35, linewidths=0)
        ax.errorbar(g.fa_per_hour.mean(), g.sensitivity.mean(), xerr=g.fa_per_hour.std(),
                    yerr=g.sensitivity.std(), fmt=MARKER[m], color=COLOR[m], ms=7,
                    mec="white", mew=0.8, elinewidth=1.2, zorder=4)
    n = ev.groupby("pipeline").seed.nunique().min()
    ax.set_xlabel("false alarms per hour")
    ax.set_ylabel("seizure-level sensitivity")
    ax.set_title(f"a  CHB-MIT: operating point at threshold 0.5 ({ev.pipeline.nunique()} "
                 f"pipelines x {n} seed sets)", loc="left")
    ax = axes[1]
    if fs.exists():
        es = pd.read_csv(fs)
        for k, m in enumerate(OPS):
            g = es[es.model == m].sensitivity
            jit = (np.arange(len(g)) % 9 - 4) * 0.025
            ax.scatter(k + jit, g, s=7, color=COLOR[m], marker=MARKER[m], alpha=0.35,
                       linewidths=0)
            ax.plot([k - 0.25, k + 0.25], [g.mean()] * 2, color=INK, lw=1.4)
        ax.set_xticks(range(len(OPS)), OPS, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("seizure-level sensitivity")
        ax.set_title("b  Siena: sensitivity", loc="left")
        ax.grid(axis="x", visible=False)
    op_legend(fig, y=1.05)
    fig.tight_layout()
    save(fig, "figure7_seizure_level")
    plt.close(fig)


# --- Figure 8 -------------------------------------------------------------------------------

def figure8(data: dict, device: str) -> None:
    from scipy.stats import kendalltau
    cr = cross_runs(RESULTS_ROOT / "cross", device)
    if not cr or not data.get("chbmit") or not data.get("siena"):
        print("figure 8: cross-dataset or a LOSO dataset missing, skipped")
        return
    plt = _plt()
    rows = []
    for _, pipe, seed, path in cr:
        pf = pd.read_csv(path / "perfold.csv")
        pf = pf[pf.model.isin(OPS)].groupby("model").f1_macro.mean()
        rows += [{"pipeline": pipe, "seed": seed, "model": m, "f1": v} for m, v in pf.items()]
    cross = pd.DataFrame(rows)
    pipes = ordered(cross.pipeline.unique())
    settings = {"CHB-MIT (LOSO)": data["chbmit"][1], "Siena (LOSO)": data["siena"][1],
                "CHB-MIT -> Siena": cross.rename(columns={"f1": "f1_macro"})}
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9),
                             gridspec_kw={"width_ratios": [2.0, 1.25, 1]})
    ax = axes[0]
    means = {}
    xt = []
    for j, (lab, rm) in enumerate(settings.items()):
        sub = rm[rm.pipeline.isin(pipes)]
        st = sub.groupby("model").f1_macro.agg(["mean", "std"]).reindex(OPS)
        means[lab] = sub.groupby(["pipeline", "model"]).f1_macro.mean()
        for k, m in enumerate(OPS):
            ax.errorbar(j + (k - 2) * 0.13, st.at[m, "mean"], yerr=st.at[m, "std"],
                        fmt=MARKER[m], color=COLOR[m], ms=4.5, elinewidth=1, mec="white",
                        mew=0.6)
        xt.append(lab)
    ax.set_xticks(range(len(xt)), xt)
    ax.set_ylabel("macro F1")
    plabel = "all nine pipelines" if len(pipes) == 9 else "pipelines " + ", ".join(pipes)
    ax.set_title(f"a  macro F1 ({plabel})", loc="left")
    ax.set_xticks(range(len(xt)), ["CHB-MIT\n(LOSO)", "Siena\n(LOSO)", "CHB-MIT\n-> Siena"])
    ax.grid(axis="x", visible=False)
    # share of runs in which each operator ranks first, per setting
    ax = axes[1]
    for j, (lab, rm) in enumerate(settings.items()):
        w = rm[rm.pipeline.isin(pipes)].pivot_table(index=["pipeline", "seed"],
                                                    columns="model", values="f1_macro")[OPS]
        won = (w.rank(axis=1, ascending=False, method="average") == 1).mean()
        for k, m in enumerate(OPS):
            ax.bar(j + (k - 2) * 0.16, 100 * won[m], width=0.15, color=COLOR[m])
    ax.set_xticks(range(3), ["CHB-MIT", "Siena", "transfer"])
    ax.set_ylabel("runs ranked first (%)")
    ax.set_title("b  Rank reversal", loc="left")
    ax.grid(axis="x", visible=False)
    ax = axes[2]
    pairs = [("CHB-MIT (LOSO)", "Siena (LOSO)"), ("CHB-MIT (LOSO)", "CHB-MIT -> Siena"),
             ("Siena (LOSO)", "CHB-MIT -> Siena")]
    plab = ["CHB vs\nSiena", "CHB vs\ntransfer", "Siena vs\ntransfer"]
    for i, (a, b) in enumerate(pairs):
        taus = []
        for p in pipes:
            ra, rb = means[a].xs(p).reindex(OPS), means[b].xs(p).reindex(OPS)
            taus.append(kendalltau(ra.values, rb.values).statistic)
        ax.scatter(np.full(len(taus), i), taus, color="#4a3aa7", s=18, zorder=3)
        ax.plot([i - 0.2, i + 0.2], [np.mean(taus)] * 2, color=INK, lw=1.5)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(range(3), plab, fontsize=6.5)
    ax.set_ylabel("Kendall tau of operator rankings")
    ax.set_title("c  Ranking agreement", loc="left")
    ax.grid(axis="x", visible=False)
    op_legend(fig, y=1.06)
    fig.tight_layout()
    save(fig, "figure8_cross_dataset")
    plt.close(fig)


def figure9(data: dict) -> None:
    """Post hoc sensitivity to evaluation choices (advisor's review of 26 October 2026), from
    the outputs of src/sensitivity_preds.py and src/posthoc_checks.py."""
    sp = RESULTS_ROOT / "sensitivity_preds"
    need = [sp / "threshold_chbmit.csv", sp / "threshold_siena.csv",
            sp / "event_threshold_chbmit.csv", sp / "posthoc_prevalence_confusion.csv"]
    if not all(f.exists() for f in need):
        print("figure 9: run python -m src.sensitivity_preds and src.posthoc_checks first")
        return
    plt = _plt()
    from matplotlib.lines import Line2D
    thr = [0.3, 0.5, 0.7]
    fig = plt.figure(figsize=(7.6, 3.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.45, 1.75], wspace=0.55)
    # a: share of runs each operator ranks first, per threshold
    for j, ds in enumerate(("chbmit", "siena")):
        ax = fig.add_subplot(gs[0, j])
        rm = pd.read_csv(sp / f"threshold_{ds}.csv")
        rm = rm[rm.model.isin(OPS)]
        winners = {}
        for i, t in enumerate(thr):
            w = rm.pivot_table(index=["pipeline", "seed"], columns="model",
                               values=f"f1_macro@{t}")[OPS]
            winners[t] = w.idxmax(axis=1)
            won = (w.rank(axis=1, ascending=False, method="average") == 1).mean()
            for k, m in enumerate(OPS):
                ax.bar(i + (k - 2) * 0.16, 100 * won[m], width=0.15, color=COLOR[m])
        for i, t in ((0, 0.3), (2, 0.7)):
            changed = 100 * (winners[t] != winners[0.5]).mean()
            ax.text(i, 99, f"{changed:.0f}%\nchanged", ha="center", va="top", fontsize=6.3,
                    color=INK2)
        ax.set_xticks(range(3), [f"{t:.1f}" for t in thr])
        ax.set_ylim(0, 100)
        ax.set_xlabel("decision threshold")
        if j == 0:
            ax.set_ylabel("runs ranked first (%)")
            ax.text(0, 1.125, "a  Operator ranking", transform=ax.transAxes, fontsize=9,
                    va="bottom")
        ax.set_title(NAMES[ds], fontsize=7.5, color=INK2)
        ax.grid(axis="x", visible=False)
    # b: seizure-level operating points of P0 and P1 (CHB-MIT), one line per operator
    ax = fig.add_subplot(gs[0, 2])
    ev = pd.read_csv(sp / "event_threshold_chbmit.csv")
    ev = ev[ev.model.isin(OPS)].groupby(["model", "threshold"])[["sens", "fa"]].mean()
    tmark = {0.3: "o", 0.5: "s", 0.7: "^"}
    for m in OPS:
        pts = ev.xs(m).reindex(thr)
        ax.plot(pts.fa, pts.sens, color=COLOR[m], lw=1.0, zorder=2)
        for t in thr:
            ax.plot(pts.at[t, "fa"], pts.at[t, "sens"], tmark[t], color=COLOR[m], ms=5,
                    mec="white", mew=0.6, zorder=3)
    ax.set_xlabel("false alarms per hour")
    ax.set_ylabel("seizure-level sensitivity")
    ax.set_title("b  Detection trade-off", loc="left", y=1.1)
    ax.legend(handles=[Line2D([], [], color=INK2, marker=tmark[t], ls="", ms=5,
                              label=f"threshold {t:.1f}") for t in thr],
              loc="lower right", fontsize=6.3, handletextpad=0.2)
    ax.text(0.03, 0.97, "CHB-MIT, P0 and P1", transform=ax.transAxes, va="top",
            fontsize=6.5, color=INK2)
    # c: macro F1 sampled (1:4) against prevalence-adjusted, per held-out person
    ax = fig.add_subplot(gs[0, 3])
    cm = pd.read_csv(sp / "posthoc_prevalence_confusion.csv")

    def macro(tp, fn, fp, tn):
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12), 0)
        f0 = np.where(2 * tn + fn + fp > 0, 2 * tn / np.maximum(2 * tn + fn + fp, 1e-12), 0)
        return (f1 + f0) / 2

    cm["sampled"] = macro(cm.tp, cm.fn, cm.fp_sampled, cm.tn_sampled)
    cm["adjusted"] = macro(cm.tp, cm.fn, cm.fp_weighted, cm.tn_weighted)
    models = OPS + ["shallow"]
    person = cm.groupby(["model", "fold"])[["sampled", "adjusted"]].mean()
    rng = np.random.default_rng(0)
    for i, m in enumerate(models):
        col = COLOR.get(m, CTRL_STYLE["shallow"][0])
        for off, key, alpha in ((-0.19, "sampled", 0.38), (0.19, "adjusted", 1.0)):
            v = person.xs(m)[key]
            ax.bar(i + off, v.mean(), width=0.36, color=col, alpha=alpha, zorder=2)
            ax.scatter(i + off + rng.uniform(-0.08, 0.08, len(v)), v, s=3.5, color=INK,
                       alpha=0.45, lw=0, zorder=3)
    ax.set_xticks(range(len(models)), models, rotation=40, ha="right", fontsize=7,
                  rotation_mode="anchor")
    ax.set_ylabel("macro F1")
    ax.set_ylim(0, 1)
    ax.set_title("c  Class-prevalence adjustment", loc="left", y=1.1)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=INK2, alpha=0.38,
                                     label="sampled (1:4)"),
                       plt.Rectangle((0, 0), 1, 1, color=INK2, label="prevalence-adjusted")],
              loc="upper right", fontsize=6.3, handlelength=1.0)
    ax.grid(axis="x", visible=False)
    op_legend(fig, y=1.1)
    save(fig, "figure9_posthoc_sensitivity")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--figures", nargs="*", type=int, default=[2, 3, 4, 5, 6, 7, 8, 9])
    args = ap.parse_args()
    data = {}
    for ds in ("chbmit", "siena"):
        long, rm = dataset_runs(ds, args.device)
        if long is not None:
            data[ds] = (long, rm)
            print(f"{NAMES[ds]}: {rm.groupby(['pipeline', 'seed']).ngroups} complete runs")
    fns = {2: figure2, 3: figure3, 4: figure4, 5: figure5, 6: figure6, 7: figure7, 9: figure9}
    for f in args.figures:
        if f == 8:
            figure8(data, args.device)
        else:
            fns[f](data)


if __name__ == "__main__":
    main()
