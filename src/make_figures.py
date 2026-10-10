"""Results figures 2 to 6 of the manuscript (paper/manuscript_v2.md), from stored runs.

  Figure 2  performance of each fusion operator across the pipelines: macro F1 and
            AUPRC, mean and SD over seed sets, CHB-MIT and Siena
  Figure 3  ranking stability: the rank of every operator in every pipeline x seed run
            (heatmap), and the share of runs each operator wins
  Figure 4  variance decomposition of macro F1, AUPRC and balanced accuracy: subject,
            preprocessing, operator, their interaction, seed, residual
  Figure 5  artefact removal (P6a/b/c): signal preservation, compute, performance change
            against P1, and reproducibility across machines
  Figure 6  cross-dataset: each operator's macro F1 on CHB-MIT (LOSO), Siena (LOSO) and
            CHB-MIT -> Siena, and the agreement of their rankings

Only complete runs enter (every LOSO fold of all five operators; make_results). Every
figure says how many seed sets it uses. Run it at any time; it uses what exists.

Usage:  python -m src.make_figures [--device cuda]
Output: paper/figures_manuscript/figure{2..6}_*.png and .pdf
"""
from __future__ import annotations

import argparse
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
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    print(f"written: {OUT / name}.png/.pdf")


def ordered(pipes) -> list[str]:
    return [p for p in PIPELINE_ORDER if p in set(pipes)]


def op_legend(fig, y: float = 1.02) -> None:
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COLOR[m], marker=MARKER[m], ls="", ms=6, label=m) for m in OPS]
    fig.legend(handles=h, loc="upper center", ncol=len(OPS), bbox_to_anchor=(0.5, y),
               handletextpad=0.3, columnspacing=1.2)


# --- data -----------------------------------------------------------------------------------

def dataset_runs(ds: str, device: str):
    inc, long, _ = complete_runs(RESULTS_ROOT / ds, ds, device)
    if not inc:
        return None, None
    return long, run_means(long)


# --- Figure 2 -------------------------------------------------------------------------------

def figure2(data: dict) -> None:
    plt = _plt()
    dss = [d for d in ("chbmit", "siena") if data.get(d)]
    fig, axes = plt.subplots(len(dss), 2, figsize=(7.2, 2.6 * len(dss)), squeeze=False)
    allp = ordered(set().union(*(data[d][1].pipeline.unique() for d in dss)))
    pos = {p: i for i, p in enumerate(allp)}
    for r, ds in enumerate(dss):
        rm = data[ds][1]
        pipes = ordered(rm.pipeline.unique())
        nseed = rm.groupby("pipeline").seed.nunique().min()
        for c, (metric, label) in enumerate((("f1_macro", "macro F1"), ("auprc", "AUPRC"))):
            ax = axes[r, c]
            st = rm.groupby(["pipeline", "model"])[metric].agg(["mean", "std"])
            for k, m in enumerate(OPS):
                x = np.array([pos[p] for p in pipes]) + (k - 2) * 0.13
                s = st.xs(m, level="model").reindex(pipes)
                ax.errorbar(x, s["mean"], yerr=s["std"], fmt=MARKER[m], color=COLOR[m],
                            ms=4.5, elinewidth=1, capsize=0, mec="white", mew=0.6)
            ax.set_xticks(range(len(allp)), allp)
            ax.set_xlim(-0.5, len(allp) - 0.5)
            ax.set_ylabel(label)
            ax.set_title(f"{NAMES[ds]}: {label} (mean and SD over {nseed} seed sets)",
                         loc="left")
    op_legend(fig)
    fig.tight_layout()
    save(fig, "figure2_performance")
    plt.close(fig)


# --- Figure 3 -------------------------------------------------------------------------------

def run_ranks(rm: pd.DataFrame) -> pd.DataFrame:
    w = rm.pivot_table(index=["pipeline", "seed"], columns="model", values="f1_macro")[OPS]
    r = w.rank(axis=1, ascending=False, method="average")
    r = r.reset_index()
    r["key"] = [PIPELINE_ORDER.index(p) for p in r.pipeline]
    return r.sort_values(["key", "seed"]).drop(columns="key")


def figure3(data: dict) -> None:
    plt = _plt()
    from matplotlib.colors import BoundaryNorm, ListedColormap
    dss = [d for d in ("chbmit", "siena") if data.get(d)]
    fig, axes = plt.subplots(len(dss), 2, figsize=(7.2, 2.3 * len(dss) + 0.5), squeeze=False,
                             gridspec_kw={"width_ratios": [5, 1.4], "wspace": 0.08,
                                          "hspace": 0.55})
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
        ax.set_title(f"{NAMES[ds]}: rank in every run ({nseed} seed sets per pipeline)",
                     loc="left")
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
    fig.legend(handles=[Patch(color=c, label=str(i + 1)) for i, c in enumerate(RANK_RAMP)],
               title="rank (1 = best macro F1)", loc="lower center", ncol=5,
               bbox_to_anchor=(0.42, -0.06), handlelength=1.2, columnspacing=1.0,
               title_fontsize=8)
    save(fig, "figure3_rank_stability")
    plt.close(fig)


# --- Figure 4 -------------------------------------------------------------------------------

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


def figure4(data: dict) -> None:
    plt = _plt()
    dss = [d for d in ("chbmit", "siena") if data.get(d)]
    fig, axes = plt.subplots(1, len(dss), figsize=(3.6 * len(dss), 2.9), squeeze=False,
                             sharey=True)
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
                ax.text(100 * v + 1, yi, f"{100 * v:.1f}", va="center", fontsize=6,
                        color=INK2)
        ax.set_yticks(range(len(order)), order)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("share of variance (%)")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{NAMES[ds]} ({int(vs.n_seeds.iloc[0])} seed sets)", loc="left")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.6, 1.07))
    save(fig, "figure4_variance")
    plt.close(fig)


# --- Figure 5 -------------------------------------------------------------------------------

def figure5(data: dict) -> None:
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
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COLOR[m], marker=MARKER[m], ls="", ms=5, label=m) for m in OPS]
    axes[1, 0].legend(handles=h, fontsize=7, ncol=5, loc="lower center",
                      bbox_to_anchor=(0.5, 1.07), handletextpad=0.2, columnspacing=0.6)
    save(fig, "figure5_artefact_removal")
    plt.close(fig)


# --- Figure 6 -------------------------------------------------------------------------------

def figure6(data: dict, device: str) -> None:
    from scipy.stats import kendalltau
    cr = cross_runs(RESULTS_ROOT / "cross", device)
    if not cr or not data.get("chbmit") or not data.get("siena"):
        print("figure 6: cross-dataset or a LOSO dataset missing, skipped")
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
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [2.2, 1]})
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
    ax.set_title(f"a  Operators in three settings (pipelines {', '.join(pipes)})", loc="left")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
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
    ax.set_xticks(range(3), plab)
    ax.set_ylabel("Kendall tau of operator rankings")
    ax.set_title("b  Ranking agreement", loc="left")
    ax.grid(axis="x", visible=False)
    op_legend(fig, y=1.06)
    fig.tight_layout()
    save(fig, "figure6_cross_dataset")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--figures", nargs="*", type=int, default=[2, 3, 4, 5, 6])
    args = ap.parse_args()
    data = {}
    for ds in ("chbmit", "siena"):
        long, rm = dataset_runs(ds, args.device)
        if long is not None:
            data[ds] = (long, rm)
            print(f"{NAMES[ds]}: {rm.groupby(['pipeline', 'seed']).ngroups} complete runs")
    fns = {2: figure2, 3: figure3, 4: figure4, 5: figure5}
    for f in args.figures:
        if f == 6:
            figure6(data, args.device)
        else:
            fns[f](data)


if __name__ == "__main__":
    main()
