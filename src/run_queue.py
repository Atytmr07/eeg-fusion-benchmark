"""Run a queue of CHB-MIT pipeline x seed runs, each followed by baselines and statistics.

For every (repeat, pipeline) pair, in that order so that each seed covers all
pipelines before the next seed starts:

  1. src.chbmit_run      unless all 23 folds of the deep models are present
                         (continues an interrupted run with --resume)
  2. src.chbmit_baselines  unless the classical baselines are present
  3. src.chbmit_stats    unless its output folder exists

so the same command can be started again after an interruption and only does what is
missing. P0 with repeat 0 on the CPU is the existing loso_grouped run.

Parallel use: with --worker k/n, this process takes every n-th item starting at k, so
two windows started with --worker 1/2 and --worker 2/2 share the queue without
overlap. Results do not depend on how many runs share the machine, only on the
thread count, which must stay the same for all runs of one analysis.

Usage:
    python -m src.run_queue --pipelines P0 P1 P2 P3 P4 P5 --repeats 0 1 2 --threads 16
    python -m src.run_queue ... --worker 1/2        # and --worker 2/2 in a second window
    python -m src.run_queue ... --dry-run           # only list what would run
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .chbmit_run import DEFAULT_MODELS, run_tag
from .config import RESULTS_ROOT
from .preprocess import PIPELINES

N_FOLDS = 23
DEEP = [m for m in DEFAULT_MODELS]


def tag_for(pipeline: str, repeat: int, device: str) -> str:
    if pipeline == "P0" and repeat == 0 and device == "cpu":
        return "loso_grouped"
    return run_tag("loso", pipeline, repeat, device)


def status(tag: str) -> tuple[bool, bool, bool]:
    """(deep models complete, baselines present, statistics present)."""
    csv = RESULTS_ROOT / "chbmit" / tag / "perfold.csv"
    if not csv.exists():
        return False, False, False
    df = pd.read_csv(csv)
    have = df.groupby("model")["fold"].nunique()
    deep = all(have.get(m, 0) == N_FOLDS for m in DEEP)
    base = all(have.get(m, 0) == N_FOLDS for m in ("logvar", "shallow"))
    stats_dir = RESULTS_ROOT / "chbmit" / ("phase0" if tag == "loso_main" else f"phase0_{tag}")
    return deep, base, (stats_dir / "summary.csv").exists()


def run(cmd: list[str]) -> None:
    print("\n>>> " + " ".join(cmd), flush=True)
    r = subprocess.run([sys.executable, "-m", *cmd])
    if r.returncode != 0:
        raise SystemExit(f"stopped: '{' '.join(cmd)}' exited with code {r.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipelines", nargs="+", default=["P0", "P1", "P2", "P3", "P4", "P5"],
                    choices=list(PIPELINES))
    ap.add_argument("--repeats", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--worker", default="1/1", help="k/n: take every n-th item from k")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    k, n = (int(x) for x in args.worker.split("/"))

    items = [(r, p) for r in args.repeats for p in args.pipelines]
    mine = items[k - 1::n]
    print(f"worker {k}/{n}: {len(mine)} of {len(items)} runs, device {args.device}, "
          f"{args.threads} threads")
    for r, p in mine:
        tag = tag_for(p, r, args.device)
        deep, base, stats = status(tag)
        state = "done" if deep and base and stats else (
            "partial" if (RESULTS_ROOT / "chbmit" / tag / "perfold.csv").exists() else "to do")
        print(f"  {tag:22s} pipeline {p} seed set {r}: {state}")
    if args.dry_run:
        return

    for r, p in mine:
        tag = tag_for(p, r, args.device)
        deep, base, stats = status(tag)
        if not deep:
            cmd = ["src.chbmit_run", "--split", "loso", "--threads", str(args.threads),
                   "--pipeline", p, "--repeat", str(r), "--device", args.device,
                   "--tag", tag]
            if (RESULTS_ROOT / "chbmit" / tag / "perfold.csv").exists():
                cmd.append("--resume")
            run(cmd)
        if not base:
            run(["src.chbmit_baselines", "--run", tag])
        if not stats:
            run(["src.chbmit_stats", "--run", tag])
    print("\nqueue finished")


if __name__ == "__main__":
    main()
