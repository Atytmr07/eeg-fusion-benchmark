"""Everything the lab PC runs after the device decision, in order, as one command.

  1. CHB-MIT queue      src.run_queue: every pipeline x seed run, baselines, statistics
  2. CHB-MIT events     src.events per pipeline (all seeds in one pass)
  3. Siena queue        src.run_queue --dataset siena
  4. Siena events       src.events --dataset siena per pipeline
  5. Cross-dataset      src.cross_dataset per pipeline and seed (trained on CHB-MIT,
                        tested on Siena), then src.events --cross
  6. Reports            src.make_results, src.multiverse --first-report per dataset

Every step is skipped when its output exists, so the same command can be started again
after an interruption and only does what is missing. A failing step is logged and the
steps that do not depend on it still run; the summary at the end lists what failed.

P6 needs the ICA packages for the event metrics (the decomposition is recomputed on
every recording): those steps run in a separate environment (--python-ica, see
requirements-ica.txt), on the CPU, with the run folders named explicitly. Everything
else runs in the environment that started this command.

Parallel CPU use: with --worker 2/2 a second window only takes its half of the CHB-MIT
queue and stops; --worker 1/2 takes the other half, waits until every CHB-MIT run is
complete, and then runs steps 2 to 6.

Usage:
    python -m src.run_all --device cuda                         # one window
    python -m src.run_all --worker 1/2   and   --worker 2/2      # two CPU windows
    python -m src.run_all --device cuda --dry-run                # only show the state
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .config import PROJECT_ROOT, RESULTS_ROOT
from .preprocess import PIPELINES
from .run_queue import status, tag_for

ALL = ["P0", "P1", "P2", "P3", "P4", "P5", "P6a", "P6b", "P6c"]
LOG = RESULTS_ROOT / "run_all_log.txt"


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def cross_tag(p: str, r: int, device: str) -> str:
    return f"chbmit_to_siena_{p}" + (f"_r{r}" if r else "") + ("_cuda" if device == "cuda" else "")


class Runner:
    def __init__(self, args):
        self.args, self.failed = args, []

    def python_for(self, pipeline: str) -> str:
        return self.args.python_ica if PIPELINES[pipeline].ica else sys.executable

    def run(self, what: str, cmd: list[str], python: str | None = None) -> bool:
        exe = python or sys.executable
        log(f"start  {what}: {' '.join(cmd)}" + ("" if exe == sys.executable else f"  [{exe}]"))
        if self.args.dry_run:
            return True
        t = time.time()
        r = subprocess.run([exe, "-m", *cmd], cwd=PROJECT_ROOT)
        ok = r.returncode == 0
        log(f"{'done  ' if ok else 'FAILED'} {what} ({(time.time() - t) / 60:.0f} min)"
            + ("" if ok else f", exit code {r.returncode}"))
        if not ok:
            self.failed.append(what)
        return ok


def queue_complete(dataset: str, args) -> tuple[int, int]:
    tags = [tag_for(p, r, args.device, dataset) for r in args.repeats for p in args.pipelines]
    return sum(all(status(t, dataset)) for t in tags), len(tags)


def events_step(R: Runner, dataset: str, args) -> None:
    """Event metrics of every pipeline whose seed sets are all complete."""
    for p in args.pipelines:
        tags = [tag_for(p, r, args.device, dataset) for r in args.repeats]
        if not all(all(status(t, dataset)) for t in tags):
            log(f"skip   {dataset} events {p}: runs not complete")
            continue
        if all((RESULTS_ROOT / dataset / t / "events" / "meta.json").exists() for t in tags):
            continue
        ica = bool(PIPELINES[p].ica)
        R.run(f"{dataset} events {p}",
              ["src.events", "--dataset", dataset, "--pipeline", p, "--runs", *tags,
               "--device", "cpu" if ica else args.device, "--threads", str(args.threads),
               "--jobs", str(args.jobs)], R.python_for(p))


def check_ica_env(args) -> bool:
    """The P6 event steps need the ICA environment; check it before anything starts."""
    if not any(PIPELINES[p].ica for p in args.pipelines + args.cross_pipelines):
        return True
    exe = Path(args.python_ica)
    if not exe.exists():
        log(f"ICA environment not found: {exe}. Install it (requirements-ica.txt) or give "
            f"--python-ica; without it the P6 event metrics fail")
        return False
    r = subprocess.run([str(exe), "-c", "import mne, gedai, jamica, torch, timescoring, sklearn"],
                       cwd=PROJECT_ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"ICA environment incomplete: {r.stderr.strip().splitlines()[-1]}")
        return False
    log(f"ICA environment ok: {exe}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--pipelines", nargs="+", default=ALL, choices=list(PIPELINES))
    ap.add_argument("--repeats", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--cross-pipelines", nargs="*", default=["P0", "P1"],
                    choices=list(PIPELINES))
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--jobs", type=int, default=16,
                    help="worker processes of the event metrics (reading recordings)")
    ap.add_argument("--python-ica", default=str(PROJECT_ROOT / ".venv-ica" / "Scripts" /
                                                "python.exe"))
    ap.add_argument("--worker", default="1/1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    k, n = (int(x) for x in args.worker.split("/"))
    R = Runner(args)
    log(f"run_all: device {args.device}, worker {k}/{n}, pipelines {' '.join(args.pipelines)}, "
        f"seeds {args.repeats}, cross {' '.join(args.cross_pipelines) or 'none'}"
        + (" (dry run)" if args.dry_run else ""))
    if k == 1 and not check_ica_env(args) and not args.dry_run:
        raise SystemExit("fix the ICA environment first (the queue has not started)")

    q = ["--pipelines", *args.pipelines, "--repeats", *map(str, args.repeats),
         "--threads", str(args.threads), "--device", args.device]

    # 1. CHB-MIT queue
    done, total = queue_complete("chbmit", args)
    log(f"CHB-MIT queue: {done}/{total} runs complete")
    if done < total:
        R.run("CHB-MIT queue", ["src.run_queue", "--dataset", "chbmit", *q,
                                "--worker", args.worker])
    if k != 1:
        log(f"worker {k}/{n}: my part of the CHB-MIT queue is finished; worker 1 does the rest")
        return
    while not args.dry_run and queue_complete("chbmit", args)[0] < total:
        if "CHB-MIT queue" in R.failed:      # my own part failed: it will never complete
            break
        log(f"waiting for the other worker(s): {queue_complete('chbmit', args)[0]}/{total} "
            f"CHB-MIT runs complete")
        time.sleep(600)

    # 2. CHB-MIT events
    events_step(R, "chbmit", args)

    # 3. Siena queue and 4. Siena events
    done, total = queue_complete("siena", args)
    log(f"Siena queue: {done}/{total} runs complete")
    if done < total:
        R.run("Siena queue", ["src.run_queue", "--dataset", "siena", *q])
    events_step(R, "siena", args)

    # 5. Cross-dataset
    for p in args.cross_pipelines:
        tags = []
        for r in args.repeats:
            t = cross_tag(p, r, args.device)
            d = RESULTS_ROOT / "cross" / t
            if not ((d / "meta.json").exists() and (d / "models").exists()):
                if not R.run(f"cross {t}", ["src.cross_dataset", "--pipeline", p, "--repeat",
                                            str(r), "--device", args.device,
                                            "--threads", str(args.threads)]):
                    continue
            tags.append(t)
        todo = [t for t in tags if not (RESULTS_ROOT / "cross" / t / "events" /
                                        "meta.json").exists()]
        if todo:
            ica = bool(PIPELINES[p].ica)
            R.run(f"cross events {p}",
                  ["src.events", "--cross", *todo, "--device", "cpu" if ica else args.device,
                   "--threads", str(args.threads), "--jobs", str(args.jobs)],
                  R.python_for(p))

    # 6. Reports
    R.run("results tables", ["src.make_results", "--device", args.device])
    for ds in ("chbmit", "siena"):
        R.run(f"{ds} first report", ["src.multiverse", "--dataset", ds, "--first-report",
                                     "--device", args.device])

    log("run_all finished" + (f"; FAILED: {', '.join(R.failed)} (start the same command "
                               f"again to retry them)" if R.failed else ", nothing failed"))


if __name__ == "__main__":
    main()
