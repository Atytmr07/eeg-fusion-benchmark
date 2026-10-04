# Results generator (`src/make_results.py`)

When the runs are finished, one command fills the results section's tables and
figures:

```bash
python -m src.make_results                     # both datasets and the cross experiment
python -m src.make_results --datasets chbmit   # one dataset
python -m src.make_results --synthetic         # self-test on synthetic runs
```

It reads only stored results. The command runs at any time and writes what exists so
far; runs that are incomplete or still missing are listed in the report.

## What it writes

The output goes to `results_v2/results/results.md`. Each table is in Markdown, ready
to paste into `paper/manuscript.md`, with its caption and source. The figures go to
`paper/figures_manuscript/`.

| Item | Content | Source | Files |
|---|---|---|---|
| Main table, per dataset | Pipeline x fusion operator: window-level macro F1 and AUPRC, mean ± SD over the seed sets. Each run is first averaged over its LOSO folds (23 CHB-MIT, 14 Siena). The best operator of each pipeline is in bold | `results_v2/<dataset>/loso_*/perfold.csv`; AUPRC from `preds/` (`src/multiverse.py: load_long`) | `main_<dataset>_f1_macro.csv`, `main_<dataset>_auprc.csv` |
| Event table | CHB-MIT: seizure-level sensitivity and false alarms per hour. Siena: sensitivity only, with the limitation note (docs/EVENTS.md). Both are pooled over persons, mean ± SD over seed sets | `<run>/events/perperson.csv` (the summary there is rounded to four decimals) | `events_<dataset>.csv`, `events_<dataset>_per_run.csv` |
| Rank figure, per dataset | Rank of each fusion operator by macro F1 in every pipeline x seed run: counts per rank and the rank map (`src/multiverse.py: rank_distribution`) | as the main table | `results_rank_distribution_<dataset>.png/.pdf`, `rank_distribution_<dataset>.csv`, `rank_summary_<dataset>.csv` |
| Cross table, per pipeline | CHB-MIT -> Siena, per Siena patient: macro F1 and AUPRC (mean ± SD over seed sets, with a mean row), and detected out of annotated seizures | `results_v2/cross/chbmit_to_siena_<P>[_r<k>]/`: `perfold.csv`, `preds.npz`, `events/perperson.csv` | `cross_<P>_f1_macro.csv`, `cross_<P>_auprc.csv`, `cross_<P>_events.csv` |

**Run selection.** A run enters the tables only when every LOSO fold has all five
fusion operators. CPU and GPU runs are kept apart, as in the multiverse analysis
(`--device cuda`).

**Table numbers.** These are placeholders (R1, R2, ...), to be matched to the
manuscript's own numbering.

## Self-test (`--synthetic`)

**How the synthetic runs are built.** They follow the formats the real runs write:

- CHB-MIT and Siena have 9 pipelines x 3 seeds each, with 23 and 14 folds.
- `perfold.csv` has every column `chbmit_run` writes, computed with `src/evaluate.py`.
- The `preds/fold<k>.npz` files hold `idx_te`, `y_te` and the (N, 2) probabilities.
- `events/` is written by `src/events.py`'s own writer, including the renamed Siena
  columns.
- Three cross runs have `perfold.csv` per patient, a `preds.npz` with the subject
  array, and `meta.json`.
- The number of annotated seizures is fixed per person, as in the data.

**What is checked.** Every table is compared with values computed directly from the
generated arrays. CHB-MIT P6c seed 2 is cut to 10 folds and must be left out. The
multiverse analysis is also run on the 14-fold Siena runs.

Result on 4 October 2026: **14/14 checks passed.**

| Check | Result |
|---|---|
| Main table, macro F1 and AUPRC, both datasets (4 checks) | 45 cells each, maximum error 3e-16 |
| Event sensitivity, both datasets; CHB-MIT false alarms per hour | maximum error 2e-16 |
| Incomplete run left out and listed | `loso_P6c_r2` (10/23 folds); P6c with 2 seeds; 26 runs |
| Siena: no false alarm table or column | sensitivity only, limitation note present |
| Rank figure and counts, both datasets | 130 and 135 ranks; PNG and PDF written |
| Cross table | 14 Siena patients; macro F1 exact; detected seizures exact and never above the annotated ones |
| `multiverse --dataset siena` with 14 folds | balanced block 9 pipelines x 3 seeds x 14 folds; corrected tests with test/train ratio 1/13 |
| `multiverse --first-report` on Siena | 27 runs, 14 folds |

## On the real data (smoke test)

On this machine, only CHB-MIT P0 seed 0 (`loso_grouped`) exists. The command writes
its main tables and rank figure, and lists the missing parts:

- event metrics not yet computed,
- no Siena run,
- no cross-dataset run.

Its macro F1 values agree with Table 3 of the manuscript (attention 0.668, late
0.649).
