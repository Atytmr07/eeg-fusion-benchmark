# Do Fusion Operators Matter? EEG Seizure Detection Benchmark

A parameter-matched, statistically controlled comparison of multimodal CNN fusion
operators (early, late, gated, attention and score fusion) that combine the raw EEG
signal with its spectrogram, run as a multiverse: nine preprocessing pipelines (including
three artefact removal algorithms), five seed sets, leave-one-subject-out evaluation on
CHB-MIT (Shoeb 2009) and Siena (Detti et al. 2020), and training on CHB-MIT with testing
on Siena. An earlier stage of the project used the Bonn dataset (Andrzejak et al. 2001).

**Status:** working draft under advisor review. The manuscript is
`paper/manuscript_v2.md`; the literature review in it is provisional.

## Research questions

1. Are differences between fusion operators preserved across preprocessing pipelines?
2. How stable are the rankings of the operators across seeds and pipelines?
3. Do the results carry over from one dataset to another?

## Findings so far

- **Operators:** no pair of operators differs significantly under a corrected test in any
  pipeline of either dataset. The held-out subject explains most of the variance (74
  percent of macro F1 on CHB-MIT); preprocessing and the operator explain almost none.
- **Rankings:** on CHB-MIT the order of the operators changes with the seed (median
  Kendall's W 0.22); on Siena it is more stable, but it does not transfer between datasets.
- **Controls:** a single-branch spectrogram network of the same size and a logistic
  regression on seven hand-crafted features reach mean scores similar to or higher than
  the fusion operators; none of these differences is significant either.
- **Reproducibility:** with the same code, package versions and seeds, ICA-based artefact
  removal (GEDAI, and Infomax less often) cleaned some recordings differently on another
  machine (`docs/ICA_REPRO.md`).
- **Evaluation practice:** the naive paired test common in this literature has a 38
  percent false positive rate under repeated cross-validation (simulation).

## Documentation

| Document | Contents |
|---|---|
| [`docs/PIPELINE.md`](docs/PIPELINE.md) | **Start here.** End to end pipeline and methodology: data, preprocessing, models, parameter matching, training, evaluation, statistics, reproducibility, how to reproduce each result, known issues |
| [`paper/manuscript_v2.md`](paper/manuscript_v2.md) | Manuscript draft (multiverse study); `paper/manuscript.md` is the earlier draft |
| [`paper/proposal.md`](paper/proposal.md) | Research proposal |
| [`docs/SIENA.md`](docs/SIENA.md) | Siena Scalp EEG: download, harmonisation to the CHB-MIT form, the six seizure-list decisions |
| [`docs/MULTIVERSE.md`](docs/MULTIVERSE.md) | Multiverse analysis: every statistic, its outputs and its synthetic validation |
| [`docs/EVENTS.md`](docs/EVENTS.md) | Seizure-level metrics, fixed before the multiverse runs |
| [`docs/ICA_REPRO.md`](docs/ICA_REPRO.md) | Reproducibility of artefact removal across machines and execution contexts |
| [`docs/RESULTS.md`](docs/RESULTS.md) | The results generator (`src/make_results.py`) |
| [`results_v2/audit/methods_audit.md`](results_v2/audit/methods_audit.md) | Methods audit: counts, partitions, statistics and reproducibility numbers checked against the code and the stored outputs |
| [`results_v2/sensitivity_preds/summary.md`](results_v2/sensitivity_preds/summary.md) | Post hoc sensitivity analyses: decision threshold, natural prevalence, subject-level bootstrap, seizure-level thresholds |
| `docs/09_LITERATUR_TARAMASI.md` | Scoping literature review (Turkish) |
| `docs/10_OLASILIK_VE_TEKRARLANABILIRLIK.md` | Thread count nondeterminism investigation (Turkish) |
| `docs/11_CHBMIT_PLANI.md` | CHB-MIT download, verification, and pipeline notes (Turkish) |
| `docs/13_CHBMIT_ISTATISTIK.md` | CHB-MIT statistical analysis notes (Turkish) |
| `docs/14_SISTEMATIK_TARAMA_PROTOKOLU.md` | Systematic literature search protocol (Turkish) |
| `docs/15_LITERATUR_SONUCLARI.md` | Systematic literature search results (Turkish) |

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; on Linux or macOS: source .venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

This environment runs the statistics, tables and figures. The multiverse runs used the
CUDA build of the same PyTorch version in a separate environment (`--device cuda`), and
the P6 artefact removal needs a newer numpy, so it has its own environment as well
(`requirements-ica.txt`, which explains the setup). The machines, operating systems and
package versions of every step are listed in Table 2 of the manuscript and recorded in
each run's `meta.json`.

## Data

Raw data is not included. Place it under `data/`:

- **Bonn** (~19 MB): sets A to E as `data/A` ... `data/E`, from the University of Bonn
  epilepsy EEG time series collection (Andrzejak et al. 2001).
- **CHB-MIT** (~43 GB): from PhysioNet (https://physionet.org/content/chbmit/1.0.0/) as
  `data/chbmit/chbNN/`. Verify with `python -m src.verify_chbmit`.
- **Siena Scalp EEG** (~20 GB): from PhysioNet
  (https://physionet.org/content/siena-scalp-eeg/1.0.0/) as `data/siena/PNxx/`;
  `python -m src.siena_download` downloads it and `python -m src.verify_siena` checks it
  (`docs/SIENA.md`).

All per fold results are stored in `results_v2/`, so the statistics and figures can be
regenerated without the raw data:

```bash
python -m src.phase0                  # Bonn statistics
python -m src.phase0_probscores       # Bonn log loss and Brier statistics
python -m src.chbmit_stats            # CHB-MIT statistics
python -m src.make_manuscript_figures # manuscript figures
```

Full training commands for every result are in `docs/PIPELINE.md`, Section 8.

## Reproducing the manuscript

Every number, table and figure of `paper/manuscript_v2.md` comes from stored runs and is
regenerated by the commands below (each script's docstring has the details). The tables
and figures need neither the EEG data nor a GPU, and can also be regenerated in the
browser:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Atytmr07/eeg-fusion-benchmark/blob/main/notebooks/reproduce.ipynb)

| Output | Command |
|---|---|
| All training runs and event metrics, in order | `python -m src.run_all --device cuda` |
| Event metrics of one pipeline | `python -m src.events --dataset chbmit --pipeline P1` |
| Multiverse statistics (variance components, rankings, equivalence) | `python -m src.multiverse --dataset chbmit --device cuda` (and `--dataset siena`) |
| Main tables (Table 4, event tables, cross-dataset tables) | `python -m src.make_results` |
| Controls against the fusion operators (Figure 2c, f) | `python -m src.controls_stats` |
| Figures 2 to 8 | `python -m src.make_figures --device cuda` |
| Supplementary Tables S1 to S8 | `python -m src.make_supplementary` |
| What artefact removal removes (Table S5) | `python -m src.p6_signal_effects` |
| Reproducibility of artefact removal across machines | `python -m src.ica_repro`, `python -m src.ica_sensitivity`, `python -m src.ica_repro_impact` (`docs/ICA_REPRO.md`) |
| PDF and Overleaf bundle | `python paper/latex/build.py` |

Three further steps read the corpus caches (`data/*/_cache/`), which are built from the
raw EDF files, and so cannot run from the repository alone:

| Output | Command |
|---|---|
| Methods audit: dataset, window and seizure counts (198, 185 and 187 seizures), identical windows in every pipeline, disjoint LOSO partitions, the statistics against independent re-implementations, the device check (`results_v2/audit/`) | `python -m src.audit_methods` |
| Post hoc sensitivity analyses from the stored predictions: thresholds 0.3, 0.5 and 0.7, CHB-MIT at the natural prevalence, subject-level bootstrap (`results_v2/sensitivity_preds/`) | `python -m src.sensitivity_preds` |
| Seizure-level metrics at several thresholds (each run's `events_thresholds/`; run before `sensitivity_preds`) | `python -m src.events --dataset chbmit --pipeline P1 --runs loso_P1_cuda loso_P1_r1_cuda ... --thresholds 0.3 0.5 0.7` |

## Repository layout

| Path | Contents |
|---|---|
| `src/config.py` | Configuration, task and normalisation definitions, per run hashing, seeds |
| `src/data.py`, `src/chbmit.py` | Bonn loading and spectrograms; CHB-MIT EDF reader and annotation parser |
| `src/chbmit_corpus.py`, `src/chbmit_prep.py` | CHB-MIT windowing, subsampling, leakage safe splits and checks, preprocessing |
| `src/models.py` | Backbones and fusion operators, parameter matched (`python -m src.models` prints the counts) |
| `src/train.py`, `src/evaluate.py` | Bonn training loop, score fusion weight; metrics shared by both corpora |
| `src/baselines.py`, `src/chbmit_baselines.py` | Classical logistic regression baselines |
| `src/run_benchmark.py`, `src/run_sensitivity.py` | Bonn experiment drivers |
| `src/chbmit_run.py` | CHB-MIT experiment driver (training loop, deep models, score fusion) |
| `src/stats.py`, `src/sim_cv_correlation.py` | Corrected tests, equivalence, Bayesian analysis; false positive simulation |
| `src/phase0.py`, `src/phase0_probscores.py`, `src/chbmit_stats.py` | Statistical analysis of stored results |
| `src/figures.py`, `src/make_manuscript_figures.py` | Figures (Bonn and the earlier draft) |
| `src/verify_chbmit.py` | CHB-MIT file integrity check |
| `src/siena.py`, `src/siena_corpus.py`, `src/siena_qc.py`, `src/siena_durations.py`, `src/siena_download.py`, `src/verify_siena.py` | Siena: seizure lists and their decisions, conversion to the CHB-MIT form, corpus, checks, download |
| `src/datasets.py` | Corpus builder and fold count per dataset |
| `src/preprocess.py` | The nine preprocessing pipelines P0 to P6c |
| `src/run_all.py`, `src/run_queue.py` | All multiverse runs in order; a queue of pipeline and seed runs |
| `src/events.py` | Seizure-level metrics (SzCORE), also at several thresholds |
| `src/cross_dataset.py` | Training on CHB-MIT, testing on Siena |
| `src/multiverse.py` | Multiverse statistics: variance components, rankings, equivalence, RM-ANOVA |
| `src/controls_stats.py` | Controls against the fusion operators |
| `src/make_results.py`, `src/make_figures.py`, `src/make_figure1.py`, `src/make_supplementary.py` | Results tables, Figures 1 to 8, Supplementary tables |
| `src/ica_repro.py`, `src/ica_sensitivity.py`, `src/ica_repro_impact.py`, `src/gedai_qc.py`, `src/p6_signal_effects.py` | Artefact removal: reproducibility, perturbation test, impact on the results, GEDAI preset, signal effects |
| `src/compare_runs.py` | Are two runs identical (repeatability, CPU against GPU)? |
| `src/audit_methods.py` | Methods audit against the code and the stored outputs |
| `src/sensitivity_preds.py` | Post hoc sensitivity analyses from the stored predictions |
| `results_v2/` | Per fold results, named by configuration hash |
| `paper/` | Manuscript, proposal, figures |
