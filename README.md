# Do Fusion Operators Matter? EEG Seizure Detection Benchmark

A parameter matched, statistically controlled comparison of multimodal CNN fusion
operators (early, late, gated, attention, score level) that combine the raw EEG signal
with its spectrogram, evaluated on two corpora: Bonn (Andrzejak et al. 2001) and CHB-MIT
(Shoeb 2009).

**Status:** working draft under advisor review. The manuscript is in
`paper/manuscript.md`; its Limitations section lists what is still open.

## Research question

With the backbone, training protocol, and parameter budget held fixed, does the choice of
fusion operator make a difference that survives a statistically valid test?

## Findings so far

- **Bonn:** late, gated, attention, and score fusion are not distinguishable under a
  corrected test in any of three task formulations, and are positively shown to be
  equivalent within about 0.02 to 0.03 macro F1. Early fusion is the one operator that
  departs, downward.
- **CHB-MIT (leave one subject out):** no operator pair differs either, but between subject
  variance is too large to establish equivalence. This is reported as a different kind of
  negative result from Bonn's.
- **Evaluation practice:** the naive paired test common in this literature has a 38
  percent false positive rate under repeated cross validation (simulation), and CPU thread
  count alone moves per fold results by amounts comparable to the operator differences.

## Documentation

| Document | Contents |
|---|---|
| [`docs/PIPELINE.md`](docs/PIPELINE.md) | **Start here.** End to end pipeline and methodology: data, preprocessing, models, parameter matching, training, evaluation, statistics, reproducibility, how to reproduce each result, known issues |
| [`paper/manuscript.md`](paper/manuscript.md) | Manuscript draft |
| [`paper/proposal.md`](paper/proposal.md) | Research proposal |
| `docs/09_LITERATUR_TARAMASI.md` | Scoping literature review (Turkish) |
| `docs/10_OLASILIK_VE_TEKRARLANABILIRLIK.md` | Thread count nondeterminism investigation (Turkish) |
| `docs/11_CHBMIT_PLANI.md` | CHB-MIT download, verification, and pipeline notes (Turkish) |
| `docs/13_CHBMIT_ISTATISTIK.md` | CHB-MIT statistical analysis notes (Turkish) |
| `docs/14_SISTEMATIK_TARAMA_PROTOKOLU.md` | Systematic literature search protocol (Turkish) |

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; on Linux or macOS: source .venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

All experiments were run on CPU (Python 3.12.7, Windows 11).

## Data

Raw data is not included. Place it under `data/`:

- **Bonn** (~19 MB): sets A to E as `data/A` ... `data/E`, from the University of Bonn
  epilepsy EEG time series collection (Andrzejak et al. 2001).
- **CHB-MIT** (~43 GB): from PhysioNet (https://physionet.org/content/chbmit/1.0.0/) as
  `data/chbmit/chbNN/`. Verify with `python -m src.verify_chbmit`.

All per fold results are stored in `results_v2/`, so the statistics and figures can be
regenerated without the raw data:

```bash
python -m src.phase0                  # Bonn statistics
python -m src.phase0_probscores       # Bonn log loss and Brier statistics
python -m src.chbmit_stats            # CHB-MIT statistics
python -m src.make_manuscript_figures # manuscript figures
```

Full training commands for every result are in `docs/PIPELINE.md`, Section 8.

## Repository layout

| Path | Contents |
|---|---|
| `src/config.py` | Configuration, task and normalisation definitions, per run hashing, seeds |
| `src/data.py`, `src/chbmit.py` | Bonn loading and spectrograms; CHB-MIT EDF reader and annotation parser |
| `src/chbmit_corpus.py`, `src/chbmit_prep.py` | CHB-MIT windowing, subsampling, leakage safe splits and checks, preprocessing |
| `src/models.py` | Backbones and fusion operators, parameter matched |
| `src/train.py`, `src/evaluate.py` | Training loop, early stopping, score fusion weight, metrics |
| `src/baselines.py`, `src/chbmit_baselines.py` | Classical logistic regression baselines |
| `src/run_benchmark.py`, `src/run_sensitivity.py` | Bonn experiment drivers |
| `src/chbmit_run.py`, `src/chbmit_score.py` | CHB-MIT experiment drivers |
| `src/stats.py`, `src/sim_cv_correlation.py` | Corrected tests, equivalence, Bayesian analysis; false positive simulation |
| `src/phase0.py`, `src/phase0_probscores.py`, `src/chbmit_stats.py` | Statistical analysis of stored results |
| `src/figures.py`, `src/make_manuscript_figures.py` | Figures |
| `src/verify_chbmit.py` | CHB-MIT file integrity check |
| `results_v2/` | Per fold results, named by configuration hash |
| `paper/` | Manuscript, proposal, figures |
