# Pipeline and Methodology

This document describes, end to end, how every number in `paper/manuscript.md` is
produced: data, preprocessing, models, training, evaluation, and statistical analysis.
Every setting below is taken from the code, not from memory, and each section names the
file that implements it. Open issues are listed in Section 9 rather than left implicit.

## 0. Overview

```mermaid
flowchart LR
    subgraph Data
        B[Bonn EEG<br/>500 segments, 1 channel]
        C[CHB-MIT<br/>24 cases, 683 recordings]
    end
    B --> PB[z-score per segment]
    C --> W[10 s windows, 18 bipolar channels<br/>4:1 non-ictal subsampling]
    W --> PC[z-score per window and channel]
    PB --> R1[raw signal]
    PB --> S1[log STFT magnitude]
    PC --> R2[raw signal]
    PC --> S2[log STFT magnitude]
    R1 & S1 & R2 & S2 --> M[11 models<br/>parameter matched]
    M --> E1[Bonn: 5x5 repeated<br/>stratified CV]
    M --> E2[CHB-MIT: leave one<br/>subject out, 24 folds]
    E1 & E2 --> PF[perfold.csv<br/>one row per model and fold]
    PF --> ST[corrected paired tests, Holm,<br/>equivalence, Bayesian ROPE]
    ST --> OUT[tables and figures]
```

The research question is narrow on purpose: with the backbone, training protocol, and
parameter budget held fixed, does the choice of fusion operator make a difference that
survives a statistically valid test? No new architecture is proposed.

## 1. Data

### 1.1 Bonn (Andrzejak et al. 2001)

| Property | Value |
|---|---|
| Sets | A, B: scalp EEG, healthy volunteers. C, D: intracranial, interictal. E: intracranial, ictal |
| Size | 100 segments per set, 500 total |
| Segment | 4097 samples, 23.6 s, single channel |
| Sampling rate | 173.61 Hz |
| Acquisition filter | 0.53 to 40 Hz band pass, already applied in the published data |
| Subject identifiers | None published |

Three task formulations (`src/config.py: TASKS`):

| Task | Classes | Purpose |
|---|---|---|
| T1 | A+B vs C+D vs E (3 classes) | Main benchmark |
| T2 | C+D vs E | Intracranial only, removes the scalp versus intracranial recording confound |
| T3 | A+B+C+D vs E | The binary split most used in the literature; kept as a reference because a single feature logistic regression already reaches 0.954 macro F1 on it |

Because Bonn publishes no subject identifiers, cross validation on Bonn is necessarily
segment wise, and no Bonn result speaks to generalisation across patients. This is why
CHB-MIT is used as an external validation corpus.

### 1.2 CHB-MIT (Shoeb 2009; Goldberger et al. 2000, PhysioNet)

| Property | Value |
|---|---|
| Cases | 24 case folders, chb01 to chb24. At most 23, and by PhysioNet's own documentation at least 22, distinct individuals: chb21 is the same subject as chb01, recorded 1.5 years later (Section 9) |
| Recordings | 686 EDF files, 982.9 hours, 198 annotated seizures |
| Sampling rate | 256 Hz |
| Montage used | Fixed 18 channel bipolar montage (`src/chbmit.py: TARGET_CHANNELS`): FP1-F7, F7-T7, T7-P7, P7-O1, FP1-F3, F3-C3, C3-P3, P3-O1, FP2-F4, F4-C4, C4-P4, P4-O2, FP2-F8, F8-T8, T8-P8, P8-O2, FZ-CZ, CZ-PZ |
| Retained | 683 of 686 recordings, 185 of 198 seizures. The dropped ones are all from part of chb12, which uses a different reference montage sharing no channel with the target |

Data integrity and loading (`src/chbmit.py`, `src/verify_chbmit.py`):

- Files are verified against sizes computed from each EDF header, not against file
  listings.
- EDF is read by a small custom reader; no external EDF library is needed.
- Seizure annotations are parsed from each case's summary file **by sequence**, not by
  the printed seizure number, and each file's parsed count is checked against its
  declared "Number of Seizures". This is needed because `chb09-summary.txt` contains a
  labelling error (a seizure end line printed with the wrong number) that a number based
  parser turns into a nonexistent 6316 second seizure.
- PhysioNet's `RECORDS-WITH-SEIZURES` index names `chb07_18.edf` where the summary and
  annotation files agree it should be `chb07_19.edf`. The summary files are treated as
  ground truth.

Windowing and labels (`src/chbmit.py: plan_windows`, `src/chbmit_corpus.py: build_corpus`):

| Step | Setting |
|---|---|
| Window | 10 s (2560 samples), non overlapping |
| Label | Ictal if the window overlaps an annotated seizure at all. 948 of the 1,276 ictal windows lie entirely within a seizure; 328 (25.7 percent) are boundary windows with a median of 5.0 s of seizure. Excluding boundary windows is implemented (`guard_s`) but not yet run |
| Class balance | Ictal windows are 0.34 percent of all windows. Non ictal windows are subsampled per case to 4 per ictal window, seed 20260727 |
| Final corpus | 6,380 windows (1,276 ictal), 24 cases, 18 channels x 2560 samples |
| Grouping | Every window carries its case id and, if ictal, a persistent seizure id (for example `chb09_08.edf#2`) used by the leakage checks |

Labels are planned from EDF headers alone in a first pass, and only the selected
windows are read from disk in a second pass, so the 43 GB corpus never has to fit in
memory.

## 2. Preprocessing and model inputs

Each example is presented to the models in two representations of the same signal:
the raw waveform and its spectrogram.

| | Bonn (`src/data.py`) | CHB-MIT (`src/chbmit_prep.py`) |
|---|---|---|
| Signal normalisation | z-score per segment | z-score per window and per channel |
| Spectrogram | log(1 + \|STFT\|), computed from the z-scored signal | log(1 + \|STFT\|), computed from the z-scored signal |
| STFT | Hann window 256, hop 128, nfft 256 | Hann window 256, hop 128, nfft 256 |
| Frequency ceiling | 40 Hz (acquisition limit) | 64 Hz (the largest measured ictal to interictal power ratio is in the 30 to 70 Hz band) |
| Line noise | none needed | optional 60 Hz notch, off by default because it overlaps the gamma band |
| Raw input shape | 1 x 4097 | 18 x 2560 |
| Spectrogram shape | 1 x 59 x 31 (freq x time) | 18 x 65 x 19 |

The Bonn normalisation choice is itself ablated (`src/config.py: NORM_MODES`): N0 no
normalisation, N1 z-scored signal but spectrogram from the raw signal (the configuration
the original notebook implementation ran by accident), N2 both from the z-scored signal
(main), N3 additionally z-scoring the spectrogram.

## 3. Models (`src/models.py`)

### 3.1 Backbones

- **CNN1D** (raw signal): three blocks of Conv1d (kernel 7) + BatchNorm + ReLU + MaxPool(2),
  widths 16, 32, 64, then global average pooling and a linear layer to a 128 dimensional
  embedding.
- **CNN2D** (spectrogram): three blocks of Conv2d (3x3) + BatchNorm + ReLU + MaxPool(2x2),
  widths 16, 32, 64, then global average pooling and a linear layer to 128 dimensions.

For CHB-MIT only the first layer changes, to accept 18 input channels. With one input
channel every Bonn parameter count is identical to before this change; this was verified
directly.

### 3.2 The eleven models

| Model | What it is |
|---|---|
| raw1d | CNN1D + dropout 0.3 + linear classifier |
| spec2d | CNN2D + dropout 0.3 + linear classifier |
| raw1d_wide | raw1d with widths scaled up so its total parameter count matches the fusion models |
| spec2d_wide | same for spec2d |
| late | concatenate the two embeddings, then MLP, then classifier |
| gated | g = sigmoid(MLP([z1, z2])), fused z = g * z1 + (1 - g) * z2 (per dimension) |
| attention | (a1, a2) = softmax(MLP([z1, z2])), fused z = a1 * z1 + a2 * z2 (per modality) |
| early | the raw signal passes through a small strided Conv1d encoder, is aligned to the spectrogram's time axis, broadcast along frequency, and appended to the spectrogram as 8 extra channels; one 2D CNN then processes both. Strictly this is intermediate fusion, since a 1D signal and a 2D spectrogram cannot be concatenated at the raw input level, and the manuscript says so |
| score | raw1d and spec2d trained independently; their softmax outputs are averaged with weight w chosen on the validation set only (41 point grid from 0 to 1, minimising validation log loss; `src/train.py: best_blend_weight`) |
| logvar | logistic regression on the log variance of the signal (one feature per channel) |
| shallow | logistic regression on 7 features per channel: log variance, log line length, and relative power in 5 bands (0.5 to 4, 4 to 8, 8 to 13, 13 to 30, and 30 to 40 Hz on Bonn or 30 to 64 Hz on CHB-MIT) |

The two classical baselines use a standard scaler and `class_weight="balanced"`
(`src/baselines.py`, `src/chbmit_baselines.py`). They are there to show how much of the
task a hand engineered linear model already solves.

### 3.3 Parameter matching

A fusion model contains two branches, so it is naturally larger than either unimodal
model; if it wins, the win cannot be attributed to fusion rather than capacity. Every
jointly trained variant is therefore sized to a common budget: the hidden width of each
fusion head is solved in closed form per operator (`_solve_hidden`), and the widened
unimodal controls use calibrated width multipliers.

| Model | Bonn T1 (3 classes) | CHB-MIT (2 classes, 18 ch) |
|---|---|---|
| raw1d | 27,075 | 28,850 |
| spec2d | 32,227 | 34,546 |
| raw1d_wide | 99,795 | 103,712 |
| spec2d_wide | 97,775 | 102,236 |
| late | 98,571 | 102,768 |
| gated | 98,698 | 102,921 |
| attention | 98,544 | 102,767 |
| early | 97,507 | 102,547 |
| score | 59,302 (sum of raw1d and spec2d) | 63,396 |

Spread across the four jointly trained fusion operators: 1.2 percent on Bonn, 0.36
percent on CHB-MIT. Score fusion is reported at its natural size, because forcing it to
the common budget would mean shrinking its two constituent classifiers.

## 4. Training

| | Bonn (`src/train.py`) | CHB-MIT (`src/chbmit_run.py: train_fold`) |
|---|---|---|
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4 | AdamW, lr 1e-3, weight decay 1e-4 (the earlier `loso_main` run used Adam; see Section 9) |
| Loss | cross entropy, class weighted | cross entropy, class weighted |
| Batch size | 32 | 32 |
| Epochs | max 60, min 20 | max 60, min 20 |
| Early stopping | validation macro F1, patience 12, only after the minimum epochs | same |
| Checkpoint | weights from the best validation epoch | same |
| Seed | deterministic per (repeat, fold, model): SHA-1 of base seed 20260727 and the triple (`src/config.py: fold_seed`) | same, per (fold, model) |

The 20 epoch minimum is not cosmetic. Without it, early stopping could trigger while a
model was still predicting a single class for every example: validation F1 stayed flat,
patience ran out around epoch 11, and the collapsed weights were kept as "best". Those
runs showed AUC 0.96 with macro F1 0.40.

## 5. Evaluation protocol

### 5.1 Bonn

- 5 x 5 repeated stratified k-fold cross validation (`StratifiedKFold`, 5 folds, 5
  repeats), giving 25 test measurements per model.
- Inside each training fold, a stratified 20 percent validation split
  (`StratifiedShuffleSplit`) is used for early stopping and for the score fusion weight.
  The test fold is never used for any choice.

### 5.2 CHB-MIT

- Leave one subject out over **persons**, not case folders: chb01 and chb21 are the same
  person (`src/chbmit_corpus.py: SAME_SUBJECT`), so there are 23 folds and the fold that
  holds out chb01 also holds out chb21. The manuscript's CHB-MIT results come from this
  corrected run, `results_v2/chbmit/loso_grouped/`. The earlier run in
  `results_v2/chbmit/loso_main/` uses 24 case based folds and is kept for the comparison
  in the manuscript's Section 5.4 (Section 9).
- The inner validation split is **subject wise**: about 20 percent of the training persons
  (4 of 22) are held out entirely, chosen evenly along a list sorted by ictal count so the
  validation set always contains seizures (`inner_split`). A window wise inner split would
  put windows from the same patient in both training and validation and inflate the early
  stopping decision.
- Two leakage rules are enforced: no person's windows in both train and test, and no single
  seizure's windows split across train and test (`src/chbmit_corpus.py: check_leakage`).
  The checker is validated two ways: a deliberately leaky random split is flagged on both
  rules, and the old case based split is flagged on exactly the two folds (chb01, chb21)
  where the same person appeared on both sides (`python -m src.chbmit_corpus --check`).

### 5.3 Metrics (`src/evaluate.py`)

Macro F1 is the primary metric. Also recorded: accuracy, macro precision and recall,
AUC, log loss, Brier score, per class F1 and recall, and on CHB-MIT sensitivity,
specificity, and false alarm counts. Every model and fold produces one row in
`perfold.csv`; everything downstream is computed from those files.

## 6. Statistical analysis (`src/stats.py`, `src/phase0.py`, `src/chbmit_stats.py`)

### 6.1 Why the usual test is not valid here

Repeated cross validation measurements are not independent: the training sets of
different folds and repeats overlap heavily. The paired Wilcoxon or t test commonly used
in this literature assumes independence. A simulation (`src/sim_cv_correlation.py`) with
two statistically identical classifiers under 5 x 5 repeated CV over 1000 datasets found
false positive rates of 37.6 percent (Wilcoxon) and 38.6 percent (paired t test), against
a nominal 5 percent. The corrected test below rejected in 2.3 percent.

### 6.2 Tests used

| Question | Method |
|---|---|
| Is there a difference? | Corrected resampled paired t test (Nadeau and Bengio 2003; Bouckaert and Frank 2004): the squared standard error of the mean difference is s² (1/n + n_test/n_train) instead of s²/n. Holm correction across all pairs within a task and metric |
| Is the difference negligible? | Equivalence bound δmin: the smallest margin within which the pair is equivalent at alpha 0.05 (two one sided tests), computed with the same corrected variance |
| How probable is practical equivalence? | Correlated Bayesian t test (Corani and Benavoli 2015) with a region of practical equivalence of ±0.01 and ±0.02 macro F1: posterior probabilities that A is better, that they are practically equivalent, that B is better |
| Robustness check | Sign test, which does not assume symmetric differences |

### 6.3 The test to train ratio

The primary analysis uses the standard ratio 1/(k-1): 0.25 for Bonn (5 folds) and 1/23
for CHB-MIT LOSO. Because both protocols also remove a validation split from the training
data, every test is recomputed with a conservative ratio based on the data actually used
for fitting: 0.3125 for Bonn and 1/18 for CHB-MIT (column `p_corrected_conservative`). No
conclusion changes: the number of significant pairs is identical in all four analyses
(10 of 55 on Bonn T1; 0 of 55 on Bonn T2, T3, and CHB-MIT) and the equivalence bounds of
the fusion operator pairs widen by at most 0.005 macro F1.

### 6.4 How to read the results

"Not significantly different" and "equivalent" are separate claims and are reported
separately. On Bonn the fusion operators are both not different and positively shown to
be close. On CHB-MIT they are not different, but between subject variance is so large
that closeness cannot be shown either. The manuscript treats these as two different kinds
of negative result.

## 7. Reproducibility

- **Configuration hashing.** Every Bonn run writes to `results_v2/<task>__<norm>__<hash>/`,
  where the hash covers every configuration field, so results produced with different
  settings cannot overwrite each other. Each run also writes `config.json` and
  `env.json` (library versions, platform, effective thread count).
- **Thread count matters.** The number of CPU threads changes the floating point
  summation order in multithreaded convolution routines. At a fixed thread count results
  are bit for bit repeatable (verified with three identical runs); changing the thread
  count alone moves per fold macro F1 by up to 0.093 on Bonn and 0.208 in one CHB-MIT
  fold, comparable to the differences between operators. Thread count is therefore part
  of the run identity (`docs/10_OLASILIK_VE_TEKRARLANABILIRLIK.md`).
- **Numerical realisations in the manuscript.** Bonn Table 2 comes from the original runs
  at 2 threads. The Bonn log loss and Brier analysis comes from a 12 thread rerun that also
  stored predicted probabilities (`results_v2/rerun_probs/`). The CHB-MIT run
  (`loso_grouped`) was trained on one machine at 16 threads throughout (Python 3.12.10,
  torch 2.8.0+cpu, about 6.3 hours). In the earlier `loso_main` run, folds 0 to 4 were
  trained at 12 threads and folds 5 to 23 at 6 threads, after a thermal slowdown forced a
  restart. These are stated in the manuscript where they apply.

## 8. Reproducing each result

Raw data goes under `data/` (not in the repository): Bonn sets as `data/A` to `data/E`,
CHB-MIT as `data/chbmit/chbNN/`. `docs/11_CHBMIT_PLANI.md` documents the CHB-MIT download
and verification. Stored results for every step below are already in `results_v2/`, so
the statistics and figures can be regenerated without the raw data or a GPU.

| Result | Command | Output |
|---|---|---|
| Bonn main runs (Table 2) | `python -m src.run_benchmark --task T1_3class --norm N2_z_zspec --threads 2` (likewise `T2_intracranial_binary`, `T3_classic_binary`), or `bash run_all.sh` | `results_v2/T*_N2_z_zspec_*/perfold.csv` |
| Bonn normalisation ablation | `python -m src.run_benchmark --task T1_3class --norm N1_z_rawspec --threads 2` | `results_v2/T1_3class__N1_*` |
| Bonn probability rerun | `python -m src.run_benchmark --task T1_3class --norm N2_z_zspec --threads 12 --outdir results_v2/rerun_probs/T1_3class__N2` | `results_v2/rerun_probs/` |
| Bonn sensitivity (STFT resolution, parameter budget) | `python -m src.run_sensitivity --list`, then `--axis <axis> --variant <variant>` | `results_v2/sensitivity/` |
| Bonn statistics | `python -m src.phase0` and `python -m src.phase0_probscores` | `results_v2/phase0/` |
| Naive test simulation | `python -m src.sim_cv_correlation` | printed |
| CHB-MIT integrity check | `python -m src.verify_chbmit` | printed |
| CHB-MIT corpus and leakage check | `python -m src.chbmit_corpus --check` | printed, cached corpus |
| CHB-MIT deep models and score fusion (Table 3) | `python -m src.chbmit_run --split loso --threads 16 --tag loso_grouped` (add `--resume` to continue an interrupted run) | `results_v2/chbmit/loso_grouped/perfold.csv` |
| CHB-MIT classical baselines | `python -m src.chbmit_baselines --run loso_grouped` | appended to the same file |
| CHB-MIT statistics | `python -m src.chbmit_stats` (default `--run loso_grouped`) | `results_v2/chbmit/phase0_loso_grouped/` |
| CHB-MIT statistics, earlier 24 fold run (Section 5.4 comparison) | `python -m src.chbmit_stats --run loso_main` | `results_v2/chbmit/phase0/` |
| Manuscript figures | `python -m src.make_manuscript_figures` | `paper/figures_manuscript/` |
| Parameter counts (Section 3.3) | `python -m src.models` | printed |

The CHB-MIT runs need only the cached corpus
`data/chbmit/_cache/corpus_w10_s10_n4_g0_seed20260727_24subj.npz` and its `.json` (about
1.2 GB), not the 43 GB of raw EDF files: the correction changes which windows count as the
same person, not which windows are selected. The score rows of the earlier
`loso_main` run were produced by a separate script, `src/chbmit_score.py`, which was
removed in the code cleanup because score fusion is now computed inside `chbmit_run`; it
remains available in the repository history (commit `804709c`).

### Running the CHB-MIT experiments on another machine

1. `git clone https://github.com/Atytmr07/eeg-fusion-benchmark` and install as in the
   README (Python 3.12, CPU build of torch, `pip install -r requirements.txt`).
2. Copy the two cache files into `data/chbmit/_cache/`.
3. `python -m src.chbmit_corpus --check` must report 23 persons, no leakage, and flag the
   old case based split on chb01 and chb21.
4. Time one fold first: `python -m src.chbmit_run --split loso --limit-folds 1 --threads <N> --tag timing_test`,
   then delete `results_v2/chbmit/timing_test/`. The full run takes about 23 times that.
5. Run the full grouping with the same `<N>` throughout. Thread count changes results
   (Section 7), so an interrupted run must be resumed with `--resume` and the same `<N>`,
   and nothing else heavy should run on the machine meanwhile.
6. Then run the baselines and statistics commands above with `--run loso_grouped`.

## 9. Known issues and open items

These are also listed in the manuscript's Limitations section.

1. **chb01 and chb21 are the same person: fixed and rerun.** The earlier `loso_main` run
   grouped by case folder, so the fold holding out chb01 trained on chb21 and vice versa
   (late fusion 0.981 on chb01, 0.444 on chb21; held out together they now give 0.841).
   Splitting and leakage checks use persons (23 folds), and every CHB-MIT number in the
   manuscript comes from the corrected `loso_grouped` run. Its conclusions match the
   earlier run, but the model ranking does not (Spearman rho 0.45; manuscript Section 5.4),
   so CHB-MIT results support the null result, not an ordering of the operators. Whether
   chb24 is a distinct person is assumed, not documented.
2. **Boundary windows.** 25.7 percent of CHB-MIT ictal windows only partly overlap a
   seizure. Whether to keep them is a design choice whose effect has not been measured.
3. **Optimiser: unified.** Bonn always used AdamW (decoupled weight decay). The earlier
   CHB-MIT `loso_main` run used Adam, whose weight decay is added to the gradient and then
   rescaled per parameter, so the same settings meant different regularisation on the two
   corpora. The `loso_grouped` run uses AdamW.
4. **No CHB-MIT sensitivity analysis yet** (window length, frequency ceiling, notch,
   subsampling ratio, boundary windows), unlike Bonn.
5. **False alarms per hour are not reported**, because the rate on the subsampled time
   base would not be clinically meaningful; computing it on the true recording duration is
   still to do.
6. **Single run per model on CHB-MIT.** The re-run changed the ranking while leaving the
   conclusions intact; ranking claims would need repeated runs with different seeds. (The
   earlier run's mixed thread counts no longer affect any reported number.)
7. **The detailed working notes in `docs/` (other than this file) are in Turkish.** The
   code, its comments, the manuscript, this document, and the README are in English.
