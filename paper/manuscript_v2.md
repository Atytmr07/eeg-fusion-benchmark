> **DRAFT v2, NOT FOR SUBMISSION.** Restructured around the three research questions the
> advisor set on 10 October 2026. This file currently holds the research questions and
> the full Methods (Section 2, in the advisor's subsections 2.1 to 2.7). The
> Introduction will be built from `paper/intro_draft.md`; Results and Discussion follow
> when the five-seed runs are complete. `paper/manuscript.md` is the previous draft
> (Bonn and CHB-MIT, single pipeline) and stays unchanged until this one replaces it;
> where the Bonn benchmark goes (supplementary material or a separate note) is still
> to be decided with the advisor. Every number below is traceable to a file in the
> repository (paths given).

# Do Fusion Operators Matter, and Does the Answer Survive Preprocessing? A Parameter-Matched Multiverse Comparison of Multimodal CNN Fusion for EEG Seizure Detection

Emre Atay Tümer, Hasan Berk Berber, Sevgi Şengül Ayan (order to be agreed)
Antalya Bilim University

## Abstract

*(to be written last)*

## 1. Introduction

*(to be integrated from `paper/intro_draft.md`: preprocessing sensitivity, fusion
benchmarking and its limits, reproducibility and ranking instability, cross-dataset
generalisation)*

Our contribution is not another fusion operator, nor only a comparison of existing
ones: it is a test of how far conclusions drawn from such a comparison hold when the
analytic decisions around it change. We ask three questions.

- **RQ1, performance robustness.** Do fusion operators maintain their predictive
  performance across different EEG preprocessing pipelines?
- **RQ2, ranking robustness.** Are the rankings of fusion operators stable across
  preprocessing choices, artefact removal algorithms and stochastic training?
- **RQ3, cross-dataset generalisation.** Do fusion performance patterns and rankings
  generalise across independent EEG seizure datasets?

Throughout, three properties are kept apart, because each can hold without the
others: **statistical significance** (is there evidence of a difference between two
operators?), **practical equivalence** (does their difference lie within a margin
fixed in advance?) and **ranking stability** (is their order preserved when
preprocessing or seeds change?). In particular, the absence of a significant
difference is not taken as evidence of equivalence.

## 2. Methods

Figure 1 (`paper/figures_manuscript/figure1_design.png`) summarises the design: the
datasets, the nine preprocessing pipelines, the two representations, the fusion
operators and controls, and the evaluation; its panel b shows where each step is
applied and how leakage between training and test data is prevented.

### 2.1 Datasets and EEG harmonisation

**CHB-MIT** (Shoeb 2009; Goldberger et al. 2000) holds 686 scalp recordings, 982.9 h,
from 24 paediatric cases at 256 Hz in a bipolar montage. Two cases, chb01 and chb21,
are documented as the same person and are treated as one subject, which gives 23
subjects. We use the standard 18-channel bipolar (double banana) montage and keep the
683 recordings that contain all 18 channels, with 185 of the 198 annotated seizures in
the windowed corpus; the three dropped recordings belong to part of subject chb12's
data recorded in a different montage. Seizure annotations were parsed by order rather
than by their printed number, which corrects a labelling error in `chb09_08.edf`
(`src/chbmit.py: parse_summary`).

**Siena Scalp EEG** (Detti et al. 2020) holds 41 recordings, 141.0 h by the EDF
headers, from 14 adult patients, with 47 annotated seizures, recorded at 512 Hz in a
referential 10-20 montage (older electrode names T3 to T6 are mapped to T7, T8, P7,
P8). Six problems in the published seizure lists (a seizure ending after its
recording, two conflicting onset or end times, two start times that disagree with the
EDF headers, an onset given only clinically) were resolved before any analysis, each
as either a correction verified against the recording or a stated decision rule;
Supplementary Table S4 lists every decision, how it was checked, its remaining
uncertainty, and how much the alternative would change the windows (`docs/SIENA.md`).

**Harmonisation.** Siena is converted to the CHB-MIT form before anything else: the
18 bipolar channels are formed as electrode differences of the referential signals,
and the continuous recording is resampled from 512 to 256 Hz with a polyphase filter
(`scipy.signal.resample_poly`, factor 1/2; `src/siena.py: read_bipolar`). Both
corpora then pass through identical code. Seizure times are converted to seconds from
each recording's first sample using the EDF header start time (modulo 24 h for
recordings past midnight).

**Windows and labels.** Recordings are cut into non-overlapping 10 s windows. A window
is ictal if it overlaps an annotated seizure at all. Non-ictal windows are subsampled
per subject to four per ictal window of that subject, with a fixed seed, and every
subject's true recording duration is kept for the event metrics (Section 2.5). This
gives 6,380 windows (1,276 ictal) for CHB-MIT and 1,640 windows (328 ictal) for Siena.
The same windows are used by every pipeline, which is checked whenever a pipeline's
corpus is built, so all comparisons between pipelines are paired window by window.

**Differences between the datasets.** The two corpora differ in age (mostly children
against adults), seizure types, recording length per patient (Siena's
recordings are cut around seizures for a seizure prediction study), centre,
equipment and mains frequency (60 Hz in Boston, 50 Hz in Italy). Results are
compared between them as an independent replication and a transfer test (RQ3), not
interpreted as an age or population effect.

### 2.2 Preprocessing multiverse

Following the multiverse approach (Steegen et al. 2016), nine defensible
preprocessing pipelines are crossed with every model, operator and seed instead of
choosing one. Each pipeline changes one hypothesis-driven aspect; windows, labels,
spectrogram settings, models, training and evaluation are fixed (`src/preprocess.py`).

| Pipeline | Band-pass | Notch | Artefact handling | Normalisation | Question |
|---|---|---|---|---|---|
| P0 | none | none | none | z-score | original benchmark |
| P1 | 0.5 to 40 Hz | none | none | z-score | standard EEG band |
| P2 | 0.5 to 70 Hz | mains (CHB-MIT 60 Hz, Siena 50 Hz) | none | z-score | information above 40 Hz |
| P3 | 1 to 40 Hz | none | none | z-score | low frequency content and drift |
| P4 | 0.5 to 40 Hz | none | technical artefact rejection (training only) | z-score | artefact rejection |
| P5 | 0.5 to 40 Hz | none | none | median and IQR | normalisation |
| P6a | 0.5 to 40 Hz | none | Extended Infomax ICA | z-score | artefact removal |
| P6b | 0.5 to 40 Hz | none | GEDAI | z-score | artefact removal |
| P6c | 0.5 to 40 Hz | none | AMICA | z-score | artefact removal |

**Filters.** Zero-phase Butterworth filters (order 4, forward and backward) are
applied to each continuous recording before it is windowed, so that no filter
transient falls at a window edge. P2's notch removes each dataset's own mains
frequency and its first harmonic; with a fixed 60 Hz notch Siena's P2 windows kept a
50 Hz line 318 times stronger than the 40 to 48 Hz background, so P2 was defined per
dataset and its Siena runs were repeated.

**Normalisation.** Each channel of each window is standardised on its own: z-score
(P0 to P4, P6) or median and interquartile range (P5). No statistic is computed across
windows, recordings or persons, so normalisation cannot carry information from test to
training data.

**P4, technical artefact rejection.** Amplitude was measured before a criterion was
chosen: every amplitude threshold removes seizures selectively (500 uV rejects 72
percent of CHB-MIT's ictal and 48 percent of its non-ictal windows), because ictal
windows have larger amplitudes. P4 therefore rejects only technical artefacts: a flat
channel (standard deviation below 1 uV in the window) or a constant run of at least
0.5 s on any channel (a signal dropout), detected on the unfiltered signal. In CHB-MIT
this rejects 42 of 1,276 ictal (3.3 percent) and 39 of 5,104 non-ictal windows (0.8
percent); the difference reflects which recordings have dropouts, not physiology
(Supplementary Table S3). In Siena it rejects one window. Rejected windows are removed
from training and validation only; the test windows are those of every other
pipeline.

**P6, three artefact removal algorithms.** P6 applies Extended Infomax ICA (Lee et al.
1999; MNE-Python, Gramfort et al. 2013), GEDAI (Ros et al. 2025) or AMICA (Palmer et
al. 2008; `jamica`) to the same 0.5 to 40 Hz signal, recording by recording.
Infomax and AMICA share every setting and one component rejection rule: 16 components
(the 18-channel bipolar montage has rank 16, because it contains two closed electrode
loops per hemisphere), fitted on every 8th sample, with components removed when their
correlation with any of the four frontopolar channels, used as EOG proxies since
neither dataset has an EOG channel, has |z| > 3 (MNE `find_bads_eog`). Muscle
components are not removed, because the gamma band carries the strongest ictal signal
in CHB-MIT. GEDAI is not an ICA: it removes what departs from a leadfield-based
reference covariance, by its own criterion, so the shared rule cannot apply. Its
reference covariance C, defined for a referential layout, is mapped onto the bipolar
montage as D C D^T (D the electrode difference matrix), and the channel mean is removed
before and restored after cleaning. GEDAI's threshold preset was fixed before any
classification run, on signal preservation alone: on six CHB-MIT subjects the default
removed more than half of the power of 39 percent of recordings, the conservative
preset of 17 percent, and neither removed more from ictal than from non-ictal windows
(Supplementary Table S2); P6b uses the conservative preset. The comparison among P6a,
P6b and P6c is therefore a comparison of whole cleaning procedures: for Infomax and
AMICA it isolates the decomposition algorithm, while GEDAI differs also in how it
decides what to remove. Signal retention and compute are in Supplementary Table S1.

**No leakage through P6.** Each recording is decomposed and cleaned on its own, from
its own unlabelled signal; seizure annotations are never used, and no decomposition,
threshold or statistic is shared between recordings or persons. A test person's
recordings are therefore cleaned exactly as they would be without any training data.

### 2.3 EEG representations and fusion strategies

**Representations.** Each 10 s window enters the network twice: as the 18-channel
waveform (2,560 samples per channel) and as its log-magnitude short-time Fourier
transform, log(1 + |STFT|), computed per channel with a 1 s Hann window, 0.5 s hop and
256-point FFT and cropped at 64 Hz (65 frequency bins by 19 frames;
`src/chbmit_prep.py`). The 64 Hz ceiling is kept in every pipeline, so input shapes and
parameter counts never change.

**Architecture.** A 1D convolutional encoder (channel widths 16, 32, 64) processes the
waveform and a 2D convolutional encoder of matching depth the spectrogram; each yields
a 128-dimensional pooled feature vector. Five interchangeable fusion operators combine
them:

- **Early:** the waveform passes a small strided 1D encoder, is broadcast along the
  spectrogram's time axis and concatenated to it as an extra channel before the 2D
  backbone (architecturally an intermediate fusion, since the two inputs share no
  tensor shape).
- **Late:** the two feature vectors are concatenated and classified by a small MLP.
- **Gated:** a learned scalar gate per branch reweights the concatenated features.
- **Attention:** a learned attention module weights the branches from both feature
  vectors jointly.
- **Score:** the two branches are trained as independent classifiers and their
  probabilities blended with a weight chosen on the validation split only.

**Parameter matching.** Every jointly trained model (the four joint operators and two
widened single-branch controls) is sized to a common budget by solving the fusion
head's hidden width numerically (`src/models.py`): with 18 input channels the models
have 102,236 to 103,712 parameters, the four joint operators within 0.36 percent of
each other. Score fusion, an ensemble of two independently sized classifiers, keeps
its natural size.

**Controls.** The two single-branch networks at their default width (raw1d, spec2d)
and widened to the fusion budget (raw1d_wide, spec2d_wide), which asks whether fusion
beats a unimodal model of the same size; and two classical baselines without
representation learning, a logistic regression on log variance (logvar) and on seven
hand-crafted features (log variance, line length and relative power in five bands;
shallow).

### 2.4 Experimental protocol

**Leave one subject out.** Each subject is held out once as the test set (23 folds on
CHB-MIT, chb01 and chb21 together; 14 on Siena). From the remaining subjects, whole
subjects are set aside as an inner validation set (4 of 22 on CHB-MIT, 3 of 13 on
Siena), chosen deterministically to cover the range of seizure burdens; it serves
early stopping and the score fusion weight only. No subject's windows and no seizure's
windows ever appear on both sides of a split, which a leakage checker confirms for
every split and which was itself validated on a deliberately leaky split
(`src/chbmit_corpus.py`).

**Training.** AdamW (Loshchilov and Hutter 2019), learning rate 1e-3, weight decay
1e-4, batch size 32, class-weighted cross-entropy; early stopping on validation macro
F1 with patience 12 after a minimum of 20 epochs, at most 60 epochs. The minimum guards
against a model stopping in a trivial single-class solution, found in an early audit.

**Seeds.** Every pipeline is run with five seed sets. A run's seed for each fold and
model is derived from the seed set, fold and model (`src/config.py: fold_seed`), so it
does not depend on execution order. The ranking of the operators was unstable across
three seed sets (Kendall's W below 0.5 in seven of nine CHB-MIT pipelines), the
condition fixed beforehand for extending to five.

**Cross-dataset experiment (RQ3).** Each model is trained once on all CHB-MIT subjects
(with the same inner validation rule) and tested on every Siena window and recording,
for every pipeline and seed set (`src/cross_dataset.py`). The pipeline is applied to
each dataset as defined, including each dataset's own mains notch in P2. Results are
reported per Siena patient.

### 2.5 Evaluation metrics

**Window level (primary).** Macro F1 and the area under the precision-recall curve
(AUPRC) are primary; balanced accuracy (the mean of sensitivity and specificity) is
reported alongside, with AUROC and the Brier score as secondary. A window is
predicted ictal when its ictal probability exceeds 0.5. Metrics are computed per fold
on the held-out subject's windows and averaged over folds within a run.

**Seizure level (secondary, clinical).** Every model saved for a fold is applied to
every 10 s window of every recording of the held-out subject, not only the subsampled
test windows, after the pipeline's own preprocessing; a window is positive when its
probability exceeds 0.5, with no further post-processing. Detections are scored
against the annotated seizures with SzCORE's event scoring (Dan et al. 2024;
`timescoring` 0.0.7, defaults: 30 s tolerance before onset and 60 s after offset, any
overlap counts, events closer than 90 s merged, events longer than 5 min split).
Seizure-level sensitivity is the share of seizures detected; the false alarm rate is
false-positive events per hour on the true recording time. The scored recordings hold
187 seizures in 979.9 h (CHB-MIT) and 47 seizures in 141.0 h (Siena); scoring the
window labels themselves as predictions detects all of them without a false alarm,
which checks the reference, the windows and the scoring together. On Siena only
sensitivity is reported: its recordings are cut around seizures, so a false alarm
rate would not describe continuous monitoring. The criterion, the merging of
consecutive detections and the counting of false alarms were fixed, and committed,
before any multiverse run (`docs/EVENTS.md`), and are the same for every operator and
pipeline.

### 2.6 Statistical analysis

**Pairwise comparisons.** Leave-one-subject-out folds share training data, so the
standard paired test overstates significance; in a simulation of two equivalent
classifiers the naive paired Wilcoxon test rejected in 37.6 percent of cases at a
nominal 5 percent (`src/sim_cv_correlation.py`). Pairs of operators are therefore
compared with the resampling-corrected paired t test (Nadeau and Bengio 2003; test to
training ratio 1/22 on CHB-MIT, 1/13 on Siena), with Holm correction over the ten
operator pairs within each pipeline.

**Practical equivalence.** For each pair the two one-sided test with the same
corrected variance gives the smallest margin within which the pair is equivalent at
alpha = 0.05; a pair is equivalent if that margin is at most 0.05 macro F1 (main
analysis) and, as a stricter sensitivity analysis, 0.02. Both margins were fixed and
committed before any multiverse result existed. A pair is robustly equivalent if it is
equivalent in every pipeline.

**Ranking stability.** In every run (pipeline and seed set) the five operators are
ranked by mean macro F1. We report each operator's full rank distribution (counts per
rank, median, best and worst rank, share of runs won and lost), Kendall's W across
seed sets within each pipeline, and Kendall's tau between the seed-averaged rankings of
every pair of pipelines. Whether preprocessing changes the ranking more than the seed
does is tested by comparing the mean tau between seed sets of the same pipeline with
the mean tau between pipelines, against a null distribution from 10,000 permutations
of the pipeline labels within each seed set; the same test is run among P6a, P6b and
P6c alone.

**Fusion x Preprocessing.** A repeated-measures ANOVA on macro F1 (seed means) treats
the held-out subjects as the repeated units and pipeline and operator as within-subject
factors; each effect is tested against its interaction with subject, with
Greenhouse-Geisser correction. Effect sizes are reported as partial eta squared.

**Variance components.** For macro F1, AUPRC and balanced accuracy, the fully crossed
subject x pipeline x operator x seed design is decomposed as y = mu + subject +
pipeline + operator + seed + every interaction, all treated as random effects
(generalizability theory) and estimated by the method of moments from the expected
mean squares. With one observation per cell the four-way interaction is the residual.
Every component is reported unrounded, as an estimated variance, as a standard
deviation in the metric's own units (a practical size) and as a share of the total;
negative estimates, which arise when an effect's mean square is smaller than its error
term, are reported as such and set to zero in the shares.

**Prediction stability.** The agreement (Cohen's kappa) of window-level predictions
between pipelines at the same seed set is compared with the agreement between seed
sets in the same pipeline.

Statistical significance is reported from the corrected tests and the ANOVA, practical
size from the equivalence margins and the variance components in metric units, and
ranking stability from the rank analyses; a result in one of these three is never
used as evidence for another. All analyses are implemented in `src/multiverse.py` and
were validated on synthetic runs with known effects (39 checks).

### 2.7 Reproducibility and computational cost

**Software and seeds.** Training used Python 3.12, numpy 1.26.4 and PyTorch 2.8.0
(CUDA 12.8 build); artefact removal numpy 2.5.3, MNE 1.13.2, GEDAI 0.60 and jamica
0.3.0, in a separate environment because they need a newer numpy (`requirements.txt`,
`requirements-ica.txt`). Seeds are fixed per run, fold and model, and per recording for
artefact removal (from the file name).

**Determinism and device.** All runs of one analysis use one machine, one device and
16 CPU threads. On the CPU a run was reproduced bit for bit, also with two runs
sharing the machine. The multiverse ran on one GPU (NVIDIA RTX 5070) with PyTorch's
deterministic algorithms; adaptive average pooling has no deterministic CUDA backward,
yet two GPU runs of the same fold were identical. Before switching to the GPU, its
agreement with the CPU was tested at identical seeds against the difference caused by
the seed alone: over five folds the mean absolute macro F1 difference between CPU and
GPU was 0.074, and between two CPU seed sets 0.066, so the GPU behaves as one more seed
rather than as a systematic shift (`docs/PIPELINE.md`). CPU and GPU runs are never
mixed in one analysis.

**Reproducibility of artefact removal.** Fixing seeds and package versions did not
make P6 reproducible across machines and execution contexts (`docs/ICA_REPRO.md`).
AMICA reproduced up to rounding. Extended Infomax removed different components in
recordings whose component lay near the |z| = 3 rule: 2.5 percent of recordings between
two machines and 1.2 percent between two runs on the same machine with different
process parallelism. GEDAI was deterministic on each machine but gave a different
result on the second machine in 27 percent of recordings, often by a large amount.
Window-level results are unaffected, because every P6 training and test window comes
from one set of signals, cleaned once. The event metrics, which need every recording
cleaned again, were computed on the machine that produced those signals; they then
reproduce the training-time cleaning exactly for GEDAI, within 1.4 percent of the
amplitude for AMICA, and differ for Infomax in 8 of 670 CHB-MIT recordings
(`events/meta.json`, `window_check`).

**Computational cost.** Artefact removal of the 670 CHB-MIT recordings with selected
windows took 4.7 (GEDAI), 8.5 (Infomax) and 73 (AMICA) CPU hours on a laptop CPU
(Supplementary Table S1). One CHB-MIT run (23 folds, nine models) took about 1.5 h on
the GPU, one Siena run about a quarter of an hour, and one cross-dataset run about
5 min; one fold took 3.7 min on the GPU against 12 min on the CPU.

**Availability.** Code, the per-fold results of every run, and all analyses are in the
public repository; Section 9 lists the commands that regenerate each table and figure.

## 3. Results

*(pending the five-seed runs; planned figures: 2, performance of each operator across
pipelines, macro F1 and AUPRC; 3, ranking stability heatmap over pipelines and seeds;
4, variance decomposition; 5, artefact removal: signal preservation, cost, performance
and reproducibility; 6, cross-dataset comparison)*

## 4. Discussion

*(pending)*
