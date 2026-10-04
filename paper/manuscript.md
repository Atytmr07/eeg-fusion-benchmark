> **DRAFT, NOT FOR SUBMISSION.** This manuscript is a working draft. The systematic
> literature search has had its first round (`docs/15_LITERATUR_SONUCLARI.md`); 122
> full texts and a human second screening are still open. The
> study is being extended, at the advisor's request, from a single comparison into a
> preprocessing and training multiverse: its design is described in Section 3.6 and
> the third corpus in Section 5.5, while their results (Section 5.6) are pending and
> marked as such. The abstract, introduction and discussion still describe the
> original two-corpus comparison and will be rewritten when those results exist.
> Numbers are traceable to stored result files (paths given throughout). See `docs/`
> for the full audit trail this draft is built from.

# Do Fusion Operators Matter, and Does the Answer Survive Preprocessing? A Parameter-Matched Multiverse Comparison of Multimodal CNN Fusion for EEG Seizure Detection

Emre Atay Tümer
Computer Engineering and Industrial Engineering, Antalya Bilim University

## Abstract

Convolutional neural networks for EEG seizure detection commonly combine two
representations of the same signal, the raw waveform and its spectrogram, using a
fusion operator such as early, late, gated, attention, or score level fusion.
Published work routinely reports that a particular operator outperforms the
alternatives, but the compared models typically differ in more than the operator:
parameter count, training protocol, and evaluation design vary alongside it, and the
paired statistical tests used to support such claims commonly assume independent
measurements that repeated cross validation violates. We hold backbone architecture,
training protocol, and parameter budget fixed to within 1.3 percent and vary only the
fusion operator, then evaluate the resulting differences with a resampling corrected
paired test, Holm correction, and two one sided equivalence testing, on two corpora:
the Bonn EEG dataset (5x5 repeated stratified cross validation) and CHB-MIT (leave
one subject out over 23 individuals). On Bonn, no pair among late, gated, attention, and
score level fusion separates under the corrected test in any of three task
formulations, and the pairs are equivalent within an F1 margin of 0.02 to 0.03; early
fusion is the one operator that departs, and only downward. On CHB-MIT, none of the
same five operators separate either, but between subject variability is large enough
that equivalence itself cannot be established with confidence, a materially different
kind of negative result that we report separately from Bonn's. Two further findings
bear on evaluation practice generally. First, a simulation shows the naive paired
Wilcoxon test commonly used in this literature has a 38 percent false positive rate
under repeated cross validation; correcting for it collapses most of the differences
we originally observed. Second, CPU thread count alone, a setting with no bearing on
the scientific question, shifts per fold macro F1 by up to 0.09 on Bonn and 0.21 on
CHB-MIT, a magnitude comparable to or larger than the differences between the
operators under comparison; re-running CHB-MIT after correcting the subject grouping
and unifying the optimiser left every conclusion intact but reordered the eleven
models almost arbitrarily (Spearman rho = 0.45 between the two runs). We also report
and correct three errors we found along the way: a training bug that let some models
collapse to a trivial single class solution while reporting high AUC, a labelling
error in CHB-MIT's own seizure annotations that a sequence naive parser would turn
into a nonexistent 6316 second seizure, and a subject grouping error in our own
CHB-MIT evaluation, where two recording cases from the same person were treated as
different subjects. We
argue that the absence of a detectable difference between fusion operators in this
literature is not yet established as a real finding so much as it is obscured by
evaluation practices too imprecise to resolve differences at this scale, and that
equivalence testing, corrected resampling statistics, and reporting of runtime
determinism should be standard rather than exceptional in this line of work.

## 1. Introduction

Electroencephalography (EEG) records the brain's electrical activity over time.
During an epileptic seizure this signal changes characteristically: amplitude rises,
the waveform becomes more rhythmic, and power concentrates in particular frequency
bands. Automatic seizure detection is therefore a signal classification problem, and
convolutional neural networks (CNNs) are the dominant modelling approach to it.

A recurring design pattern in this literature represents the same EEG segment two
ways at once: the raw waveform, processed by a 1D CNN, and its spectrogram, processed
by a 2D CNN. The two branches are combined by a fusion operator, most commonly one of
early fusion (combining inputs or early features), late fusion (combining final
feature vectors before classification), gated fusion (a learned per branch weight),
attention based fusion, or score level fusion (averaging the two branches' independent
predictions). Published comparisons of these operators typically report that the
authors' chosen operator wins. The comparisons, however, usually differ in more than
the operator under study: model capacity, training schedule, and data handling vary
alongside it. When a larger fusion model outperforms a smaller single branch model, it
is not clear whether the gain comes from the fusion mechanism or simply from the added
capacity.

This paper asks a narrower and more mechanical question than most work in this space:
when architecture size, training protocol, and data are held fixed, and only the
fusion operator is varied, does the operator choice produce a difference that survives
a statistical test appropriate to the way the data was resampled? We treat this as a
comparative and methodological study rather than a proposal for a new architecture,
and we hold ourselves to the same scrutiny we apply to the literature: every reported
number in this manuscript traces to a file under `results_v2/`, and every claim about
what has or has not been shown before is qualified by how the underlying literature
search was actually conducted (Section 2 and `docs/09`, `docs/14`).

### 1.1 Contributions

1. A parameter matched benchmark of five fusion operators plus single branch and
   classical feature controls, on the Bonn EEG corpus, under three task formulations,
   evaluated with 5x5 repeated stratified cross validation (Section 4).
2. An audit of the statistical practice common in this literature: a simulation
   quantifying the false positive rate of the naive paired test under repeated cross
   validation, and a corrected testing framework (resampled t test, Holm correction,
   equivalence testing, and a correlated Bayesian t test) applied throughout
   (Section 3.5).
3. External validation on CHB-MIT with subject wise leave one subject out evaluation,
   which the Bonn corpus does not support because it publishes no subject identifiers
   (Section 5), with subjects grouped by person rather than by recording case, after
   finding that two of CHB-MIT's 24 cases are documented as the same individual
   (Section 5.2).
4. Two methodological findings that generalise beyond this specific comparison: naive
   paired testing under repeated cross validation and CPU thread count as an
   unreported source of variance comparable in magnitude to the effect under study
   (Section 6).
5. A corrected, order based parser for CHB-MIT's seizure annotation files, after
   finding that the standard number based parsing is vulnerable to a labelling error
   present in the released files (Section 5.1).
6. A preprocessing and training multiverse (Section 3.6): nine preprocessing pipelines,
   including three artefact removal algorithms, each crossed with three seed sets, to
   ask whether conclusions about fusion operators, not only their absolute
   performance, are robust to preprocessing decisions and to stochastic training
   (results pending, Section 5.6).

## 2. Related Work

Table 1 summarises the closest published work to this comparison, restricted to what
we have verified through Crossref by DOI. It draws on a systematic search run on 3
October 2026 under a protocol fixed beforehand (`docs/14_SISTEMATIK_TARAMA_PROTOKOLU.md`):
Scopus, Web of Science, IEEE Xplore and arXiv returned 902 records, 555 after
deduplication, of which 128 passed title and abstract screening; the studies closest
to ours were read in full, and every claim Table 1 makes about them is traced to a
section or table of the paper (`paper/literature/extraction.csv`, `docs/15_LITERATUR_SONUCLARI.md`).
Title and abstract screening was done by an AI assistant applying the protocol's
criteria; an independent AI pass over a random 20 percent agreed on 91.9 percent of
records (Cohen's kappa 0.81). A human second screening and 122 full texts are still
outstanding (Section 7).

**Table 1. Closest related work.**

| Work | Contribution | What it leaves open |
|---|---|---|
| Andrzejak et al. (2001) | Introduces the Bonn EEG corpus | No per segment subject identifiers |
| Acharya et al. (2018) | Early deep CNN for Bonn seizure detection | Single architecture, no operator comparison |
| Truong et al. (2018) | CNN over spectrograms, scalp and intracranial | No raw signal branch to fuse against |
| Wang et al. (2020) | Compares 1D and 2D CNNs | Compared separately, not fused |
| Chatzichristos et al. (2020) | Attention gated multi view fusion | One operator, no alternatives compared |
| Das et al. (2024) | 1D+2D fusion of decomposed EEG | Fixed design, no operator ablation |
| Huang et al. (2024) | Multi head attention fusion | Reports gains without matching parameter count |
| Golrizkhatami and Acan (2018) | Three level fusion for ECG | Different signal domain, related fusion methodology |
| Narotamo et al. (2024) | Compares 1D and 2D representations plus early, late, and joint fusion, the same design question as ours | ECG, not EEG; no parameter matching or corrected/equivalence statistics |
| Roy (2019), Shoeibi (2021), Xu (2024) | Reviews of deep learning for EEG | Document inconsistent evaluation protocols, do not resolve them |
| Ali et al. (2024) | Shows CHB-MIT results depend heavily on evaluation choices | Concerns evaluation of CHB-MIT generally, not fusion operators |
| Dan et al. (2024), SzCORE | Standard datasets, event based scoring and metrics for EEG seizure detection | An evaluation framework, not a fusion operator comparison |
| Rheude et al. (2025) | Multimodal complexity often does not pay off | General multimodal setting, not EEG |
| **Daşdemir and Örnek (2025)** | **Self described fair, protocol controlled comparison of two fusion strategies for EEG** | Seizure *prediction*, not detection; two operators; reports a 0.20 point accuracy difference (97.50 vs 97.70) without a stated statistical test or parameter matching |
| **An et al. (2026), fastSeizureNet** | **Compares five fusion strategies for EEG seizure detection**, within and across patients, with several backbones (semi-supervised) | Fuses hand-crafted feature knowledge with deep features rather than a raw waveform with its spectrogram; the strategies are not parameter matched and are not compared with each other by a statistical test (a Wilcoxon test with Holm correction is used only against prior methods) |
| Einizade et al. (2023), fAttNet | Compares four fusion operators (majority vote, averaging, concatenation, attention) over raw EEG, wavelet packet and hand-engineered views, leave one subject out on TUH | No parameter matching and no statistical test; the operators lie within one standard deviation of each other |
| Basheer and Mishra (2026) | Compares concatenation, element-wise addition and KAN gating for seizure detection, with paired t-tests over six seeds | Both streams come from the same raw EEG rather than two representations; uncorrected tests against concatenation only; no parameter matching |
| Wang et al. (2026), *Neurocomputing* | Cross-attention fusion of time domain and S-transform views, CHB-MIT and Siena | One ablation against concatenation, patient specific evaluation, no statistical test |
| Camastra et al. (2026), *Brain Sciences* | Controlled fusion benchmark, concludes strategy matters more than architecture | Tabular neuroimaging features, not EEG time series; no paired or equivalence testing |
| Mohamady et al. (2026) | Seven fusion techniques compared on one benchmark | Human activity recognition, not EEG; states this kind of head to head comparison did not previously exist in that field either |
| Kontras et al. (2026), NeuroAtlas | Reports Bonn is saturated (11 models at AUROC ≥ 0.99) and carries no subject identifiers | Foundation model benchmark, not a fusion operator comparison |

Reading this table, fusion operators for EEG seizure detection are proposed
frequently, often one at a time against baselines that differ in capacity from the
proposed model, and several studies now compare two to five operators (Einizade et
al. 2023; An et al. 2026; Basheer and Mishra 2026), but none we read in full matches
parameter counts or tests the operators against each other with corrected or
equivalence tests. Two studies come closest. Daşdemir
and Örnek (2025) holds two branches fixed and compares two fusion points, but on a prediction
rather than detection task, without a stated parameter matching procedure or
significance test on the 0.20 point difference it reports. An et al. (2026) compare
five fusion strategies for seizure detection, the closest design to ours in breadth,
but between hand-crafted and deep features rather than between a raw waveform and its
spectrogram, without matching the strategies' parameter counts and without a
statistical test between them. We have not found a study that compares three or more
fusion operators for EEG seizure detection under a matched parameter budget with
paired statistical testing and equivalence testing. This
is the gap this paper addresses, stated at this narrower scope rather than as a
categorical absence, and subject to revision once the remaining full texts of the
systematic search (`docs/15`) have been read. An independent review across the wider EEG-based multimodal
human-computer interface literature is consistent with this being a real gap rather
than a search artifact: Lee et al. (2025), surveying over 120 hybrid EEG studies,
report that only two explored more than one fusion strategy and only one directly
investigated which fusion approach was best, and name fusion strategy optimisation
as a direction the field has largely left open.

Two further points from Table 1 shape our design directly. First, NeuroAtlas reports
that Bonn's conventional binary split is saturated, eleven foundation models reaching
AUROC at or above 0.99, and recommends treating it as a sanity check rather than a
discriminating benchmark; this is not our own inference but an independent finding we
build on (Section 4.1, Section 6.3). Second, the same benchmark confirms Bonn
publishes no per segment subject identifier, which is why external validation with
genuine subject wise cross validation (Section 5) is necessary rather than optional
for any claim this paper makes about generalisation across patients.

## 3. Method

### 3.1 Architecture

Two branch CNN: a 1D convolutional encoder (channel widths 16, 32, 64) over the raw
waveform, and a 2D convolutional encoder of matching depth over the log magnitude
short time Fourier transform (STFT) spectrogram of the same segment. The two
encoders' pooled features (dimension 128 each) are combined by one of five
interchangeable fusion heads:

- **Early fusion.** The raw signal is passed through a small strided 1D encoder,
  broadcast across the spectrogram's time axis, and concatenated as an additional
  channel before the 2D backbone. We note for precision that this is what the
  literature calls "early" fusion but is, architecturally, an intermediate fusion:
  a genuinely input level combination of a 1D waveform and a 2D spectrogram is not
  well defined, since the two representations do not share a common tensor shape.
- **Late fusion.** The two branches' pooled feature vectors are concatenated and
  passed through a small MLP classifier head.
- **Gated fusion.** A learned scalar gate per branch reweights the concatenated
  features before classification.
- **Attention fusion.** A learned attention mechanism computes per branch weights
  from both feature vectors jointly.
- **Score level fusion.** The two branches are trained as independent unimodal
  classifiers; their softmax outputs are blended with a weight selected by grid
  search on the validation split only, never on test data (`src/train.py:
  best_blend_weight`).

Reference models: **raw1d** and **spec2d**, the same two backbones trained and
classified independently at their default width; **raw1d_wide** and **spec2d_wide**,
the same single branch models widened to match the fusion models' parameter count,
which is the correct comparison for asking whether fusion outperforms a unimodal
model rather than simply outperforming a smaller one; **logvar**, a one feature
logistic regression on the segment's log variance; and **shallow**, a logistic
regression on seven hand crafted features (log variance, line length, and relative
power in five frequency bands). The two classical baselines are not deep models and
are included to establish how much of any reported performance is attributable to
representation learning at all, rather than to the discriminability of the task
itself (`src/baselines.py`, `src/chbmit_baselines.py`).

### 3.2 Parameter matching

Every jointly trained variant (early, late, gated, attention, and the two widened
single branch controls) is sized to a common budget with a fusion head hidden
dimension solved numerically per operator (`src/models.py: _solve_hidden`) to land
within 1.3 percent of the target. On Bonn (single channel input) this budget is
approximately 98,500 parameters (late 98,571; gated 98,698; attention 98,544; early
97,507; spec2d_wide 97,775; raw1d_wide 99,795). Score level fusion, being an ensemble
of two independently sized unimodal models rather than a jointly parameterised head,
is reported at its own natural parameter count (59,302 on Bonn) rather than forced
into the same budget, since forcing it there would require shrinking its two
constituent classifiers below their own useful capacity. On CHB-MIT, the same budget
solved for 18 input channels lands between 102,236 and 103,712 parameters across the
same model set (Section 5.2), with a spread of 0.36 percent across the four jointly
trained fusion operators (late 102,768; gated 102,921; attention 102,767; early
102,547).

### 3.3 Preprocessing

Bonn's segments are band limited at acquisition to 0.53 to 40 Hz (Andrzejak et al.
2001); the spectrogram is computed with a 256 point STFT (window 256, hop 128) and
capped at the same 40 Hz. CHB-MIT is not band limited, and we measured (Section 5.1)
the largest ictal to interictal power ratio in the gamma band (30 to 70 Hz); the
frequency ceiling is therefore raised to 64 Hz for CHB-MIT rather than reused at 40 Hz
(`src/chbmit_prep.py`), and line noise removal at the US mains frequency (60 Hz) is
implemented as an optional, separately reported arm rather than baked into the
default pipeline, since it falls inside the informative gamma band.

Amplitude normalisation on Bonn is treated as an experimental factor with four arms
(N0 through N3, `src/config.py: NORM_MODES`) rather than a single fixed choice,
because an audit of the original implementation this project began from found the
cached spectrogram had in fact been computed from the unnormalised signal while the
1D branch saw a z scored signal, an inconsistency that biased comparisons involving
early fusion specifically (Section 6.4). The default arm used throughout (N2) z
scores both branches from the same normalised signal.

### 3.4 Training and evaluation protocol

AdamW optimiser (decoupled weight decay; Loshchilov and Hutter 2019), learning rate
1e-3, weight decay 1e-4, batch size 32, class weighted cross entropy loss, on both
corpora. An earlier CHB-MIT run, which we compare against in Section 5.4, used Adam
with coupled (L2) weight decay instead. A minimum epoch budget of 20 is enforced before early stopping on
validation macro F1 (patience 12, ceiling 60 epochs); this guard was added after an
audit found 6 of 55 early runs had frozen in a trivial single class solution while
still reporting AUC near 0.96, because early stopping triggered while validation
macro F1 was flat rather than genuinely converged (Section 6.4). Seeds are
deterministic per (repeat, fold, model) via a hash of those three values
(`src/config.py: fold_seed`), removing a dependency on execution order that the
original implementation had.

On Bonn, evaluation is 5x5 repeated stratified cross validation (25 paired
measurements per model) with an inner 20 percent validation split carved from the
training fold, never touching the test fold. On CHB-MIT, evaluation is 23 fold leave
one subject out over 24 recording cases, with the same inner validation split
constructed subject wise (Section 5.2), since Bonn's lack of subject identifiers makes
this comparison impossible there. All CHB-MIT models were trained on one machine at a
fixed thread count of 16 (Section 6.2).

### 3.5 Statistical framework

Repeated cross validation measurements are not independent: the folds share
training data across repeats, which the standard paired test (independent samples,
or naive paired Wilcoxon or t test) does not model. We quantified the consequence by
simulation (`src/sim_cv_correlation.py`): two statistically identical logistic
regression classifiers, differing only in which of two equally informative feature
sets they see, evaluated by 5x5 repeated cross validation over 1000 simulated
datasets. Naive Wilcoxon rejected the (true) null of no difference in 37.6 percent
of simulations, and a naive paired t test in 38.6 percent, against a nominal 5
percent. A resampling corrected t test (following Nadeau and Bengio 2003 and
Bouckaert and Frank 2004, which inflate the standard error by the fraction of data
shared between the resampled training sets) rejected in 2.3 percent of simulations,
close to nominal. This matches an independent, external result: Jafrasteh et al.
(2025), studying neuroimaging classifiers rather than EEG fusion specifically, show
by a different route that whether a cross validated comparison comes out
significant depends on the number of folds and repeats chosen, which is exactly the
dependence structure that makes p-hacking possible under the naive test.

All significance claims in this paper therefore use the corrected test
(`src/stats.py: corrected_ttest`), with Holm correction across all pairwise
comparisons within a task and metric. Where a null result is claimed, we support it
with a two one sided equivalence test computed with the same corrected variance
(`corrected_equivalence_bound`, reporting the minimum margin δmin at which
equivalence would be established at α = 0.05), a correlated Bayesian t test
(following Corani and Benavoli 2015; Corani et al. 2017)
reporting the posterior probability of practical equivalence within a stated region
of practical equivalence (ROPE), and a sign test as a distribution free check. We
report primary results on macro F1 and Brier score; log loss is reported but treated
as secondary, because it proved heavy tailed and unstable across both corpora
(Section 4.3, Section 5.2), inflating variance for any model with even a few
confidently wrong predictions and correspondingly losing statistical power.

### 3.6 Preprocessing and training multiverse

The re-run of Section 5.4 showed that settings unrelated to the fusion operator can
reorder the operators while leaving the null result intact. We therefore ask a
broader question than "which operator is best": how robust are conclusions about
fusion operators to preprocessing decisions and to stochastic training? The design
follows the multiverse approach (Steegen et al. 2016): a fixed set of defensible
analysis choices is crossed and every combination is run, instead of reporting one.

**Pipelines.** Each pipeline changes one hypothesis driven aspect of preprocessing;
windows, labels, subsampling, spectrogram settings (64 Hz ceiling, so input shapes
and parameter counts do not change), models, training and evaluation stay fixed
(`src/preprocess.py`).

| Pipeline | Band-pass | Notch | Artefact handling | Normalisation | Question |
|---|---|---|---|---|---|
| P0 | none | none | none | z-score | original benchmark (Section 5.3) |
| P1 | 0.5 to 40 Hz | none | none | z-score | standard EEG band |
| P2 | 0.5 to 70 Hz | 60 Hz | none | z-score | information above 40 Hz |
| P3 | 1 to 40 Hz | none | none | z-score | low frequency content and drift |
| P4 | 0.5 to 40 Hz | none | technical artefact rejection | z-score | artefact rejection |
| P5 | 0.5 to 40 Hz | none | none | median and IQR | normalisation |
| P6a | 0.5 to 40 Hz | none | Extended Infomax ICA | z-score | aggressive cleaning |
| P6b | 0.5 to 40 Hz | none | GEDAI | z-score | aggressive cleaning |
| P6c | 0.5 to 40 Hz | none | AMICA | z-score | aggressive cleaning |

Filters are zero phase Butterworth filters (order 4, applied forward and backward)
run on each continuous recording before it is cut into windows, since filtering 10 s
windows separately would place filter transients at every window edge. Every
pipeline's corpus holds exactly the windows of P0, which is checked when it is built,
so all pipelines are paired fold by fold.

**P4: rejecting technical artefacts, not amplitude.** We measured amplitude
distributions before choosing a criterion. The median worst channel peak to peak
amplitude is 482 uV in non-ictal and 890 uV in ictal windows, so every amplitude
threshold removes seizures selectively: 500 uV rejects 72 percent of ictal against 48
percent of non-ictal windows, 1000 uV 44 against 12, and a per subject robust z score
above 10 still 21 against 4.5. Amplifier clipping could not serve as a criterion
either, because the stored integer data exceed the declared physical range and no
recording piles up at a limit. P4 therefore rejects only technical artefacts: a flat
channel (standard deviation below 1 uV over the window) or a constant segment of at
least 0.5 s on any channel (a signal dropout, at 0 uV in CHB-MIT), detected on the
unfiltered signal. This rejects 3.3 percent of ictal and 0.8 percent of non-ictal
windows. The difference is not physiological: the 0.5 s before a dropout have the
same amplitude in both classes (median about 40 uV); dropouts occur in a few
recordings that also contain seizures. Rejected windows are removed from training and
validation only, so the test set is identical across pipelines.

**P6: three artefact removal algorithms.** To separate the effect of the algorithm
from the rest of the pipeline, P6 is run with Extended Infomax ICA (Lee et al. 1999,
as implemented in MNE-Python, Gramfort et al. 2013), GEDAI (Ros et al. 2025), and
AMICA (Palmer et al. 2008; Python
implementation `jamica`), all on the same 0.5 to 40 Hz signal and per recording.
Infomax and AMICA share every setting and the same automatic component rule: 16
components, fitted on every 8th sample, and MNE's correlation based ocular component
detection (`find_bads_eog`, |z| > 3) with the four frontal FP channels as EOG proxies,
since CHB-MIT has no EOG channel. Sixteen, not eighteen, components are used because
the 18 channel bipolar montage contains two closed electrode loops on each side (FP1
to O1 along the temporal and along the parasagittal chain), so the data have rank 16;
the remaining two singular values are quantisation noise. Muscle components are not
removed, because the gamma band carries the strongest ictal signal in this corpus
(Section 3.3). GEDAI is not an ICA: it removes what departs from a leadfield based
reference covariance, chosen by its own criterion, so the shared component rule
cannot apply to it. Its bundled reference covariance C, defined for a referential
10-05 layout, is mapped onto the bipolar montage as D C D^T, with D the electrode
difference matrix (a bipolar difference cancels the common reference), and the
channel mean is removed before and restored after cleaning, because GEDAI would
otherwise average reference the bipolar channels. GEDAI's threshold preset was chosen
before any classification run and on signal preservation only, as the advisor
required: on six subjects its default removed more than half of the power of 39
percent of recordings and of 33 percent of ictal windows, its conservative preset
("auto-") of 17 and 13 percent, and neither removed more from ictal than from
non-ictal windows within a recording (`src/gedai_qc.py`). P6b uses the conservative
preset; the default is kept for a robustness report. All three methods were bit for
bit reproducible with a fixed seed. Across all 670 recordings with selected windows,
the median recording keeps 78 percent of its power under Infomax and AMICA (about 1.5
components removed per recording) and 99.9 percent under GEDAI, but a minority is
cleaned heavily by every method: 22, 23 and 21 percent of recordings lose more than
half of their power under Infomax, AMICA and GEDAI (43 percent under GEDAI's
default). Compute per one hour recording differs by an order of magnitude (median
18 s for GEDAI, 33 s for Infomax, 213 s for AMICA; 4.7, 8.5 and 73 CPU hours in
total); per recording logs are in `results_v2/qc/p6_recording_logs.csv`.

**Seeds and device.** Every pipeline is trained with three seed sets (`--repeat`),
seed set 0 being the one used throughout Section 5. If the ranking of the fusion
operators proves unstable across seeds, the affected pipelines are extended to five.
All runs of one analysis use one machine, one thread count and one device. Before GPU
training is used, its agreement with the CPU is tested on one fold: two GPU runs must
repeat, and the CPU to GPU difference at identical seeds must not exceed the
difference between two CPU seed sets (`docs/PIPELINE.md`).

**Analyses** (`src/multiverse.py`, validated on synthetic runs with known effects):
(1) performance per pipeline, seed and model, with AUPRC added to the metrics of
Section 3.5; (2) rank stability of the five fusion operators, comparing Kendall's tau
between pipelines with Kendall's tau between seeds of the same pipeline, with a
permutation test that permutes pipeline labels within each seed set, and the same
comparison among P6a, P6b and P6c to ask whether the artefact algorithm alone changes
the ranking; (3) the Fusion x Preprocessing interaction in a repeated measures ANOVA
over folds (Greenhouse-Geisser corrected); (4) variance components for subject,
pipeline, fusion, their interaction and seed; (5) robust equivalence, a fusion pair
being equivalent in every pipeline under the corrected test of Section 3.5;
(6) robustness against performance, the mean macro F1 of each operator against the
spread of its pipeline means; and (7) prediction stability, the agreement of window
level predictions between pipelines compared with the agreement between seeds.

## 4. Bonn Results

### 4.1 Task formulations

The Bonn corpus (Andrzejak et al. 2001) comprises five sets of 100 single channel
segments each (23.6 s, 4097 samples, 173.61 Hz): A and B are surface (scalp)
recordings from healthy volunteers, C, D, and E are intracranial depth electrode
recordings from epilepsy patients, with E capturing ictal activity. We evaluate three
label groupings: **T1**, a three class task (Normal = A,B; Interictal = C,D; Ictal =
E); **T2**, an intracranial only binary task (C,D vs E) that removes the scalp versus
intracranial recording type as a confound; and **T3**, the conventional binary split
used in most published work (A,B,C,D vs E). We retain T3 as a reference point rather
than a headline result, because a single feature logistic regression (`logvar`)
reaches 0.954 macro F1 on it against 0.986 for the best of eleven models (Table 2),
independently consistent with NeuroAtlas's report that this split saturates.

### 4.2 Main results

**Table 2. Bonn, macro F1 and AUC, mean over 25 paired measurements (5x5 repeated
stratified CV).** Source: `results_v2/T{1,2,3}_*_N2_z_zspec_*/perfold.csv`.

| Model | T1 F1 | T1 AUC | T2 F1 | T2 AUC | T3 F1 | T3 AUC | Params |
|---|---|---|---|---|---|---|---|
| late | 0.974 | 0.999 | 0.987 | 1.000 | 0.986 | 1.000 | 98,571 |
| gated | 0.973 | 0.999 | 0.987 | 0.999 | 0.980 | 1.000 | 98,698 |
| attention | 0.969 | 0.997 | 0.985 | 0.999 | 0.977 | 0.998 | 98,544 |
| spec2d_wide | 0.966 | 0.998 | 0.984 | 0.999 | 0.982 | 1.000 | 97,775 |
| score | 0.964 | 0.998 | 0.981 | 0.999 | 0.976 | 0.999 | 59,302 |
| raw1d_wide | 0.963 | 0.997 | 0.973 | 0.998 | 0.976 | 0.992 | 99,795 |
| spec2d | 0.960 | 0.998 | 0.979 | 0.999 | 0.972 | 0.999 | 32,227 |
| shallow | 0.954 | 0.996 | 0.964 | 0.995 | 0.964 | 0.995 | 8 |
| raw1d | 0.951 | 0.996 | 0.964 | 0.997 | 0.969 | 0.995 | 27,075 |
| early | **0.896** | 0.977 | **0.923** | 0.981 | **0.899** | 0.965 | 97,507 |
| logvar | 0.646 | 0.755 | 0.933 | 0.986 | 0.954 | 0.992 | 1 |

![Bonn T1, mean macro F1 with 95% CI per model, grouped by family.](figures_manuscript/bonn_T1_forest.png)

**Figure 1.** Bonn T1, mean macro F1 with 95% CI per model, grouped by family.
Early/intermediate fusion (bold, Table 2) is the one operator visibly separated
from the fusion cluster; logvar sits far off axis and is shown pinned to the left
edge with its value labelled, rather than compressing the rest of the plot.

Two patterns hold across all three tasks. First, **early fusion is the one operator
that separates from the rest, and only downward** (bold in Table 2). Second, on T2
and T3, **naive Wilcoxon finds 20 and 13 "significant" pairs respectively out of 55,
and the corrected test finds zero on both** (`results_v2/phase0/summary.csv`); on T1,
naive finds 22 and corrected finds 10, all ten involving `logvar`, none involving a
pairwise comparison among the deep models (Section 4.3). We treat this collapse from
naive to corrected significance as itself a central finding, not a footnote (Section
6.1).

![Bonn T1, pairwise corrected significance and equivalence matrix, all 11 models.](figures_manuscript/bonn_T1_significance_corrected.png)

**Figure 2.** Bonn T1, all 11 models, pairwise outcome of the corrected test at
delta_min <= 0.03 macro F1. Green cells (late, gated, attention, score, and
the two widened unimodal controls) are established equivalent to each other; orange
cells are `logvar` against everything else, the only pairs that reach corrected
significance in this dataset. Grey cells, including every comparison involving early
fusion, are inconclusive at this margin and sample size, not equivalent and not
significantly different. Note that this figure sources the same 2-thread numerical
realisation as Table 2 (Section 6.2); the one significant early-vs-late pair reported
in Section 4.3 comes from a separate 12-thread realisation and does not appear here,
itself a small illustration of the thread count finding.

### 4.3 Are the fusion operators distinguishable from each other?

Restricting to the five fusion operators (early, late, gated, attention, score) and
their ten pairwise comparisons on T1, using the numerical realisation with recorded
log loss and Brier (Section 6.2 explains why two realisations exist), the corrected
test finds **one significant pair on macro F1** (early vs late, p_Holm = 0.048)
and **three on Brier** (early vs gated, early vs late, early vs score, all
p_Holm < 0.04); log loss finds none, consistent with its instability
(Section 3.5). Every other pair among late, gated, attention, and score is
statistically indistinguishable under the corrected test, with equivalence margins
δmin between 0.022 and 0.028 macro F1, and Bayesian posterior probability
of equivalence within a ±0.01 ROPE between 0.48 and 0.61 for these pairs, against
0.001 to 0.005 for pairs involving early. In short: **late, gated, attention, and
score are equivalent to each other within about 0.02 to 0.03 macro F1; early is not
equivalent to any of them, and is worse.**

### 4.4 Robustness of the null result

Repeating the T1 comparison at three spectrogram resolutions and three fusion
parameter budgets (`results_v2/phase0/sens_*`) leaves the conclusion intact at five
of six design points (9 of 45 pairs significant at each, all `logvar` related as in
Section 4.2); the one partial exception is that score level fusion, the one operator
that does not train its two branches jointly, degrades when one branch is
deliberately weakened, which is the expected behaviour of an operator that cannot
compensate for an underperforming branch during training.

The amplitude normalisation ablation (N1: spectrogram computed from the
unnormalised signal, against N2: both branches normalised) shows early fusion shifts
by 0.062 macro F1 between the two arms (naive p < 1e-6; corrected
p_Holm = 0.067, borderline after correction) while every other model is
statistically untouched (`results_v2/phase0/ablation_N1_vs_N2.csv`). This localises
an amplitude related shortcut to exactly the one architecture that concatenates the
raw signal into the spectrogram's channel dimension (Section 3.1), and is a plausible
mechanistic explanation for why early fusion is the one operator that separates.

## 5. CHB-MIT: External Validation

Bonn publishes no per segment subject identifier, so no claim in Section 4 bears on
whether a model generalises across patients rather than across segments drawn from a
small, possibly overlapping pool. CHB-MIT (Shoeb 2009; Goldberger et al. 2000),
publicly available via PhysioNet, provides 24 pediatric recording cases with subject
identifiers, making leave one subject out (LOSO) evaluation possible for the first
time in this comparison. Two of the 24 cases are documented as the same person, so
the evaluation has 23 folds rather than 24 (Section 5.2).

### 5.1 Data preparation and two errors found in the source data

We downloaded and verified the full corpus (686 recordings, 982.9 hours,
`src/verify_chbmit.py`), confirming file integrity against sizes computed from each
EDF header rather than trusting file listings.

**Annotation labelling error.** Summing seizure counts from each subject's summary
file gives 198 seizures, matching PhysioNet's published count, but the naive
approach of pairing "Seizure N Start/End Time" lines by their printed number produces
a total ictal duration of 17,856 seconds. One file, `chb09_08.edf`, contains a
labelling error in the source data: its second seizure's end line is itself printed
as "Seizure 1 End Time" rather than "Seizure 2", so a number based parser pairs the
first seizure's start (2951 s) with the second seizure's end (9267 s), fabricating a
6316 second seizure that does not exist. We replaced number based pairing with
sequence based pairing (each "Start Time" opens a seizure, each "End Time" closes the
most recently opened one, and the parsed count is cross checked against each file's
declared "Number of Seizures") and this corrects total ictal duration to 11,611
seconds, a 35 percent reduction, with every subject's parsed count now matching its
declaration (`src/chbmit.py: parse_summary`).

**Index file error.** PhysioNet's `RECORDS-WITH-SEIZURES` index lists
`chb07/chb07_18.edf` as containing a seizure, but that file's own summary declares
zero seizures and no `.seizures` annotation file exists for it on the server (HTTP
404); `chb07_19.edf` has both a declared seizure and a server side annotation file.
Both independent sources agree the index file names the wrong recording for this one
entry.

**Channel montage.** Channel sets vary not only between subjects but between
recordings of the same subject: part of subject 12's recordings use a different
reference montage (channels such as `C3-CS2`) that shares no channel with the
standard bipolar montage used elsewhere, so a naive intersection across all
recordings returns zero channels. We fix the standard 18 channel bipolar montage as
a target and drop recordings that do not contain it, retaining 683 of 686 recordings
and 185 of 198 seizures; the three dropped recordings and 13 dropped seizures are
entirely from the affected part of subject 12's data.

### 5.2 Windowing, class balance, and leakage control

Ten second, non overlapping windows. Non overlapping windows are an explicit choice:
overlapping, sub window strided windowing (which we considered to increase the
positive count) risks placing near identical copies of the same seizure on both
sides of a train or test split, an error documented in this literature (Ali et al.
2024). A window is labelled ictal if it overlaps an annotated seizure interval at
all, so a window that only partly covers a seizure's onset or offset is labelled
ictal. Of the 1,276 ictal windows, 948 lie entirely within a seizure and 328 (25.7
percent) are such boundary windows, containing a median of 5.0 seconds of annotated
seizure (minimum 1.0 second). The alternative, excluding boundary windows, is
implemented (`src/chbmit.py: plan_windows`, `guard_s`) but has not been run, and is
listed among the CHB-MIT sensitivity analyses still outstanding (Section 7).
Because ictal windows are 0.34 percent of all windows before subsampling, non
ictal windows are subsampled per subject to a 4:1 ratio against that subject's own
ictal count, and both this ratio and each subject's true unsampled recording duration
are retained so that clinically meaningful rates (false alarms per hour) can in
principle be computed on the true time base rather than the subsampled one (a
computation we have not yet performed; Section 7).

Two leakage rules are enforced and checked: no subject's windows may appear in both
train and test, and no single seizure's windows may be split across train and test.
We validated the leakage checker itself by constructing a deliberately leaky split
(windows shuffled at random, ignoring subject and seizure identity) and confirming it
is flagged on both rules; the real LOSO and grouped k-fold splits used throughout are
confirmed leakage free by the same checker (`src/chbmit_corpus.py`), under the
person level grouping described next.

**Two cases, one person.** PhysioNet's own dataset documentation states that "case
chb21 was obtained 1.5 years after case chb01, from the same female subject." We
therefore group subjects by person rather than by case folder: chb01 and chb21 form
one subject, held out together, which gives 23 LOSO folds over 24 cases
(`src/chbmit_corpus.py: SAME_SUBJECT`). The leakage checker validates the grouping
directly: applied to the earlier, case based split, it flags exactly the two folds
in which one of the pair was tested while the other was in training. One residual
uncertainty remains: case chb24 was added to the corpus later and is not listed in
its `SUBJECT-INFO` file, so we cannot confirm from the documentation that it is a
distinct person; we treat it as one. Within each training fold, four of the 22
training subjects are held out, as whole subjects, for the inner validation split
used for early stopping and for the score fusion weight.

The resulting corpus is 6,380 windows (1,276 ictal) from 24 cases, 23 subjects, and
18 channels. Model architectures are unchanged from Section 3.1 except that the first
convolutional layer of each backbone accepts 18 input channels rather than one; this
adds channels only to the input layer and leaves every reported Bonn parameter count
identical when re run at 1 input channel, which we verified directly rather than
assuming.

### 5.3 Results

**Table 3. CHB-MIT, macro F1, log loss, and Brier, mean over 23 leave one subject
out folds.** Source: `results_v2/chbmit/loso_grouped/perfold.csv`.

| Model | F1 | F1 SD | Log loss | Brier |
|---|---|---|---|---|
| spec2d_wide | 0.673 | 0.132 | 0.752 | 0.291 |
| shallow | 0.671 | 0.180 | 0.634 | 0.373 |
| spec2d | 0.671 | 0.121 | 1.012 | 0.330 |
| attention | 0.668 | 0.164 | 0.982 | 0.318 |
| raw1d | 0.668 | 0.152 | 0.700 | 0.311 |
| score | 0.666 | 0.154 | **0.499** | **0.274** |
| early | 0.666 | 0.164 | 1.047 | 0.354 |
| gated | 0.659 | 0.165 | 0.965 | 0.344 |
| raw1d_wide | 0.657 | 0.130 | 0.806 | 0.329 |
| late | 0.649 | 0.150 | 0.804 | 0.318 |
| logvar | 0.622 | 0.162 | 0.615 | 0.416 |

![CHB-MIT LOSO, mean macro F1 with 95% CI per model.](figures_manuscript/chbmit_forest.png)

**Figure 3.** CHB-MIT LOSO, mean macro F1 with 95% CI per model, same axis
conventions as Figure 1. Compare the width of these intervals to Figure 1's: every
model's 95% CI here spans roughly 0.10 to 0.16 macro F1, wide enough that all eleven
overlap substantially, which is the between subject variability problem stated in
prose below made visible.

**No pair among the five fusion operators separates under the corrected test on any
metric** (0 of 10 pairs on macro F1, log loss, or Brier;
`results_v2/chbmit/phase0_loso_grouped/chbmit_*.csv`), and in fact **no pair among
all 55 comparisons across all 11 models separates on macro F1, even under the
uncorrected naive test** (0 of 55). This is a stronger non separation than Bonn's,
where the naive test found 22 of 55 pairs "significant" before correction. The reason
is visible in the standard deviation column of Table 3: between subject variability
(SD 0.12 to 0.18 in macro F1) is large relative to the mean differences between
models (spread across all 11 models of 0.051, and across the five fusion operators of
0.020), so no resampling based test, corrected or not, has the power to resolve
differences at this scale from 23 subjects. Across all 55 pairs and all three
metrics, the corrected test finds a single difference: score fusion's lower log loss
than raw1d_wide (p_Holm = 0.048), which does not survive the conservative test to
train ratio (p_Holm = 0.079; Section 7) and which we therefore treat as fragile.

This is not the same finding as Bonn's, and we report it separately rather than
folding both into one sentence about operators being "indistinguishable". On Bonn,
equivalence is positively established: the Bayesian posterior probability that late,
gated, attention, and score are practically equivalent (within a ±0.01 macro F1
ROPE) reaches as high as 0.61 for some pairs. On CHB-MIT, the same posterior
probability for the same five operators' ten pairs ranges only 0.19 to 0.31, and the
corrected equivalence margin δmin ranges 0.049 to 0.079, wider than any
Bonn pair. **CHB-MIT does not show that the operators are equivalent; it shows only
that this design, at 23 subjects, cannot resolve whether they are equivalent or
different.** This is a materially weaker and more honest claim, and the distinction
matters for anyone citing this result: "we could not tell the difference" is not
"there is no difference."

![CHB-MIT LOSO, pairwise corrected significance and equivalence matrix, all 11 models.](figures_manuscript/chbmit_significance_corrected.png)

**Figure 4.** CHB-MIT LOSO, all 11 models, the same corrected test and the same
delta_min <= 0.03 margin as Figure 2. Every cell but one is inconclusive. The
exception, raw1d and score fusion established as equivalent (δmin = 0.026), is
expected rather than informative: score fusion is a weighted average of raw1d's and
spec2d's predicted probabilities in which raw1d receives a mean validation selected
weight of 0.65, so the two make largely the same decisions. Contrasted directly
with Figure 2, which is mostly green (equivalent) with a clear orange block
(`logvar` significantly worse), this is the clearest single illustration in this
paper of the difference between "no difference found" and "equivalence established":
identical method, identical margin, and CHB-MIT's between subject variance is enough
to erase every distinction Bonn's design could draw, including the one, `logvar`
being worse, that Bonn establishes most strongly of all.

A further consequence is that **early fusion's Bonn specific weakness does not
replicate on CHB-MIT**: it ranks seventh of eleven models by macro F1, not last, and
the lowest ranked deep model is instead late fusion, the operator with the highest
mean on Bonn T1, although no CHB-MIT difference is significant and Section 5.4 shows
that this ordering is not stable across runs. We cannot distinguish, with the present data, whether early fusion's Bonn
result reflects something genuine about that architecture or reflects a Bonn specific
artefact (Section 4.4's amplitude shortcut finding is itself Bonn specific, since it
concerns the interaction between early fusion and a spectrogram caching bug in the
Bonn pipeline the current implementation does not share).

**Classical features remain competitive.** The `shallow` classical baseline (a
logistic regression on seven hand engineered features per channel) ranks second of
eleven models by macro F1, 0.002 behind the best model, with a higher mean than all
five fusion operators and the raw1d_wide control, though none of these differences is
significant. It ranked third in the earlier run as well (Section 5.4). This is a less
extreme version of Bonn's saturation finding (Section 4.1) but the same direction:
the representation learning component of these models is not obviously earning its
keep over a linear classifier on hand engineered features, on either corpus.

**Score level fusion is best calibrated, not necessarily most accurate.** Score
fusion, the simplest of the five operators, an ensemble of two independently trained
unimodal models with no additional training, achieves the lowest log loss of all
eleven models (0.499, next best 0.615) and the lowest Brier score (0.274, next best
0.291), the latter not significantly different from any other model. Its macro F1
(0.666) is unremarkable and not separable from the other operators. This is also one
of the few CHB-MIT patterns that held across both runs (Section 5.4: log loss 0.501
and joint lowest Brier in the earlier run). We flag it as an interesting but narrow
finding: classification accuracy and probabilistic calibration are different
properties, and an operator that wins on one need not win on the other. This has a
plausible theoretical explanation rather than being an accident of this dataset:
Mattei and Garreau (2025) show that averaging predictions from an ensemble of
models is guaranteed to help a convex loss, by Jensen's inequality, but carries no
such guarantee for a non convex one. Log loss and Brier score are convex, macro F1
is not, which is consistent with score fusion (the one operator here that is a
literal post hoc average of two independent models) improving on the two convex
metrics without improving on the non convex one.

### 5.4 Re-running CHB-MIT: what changed and what did not

The results above come from a full re-run. An earlier CHB-MIT run
(`results_v2/chbmit/loso_main/`), on which a previous draft of this manuscript was
based, differed in four ways: it grouped subjects by case folder, giving 24 folds and
letting chb01 and chb21 appear on opposite sides of a split; it used Adam rather than
AdamW; it ran on a different machine with mixed thread counts (12 threads for the
first five folds, 6 for the rest); and it computed score fusion in a separate script.
The corrected run changes all four at once, so the differences below cannot be
attributed to any single one of them.

**The conclusions did not change.** In both runs, 0 of 55 pairs separate on macro F1
under either the naive or the corrected test, no pair of fusion operators separates
on any metric, equivalence is not established for any fusion pair, and score fusion
has the lowest log loss.

**The ranking did.** Mean macro F1 per model moved by between −0.030 and +0.027, and
the rank correlation of the eleven models between the two runs is weak and not
significant (Spearman rho = 0.45, p = 0.17). The best fusion operator was gated in
the earlier run (second of eleven) and is attention in this one; gated fell to eighth
and late fusion from fifth to tenth.

**The cause lies in training, not in the data.** On the 22 subjects whose test folds
are identical in the two runs (every subject except chb01 and chb21), the per fold
macro F1 of the nine deep models changed by a median of 0.056 and by up to 0.364 (gated
fusion on chb05), while the two classical baselines, which involve no neural network
training, changed by at most 0.004. The windows, labels, and features are therefore
the same; what changed is how the networks trained.

**The grouping error distorted two folds, in opposite directions.** In the earlier
run, the fold testing chb01 (with chb21 in training) was among the best of all folds
(late fusion 0.981, its best fold), while the fold testing chb21 (with chb01 in
training) was the second worst averaged over the deep models: six of the nine
detected no ictal window at all (late fusion 0.444). Held out together, the same person now scores 0.841 with late
fusion. The leak therefore did not simply inflate results, and its net effect on the
mean was small, but it made two of 24 folds unrepresentative.

We read this as the thread count finding of Section 6.2 at a larger scale: a set of
changes with no bearing on which fusion operator is better moves individual models by
as much as the entire spread between them. A single CHB-MIT run supports the null
result, which is about the absence of separable differences, but it does not support
any ranking of the operators, and neither run's ordering should be cited as one.

### 5.5 Third corpus: Siena Scalp EEG (in progress)

The Siena Scalp EEG Database (Detti et al. 2020; PhysioNet) holds 41 recordings of 14
adult patients (47 seizures, 512 Hz, referential 10-20 montage; 141.0 hours by the EDF
headers). The database description gives about 128 hours, the total of its
`subject_info.csv`, which differs from the recordings for four patients: for PN14 it
inherits a three-hour error in the seizure list, and for PN03, PN10 and PN12 it is
11.7, 2.0 and 2.0 hours shorter than the files, without an explanation in the data
(the files are complete and contain no filler; `docs/SIENA.md`, Section 8). After
windowing and the same
1:4 subsampling as CHB-MIT, the corpus has 1,640 windows (328 ictal) and 14 leave one
subject out folds, confirmed free of leakage by the same checker. It is
converted to exactly the CHB-MIT form, so every driver is shared: the 18 channels of
the CHB-MIT bipolar montage are derived from the referential electrodes, the signal
is resampled to 256 Hz on the continuous recording, and windowing, labelling and
subsampling are CHB-MIT's own (`src/siena.py`, `src/siena_corpus.py`). Siena serves
two purposes: an external replication of the multiverse on adult patients, and,
separately from the multiverse, a cross dataset experiment in which models trained on
CHB-MIT are tested on Siena. Differences between the corpora will not be interpreted
as an age effect, because seizure types, recording structure and acquisition also
differ.

The published seizure lists required checks before use (`docs/SIENA.md`). All 41
headers are internally consistent; the listed recording start agrees with the header
in 39 recordings. In PN14-3 the list is three hours off and the header is verifiably
right (its start plus the recording's duration gives the listed end exactly). In PN05-3
they differ by 10 s and neither source can be confirmed; using the header there is a
decision rule (the header's start is used for every recording), not a verified
correction, and moves the labels of two windows. One seizure end in PN00 lies 32
minutes after its recording ends; we accepted the evident correction (one hour
earlier, a 60 s seizure) only after the signal confirmed it: measured with the same
rule as the patient's other four seizures, the ictal rhythm ends 14 s after the
corrected end, within the range of those four (0 to 13 s). For PN10 the electrical
onset is used where both onsets are listed; it changes the label of one window.

### 5.6 Multiverse results (pending)

[Pending: the nine pipelines x three seed sets on CHB-MIT are being run; tables and
figures will be produced by `python -m src.multiverse --dataset chbmit`: performance
per pipeline, rank stability heatmap, interaction plot, variance components, robust
equivalence, robustness against performance, and prediction stability.]

## 6. Discussion: What Does "Cannot Distinguish" Mean Here

### 6.1 The naive test is not a minor technicality

Across both corpora, the gap between naive and corrected significance is the single
largest effect size in this paper. On Bonn T2 and T3, naive testing reports 20 and 13
"significant" differences respectively that the corrected test reduces to zero. Our
simulation (Section 3.5) shows this is expected: the naive test's false positive rate
under repeated cross validation is 37 to 39 percent against a nominal 5 percent, an
error rate large enough that a substantial fraction of "significant" findings
reported under this common practice, in this literature and adjacent ones, should be
presumed spurious until shown otherwise by a corrected re-analysis. We are not aware
of prior work in the EEG seizure detection literature specifically that quantifies
this by simulation, though the underlying statistical point (Nadeau and Bengio 2003;
Bouckaert and Frank 2004) predates this application by two decades.

### 6.2 Thread count as an unreported confound

While verifying determinism of the CHB-MIT pipeline we found that CPU thread count
alone, set via `torch.set_num_threads` and otherwise invisible in any results table,
shifts per fold macro F1 by up to 0.093 on Bonn and 0.208 on CHB-MIT (a single
outlier fold; median effect near zero, mean effect 0.05 across the 24 fold by 2 model
comparisons we made, on the earlier case grouped configuration), holding code, data,
and random seed fixed
(`docs/10_OLASILIK_VE_TEKRARLANABILIRLIK.md`, `docs/13_CHBMIT_ISTATISTIK.md`). We
confirmed this is deterministic at fixed thread count (three repeated runs at
identical settings reproduce identical results to the last recorded decimal) and
that the effect appears only when thread count changes, which localises the cause to
floating point summation order in multi threaded BLAS and convolution routines
rather than to any nondeterministic scheduling. The consequence for interpretation is
direct: on Bonn, the spread across our five fusion operators (0.078 macro F1, Table
2) is comparable to the spread this single, scientifically meaningless setting can
introduce on its own (0.093, measured on a separate numerical realisation of the same
comparison); on CHB-MIT the thread effect can exceed the entire spread across all
eleven models (0.051, Table 3). The re-run of Section 5.4 shows the same phenomenon
from a different direction: changing the optimiser, thread count, and machine
together reordered the CHB-MIT models while leaving every conclusion intact. The final
CHB-MIT run was therefore made on a single machine at one fixed thread count, and the
thread count is recorded with every run. Thread count, batch order and library version
are not routinely reported in this literature; our finding suggests they should be, and that
result tables from single, unreported hardware configurations warrant more caution
than they are usually given.

### 6.3 Two corpora, two different reasons for a null result

We deliberately avoid summarising Bonn and CHB-MIT with the same sentence. Bonn's
task, particularly its conventional binary split, is saturated (Section 4.1;
independently confirmed by Kontras et al. 2026), and within that regime the five
fusion operators are shown, positively, to be statistically equivalent to each other
at a fairly tight margin. CHB-MIT is not saturated in the same way (macro F1 in the
0.62 to 0.67 range, well below ceiling) but its between subject variability is large
enough that with 23 subjects, this design has too little power to establish either
equivalence or difference between the operators. Both are negative results for the
practical question "does the fusion operator matter", but they are negative for
different statistical reasons, and a reader who only takes away "no difference was
found" has lost the more useful, more falsifiable content of each result.

### 6.4 Defects found and corrected in this project's own history

Consistent with the standard we apply to the published literature, we report the
defects found during this project's own development rather than omitting them.

| Defect | How it was found | Consequence |
|---|---|---|
| Spectrogram cache built from the unnormalised signal | Cached values matched the unnormalised spectrogram to numerical precision | Confounded early fusion's operator effect with an amplitude shortcut (Section 4.4) |
| Early stopping froze 6 of 55 runs in a trivial single class solution | F1 near 0.40 co-occurring with AUC near 0.96 | Fixed with a minimum epoch floor; affected runs recovered to 0.96 to 1.00 |
| Naive paired testing under repeated cross validation | Simulation (Section 3.5) | Most originally reported "significant" differences did not survive correction |
| CHB-MIT annotation number based parsing | A 27 minute average seizure duration in one subject was not physiologically plausible | Total ictal duration corrected downward by 35 percent (Section 5.1) |
| CPU thread count changes results | Direct controlled test after an unusual wall clock slowdown was investigated | Documented as a confound (Section 6.2), not fully eliminated |
| CHB-MIT subjects grouped by case folder | PhysioNet's documentation states chb21 is chb01 recorded 1.5 years later | Two folds distorted in opposite directions; corrected by grouping by person and re-running all models (Sections 5.2, 5.4) |
| CHB-MIT trained with Adam, Bonn with AdamW | Documentation of the training protocol | Optimiser unified to AdamW in the re-run (Section 3.4) |

## 7. Limitations

- **The CHB-MIT results of Section 5.3 are a single training run per model.** Section
  5.4 shows that the ranking of models is not stable across runs that differ only in
  settings unrelated to the fusion operator. The multiverse of Section 3.6 trains every
  pipeline with three seed sets for this reason; until its results are in, no ranking
  of operators on CHB-MIT should be cited. Case chb24's identity as a distinct person
  is assumed rather than documented (Section 5.2).
- **The artefact removal algorithms are not compared on equal terms in one respect.**
  Infomax and AMICA share one automatic component rule, but GEDAI selects what to
  remove by its own criterion (Section 3.6), so differences involving GEDAI combine
  the algorithm with its selection rule. Ocular components are identified without an
  EOG channel, from frontal proxies, which may also capture frontal ictal activity.
- **The corrected resampled t test assumes a fixed test to train ratio, which an
  inner validation split complicates.** Del Pup et al. (2025) argue this assumption
  is frequently violated in deep learning pipelines that hold out an internal
  validation set, as ours does for early stopping (Section 3.4). Our primary analysis
  uses the standard ratio 1/(k-1) (0.25 for Bonn, 1/22 for CHB-MIT LOSO). As a
  sensitivity check we recomputed every test with a conservative ratio based on the
  data actually used for fitting once the validation split is removed (0.3125 for
  Bonn; 1/18 for CHB-MIT, where four of the 22 training subjects are held out for
  validation). No conclusion on macro F1 changes: the number of Holm significant
  pairs is identical in all four analyses (10 of 55 on Bonn T1; 0 of 55 on Bonn T2,
  Bonn T3, and CHB-MIT), and the corrected equivalence bounds for the fusion operator
  pairs widen by at most 0.005 macro F1 (`src/phase0.py`, `src/chbmit_stats.py`,
  column `p_corrected_conservative`). The one CHB-MIT pair significant under the
  primary ratio, on log loss (Section 5.3), is not significant under the
  conservative one. This addresses the objection empirically for our
  design; it does not replace a correction derived for nested splits.
- **The systematic literature search is incomplete.** Its title and abstract
  screening was done by an AI assistant, checked by a second, independent AI pass
  (agreement 91.9 percent, kappa 0.81) but not yet by a human reviewer, which a PRISMA
  conforming review would require; and 122 of the 128 candidates have not been read
  in full. The gap claimed in Section 2 rests on the closest studies, which were read
  in full (`docs/15_LITERATUR_SONUCLARI.md`).
- **CHB-MIT's clinically relevant false alarm rate has not been computed on the true,
  unsubsampled time base**; the subsampled rate we could report would not be
  clinically meaningful and we have chosen not to report it rather than report a
  number likely to be misread (Section 5.2).
- **No sensitivity analysis has been run on CHB-MIT** corresponding to Section 4.4's
  Bonn analysis (window length, frequency ceiling, line noise removal, subsampling
  ratio, and whether the 25.7 percent of ictal windows that only partly overlap a
  seizure are kept; Section 5.2); the CHB-MIT results in Section 5.3 should be read
  as a single design point.
- **Hardware is CPU only**, which bounds both the scale of models we could evaluate
  and, per Section 6.2, appears to bound the numerical reproducibility of the results
  themselves.
- **The current implementation is treated by its own authors as preliminary.** The
  proposal this project is attached to states an intention to rebuild the pipeline
  cleanly under advisor supervision before treating any of these numbers as final;
  this manuscript should be read as reporting what that preliminary implementation
  currently shows, not as a final account.

## 8. Use of AI Assistance

An AI assistant (Claude) was used substantially in this work: implementation of the
data pipeline, model code, and statistical framework; execution of the experiments
reported here; drafting of this manuscript; and verification of the reference list
against Crossref. The choice of topic, dataset, and research question, the original
implementation that was later audited, and the decision at each branch point in the
analysis (including which design choices to treat as ablations, and which findings
to report as limitations rather than omit) were made by the author, who takes
responsibility for understanding and being able to defend every part of this work.

## 9. Data and Code Availability

All code, the per fold results underlying every number in this manuscript, and the
analysis and figure generation scripts are publicly available at
github.com/Atytmr07/eeg-fusion-benchmark. `docs/PIPELINE.md` in that repository
documents the full pipeline and the command that reproduces each table and figure;
the statistics and figures regenerate identically from the stored results with the
pinned package versions. Raw EEG data (Bonn, CHB-MIT, Siena) is not redistributed;
all three corpora are publicly available from their original sources (Andrzejak et al.
2001; PhysioNet), and the repository documents how to obtain and verify them
(`src/chbmit.py`, `src/verify_chbmit.py`, `docs/11_CHBMIT_PLANI.md`,
`src/siena_download.py`, `src/verify_siena.py`, `docs/SIENA.md`). The packages used
only to build the P6 caches are listed in `requirements-ica.txt`.

## References

- Ali, E. et al. (2024) Epileptic seizure detection using CHB-MIT dataset: The overlooked perspectives. *Royal Society Open Science* 11(5):230601. doi:10.1098/rsos.230601
- Acharya, U. R. et al. (2018) Deep convolutional neural network for the automated detection and diagnosis of seizure using EEG signals. *Computers in Biology and Medicine* 100:270-278. doi:10.1016/j.compbiomed.2017.09.017
- An, J., Peng, R., Yang, X., Wu, D. (2026) fastSeizureNet: Accurate and efficient knowledge-data fusion for semi-supervised seizure detection. *Neural Networks* 202:109078. doi:10.1016/j.neunet.2026.109078
- Andrzejak, R. G. et al. (2001) Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity: Dependence on recording region and brain state. *Physical Review E* 64(6):061907. doi:10.1103/PhysRevE.64.061907
- Basheer, R., Mishra, D. (2026) Learning feature-wise temporal-spatial gating with Kolmogorov-Arnold networks for EEG seizure detection. *Pattern Recognition Letters* 208:1-6. doi:10.1016/j.patrec.2026.07.016
- Camastra, C., Pelagi, A., Quattrone, A., Sarica, A. (2026) Benchmarking Multimodal Deep Fusion Strategies for Heterogeneous Neuroimaging and Cognitive Data Using a Controlled Sex Classification Task. *Brain Sciences* 16(4):405. doi:10.3390/brainsci16040405
- Bouckaert, R. R., Frank, E. (2004) Evaluating the Replicability of Significance Tests for Comparing Learning Algorithms. *Lecture Notes in Computer Science* (PAKDD 2004). doi:10.1007/978-3-540-24775-3_3
- Chatzichristos, C. et al. (2020) Epileptic Seizure Detection in EEG via Fusion of Multi-View Attention-Gated U-Net Deep Neural Networks. *IEEE SPMB*. doi:10.1109/spmb50085.2020.9353630
- Corani, G., Benavoli, A. (2015) A Bayesian approach for comparing cross-validated algorithms on multiple data sets. *Machine Learning* 100:285-304. doi:10.1007/s10994-015-5486-z
- Corani, G., Benavoli, A., Demšar, J., Mangili, F., Zaffalon, M. (2017) Statistical comparison of classifiers through Bayesian hierarchical modelling. *Machine Learning* 106:1817-1837. doi:10.1007/s10994-017-5641-9
- Dan, J. et al. (2024) SzCORE: Seizure Community Open-Source Research Evaluation framework for the validation of electroencephalography-based automated seizure detection algorithms. *Epilepsia* 66(S3). doi:10.1111/epi.18113
- Daşdemir, A., Örnek, H. K. (2025) Epileptic seizure prediction with deep learning-based fusion methods. *Engineering Science and Technology, an International Journal* 72:102212. doi:10.1016/j.jestch.2025.102212
- Das, S. et al. (2024) Epileptic Seizure Detection from Decomposed EEG Signal through 1D and 2D Feature Representation and Convolutional Neural Network. *Information* 15(5):256. doi:10.3390/info15050256
- Detti, P., Vatti, G., Zabalo Manrique de Lara, G. (2020) EEG Synchronization Analysis for Seizure Prediction: A Study on Data of Noninvasive Recordings. *Processes* 8(7):846. doi:10.3390/pr8070846
- Del Pup, F., Zanola, A., Tshimanga, L. F., Bertoldo, A., Atzori, M. (2025) The More, the Better? Evaluating the Role of EEG Preprocessing for Deep Learning Applications. *IEEE Transactions on Neural Systems and Rehabilitation Engineering* 33:1061-1070. doi:10.1109/TNSRE.2025.3547616
- Goldberger, A. L. et al. (2000) PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. *Circulation* 101(23):e215-e220. doi:10.1161/01.cir.101.23.e215
- Einizade, A., Nasiri, S., Mozafari, M., Sardouie, S. H., Clifford, G. D. (2023) Explainable automated seizure detection using attentive deep multi-view networks. *Biomedical Signal Processing and Control* 79:104076. doi:10.1016/j.bspc.2022.104076
- Golrizkhatami, Z., Acan, A. (2018) ECG classification using three-level fusion of different feature descriptors. *Expert Systems with Applications* 114:54-64. doi:10.1016/j.eswa.2018.07.030
- Jafrasteh, B., Adeli, E., Pohl, K. M., Kuceyeski, A., Sabuncu, M. R., Zhao, Q. (2025) Statistical variability in comparing accuracy of neuroimaging based classification models via cross validation. *Scientific Reports* 15:28745. doi:10.1038/s41598-025-12026-2
- Gramfort, A. et al. (2013) MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience* 7:267. doi:10.3389/fnins.2013.00267
- Huang, J. et al. (2024) Multi-modal feature fusion with multi-head self-attention for epileptic EEG signals. *Mathematical Biosciences and Engineering* 21(2). doi:10.3934/mbe.2024304
- Kontras, K. et al. (2026) NeuroAtlas: Benchmarking Foundation Models for Clinical EEG and Brain-Computer Interfaces. arXiv:2605.14698.
- Lee, H.-T., Shim, M., Liu, X., Cheon, H.-R., Kim, S.-G., Han, C.-H., Hwang, H.-J. (2025) A review of hybrid EEG-based multimodal human-computer interfaces using deep learning: applications, advances, and challenges. *Biomedical Engineering Letters* 15:587-618. doi:10.1007/s13534-025-00469-5
- Lee, T.-W., Girolami, M., Sejnowski, T. J. (1999) Independent component analysis using an extended infomax algorithm for mixed subgaussian and supergaussian sources. *Neural Computation* 11(2):417-441. doi:10.1162/089976699300016719
- Loshchilov, I., Hutter, F. (2019) Decoupled Weight Decay Regularization. *International Conference on Learning Representations (ICLR)*. arXiv:1711.05101.
- Mattei, P.-A., Garreau, D. (2025) Are Ensembles Getting Better all the Time? *Journal of Machine Learning Research* 26(201):1-46. arXiv:2311.17885.
- Mohamady, A., Burchard, R., Van Laerhoven, K. (2026) A Comparison of Fusion Techniques for Multi-Modal Human Activity Recognition on the HARMES Dataset. arXiv:2606.27886.
- Nadeau, C., Bengio, Y. (2003) Inference for the Generalization Error. *Machine Learning* 52:239-281. doi:10.1023/a:1024068626366
- Narotamo, H., Dias, M., Santos, R., Carreiro, A. V., Gamboa, H., Silveira, M. (2024) Deep learning for ECG classification: A comparative study of 1D and 2D representations and multimodal fusion approaches. *Biomedical Signal Processing and Control* 93:106141. doi:10.1016/j.bspc.2024.106141
- Palmer, J. A., Makeig, S., Kreutz-Delgado, K., Rao, B. D. (2008) Newton method for the ICA mixture model. *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 1805-1808. doi:10.1109/ICASSP.2008.4517982
- Rheude, T., Eils, R., Wild, B. (2025) Fusion or Confusion? Multimodal Complexity Is Not All You Need. arXiv:2512.22991.
- Ros, T., Férat, V., Huang, Y., Colangelo, C., Kia, S. M., Wolfers, T., Vulliemoz, S., Michela, A. (2025) Return of the GEDAI: Unsupervised EEG Denoising based on Leadfield Filtering. *bioRxiv*. doi:10.1101/2025.10.04.680449
- Roy, Y. et al. (2019) Deep learning-based electroencephalography analysis: a systematic review. *Journal of Neural Engineering* 16:051001. doi:10.1088/1741-2552/ab260c
- Shoeb, A. H. (2009) Application of machine learning to epileptic seizure onset detection and treatment. PhD thesis, Massachusetts Institute of Technology.
- Shoeibi, A. et al. (2021) Epileptic Seizures Detection Using Deep Learning Techniques: A Review. *International Journal of Environmental Research and Public Health* 18(11):5780. doi:10.3390/ijerph18115780
- Steegen, S., Tuerlinckx, F., Gelman, A., Vanpaemel, W. (2016) Increasing Transparency Through a Multiverse Analysis. *Perspectives on Psychological Science* 11(5):702-712. doi:10.1177/1745691616658637
- Truong, N. D. et al. (2018) Convolutional neural networks for seizure prediction using intracranial and scalp electroencephalogram. *Neural Networks* 105:104-111. doi:10.1016/j.neunet.2018.04.018
- Wang, X. et al. (2020) One and Two Dimensional Convolutional Neural Networks for Seizure Detection Using EEG Signals. *EUSIPCO 2020*. doi:10.23919/eusipco47968.2020.9287640
- Wang, J., Wei, L., Qian, Z., Shi, C., Liu, Y., Xu, Y. (2026) An explainable multi-view representation fusion learning framework with hybrid MetaFormer for EEG-based epileptic seizure detection. *Neurocomputing* 675:132929. doi:10.1016/j.neucom.2026.132929
- Xu, J. et al. (2024) EEG-based epileptic seizure detection using deep learning techniques: A survey. *Neurocomputing* 610:128644. doi:10.1016/j.neucom.2024.128644
