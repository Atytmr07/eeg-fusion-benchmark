# Senior Project Proposal

**Do Fusion Strategies Matter? A Controlled Comparison of Multimodal CNN Fusion for EEG Seizure Detection**

Emre Atay Tümer
Computer Engineering and Industrial Engineering, double major
Antalya Bilim University

---

## 1. Topic

Electroencephalography (EEG) records the electrical activity of the brain over time. During
an epileptic seizure the signal changes in a characteristic way: amplitude rises, the
waveform becomes rhythmic, and power concentrates in particular frequency bands. Automatic
seizure detection is therefore a signal classification problem, and convolutional neural
networks (CNNs) are the dominant approach to it.

A recurring design pattern in this literature is to represent the same EEG segment in two
ways at once:

- the **raw waveform**, processed by a 1D CNN, which preserves the exact shape of the
  signal in time;
- the **spectrogram**, processed by a 2D CNN, which shows how the frequency content changes
  over time.

The two branches are then combined by a **fusion operator**. The common variants are early
fusion (combine the inputs or the early features), late fusion (combine the final feature
vectors), gated fusion (the network learns how much to trust each branch), attention based
fusion, and score level fusion (average the two independent predictions).

## 2. The question

Published studies typically report that their chosen fusion operator outperforms the
alternatives. However, in these comparisons the models usually differ in more than the
operator: parameter count, training schedule, regularisation and data splits vary between
the compared systems. When a fusion model with more parameters beats a unimodal model with
fewer, it is not clear whether the gain comes from the fusion strategy or simply from the
added capacity.

**Research question.** When architecture size, training protocol and data are held fixed
and only the fusion operator is varied, does the choice of fusion operator produce a
measurable and statistically defensible difference in seizure detection performance?

This is a comparative and methodological study rather than a proposal for a new
architecture. Its intended contribution is a fair and reproducible answer to a question the
field currently answers informally.

## 3. Where the literature stands

The works below were verified through Crossref; each was resolved by DOI and its metadata
checked against the publisher record.

| Work | What it contributes | What it leaves open |
|---|---|---|
| Andrzejak et al. 2001 | The Bonn EEG corpus, the most widely used benchmark for this task | No per segment subject identifiers, so subject wise validation is not possible |
| Acharya et al. 2018 | One of the first deep CNNs for seizure detection on Bonn | Single architecture, no comparison of design choices |
| Truong et al. 2018 | CNN over spectrograms, both scalp and intracranial data | Spectrogram only, no raw signal branch to compare against |
| Wang et al. 2020 | Compares 1D and 2D CNNs for seizure detection | Compares the two separately, does not fuse them |
| Chatzichristos et al. 2020 | Attention gated fusion of multiple EEG views | Proposes one operator, does not compare against alternatives |
| Das et al. 2024 | Combines 1D and 2D representations of decomposed EEG | Fixed fusion design, no ablation of the operator |
| Huang et al. 2024 | Multi head self attention fusion for epileptic EEG | Reports gains but does not match parameter counts |
| Golrizkhatami and Acan 2018 | Three level fusion of feature descriptors for ECG | Different signal domain, but directly relevant fusion methodology |
| Roy 2019; Shoeibi 2021; Xu 2024 | Systematic reviews of deep learning for EEG | Report widespread inconsistency in evaluation protocols across studies |
| Ali et al. 2024 | Shows that reported seizure detection results depend heavily on evaluation choices | Concerns CHB-MIT, not fusion operators |
| Rheude et al. 2025 | Argues that multimodal complexity often does not pay off | General multimodal setting, not EEG seizure detection |
| **Daşdemir 2025** | The closest work: a self described fair, protocol controlled comparison of fusion strategies for EEG, with the two branches held fixed | Compares two fusion points only, addresses seizure prediction rather than detection, and reports 97.50 against 97.70 accuracy without statistical testing or stated parameter matching |
| Brain Sciences 2026 | A controlled benchmark of fusion strategies in a unified framework, concluding that fusion strategy matters more than architectural complexity | Tabular neuroimaging and cognitive features rather than time series; parameter counts not matched; no paired or equivalence testing |
| Mohamady et al. 2026 | Head to head comparison of seven fusion techniques on a common benchmark, in human activity recognition | Different domain; states that no such head to head comparison existed in that field as of 2026, which indicates the design is still uncommon |
| Kontras et al. 2026 (NeuroAtlas) | Large scale EEG benchmark across 42 datasets | Reports that the Bonn benchmark saturates and should be treated as a sanity check, and that Bonn carries no subject identifiers |

**Reading of this table.** Fusion operators for EEG seizure detection are proposed
frequently, but they are usually proposed one at a time and evaluated against baselines
that differ in capacity. The closest existing work, Daşdemir (2025), does hold the branches
fixed, but compares two fusion points on a prediction task and reports a difference of
0.20 accuracy points without any statistical test of whether that difference is real.

The gap I intend to address is therefore narrower than a simple absence, and I would state
it as follows: I have not found a study that compares three or more fusion operators for
EEG seizure detection under a matched parameter budget, with paired statistical testing and
equivalence testing, so that the absence of a difference can be demonstrated rather than
inferred from a null result.

I should state clearly that my search is not yet systematic. Before claiming novelty I
intend to run a registered search on Scopus and Web of Science with a documented screening
procedure, and to present the outcome as a proper comparison table. I also intend to obtain
and read the Daşdemir paper in full, since it is the single work closest to my design.

## 4. Proposed method

1. **Data.** Start with the Bonn corpus (five sets of 100 single channel segments,
   173.61 Hz). It is small, public and widely used, which makes controlled comparison
   feasible. Its limitations are stated in Section 7.
2. **Task definitions.** Evaluate under more than one label grouping, since different
   papers group the Bonn sets differently and this choice alone changes the reported
   difficulty of the task.
3. **Fixed backbones.** One 1D CNN over the raw segment and one 2D CNN over the
   spectrogram, identical across all conditions.
4. **Parameter matching.** Size every fusion head so that total parameter counts agree
   within a small tolerance, so that the operator is not confounded with capacity.
5. **Controlled variation.** Vary only the fusion operator: early, late, gated, attention
   and score level.
6. **Reference points.** Include each single branch alone, a capacity matched widened
   single branch, and a simple classical baseline on hand crafted features, to establish
   how much of the performance is attributable to the deep models at all.
7. **Evaluation.** Repeated stratified cross validation with a held out inner validation
   split, reporting macro F1 together with probabilistic scores such as log loss, and
   comparing models with paired statistical tests and equivalence testing, so that the
   absence of a difference can be demonstrated rather than assumed.
8. **Reproducibility.** Deterministic seeding per fold, stored configurations, and every
   reported number traceable to a stored result file.

## 5. Timeline

| Period | Work |
|---|---|
| Fall 2026, weeks 1 to 4 | Systematic literature search with a documented screening procedure and a comparison table |
| Fall 2026, weeks 5 to 8 | Clean re-implementation, with the design decisions fixed in advance rather than settled during analysis |
| Fall 2026, weeks 9 to 14 | Main experiments on both corpora and statistical analysis |
| Spring 2027, weeks 1 to 6 | Sensitivity analysis: window length, frequency ceiling, line noise removal, subsampling ratio, normalisation scheme |
| Spring 2027, weeks 7 to 12 | Manuscript, figures, defence preparation |

The external validation originally planned for the spring is already built and running,
which is why the second half of the schedule is given to sensitivity analysis instead.

## 6. What already exists

I began working on this topic last year with Dr. Yusuf Öztürk in Electrical Engineering.
The work stalled because I could not maintain it alongside my course load, and I put it on
hold. Since returning to it I have built a working implementation and taken it through a
full audit.

**On the Bonn corpus.** A complete pipeline: the two backbones, all five fusion operators
at a matched parameter budget, capacity matched single branch controls, classical feature
baselines, and repeated stratified cross validation with deterministic per fold seeding.

**On CHB-MIT.** The external validation described in Section 5 is no longer only a plan.
The corpus is downloaded and verified (686 recordings, 983 hours, 24 subjects), and a
subject wise evaluation pipeline is built and tested: leave one subject out splitting, an
inner validation split that is also subject wise, and a leakage check that I validated by
deliberately feeding it a leaky split to confirm it fails when it should. A full leave one
subject out comparison is currently running.

**I am treating the numbers as preliminary and expect to rebuild the study cleanly under
supervision.** During the audit I found defects that changed the conclusions, and I would
rather present them than hide them:

- A training bug let some models freeze in a trivial single class solution while the
  reported AUC still looked high.
- The two branches were receiving inconsistently normalised inputs.
- The statistical test I had used assumes independent measurements, which repeated cross
  validation violates. I estimated its false positive rate by simulation at 38 percent,
  and most of the differences I had reported did not survive a corrected test.
- Thread count alone, an experimentally irrelevant runtime setting, shifted per fold macro
  F1 by up to 0.09 and reversed the direction of 3 of 55 model comparisons.
- In CHB-MIT itself, one annotation file mislabels a seizure boundary in a way that makes a
  naive parser produce a 6316 second seizure that does not exist, and the official index of
  seizure records points to the wrong recording for one subject.

Finding these is most of why I would like supervision on the experimental design before
committing to results.

## 7. Known limitations

- **Bonn publishes no per segment subject identifiers**, so subject wise cross validation
  is not merely omitted, it is not applicable. This is not my own inference: the NeuroAtlas
  benchmark states that "Bonn provides no per-trial subject identifier, so subject-grouped
  cross-validation is not applicable" (Kontras et al. 2026). A second corpus is therefore
  needed for any claim about generalisation across patients.
- **The conventional Bonn task is saturated.** The same benchmark reports that eleven
  models reach AUROC at or above 0.99 on it and concludes, in their words, that they
  "treat Bonn as a sanity check rather than a discriminating benchmark". This shapes the
  design directly: the conventional binary split cannot be the headline task, the
  reference points in Section 4 are needed to show where the ceiling is, and external
  validation on a second corpus becomes necessary rather than optional.
- **The literature gap is not yet formally established** (Section 3).
- **Hardware is CPU only**, which constrains the scale of the second corpus experiment.

## 8. Use of AI assistance

I used an AI assistant (Claude) substantially in this work, for implementation, for running
experiments, and for parts of the analysis and writing. I chose the topic, the dataset and
the research question, wrote the original implementation, and made the decisions at each
branch point. I am stating this at the outset rather than leaving it ambiguous, and I will
follow whatever disclosure format the department requires. I am aware that this places the
burden on me to understand every part of the work well enough to defend it independently.

## 9. What I am asking

I would like to take the CS senior project one year early, during my junior year. The
reason is scheduling: I move to senior year in Industrial Engineering this year and will
complete the IE senior project on time. If the CS project waits for my fourth CS year my
graduation extends by a year, which conflicts with applying for a master's programme. I
understand this is out of the normal sequence and that it depends on whether the work is
far enough along to justify it.

My IE senior project will be separate work on a different problem.

Questions I would like your opinion on:

1. Is the early senior project feasible from the department's side, and what would you need
   to see before supporting it?
2. Is the scope right, or should it be narrowed to a single task formulation and a single
   corpus? I would rather do one thing properly than three things thinly.
3. The statistical side is where I am least confident. Repeated cross validation violates
   the independence assumption of the usual paired tests, and I have moved to a corrected
   test and equivalence testing. I would value a check on whether that is the right
   treatment.
4. Would the work benefit from a second advisor, so that the machine learning methodology
   and the neurophysiological interpretation are each covered by someone who works in that
   area? The scalp against intracranial distinction and the choice of frequency band are
   the points where I am reasoning from dataset documentation rather than domain knowledge.

---

## References

- Ali, E. et al. (2024) Epileptic seizure detection using CHB-MIT dataset: The overlooked perspectives. *Royal Society Open Science* 11(5):230601. doi:10.1098/rsos.230601
- Acharya, U. R. et al. (2018) Deep convolutional neural network for the automated detection and diagnosis of seizure using EEG signals. *Computers in Biology and Medicine* 100:270-278. doi:10.1016/j.compbiomed.2017.09.017
- Andrzejak, R. G. et al. (2001) Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity. *Physical Review E* 64(6):061907. doi:10.1103/PhysRevE.64.061907
- Benchmarking Multimodal Deep Fusion Strategies for Heterogeneous Neuroimaging and Cognitive Data Using a Controlled Sex Classification Task (2026). *Brain Sciences* 16(4):405. doi:10.3390/brainsci16040405
- Chatzichristos, C. et al. (2020) Epileptic Seizure Detection in EEG via Fusion of Multi-View Attention-Gated U-Net Deep Neural Networks. *IEEE SPMB*. doi:10.1109/spmb50085.2020.9353630
- Daşdemir, A. (2025) Epileptic seizure prediction with deep learning-based fusion methods. *Engineering Science and Technology, an International Journal* 72:102212. doi:10.1016/j.jestch.2025.102212
- Kontras, K. et al. (2026) NeuroAtlas: Benchmarking Foundation Models for Clinical EEG and Brain-Computer Interfaces. arXiv:2605.14698.
- Mohamady, A., Burchard, R., Van Laerhoven, K. (2026) A Comparison of Fusion Techniques for Multi-Modal Human Activity Recognition on the HARMES Dataset. arXiv:2606.27886.
- Das, S. et al. (2024) Epileptic Seizure Detection from Decomposed EEG Signal through 1D and 2D Feature Representation and Convolutional Neural Network. *Information* 15(5):256. doi:10.3390/info15050256
- Golrizkhatami, Z., Acan, A. (2018) ECG classification using three-level fusion of different feature descriptors. *Expert Systems with Applications* 114:54-64.
- Huang, J. et al. (2024) Multi-modal feature fusion with multi-head self-attention for epileptic EEG signals. *Mathematical Biosciences and Engineering* 21(2). doi:10.3934/mbe.2024304
- Rheude, T., Eils, R., Wild, B. (2025) Fusion or Confusion? Multimodal Complexity Is Not All You Need. arXiv:2512.22991.
- Roy, Y. et al. (2019) Deep learning-based electroencephalography analysis: a systematic review. *Journal of Neural Engineering* 16:051001. doi:10.1088/1741-2552/ab260c
- Shoeibi, A. et al. (2021) Epileptic Seizures Detection Using Deep Learning Techniques: A Review. *IJERPH* 18(11):5780. doi:10.3390/ijerph18115780
- Truong, N. D. et al. (2018) Convolutional neural networks for seizure prediction using intracranial and scalp electroencephalogram. *Neural Networks* 105:104-111. doi:10.1016/j.neunet.2018.04.018
- Wang, X. et al. (2020) One and Two Dimensional Convolutional Neural Networks for Seizure Detection Using EEG Signals. *EUSIPCO 2020*. doi:10.23919/eusipco47968.2020.9287640
- Xu, J. et al. (2024) EEG-based epileptic seizure detection using deep learning techniques: A survey. *Neurocomputing* 610:128644. doi:10.1016/j.neucom.2024.128644
