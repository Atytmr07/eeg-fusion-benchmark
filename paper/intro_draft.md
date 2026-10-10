# Introduction draft: four topics requested by the advisor

Draft for the manuscript's introduction, from the sources in `docs/15_LITERATUR_SONUCLARI.md`
Section 10. It does not change `paper/manuscript.md`; Atay integrates it.

Every sentence that states a finding rests on one of these sources:

- A source in `paper/literature/references.bib`, verified via Crossref or the arXiv API.
- What the sentence states comes from that source's abstract, or, for works already in
  the manuscript, from the manuscript's own description of them.

The claim-to-source table at the end gives the basis of each sentence.

Sentences that only link the literature to this study's design are marked *[link]*.
They make no claim about other work.

---

## 1. Preprocessing changes what deep models achieve

Deep networks are often expected to learn from minimally processed EEG, but the choice of
preprocessing changes their results.

- **Del Pup et al. (2025).** In the first systematic study of EEG preprocessing for deep
  learning, they trained 4,800 models on six classification tasks with four
  architectures. They found statistically significant differences between preprocessing
  pipelines. Models trained on raw data ranked last on average, and minimal pipelines
  without artefact handling tended to serve the models best.
- **Kang et al. (2024).** Artefact removal by independent component analysis did not
  consistently improve deep network decoding. They crossed two decomposition algorithms
  (Infomax and AMICA) with three component rejection strategies on three
  brain-computer interface datasets.
- **Truong and Delorme (2025).** Even the granularity and scope of normalisation matter:
  the best strategy differed between supervised and self-supervised training.

Outside deep learning, multiverse analyses of EEG and event-related potentials show the
same sensitivity:

- Across 43 preprocessing pipelines no single pipeline was best on every quality criterion
  (Huang et al. 2025).
- Fourteen pipelines drawn from the N400 literature changed statistical outcomes, effect
  sizes and power (Šoškić et al. 2024).
- Only 13 of 270 analyses of frontal alpha asymmetry in depression were significant
  (Kołodziej et al. 2021).

A multiverse analysis runs and reports every defensible combination of analysis choices
instead of one (Steegen et al. 2016; Clayson 2024). For event-related potentials it has
been used to choose processing pipelines by their effect on data quality and reliability
(Clayson et al. 2021).

*[link]* We apply the same idea to fusion operators for seizure detection: nine
preprocessing pipelines, including three artefact removal algorithms, are crossed with
every fusion operator (Section 3.6).

## 2. Fusion operators are compared, but not under controlled conditions

Combining two representations of the same EEG segment, typically the raw waveform and a
time-frequency image, is a common design for seizure detection. Several recent studies
compare fusion operators directly:

- **An et al. (2026):** five fusion strategies, within and across patients.
- **Einizade et al. (2023):** four operators (majority vote, averaging, concatenation and
  attention), with leave-one-subject-out evaluation.
- **Basheer and Mishra (2026):** three operators, with paired tests over six random seeds.
- **Daşdemir and Örnek (2025):** feature-level and decision-level fusion for seizure
  prediction.

None of these studies does all three of the following:

- matching the parameter counts of the compared operators,
- testing the differences between operators with a correction for repeated resampling,
- testing the operators for equivalence.

The evidence for each work is in Table 1 and `paper/literature/extraction.csv`. In
Einizade et al. (2023), the four operators lie within one standard deviation of each
other.

Across the wider literature on hybrid EEG interfaces, only two of more than 120 surveyed
studies explored more than one fusion strategy, and only one directly investigated which
fusion approach was best (Lee et al. 2025).

Controlled comparisons outside EEG point the same way:

- In ECG classification, Narotamo et al. (2024) compared 1D and 2D representations with
  early, late and joint fusion.
- A controlled benchmark on tabular neuroimaging and cognitive data concluded that the
  fusion strategy mattered more than the architecture (Camastra et al. 2026).
- Added multimodal complexity often did not pay off (Rheude et al. 2025).

*[link]* We therefore hold the backbone, training protocol and parameter budget fixed,
vary only the fusion operator, and test the differences with corrected and equivalence
tests (Sections 3.1-3.5).

## 3. A single seed or a single pipeline is a weak basis for a ranking

A comparison built on one training run can be decided by factors unrelated to the
methods compared.

**Identical training runs differ.**

- **Pham et al. (2020).** Identical training runs of deep networks produced different
  models. Even after weak models were excluded, accuracy differed by up to 10.8% between
  identical runs, and implementation-level factors alone caused differences of up to
  2.9%. Only about one fifth of the papers they surveyed used repeated runs to quantify
  this variance.
- **Summers and Dinneen (2021).** All sources of nondeterminism in neural network
  optimisation, down to a one-bit change in the initial parameters, produced similar run
  to run variability. This makes small improvements hard to separate from noise.
- **Picard (2021).** Seeds that perform much better or much worse than average are easy
  to find.

**The computing environment adds variance.**

- CPU multithreading makes deep learning training nondeterministic (Xiao et al. 2021).
- Floating-point non-associativity in parallel computation causes run to run variability
  (Shanmugavelu et al. 2024).
- Reproducible training requires dedicated control of both software randomness and
  hardware nondeterminism (Chen et al. 2022).

**Reporting.** In speech recognition, training variance was large enough to call for a
change in how results are reported (van den Berg et al. 2017). Seed-induced variability
can be measured with dedicated hypothesis tests (Banerjee et al. 2025).

**Statistical testing.** Under repeated cross-validation the folds share training data,
so standard paired tests are overconfident unless the variance is corrected (Nadeau and
Bengio 2003; Bouckaert and Frank 2004). Whether a cross-validated comparison comes out
significant can depend on the number of folds and repeats chosen (Jafrasteh et al.
2025).

*[link]* Each pipeline is therefore trained with several seed sets. Rank stability across
seeds is compared with rank stability across pipelines (Section 3.6), and the thread count
is fixed and reported (Section 6.2).

## 4. Results on one dataset rarely carry over to another

High accuracies reported for automatic seizure detection often do not hold on new patients
or recording sites.

- **Dan et al. (2025).** Twenty-eight algorithms from the SzCORE challenge were evaluated
  on a strictly held-out set of 4,360 hours of continuous EEG from 65 subjects. The best
  F1 score was 32% (sensitivity 37%, precision 29%). There was a notable gap to
  self-reported efficacies, and the algorithms with the highest aggregate F1 were not the
  most consistent across subjects.
- **Dan et al. (2024).** Proposed standard datasets, cross-validation strategies and
  event-based metrics to make such comparisons possible.
- **Lee et al. (2022).** Compared real-time models under one realistic setting because
  earlier models had been tested in distinct experimental settings.
- **Yang et al. (2021).** A prospective inference test of a seizure recognition system on
  nearly 14,590 hours of adult EEG from a hospital in Sydney, presented as continental
  generalisation, achieved 76.68% with about 56 false alarms per 24 hours against legacy
  expert annotations.
- **Moutonnet et al. (2024).** A systematic review names the generalisability of
  algorithms across training data and acquisition hardware among the main obstacles to
  clinical translation.
- **Ali et al. (2024).** Within one public corpus, CHB-MIT results depend heavily on
  evaluation choices.
- **Kontras et al. (2026).** The conventional Bonn split is saturated: eleven foundation
  models reach AUROC at or above 0.99.

*[link]* We therefore evaluate person-wise on CHB-MIT and on Siena, a second corpus from
another hospital, country and age group, with the same windows, labels and models, and
test models trained on CHB-MIT on Siena (Sections 5 and 5.5).

---

## Claim-to-source table

| Topic | Claim (short) | Source (`references.bib` key) | Basis |
|---|---|---|---|
| 1 | First systematic study of EEG preprocessing for DL; 4,800 models, six tasks, four architectures; significant differences between pipelines; raw data last; minimal pipelines without artefact handling best | `Del_Pup_2025` | abstract |
| 1 | ICA-based artefact removal (Infomax, AMICA x none/ICLabel/MARA) did not consistently improve deep decoding on three BCI datasets | `Kang_2024` | abstract |
| 1 | Best normalisation strategy differs between supervised and self-supervised training | `Truong_2025` | abstract |
| 1 | 43 pipelines, no single best | `Huang_2025` | abstract |
| 1 | 14 N400 pipelines changed test outcomes, effect sizes, power | `Soskic_2024` | abstract |
| 1 | 13 of 270 analyses significant | `Kolodziej_2021` | abstract |
| 1 | Multiverse analysis: definition | `Steegen_2016`, `Clayson_2024` | manuscript Section 3.6; abstract |
| 1 | ERP pipelines chosen by data quality and reliability | `Clayson_2021` | abstract |
| 2 | Five strategies, within and across patients | `An_2026` | full text, `extraction.csv` A0087 |
| 2 | Four operators, LOSO; within one SD | `Einizade_2023` | full text, `extraction.csv` A0415 |
| 2 | Three operators, paired tests over six seeds | `Basheer_2026` | full text, `extraction.csv` A0102 |
| 2 | Feature- vs decision-level fusion, prediction | `Dasdemir_2025` | full text, `extraction.csv` A0239 |
| 2 | No parameter matching, no corrected or equivalence test between operators | the four above | `extraction.csv` evidence column; Table 1 |
| 2 | Two of more than 120 hybrid EEG studies explored more than one fusion strategy; one investigated which was best | `Lee_2025` | manuscript Section 2 |
| 2 | ECG: 1D/2D with early, late, joint fusion | `Narotamo_2024` | manuscript Table 1 |
| 2 | Fusion strategy mattered more than architecture (tabular neuroimaging) | `Camastra_2026` | manuscript Table 1 |
| 2 | Multimodal complexity often does not pay off | `Rheude_2025` | manuscript Table 1 |
| 3 | Up to 10.8% between identical runs; up to 2.9% implementation-level; about one fifth of papers use repeated runs | `Pham_2020` | abstract |
| 3 | All nondeterminism sources similar; one-bit change; small improvements hard to discern | `Summers2021arxiv` | abstract |
| 3 | Outlier seeds easy to find | `Picard2021arxiv` | abstract |
| 3 | CPU multithreading causes nondeterminism | `Xiao_2021` | abstract |
| 3 | Floating-point non-associativity causes run to run variability | `Shanmugavelu_2024` | abstract |
| 3 | Reproducible training needs control of software randomness and hardware nondeterminism | `Chen_2022` | abstract |
| 3 | Training variance calls for rethinking how results are reported (speech) | `van_den_Berg_2017` | abstract |
| 3 | Hypothesis tests for seed-induced variability | `Banerjee_2025` | abstract |
| 3 | Corrected resampled t test | `Nadeau_2003`, `Bouckaert_2004` | manuscript Section 3.5 |
| 3 | Significance depends on folds and repeats | `Jafrasteh_2025` | manuscript Section 3.5 |
| 4 | 28 algorithms, 4,360 h, 65 subjects; best F1 32% (sens 37%, prec 29%); gap to self-reported results; best aggregate F1 not most consistent | `Dan2025arxiv` | abstract (arXiv preprint) |
| 4 | Standard datasets, CV strategies, event-based metrics | `Dan_2024` | abstract |
| 4 | Models tested in distinct settings; compared under one realistic setting | `Lee2022arxiv` | abstract (arXiv preprint) |
| 4 | 14,590 h, Sydney; 76.68% with about 56 false alarms per 24 h | `Yang2021arxiv` | abstract (arXiv preprint) |
| 4 | Generalisability across training data and hardware an obstacle to clinical translation | `Moutonnet2024arxiv` | abstract (arXiv preprint) |
| 4 | CHB-MIT results depend on evaluation choices | `Ali_2024` | manuscript Table 1 |
| 4 | Bonn saturated (eleven models AUROC >= 0.99) | `Kontras_2026` | manuscript Table 1 and Section 2 |

## Notes for integration

- **References missing from the manuscript.** These entries are not yet in the
  manuscript's reference list. They are verified in `references.bib`, so the PDF build
  takes them from there once they are cited:
  - `Kang_2024`, `Truong_2025`, `Huang_2025`, `Soskic_2024`, `Kolodziej_2021`,
    `Clayson_2021`, `Clayson_2024`;
  - `Pham_2020`, `Summers2021arxiv`, `Picard2021arxiv`, `Xiao_2021`,
    `Shanmugavelu_2024`, `Chen_2022`, `van_den_Berg_2017`, `Banerjee_2025`;
  - `Dan2025arxiv`, `Yang2021arxiv`, `Lee2022arxiv`, `Moutonnet2024arxiv`.

  The plain reference list in `manuscript.md` needs the same entries.
- **Preprints.** Four sources are arXiv preprints, not peer reviewed: Dan 2025,
  Yang 2021, Lee 2022 and Moutonnet 2024. Summers and Dinneen 2021 and Picard 2021 are
  in `references.bib` as arXiv records. Summers and Dinneen's abstract carries a
  conference-style copyright line, so a published version may exist; check its venue
  before submission.
- **Abstract-level claims.** Topics 1, 3 and 4 rest on abstracts. A reader of the full
  texts may refine the wording, for example Yang et al.'s "76.68%", which the abstract
  does not name as sensitivity.
- **Seed sets.** The manuscript's Section 3.6 still says three seed sets per pipeline;
  Figure 1 now says five.
