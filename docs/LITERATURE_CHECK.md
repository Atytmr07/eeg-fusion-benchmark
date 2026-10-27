# Literature check before submission

Advisor's review of 27 October 2026, item 4: every claim that rests on a source's
abstract only, every limitation that Table 1 attributes to a study, and the Discussion
sources not yet checked against their full texts are to be verified by the authors
(Emre Atay Tümer, Hasan Berk Berber) against the full text. Nothing in this list has
been verified by an AI assistant; the column "Result" is filled in by the person who
read the full text. When a claim is confirmed, its `[abstract]` tag in
`paper/manuscript_v2.md` is removed; when it is not, the sentence is corrected or the
source dropped.

Columns: **Where** = section of `paper/manuscript_v2.md`; **Claim** = what the
manuscript states (shortened); **Check** = what the full text has to show.

## A. Introduction: claims tagged [abstract] (19)

| # | Source | Where | Claim | Check | Checked by | Result |
|---|---|---|---|---|---|---|
| A1 | Del Pup et al. 2025, doi:10.1109/tnsre.2025.3547616 | 1.1 | 4,800 models, six tasks, four architectures; pipelines differ significantly; raw data ranked last on average; minimal pipelines without artefact handling tended to serve best | All four numbers; "raw last on average"; "tended" (not always) | | |
| A2 | Kang et al. 2024, doi:10.1088/1741-2552/ad788e | 1.1 | ICA with Infomax or AMICA and three rejection strategies did not consistently improve deep decoding on three BCI datasets | Infomax and AMICA; three strategies; three datasets; "not consistently" | | |
| A3 | Truong and Delorme 2025, doi:10.1109/ner61569.2025.11589116 | 1.1 | The scope of normalisation changed which strategy worked best | The finding as stated | | |
| A4 | Huang et al. 2025, doi:10.1111/psyp.70197 | 1.1 | Across 43 preprocessing pipelines, none best on every quality criterion | 43 pipelines; "every quality criterion" | | |
| A5 | Šoškić et al. 2024, doi:10.1111/psyp.14628 | 1.1 | 14 pipelines from the N400 literature changed statistical outcomes, effect sizes and power | 14 pipelines; all three outcomes | | |
| A6 | Kołodziej et al. 2021, doi:10.7554/elife.60595 | 1.1 | Only 13 of 270 analyses of frontal alpha asymmetry in depression significant | 13 of 270 | | |
| A7 | Clayson et al. 2021, doi:10.1016/j.neuroimage.2021.118712 | 1.1 | Multiverse analyses used to choose ERP pipelines by data quality and reliability | The finding as stated | | |
| A8 | Pham et al. 2020, doi:10.1145/3324884.3416545 | 1.3 | Up to 10.8 percent accuracy difference between identical runs after excluding weak models; up to 2.9 percent from implementation-level factors; about one fifth of papers use repeated runs | 10.8, 2.9, "one fifth" | | |
| A9 | Summers and Dinneen 2021 (arXiv) | 1.3 | Every source of nondeterminism, down to a one-bit change in initial parameters, gives similar run-to-run variability | "one bit"; "similar variability" | | |
| A10 | Picard 2021 (arXiv) | 1.3 | Seeds far better or worse than average are easy to find | The finding as stated | | |
| A11 | van den Berg et al. 2017, doi:10.1109/icassp.2017.7952564 | 1.3 | Training variance in speech recognition large enough to call for a change in reporting | The finding as stated | | |
| A12 | Banerjee et al. 2025, doi:10.1109/jstsp.2025.3583140 | 1.3 | Seed-induced variability can be tested for directly | The method as stated | | |
| A13 | Dan et al. 2025 (preprint) | 1.4 | 28 SzCORE challenge algorithms; 4,360 h of held-out EEG from 65 subjects; best F1 32 percent; highest aggregate score not the most consistent across subjects | All numbers; the consistency statement | | |
| A14 | Lee et al. 2022 (preprint) | 1.4 | Real-time detectors tested in distinct settings until compared under one | The finding as stated | | |
| A15 | Yang et al. 2021 (preprint) | 1.4 | Prospective test on about 14,590 h of adult EEG from another continent; 76.68 percent; about 56 false alarms per 24 h | All numbers; what 76.68 percent measures (sensitivity?) | | |
| A16 | Moutonnet et al. 2024 (preprint) | 1.4 | Systematic review: generalisability across training data and acquisition hardware among the main obstacles to clinical translation | The finding as stated | | |
| A17 | Xiao et al. 2021, doi:10.1109/issre52982.2021.00063 | 1.4 | CPU multithreading makes training nondeterministic | The finding as stated | | |
| A18 | Shanmugavelu et al. 2024, doi:10.1109/scw63240.2024.00028 | 1.4 | Floating-point non-associativity in parallel computation causes run-to-run differences | The finding as stated | | |
| A19 | Chen et al. 2022, doi:10.1145/3510003.3510163 | 1.4 | Reproducible training requires control of software randomness and hardware nondeterminism | The finding as stated | | |

## B. Table 1 and the novelty claim

The sentence in 1.2, "Among the studies we have read in full, none matches the parameter
counts of the compared operators, tests the differences between them with a correction
for repeated resampling, and tests them for equivalence", is the basis of the novelty
claim. Before submission: (1) list which Table 1 studies have been read in full, (2) for
each, confirm the entry in "What it leaves open", (3) if a study cannot be read in full,
either soften its entry or move it out of the sentence above. The caption's note that
full texts are still being read is removed once this is done.

| # | Source | Table 1 says it leaves open | Read in full? | Confirmed? |
|---|---|---|---|---|
| B1 | An et al. 2026, doi:10.1016/j.neunet.2026.109078 | Hand-crafted with deep features, not waveform with spectrogram; strategies not parameter matched; not tested against each other (Wilcoxon with Holm only against prior methods) | | |
| B2 | Einizade et al. 2023, doi:10.1016/j.bspc.2022.104076 | No parameter matching, no statistical test; operators within one SD of each other (LOSO on TUH, four operators) | | |
| B3 | Basheer and Mishra 2026, doi:10.1016/j.patrec.2026.07.016 | Both streams from the same raw EEG; uncorrected t tests against concatenation only; no parameter matching; six seeds | | |
| B4 | Daşdemir and Örnek 2025, doi:10.1016/j.jestch.2025.102212 | Prediction, not detection; two operators; 0.20 point accuracy difference without a stated test or parameter matching | | |
| B5 | Wang et al. 2026, doi:10.1016/j.neucom.2026.132929 | One ablation against concatenation; patient-specific evaluation; no statistical test; CHB-MIT and Siena | | |
| B6 | Chatzichristos et al. 2020, doi:10.1109/spmb50085.2020.9353630 | One operator, no alternatives | | |
| B7 | Das et al. 2024, doi:10.3390/info15050256 | Fixed 1D+2D design, no operator ablation, no matched parameter counts | | |
| B8 | Huang et al. 2024, doi:10.3934/mbe.2024304 | Fixed multi-head attention design, no operator ablation, no matched parameter counts | | |
| B9 | Narotamo et al. 2024, doi:10.1016/j.bspc.2024.106141 | ECG; early, late and joint fusion; no parameter matching, no corrected or equivalence statistics | | |
| B10 | Camastra et al. 2026, doi:10.3390/brainsci16040405 | Tabular neuroimaging; strategy matters more than architecture; no paired or equivalence testing | | |
| B11 | Ali et al. 2024, doi:10.1098/rsos.230601 | CHB-MIT results depend heavily on evaluation choices | | |
| B12 | Dan et al. 2024, doi:10.1111/epi.18113 | SzCORE: standard datasets, event-based scoring and metrics | | |
| B13 | Kontras et al. 2026 | Bonn saturated (eleven models at AUROC >= 0.99), no subject identifiers | | |
| B14 | Lee et al. 2025, doi:10.1007/s13534-025-00469-5 (1.2, not in Table 1) | More than 120 studies of hybrid EEG interfaces; only two explored more than one fusion strategy; one asked which was best | | |

## C. Discussion sources still to be checked

| # | Source | Where | Claim | Check | Checked by | Result |
|---|---|---|---|---|---|---|
| C1 | Baumgartner et al. 2018, doi:10.3389/fneur.2018.00639 | 4.4 | Generic scalp EEG detectors evaluated for clinical use: 0.1 to 5 false alarms per hour, most about 0.2 to 0.3; patient-specific detectors fewer | The range and "most" | | |
| C2 | Beniczky and Ryvlin 2018, doi:10.1111/epi.14049 | 4.4 | Low false alarm rate essential for clinical acceptance; validation standards require it to be reported | The finding as stated | | |
| C3 | Shoeb and Guttag 2010 (ICML; PhysioNet PDF) | 4.4 | Patient-specific classifiers detected 96 percent of 173 test seizures in CHB-MIT, median two false detections per day | 96 percent, 173, "median of two per day" | | |
| C4 | Glatard et al. 2015, doi:10.3389/fninf.2015.00012 | 1.4, 4.5 | Neuroimaging pipelines, including ICA decompositions of fMRI data, give different results on different operating systems | ICA of fMRI among the affected steps | | |
| C5 | Rheude et al. 2025 (preprint) | 1.2, 4.1 | In a broader evaluation of multimodal models, added fusion complexity often did not pay off | The finding as stated; whether a published version exists | | |
| C6 | Sendelbach and Funk 2013, doi:10.4037/nci.0b013e3182a903f9 | 4.4 | Alarm fatigue | That it supports the clinical trade-off sentence | | |

## D. Summary sentences that rest on several sources

| # | Where | Sentence | Rests on |
|---|---|---|---|
| D1 | 1.1 | "In none of these studies did a single approach consistently give the best outcome across tasks and models." | A1 to A3 |
| D2 | 1.2 | "Among the studies we have read in full, none matches ..." (novelty claim) | B1 to B13 |
| D3 | 1.4 | Standard event-based scoring proposed to make results comparable | B12 |
