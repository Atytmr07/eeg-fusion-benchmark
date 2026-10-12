# Multiverse analysis: siena_cuda

Runs found (45): loso_P0_cuda (P0, seed 0), loso_P0_r1_cuda (P0, seed 1), loso_P0_r2_cuda (P0, seed 2), loso_P0_r3_cuda (P0, seed 3), loso_P0_r4_cuda (P0, seed 4), loso_P1_cuda (P1, seed 0), loso_P1_r1_cuda (P1, seed 1), loso_P1_r2_cuda (P1, seed 2), loso_P1_r3_cuda (P1, seed 3), loso_P1_r4_cuda (P1, seed 4), loso_P2_cuda (P2, seed 0), loso_P2_r1_cuda (P2, seed 1), loso_P2_r2_cuda (P2, seed 2), loso_P2_r3_cuda (P2, seed 3), loso_P2_r4_cuda (P2, seed 4), loso_P3_cuda (P3, seed 0), loso_P3_r1_cuda (P3, seed 1), loso_P3_r2_cuda (P3, seed 2), loso_P3_r3_cuda (P3, seed 3), loso_P3_r4_cuda (P3, seed 4), loso_P4_cuda (P4, seed 0), loso_P4_r1_cuda (P4, seed 1), loso_P4_r2_cuda (P4, seed 2), loso_P4_r3_cuda (P4, seed 3), loso_P4_r4_cuda (P4, seed 4), loso_P5_cuda (P5, seed 0), loso_P5_r1_cuda (P5, seed 1), loso_P5_r2_cuda (P5, seed 2), loso_P5_r3_cuda (P5, seed 3), loso_P5_r4_cuda (P5, seed 4), loso_P6a_cuda (P6a, seed 0), loso_P6a_r1_cuda (P6a, seed 1), loso_P6a_r2_cuda (P6a, seed 2), loso_P6a_r3_cuda (P6a, seed 3), loso_P6a_r4_cuda (P6a, seed 4), loso_P6b_cuda (P6b, seed 0), loso_P6b_r1_cuda (P6b, seed 1), loso_P6b_r2_cuda (P6b, seed 2), loso_P6b_r3_cuda (P6b, seed 3), loso_P6b_r4_cuda (P6b, seed 4), loso_P6c_cuda (P6c, seed 0), loso_P6c_r1_cuda (P6c, seed 1), loso_P6c_r2_cuda (P6c, seed 2), loso_P6c_r3_cuda (P6c, seed 3), loso_P6c_r4_cuda (P6c, seed 4)

## 1. Performance

Means over folds per pipeline, seed and model: `performance.csv`; seed mean and SD per pipeline and model: `performance_by_pipeline.csv`. AUPRC is computed from the stored test probabilities (logvar and shallow store none).

Balanced block for the analyses below: pipelines P0, P1, P2, P3, P4, P5, P6a, P6b, P6c; seeds 0, 1, 2, 3, 4; 14 folds; fusion models early, late, gated, attention, score.

## Effects, separately

### Preprocessing effect (pipeline main effect, rank agreement between pipelines)

Pipeline main effect on macro F1: F(8, 104) = 1.09, p_GG = 0.365, 0.1 % of the variance; the fusion ranking agrees between runs of different pipelines with mean Kendall tau 0.34, against 0.37 between seeds of the same pipeline (D = 0.03, permutation p = 0.074).

### Seed effect (seed variance components, rank agreement between seeds)

All seed terms together hold 4.2 % of the variance of macro F1; rankings from different seeds of the same pipeline agree with mean Kendall tau 0.37 (Kendall's W over seeds, median over 9 pipelines: 0.58).

### Fusion x Preprocessing interaction (RM-ANOVA, variance share)

Fusion x Preprocessing: F(32, 416) = 1.05, p_GG = 0.395 (GG epsilon 0.15), partial eta squared 0.075; the fusion x pipeline component holds 0.2 % of the variance.

## 2. Rank stability (main analysis)

Ranking of the fusion operators by mean macro F1 per run: `rank_per_run.csv`.

| | value |
|---|---|
| mean Kendall tau, same pipeline, different seeds | 0.373 |
| mean Kendall tau, different pipelines | 0.341 |
| D = within - between | 0.032 |
| permutation p (pipeline labels permuted within seeds, 10000 permutations) | 0.0745 |

- P0: Kendall's W over 5 seeds = 0.640, mean pairwise tau = 0.440
- P1: Kendall's W over 5 seeds = 0.440, mean pairwise tau = 0.240
- P2: Kendall's W over 5 seeds = 0.776, mean pairwise tau = 0.640
- P3: Kendall's W over 5 seeds = 0.584, mean pairwise tau = 0.400
- P4: Kendall's W over 5 seeds = 0.536, mean pairwise tau = 0.320
- P5: Kendall's W over 5 seeds = 0.400, mean pairwise tau = 0.160
- P6a: Kendall's W over 5 seeds = 0.464, mean pairwise tau = 0.280
- P6b: Kendall's W over 5 seeds = 0.632, mean pairwise tau = 0.440
- P6c: Kendall's W over 5 seeds = 0.640, mean pairwise tau = 0.440

### Rank distribution

Rank of each operator in each of the 45 runs (1 = best). Not only the mean rank: the counts per rank over all runs, the median, the best and worst rank, and the share of runs the operator wins. Counts per pipeline (over seeds) and per seed (over pipelines) are in `rank_distribution.csv`; `rank_distribution.png` shows the counts and the rank map of every run.

| model | rank 1 | rank 2 | rank 3 | rank 4 | rank 5 |
|---|---|---|---|---|---|
| early | 0 | 8 | 7 | 10 | 20 |
| late | 3 | 19 | 8 | 12 | 3 |
| gated | 4 | 10 | 14 | 5 | 12 |
| attention | 1 | 4 | 13 | 17 | 10 |
| score | 37 | 4 | 3 | 1 | 0 |

| model | n_runs | mean_rank | median_rank | best_rank | worst_rank | share_first | share_last |
|---|---|---|---|---|---|---|---|
| early | 45 | 3.93 | 4.00 | 2 | 5 | 0.00 | 0.44 |
| late | 45 | 2.84 | 3.00 | 1 | 5 | 0.07 | 0.07 |
| gated | 45 | 3.24 | 3.00 | 1 | 5 | 0.09 | 0.27 |
| attention | 45 | 3.69 | 4.00 | 1 | 5 | 0.02 | 0.22 |
| score | 45 | 1.29 | 1.00 | 1 | 4 | 0.82 | 0.00 |

ICA algorithm (P6a, P6b, P6c): within tau 0.387, between tau 0.211, D 0.176, permutation p 0.0293; seed-averaged tau matrix:

| pipeline | P6a | P6b | P6c |
|---|---|---|---|
| P6a | 1.000 | 0.200 | 0.600 |
| P6b | 0.200 | 1.000 | -0.200 |
| P6c | 0.600 | -0.200 | 1.000 |

## 3. Fusion x Preprocessing

Repeated-measures ANOVA on macro F1 (seed means; folds as subjects; each effect tested against its interaction with fold; Greenhouse-Geisser):

| effect | df | df_error | F | p | gg_epsilon | p_gg | partial_eta2 |
|---|---|---|---|---|---|---|---|
| pipeline | 8 | 104 | 1.0885 | 0.3771 | 0.3668 | 0.3647 | 0.0773 |
| model | 4 | 52 | 2.8726 | 0.0318 | 0.3778 | 0.0921 | 0.1810 |
| pipeline x model | 32 | 416 | 1.0512 | 0.3946 | 0.1490 | 0.3945 | 0.0748 |

## 4. Variance components

Model: every fold (held-out person) x pipeline x fusion operator x seed cell holds one value of the metric, y_fpms = mu + a_f + b_p + c_m + d_s + every two-, three- and four-way interaction of these terms. All terms are treated as random effects (generalizability theory) and estimated by the method of moments from the expected mean squares of the fully crossed, balanced design. With one observation per cell the four-way interaction is confounded with the residual. A fixed factor's component (pipeline, operator) is then the variance of its effects. Negative estimates are set to 0 for the shares and listed. `sd` is the square root of the component, in the metric's own units, as a practical size; `share` is its part of the summed components. Grouped: 'fold x (pipeline, fusion)' sums the interactions of fold with pipeline, operator and both; 'seed (all seed terms)' sums every term containing seed.

### f1_macro

Grouped:

| group | sigma2 | sd | share_pct |
|---|---|---|---|
| fold (subject) | 0.005479 | 0.074020 | 32.478706 |
| pipeline | 0.000010 | 0.003210 | 0.061080 |
| fusion | 0.000260 | 0.016126 | 1.541434 |
| fusion x pipeline | 0.000028 | 0.005285 | 0.165554 |
| seed (all seed terms) | 0.000708 | 0.026600 | 4.194321 |
| fold x (pipeline, fusion) | 0.004536 | 0.067348 | 26.887218 |
| residual | 0.005849 | 0.076479 | 34.671688 |

Every component, unrounded estimate first:

| component | sigma2_raw | sigma2 | sd | share_pct |
|---|---|---|---|---|
| fold | 5.479018e-03 | 5.479018e-03 | 0.074020 | 32.478706 |
| pipeline | 1.030391e-05 | 1.030391e-05 | 0.003210 | 0.061080 |
| model | 2.600332e-04 | 2.600332e-04 | 0.016126 | 1.541434 |
| seed | -9.432926e-06 | 0.000000e+00 | 0.000000 | 0.000000 |
| fold x pipeline | 2.205648e-03 | 2.205648e-03 | 0.046964 | 13.074714 |
| fold x model | 1.756046e-03 | 1.756046e-03 | 0.041905 | 10.409547 |
| fold x seed | 3.296238e-05 | 3.296238e-05 | 0.005741 | 0.195395 |
| pipeline x model | 2.792824e-05 | 2.792824e-05 | 0.005285 | 0.165554 |
| pipeline x seed | 2.394969e-05 | 2.394969e-05 | 0.004894 | 0.141970 |
| model x seed | 5.695876e-05 | 5.695876e-05 | 0.007547 | 0.337642 |
| fold x pipeline x model | 5.740642e-04 | 5.740642e-04 | 0.023960 | 3.402957 |
| fold x pipeline x seed | 1.703098e-04 | 1.703098e-04 | 0.013050 | 1.009568 |
| fold x model x seed | 4.233833e-04 | 4.233833e-04 | 0.020576 | 2.509746 |
| pipeline x model x seed | -1.077821e-04 | 0.000000e+00 | 0.000000 | 0.000000 |
| residual (fold x pipeline x model x seed) | 5.848965e-03 | 5.848965e-03 | 0.076479 | 34.671688 |

Negative raw estimates (set to 0): seed, pipeline x model x seed

### auprc

Grouped:

| group | sigma2 | sd | share_pct |
|---|---|---|---|
| fold (subject) | 0.022733 | 0.150773 | 52.421853 |
| pipeline | 0.000021 | 0.004605 | 0.048892 |
| fusion | 0.002364 | 0.048617 | 5.450590 |
| fusion x pipeline | 0.000065 | 0.008090 | 0.150919 |
| seed (all seed terms) | 0.001856 | 0.043082 | 4.280069 |
| fold x (pipeline, fusion) | 0.008490 | 0.092143 | 19.578952 |
| residual | 0.007835 | 0.088518 | 18.068725 |

Every component, unrounded estimate first:

| component | sigma2_raw | sigma2 | sd | share_pct |
|---|---|---|---|---|
| fold | 2.273251e-02 | 2.273251e-02 | 0.150773 | 52.421853 |
| pipeline | 2.120193e-05 | 2.120193e-05 | 0.004605 | 0.048892 |
| model | 2.363625e-03 | 2.363625e-03 | 0.048617 | 5.450590 |
| seed | 6.368998e-05 | 6.368998e-05 | 0.007981 | 0.146871 |
| fold x pipeline | 6.509594e-03 | 6.509594e-03 | 0.080682 | 15.011320 |
| fold x model | 1.369547e-03 | 1.369547e-03 | 0.037007 | 3.158218 |
| fold x seed | 1.433017e-04 | 1.433017e-04 | 0.011971 | 0.330458 |
| pipeline x model | 6.544542e-05 | 6.544542e-05 | 0.008090 | 0.150919 |
| pipeline x seed | -2.821855e-05 | 0.000000e+00 | 0.000000 | 0.000000 |
| model x seed | 5.472327e-05 | 5.472327e-05 | 0.007398 | 0.126194 |
| fold x pipeline x model | 6.111862e-04 | 6.111862e-04 | 0.024722 | 1.409414 |
| fold x pipeline x seed | 2.073297e-04 | 2.073297e-04 | 0.014399 | 0.478109 |
| fold x model x seed | 1.386989e-03 | 1.386989e-03 | 0.037242 | 3.198438 |
| pipeline x model x seed | -3.526978e-05 | 0.000000e+00 | 0.000000 | 0.000000 |
| residual (fold x pipeline x model x seed) | 7.835424e-03 | 7.835424e-03 | 0.088518 | 18.068725 |

Negative raw estimates (set to 0): pipeline x seed, pipeline x model x seed

### balanced_accuracy

Grouped:

| group | sigma2 | sd | share_pct |
|---|---|---|---|
| fold (subject) | 0.005458 | 0.073876 | 40.882734 |
| pipeline | 0.000000 | 0.000000 | 0.000000 |
| fusion | 0.000248 | 0.015757 | 1.859979 |
| fusion x pipeline | 0.000000 | 0.000000 | 0.000000 |
| seed (all seed terms) | 0.000502 | 0.022404 | 3.760119 |
| fold x (pipeline, fusion) | 0.003030 | 0.055049 | 22.700536 |
| residual | 0.004111 | 0.064119 | 30.796632 |

Every component, unrounded estimate first:

| component | sigma2_raw | sigma2 | sd | share_pct |
|---|---|---|---|---|
| fold | 5.457631e-03 | 5.457631e-03 | 0.073876 | 40.882734 |
| pipeline | -4.479171e-05 | 0.000000e+00 | 0.000000 | 0.000000 |
| model | 2.482975e-04 | 2.482975e-04 | 0.015757 | 1.859979 |
| seed | -3.923956e-06 | 0.000000e+00 | 0.000000 | 0.000000 |
| fold x pipeline | 1.944204e-03 | 1.944204e-03 | 0.044093 | 14.563900 |
| fold x model | 7.276387e-04 | 7.276387e-04 | 0.026975 | 5.450691 |
| fold x seed | 2.862707e-05 | 2.862707e-05 | 0.005350 | 0.214443 |
| pipeline x model | -2.034800e-05 | 0.000000e+00 | 0.000000 | 0.000000 |
| pipeline x seed | 4.763560e-06 | 4.763560e-06 | 0.002183 | 0.035683 |
| model x seed | 3.385268e-05 | 3.385268e-05 | 0.005818 | 0.253588 |
| fold x pipeline x model | 3.585596e-04 | 3.585596e-04 | 0.018936 | 2.685945 |
| fold x pipeline x seed | 8.076071e-05 | 8.076071e-05 | 0.008987 | 0.604973 |
| fold x model x seed | 3.539522e-04 | 3.539522e-04 | 0.018814 | 2.651431 |
| pipeline x model x seed | -4.862325e-06 | 0.000000e+00 | 0.000000 | 0.000000 |
| residual (fold x pipeline x model x seed) | 4.111189e-03 | 4.111189e-03 | 0.064119 | 30.796632 |

Negative raw estimates (set to 0): pipeline, seed, pipeline x model, pipeline x model x seed

## 5. Equivalence and robust equivalence

Equivalent = corrected equivalence bound delta_min <= 0.05 macro F1 (test/train ratio 1/13). Robust equivalence = equivalent in every pipeline.

| pair | n_pipelines | equivalent_in | robust_equivalent | never_significant | max_delta_min |
|---|---|---|---|---|---|
| early-late | 9 | 0 | False | True | 0.1104 |
| early-gated | 9 | 1 | False | True | 0.1242 |
| early-attention | 9 | 0 | False | True | 0.1060 |
| early-score | 9 | 0 | False | True | 0.1265 |
| late-gated | 9 | 5 | False | True | 0.0876 |
| late-attention | 9 | 5 | False | True | 0.0882 |
| late-score | 9 | 2 | False | True | 0.1482 |
| gated-attention | 9 | 8 | False | True | 0.0783 |
| gated-score | 9 | 0 | False | True | 0.1463 |
| attention-score | 9 | 0 | False | True | 0.1445 |

Strict sensitivity analysis, margin 0.02:

| pair | n_pipelines | equivalent_in | robust_equivalent | never_significant | max_delta_min |
|---|---|---|---|---|---|
| early-late | 9 | 0 | False | True | 0.1104 |
| early-gated | 9 | 0 | False | True | 0.1242 |
| early-attention | 9 | 0 | False | True | 0.1060 |
| early-score | 9 | 0 | False | True | 0.1265 |
| late-gated | 9 | 0 | False | True | 0.0876 |
| late-attention | 9 | 0 | False | True | 0.0882 |
| late-score | 9 | 0 | False | True | 0.1482 |
| gated-attention | 9 | 0 | False | True | 0.0783 |
| gated-score | 9 | 0 | False | True | 0.1463 |
| attention-score | 9 | 0 | False | True | 0.1445 |

Both margins side by side: robustly equivalent pairs 0/10 at +-0.05 and 0/10 at +-0.02.

| pair | n_pipelines | equivalent in (+-0.05) | robust (+-0.05) | equivalent in (+-0.02) | robust (+-0.02) | max delta_min | never significant |
|---|---|---|---|---|---|---|---|
| early-late | 9 | 0 | no | 0 | no | 0.110 | yes |
| early-gated | 9 | 1 | no | 0 | no | 0.124 | yes |
| early-attention | 9 | 0 | no | 0 | no | 0.106 | yes |
| early-score | 9 | 0 | no | 0 | no | 0.127 | yes |
| late-gated | 9 | 5 | no | 0 | no | 0.088 | yes |
| late-attention | 9 | 5 | no | 0 | no | 0.088 | yes |
| late-score | 9 | 2 | no | 0 | no | 0.148 | yes |
| gated-attention | 9 | 8 | no | 0 | no | 0.078 | yes |
| gated-score | 9 | 0 | no | 0 | no | 0.146 | yes |
| attention-score | 9 | 0 | no | 0 | no | 0.144 | yes |

## 6. Robustness and performance

| model | mean_f1 | sd_across_pipelines | cv_across_pipelines | robustness |
|---|---|---|---|---|
| early | 0.6171 | 0.0105 | 0.0170 | 0.9895 |
| late | 0.6404 | 0.0185 | 0.0289 | 0.9815 |
| gated | 0.6306 | 0.0235 | 0.0372 | 0.9765 |
| attention | 0.6252 | 0.0180 | 0.0288 | 0.9820 |
| score | 0.6699 | 0.0143 | 0.0213 | 0.9857 |

## 7. Prediction stability

Pairwise agreement and Cohen's kappa of predicted classes (p > 0.5), between pipelines (same seed) and between seeds (same pipeline):

| model | between | agreement | kappa | all_agree | pairs |
|---|---|---|---|---|---|
| attention | pipelines | 0.7907 | 0.4697 | 0.4549 | 180 |
| attention | seeds | 0.8179 | 0.5389 | 0.6264 | 90 |
| early | pipelines | 0.7620 | 0.3115 | 0.3763 | 180 |
| early | seeds | 0.7557 | 0.2941 | 0.4988 | 90 |
| gated | pipelines | 0.8026 | 0.5024 | 0.4804 | 180 |
| gated | seeds | 0.8365 | 0.5875 | 0.6638 | 90 |
| late | pipelines | 0.7966 | 0.4795 | 0.4630 | 180 |
| late | seeds | 0.8287 | 0.5607 | 0.6494 | 90 |
| raw1d | pipelines | 0.7827 | 0.4726 | 0.4274 | 180 |
| raw1d | seeds | 0.8229 | 0.5711 | 0.6359 | 90 |
| raw1d_wide | pipelines | 0.8021 | 0.5042 | 0.4679 | 180 |
| raw1d_wide | seeds | 0.8462 | 0.6136 | 0.6837 | 90 |
| score | pipelines | 0.8219 | 0.4616 | 0.5359 | 180 |
| score | seeds | 0.8489 | 0.5426 | 0.6893 | 90 |
| spec2d | pipelines | 0.8184 | 0.4647 | 0.5172 | 180 |
| spec2d | seeds | 0.8352 | 0.5130 | 0.6596 | 90 |
| spec2d_wide | pipelines | 0.8223 | 0.4803 | 0.5241 | 180 |
| spec2d_wide | seeds | 0.8468 | 0.5510 | 0.6833 | 90 |

## 8. P4: technical artefact rejection

Rejected by the rule (from meta.json of 5 P4 run(s), identical in all): 0 of 328 ictal windows (0.0 %) and 1 of 1312 non-ictal windows (0.1 %). Rejected windows leave training and validation only; the test windows are those of every other pipeline.

Per person (14 persons; 1 with at least one rejected window; median share 0.00 %, maximum 0.4 % (PN10)):

| person | windows | ictal_total | ictal_rejected | nonictal_total | nonictal_rejected | rejected_pct |
|---|---|---|---|---|---|---|
| PN00 | 185 | 37 | 0 | 148 | 0 | 0.00 |
| PN01 | 75 | 15 | 0 | 60 | 0 | 0.00 |
| PN03 | 130 | 26 | 0 | 104 | 0 | 0.00 |
| PN05 | 65 | 13 | 0 | 52 | 0 | 0.00 |
| PN06 | 160 | 32 | 0 | 128 | 0 | 0.00 |
| PN07 | 40 | 8 | 0 | 32 | 0 | 0.00 |
| PN09 | 115 | 23 | 0 | 92 | 0 | 0.00 |
| PN10 | 235 | 47 | 0 | 188 | 1 | 0.43 |
| PN11 | 30 | 6 | 0 | 24 | 0 | 0.00 |
| PN12 | 155 | 31 | 0 | 124 | 0 | 0.00 |
| PN13 | 145 | 29 | 0 | 116 | 0 | 0.00 |
| PN14 | 100 | 20 | 0 | 80 | 0 | 0.00 |
| PN16 | 125 | 25 | 0 | 100 | 0 | 0.00 |
| PN17 | 80 | 16 | 0 | 64 | 0 | 0.00 |

Figure: `p4_rejection.png`.

## 9. P6: signal retention and computational cost

Source: C:\Projects\python\eeg-fusion-benchmark\data\ica_logs (123 logs). Power kept = signal power after cleaning / before, per recording.

| pipeline | method | recordings | power_kept_median | q25 | q75 | min | share_below_0.5 | share_below_0.9 | components_removed_mean | share_none_removed |
|---|---|---|---|---|---|---|---|---|---|---|
| P6a | Infomax (P6a) | 41 | 0.658 | 0.484 | 0.864 | 0.085 | 0.293 | 0.780 | 2.049 | 0.024 |
| P6b | GEDAI auto- (P6b, used) | 41 | 0.573 | 0.204 | 0.999 | 0.005 | 0.488 | 0.561 |  |  |
| P6c | AMICA (P6c) | 41 | 0.735 | 0.445 | 0.926 | 0.086 | 0.317 | 0.732 | 1.732 | 0.073 |

Computational cost (for the supplementary material; `p6_cost.csv`). Seconds are the logged processing time of each recording, summed for the total:

| pipeline | method | recordings | median_s_per_recording | mean_s_per_recording | total_compute_h | components_removed_median | components_removed_mean |
|---|---|---|---|---|---|---|---|
| P6a | Infomax (P6a) | 41 | 125.40 | 175.27 | 2.00 | 2.00 | 2.05 |
| P6b | GEDAI auto- (P6b, used) | 41 | 49.60 | 74.54 | 0.85 |  |  |
| P6c | AMICA (P6c) | 41 | 676.90 | 984.80 | 11.22 | 2.00 | 1.73 |

Figure: `p6_signal_retention.png`.
