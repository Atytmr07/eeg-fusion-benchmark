# Multiverse analysis: chbmit_cuda

Runs found (45): loso_P0_cuda (P0, seed 0), loso_P0_r1_cuda (P0, seed 1), loso_P0_r2_cuda (P0, seed 2), loso_P0_r3_cuda (P0, seed 3), loso_P0_r4_cuda (P0, seed 4), loso_P1_cuda (P1, seed 0), loso_P1_r1_cuda (P1, seed 1), loso_P1_r2_cuda (P1, seed 2), loso_P1_r3_cuda (P1, seed 3), loso_P1_r4_cuda (P1, seed 4), loso_P2_cuda (P2, seed 0), loso_P2_r1_cuda (P2, seed 1), loso_P2_r2_cuda (P2, seed 2), loso_P2_r3_cuda (P2, seed 3), loso_P2_r4_cuda (P2, seed 4), loso_P3_cuda (P3, seed 0), loso_P3_r1_cuda (P3, seed 1), loso_P3_r2_cuda (P3, seed 2), loso_P3_r3_cuda (P3, seed 3), loso_P3_r4_cuda (P3, seed 4), loso_P4_cuda (P4, seed 0), loso_P4_r1_cuda (P4, seed 1), loso_P4_r2_cuda (P4, seed 2), loso_P4_r3_cuda (P4, seed 3), loso_P4_r4_cuda (P4, seed 4), loso_P5_cuda (P5, seed 0), loso_P5_r1_cuda (P5, seed 1), loso_P5_r2_cuda (P5, seed 2), loso_P5_r3_cuda (P5, seed 3), loso_P5_r4_cuda (P5, seed 4), loso_P6a_cuda (P6a, seed 0), loso_P6a_r1_cuda (P6a, seed 1), loso_P6a_r2_cuda (P6a, seed 2), loso_P6a_r3_cuda (P6a, seed 3), loso_P6a_r4_cuda (P6a, seed 4), loso_P6b_cuda (P6b, seed 0), loso_P6b_r1_cuda (P6b, seed 1), loso_P6b_r2_cuda (P6b, seed 2), loso_P6b_r3_cuda (P6b, seed 3), loso_P6b_r4_cuda (P6b, seed 4), loso_P6c_cuda (P6c, seed 0), loso_P6c_r1_cuda (P6c, seed 1), loso_P6c_r2_cuda (P6c, seed 2), loso_P6c_r3_cuda (P6c, seed 3), loso_P6c_r4_cuda (P6c, seed 4)

## 1. Performance

Means over folds per pipeline, seed and model: `performance.csv`; seed mean and SD per pipeline and model: `performance_by_pipeline.csv`. AUPRC is computed from the stored test probabilities (logvar and shallow store none).

Balanced block for the analyses below: pipelines P0, P1, P2, P3, P4, P5, P6a, P6b, P6c; seeds 0, 1, 2, 3, 4; 23 folds; fusion models early, late, gated, attention, score.

## Effects, separately

### Preprocessing effect (pipeline main effect, rank agreement between pipelines)

Pipeline main effect on macro F1: F(8, 176) = 0.29, p_GG = 0.834, 0.0 % of the variance; the fusion ranking agrees between runs of different pipelines with mean Kendall tau 0.03, against 0.10 between seeds of the same pipeline (D = 0.07, permutation p = 0.073).

### Seed effect (seed variance components, rank agreement between seeds)

All seed terms together hold 1.1 % of the variance of macro F1; rankings from different seeds of the same pipeline agree with mean Kendall tau 0.10 (Kendall's W over seeds, median over 9 pipelines: 0.22).

### Fusion x Preprocessing interaction (RM-ANOVA, variance share)

Fusion x Preprocessing: F(32, 704) = 1.73, p_GG = 0.069 (GG epsilon 0.34), partial eta squared 0.073; the fusion x pipeline component holds 0.1 % of the variance.

## 2. Rank stability (main analysis)

Ranking of the fusion operators by mean macro F1 per run: `rank_per_run.csv`.

| | value |
|---|---|
| mean Kendall tau, same pipeline, different seeds | 0.098 |
| mean Kendall tau, different pipelines | 0.028 |
| D = within - between | 0.070 |
| permutation p (pipeline labels permuted within seeds, 10000 permutations) | 0.0731 |

- P0: Kendall's W over 5 seeds = 0.184, mean pairwise tau = 0.040
- P1: Kendall's W over 5 seeds = 0.616, mean pairwise tau = 0.400
- P2: Kendall's W over 5 seeds = 0.176, mean pairwise tau = -0.040
- P3: Kendall's W over 5 seeds = 0.136, mean pairwise tau = -0.040
- P4: Kendall's W over 5 seeds = 0.456, mean pairwise tau = 0.240
- P5: Kendall's W over 5 seeds = 0.216, mean pairwise tau = 0.040
- P6a: Kendall's W over 5 seeds = 0.168, mean pairwise tau = -0.040
- P6b: Kendall's W over 5 seeds = 0.272, mean pairwise tau = 0.120
- P6c: Kendall's W over 5 seeds = 0.312, mean pairwise tau = 0.160

### Rank distribution

Rank of each operator in each of the 45 runs (1 = best). Not only the mean rank: the counts per rank over all runs, the median, the best and worst rank, and the share of runs the operator wins. Counts per pipeline (over seeds) and per seed (over pipelines) are in `rank_distribution.csv`; `rank_distribution.png` shows the counts and the rank map of every run.

| model | rank 1 | rank 2 | rank 3 | rank 4 | rank 5 |
|---|---|---|---|---|---|
| early | 5 | 9 | 12 | 10 | 9 |
| late | 9 | 10 | 8 | 11 | 7 |
| gated | 17 | 7 | 5 | 7 | 9 |
| attention | 11 | 13 | 8 | 7 | 6 |
| score | 3 | 6 | 12 | 10 | 14 |

| model | n_runs | mean_rank | median_rank | best_rank | worst_rank | share_first | share_last |
|---|---|---|---|---|---|---|---|
| early | 45 | 3.20 | 3.00 | 1 | 5 | 0.11 | 0.20 |
| late | 45 | 2.93 | 3.00 | 1 | 5 | 0.20 | 0.16 |
| gated | 45 | 2.64 | 2.00 | 1 | 5 | 0.38 | 0.20 |
| attention | 45 | 2.64 | 2.00 | 1 | 5 | 0.24 | 0.13 |
| score | 45 | 3.58 | 4.00 | 1 | 5 | 0.07 | 0.31 |

ICA algorithm (P6a, P6b, P6c): within tau 0.080, between tau 0.152, D -0.072, permutation p 0.8813; seed-averaged tau matrix:

| pipeline | P6a | P6b | P6c |
|---|---|---|---|
| P6a | 1.000 | 0.800 | 1.000 |
| P6b | 0.800 | 1.000 | 0.800 |
| P6c | 1.000 | 0.800 | 1.000 |

## 3. Fusion x Preprocessing

Repeated-measures ANOVA on macro F1 (seed means; folds as subjects; each effect tested against its interaction with fold; Greenhouse-Geisser):

| effect | df | df_error | F | p | gg_epsilon | p_gg | partial_eta2 |
|---|---|---|---|---|---|---|---|
| pipeline | 8 | 176 | 0.2876 | 0.9694 | 0.3738 | 0.8336 | 0.0129 |
| model | 4 | 88 | 0.5678 | 0.6866 | 0.5098 | 0.5740 | 0.0252 |
| pipeline x model | 32 | 704 | 1.7296 | 0.0080 | 0.3373 | 0.0692 | 0.0729 |

## 4. Variance components

Model: every fold (held-out person) x pipeline x fusion operator x seed cell holds one value of the metric, y_fpms = mu + a_f + b_p + c_m + d_s + every two-, three- and four-way interaction of these terms. All terms are treated as random effects (generalizability theory) and estimated by the method of moments from the expected mean squares of the fully crossed, balanced design. With one observation per cell the four-way interaction is confounded with the residual. A fixed factor's component (pipeline, operator) is then the variance of its effects. Negative estimates are set to 0 for the shares and listed. `sd` is the square root of the component, in the metric's own units, as a practical size; `share` is its part of the summed components. Grouped: 'fold x (pipeline, fusion)' sums the interactions of fold with pipeline, operator and both; 'seed (all seed terms)' sums every term containing seed.

### f1_macro

Grouped:

| group | sigma2 | sd | share_pct |
|---|---|---|---|
| fold (subject) | 0.015870 | 0.125978 | 74.228268 |
| pipeline | 0.000000 | 0.000000 | 0.000000 |
| fusion | 0.000000 | 0.000000 | 0.000000 |
| fusion x pipeline | 0.000019 | 0.004403 | 0.090686 |
| seed (all seed terms) | 0.000237 | 0.015383 | 1.106811 |
| fold x (pipeline, fusion) | 0.001545 | 0.039309 | 7.227184 |
| residual | 0.003709 | 0.060901 | 17.347052 |

Every component, unrounded estimate first:

| component | sigma2_raw | sigma2 | sd | share_pct |
|---|---|---|---|---|
| fold | 1.587044e-02 | 1.587044e-02 | 0.125978 | 74.228268 |
| pipeline | -2.718347e-05 | 0.000000e+00 | 0.000000 | 0.000000 |
| model | -2.375838e-05 | 0.000000e+00 | 0.000000 | 0.000000 |
| seed | -4.013748e-06 | 0.000000e+00 | 0.000000 | 0.000000 |
| fold x pipeline | 5.247783e-04 | 5.247783e-04 | 0.022908 | 2.454462 |
| fold x model | 9.253584e-04 | 9.253584e-04 | 0.030420 | 4.328031 |
| fold x seed | 7.192963e-06 | 7.192963e-06 | 0.002682 | 0.033642 |
| pipeline x model | 1.938918e-05 | 1.938918e-05 | 0.004403 | 0.090686 |
| pipeline x seed | 4.207567e-06 | 4.207567e-06 | 0.002051 | 0.019679 |
| model x seed | 4.904720e-06 | 4.904720e-06 | 0.002215 | 0.022940 |
| fold x pipeline x model | 9.507758e-05 | 9.507758e-05 | 0.009751 | 0.444691 |
| fold x pipeline x seed | -6.560921e-05 | 0.000000e+00 | 0.000000 | 0.000000 |
| fold x model x seed | 1.845425e-04 | 1.845425e-04 | 0.013585 | 0.863131 |
| pipeline x model x seed | 3.579479e-05 | 3.579479e-05 | 0.005983 | 0.167417 |
| residual (fold x pipeline x model x seed) | 3.708901e-03 | 3.708901e-03 | 0.060901 | 17.347052 |

Negative raw estimates (set to 0): pipeline, model, seed, fold x pipeline x seed

### auprc

Grouped:

| group | sigma2 | sd | share_pct |
|---|---|---|---|
| fold (subject) | 0.033324 | 0.182548 | 76.805016 |
| pipeline | 0.000056 | 0.007451 | 0.127943 |
| fusion | 0.000337 | 0.018357 | 0.776671 |
| fusion x pipeline | 0.000010 | 0.003211 | 0.023764 |
| seed (all seed terms) | 0.000665 | 0.025783 | 1.532197 |
| fold x (pipeline, fusion) | 0.003284 | 0.057310 | 7.570132 |
| residual | 0.005712 | 0.075575 | 13.164276 |

Every component, unrounded estimate first:

| component | sigma2_raw | sigma2 | sd | share_pct |
|---|---|---|---|---|
| fold | 3.332362e-02 | 3.332362e-02 | 0.182548 | 76.805016 |
| pipeline | 5.551109e-05 | 5.551109e-05 | 0.007451 | 0.127943 |
| model | 3.369768e-04 | 3.369768e-04 | 0.018357 | 0.776671 |
| seed | -7.556732e-06 | 0.000000e+00 | 0.000000 | 0.000000 |
| fold x pipeline | 1.347201e-03 | 1.347201e-03 | 0.036704 | 3.105057 |
| fold x model | 1.805094e-03 | 1.805094e-03 | 0.042486 | 4.160421 |
| fold x seed | -1.802889e-05 | 0.000000e+00 | 0.000000 | 0.000000 |
| pipeline x model | 1.031077e-05 | 1.031077e-05 | 0.003211 | 0.023764 |
| pipeline x seed | -1.694051e-06 | 0.000000e+00 | 0.000000 | 0.000000 |
| model x seed | 1.364270e-05 | 1.364270e-05 | 0.003694 | 0.031444 |
| fold x pipeline x model | 1.321811e-04 | 1.321811e-04 | 0.011497 | 0.304654 |
| fold x pipeline x seed | 2.009626e-05 | 2.009626e-05 | 0.004483 | 0.046318 |
| fold x model x seed | 6.310401e-04 | 6.310401e-04 | 0.025121 | 1.454435 |
| pipeline x model x seed | -9.942372e-09 | 0.000000e+00 | 0.000000 | 0.000000 |
| residual (fold x pipeline x model x seed) | 5.711624e-03 | 5.711624e-03 | 0.075575 | 13.164276 |

Negative raw estimates (set to 0): seed, fold x seed, pipeline x seed, pipeline x model x seed

### balanced_accuracy

Grouped:

| group | sigma2 | sd | share_pct |
|---|---|---|---|
| fold (subject) | 0.010932 | 0.104555 | 72.849576 |
| pipeline | 0.000000 | 0.000000 | 0.000000 |
| fusion | 0.000000 | 0.000000 | 0.000000 |
| fusion x pipeline | 0.000009 | 0.003005 | 0.060167 |
| seed (all seed terms) | 0.000209 | 0.014459 | 1.393209 |
| fold x (pipeline, fusion) | 0.001114 | 0.033379 | 7.424850 |
| residual | 0.002742 | 0.052363 | 18.272197 |

Every component, unrounded estimate first:

| component | sigma2_raw | sigma2 | sd | share_pct |
|---|---|---|---|---|
| fold | 1.093166e-02 | 1.093166e-02 | 0.104555 | 72.849576 |
| pipeline | -1.888610e-05 | 0.000000e+00 | 0.000000 | 0.000000 |
| model | -2.802127e-05 | 0.000000e+00 | 0.000000 | 0.000000 |
| seed | -2.115494e-06 | 0.000000e+00 | 0.000000 | 0.000000 |
| fold x pipeline | 4.037267e-04 | 4.037267e-04 | 0.020093 | 2.690472 |
| fold x model | 6.241344e-04 | 6.241344e-04 | 0.024983 | 4.159288 |
| fold x seed | 1.263738e-05 | 1.263738e-05 | 0.003555 | 0.084217 |
| pipeline x model | 9.028592e-06 | 9.028592e-06 | 0.003005 | 0.060167 |
| pipeline x seed | 5.211764e-06 | 5.211764e-06 | 0.002283 | 0.034732 |
| model x seed | -1.649124e-06 | 0.000000e+00 | 0.000000 | 0.000000 |
| fold x pipeline x model | 8.629689e-05 | 8.629689e-05 | 0.009290 | 0.575090 |
| fold x pipeline x seed | -4.628647e-05 | 0.000000e+00 | 0.000000 | 0.000000 |
| fold x model x seed | 1.586340e-04 | 1.586340e-04 | 0.012595 | 1.057152 |
| pipeline x model x seed | 3.257901e-05 | 3.257901e-05 | 0.005708 | 0.217109 |
| residual (fold x pipeline x model x seed) | 2.741889e-03 | 2.741889e-03 | 0.052363 | 18.272197 |

Negative raw estimates (set to 0): pipeline, model, seed, model x seed, fold x pipeline x seed

## 5. Equivalence and robust equivalence

Equivalent = corrected equivalence bound delta_min <= 0.05 macro F1 (test/train ratio 1/22). Robust equivalence = equivalent in every pipeline.

| pair | n_pipelines | equivalent_in | robust_equivalent | never_significant | max_delta_min |
|---|---|---|---|---|---|
| early-late | 9 | 6 | False | True | 0.0644 |
| early-gated | 9 | 5 | False | True | 0.0661 |
| early-attention | 9 | 7 | False | True | 0.0620 |
| early-score | 9 | 7 | False | True | 0.0648 |
| late-gated | 9 | 8 | False | True | 0.0517 |
| late-attention | 9 | 9 | True | True | 0.0434 |
| late-score | 9 | 7 | False | True | 0.0542 |
| gated-attention | 9 | 9 | True | True | 0.0407 |
| gated-score | 9 | 7 | False | True | 0.0639 |
| attention-score | 9 | 8 | False | True | 0.0505 |

Strict sensitivity analysis, margin 0.02:

| pair | n_pipelines | equivalent_in | robust_equivalent | never_significant | max_delta_min |
|---|---|---|---|---|---|
| early-late | 9 | 0 | False | True | 0.0644 |
| early-gated | 9 | 0 | False | True | 0.0661 |
| early-attention | 9 | 0 | False | True | 0.0620 |
| early-score | 9 | 0 | False | True | 0.0648 |
| late-gated | 9 | 0 | False | True | 0.0517 |
| late-attention | 9 | 1 | False | True | 0.0434 |
| late-score | 9 | 0 | False | True | 0.0542 |
| gated-attention | 9 | 0 | False | True | 0.0407 |
| gated-score | 9 | 0 | False | True | 0.0639 |
| attention-score | 9 | 0 | False | True | 0.0505 |

Both margins side by side: robustly equivalent pairs 2/10 at +-0.05 and 0/10 at +-0.02.

| pair | n_pipelines | equivalent in (+-0.05) | robust (+-0.05) | equivalent in (+-0.02) | robust (+-0.02) | max delta_min | never significant |
|---|---|---|---|---|---|---|---|
| early-late | 9 | 6 | no | 0 | no | 0.064 | yes |
| early-gated | 9 | 5 | no | 0 | no | 0.066 | yes |
| early-attention | 9 | 7 | no | 0 | no | 0.062 | yes |
| early-score | 9 | 7 | no | 0 | no | 0.065 | yes |
| late-gated | 9 | 8 | no | 0 | no | 0.052 | yes |
| late-attention | 9 | 9 | yes | 1 | no | 0.043 | yes |
| late-score | 9 | 7 | no | 0 | no | 0.054 | yes |
| gated-attention | 9 | 9 | yes | 0 | no | 0.041 | yes |
| gated-score | 9 | 7 | no | 0 | no | 0.064 | yes |
| attention-score | 9 | 8 | no | 0 | no | 0.050 | yes |

## 6. Robustness and performance

| model | mean_f1 | sd_across_pipelines | cv_across_pipelines | robustness |
|---|---|---|---|---|
| early | 0.6768 | 0.0079 | 0.0117 | 0.9921 |
| late | 0.6806 | 0.0072 | 0.0105 | 0.9928 |
| gated | 0.6860 | 0.0082 | 0.0120 | 0.9918 |
| attention | 0.6829 | 0.0070 | 0.0102 | 0.9930 |
| score | 0.6730 | 0.0080 | 0.0119 | 0.9920 |

## 7. Prediction stability

Pairwise agreement and Cohen's kappa of predicted classes (p > 0.5), between pipelines (same seed) and between seeds (same pipeline):

| model | between | agreement | kappa | all_agree | pairs |
|---|---|---|---|---|---|
| attention | pipelines | 0.8842 | 0.5659 | 0.6885 | 180 |
| attention | seeds | 0.8888 | 0.5834 | 0.7693 | 90 |
| early | pipelines | 0.8697 | 0.5387 | 0.6359 | 180 |
| early | seeds | 0.8703 | 0.5411 | 0.7275 | 90 |
| gated | pipelines | 0.8835 | 0.5507 | 0.6894 | 180 |
| gated | seeds | 0.8914 | 0.5802 | 0.7773 | 90 |
| late | pipelines | 0.8816 | 0.5516 | 0.6869 | 180 |
| late | seeds | 0.8863 | 0.5706 | 0.7659 | 90 |
| raw1d | pipelines | 0.8686 | 0.5861 | 0.6422 | 180 |
| raw1d | seeds | 0.8829 | 0.6300 | 0.7568 | 90 |
| raw1d_wide | pipelines | 0.8635 | 0.5589 | 0.6333 | 180 |
| raw1d_wide | seeds | 0.8837 | 0.6238 | 0.7599 | 90 |
| score | pipelines | 0.8914 | 0.5970 | 0.7069 | 180 |
| score | seeds | 0.8998 | 0.6270 | 0.7915 | 90 |
| spec2d | pipelines | 0.8736 | 0.5279 | 0.6562 | 180 |
| spec2d | seeds | 0.8779 | 0.5440 | 0.7462 | 90 |
| spec2d_wide | pipelines | 0.8840 | 0.5484 | 0.6829 | 180 |
| spec2d_wide | seeds | 0.8922 | 0.5797 | 0.7758 | 90 |

## 8. P4: technical artefact rejection

Rejected by the rule (from meta.json of 5 P4 run(s), identical in all): 42 of 1276 ictal windows (3.3 %) and 39 of 5104 non-ictal windows (0.8 %). Rejected windows leave training and validation only; the test windows are those of every other pipeline.

Per person (23 persons; 18 with at least one rejected window; median share 1.43 %, maximum 6.1 % (chb10)):

| person | windows | ictal_total | ictal_rejected | nonictal_total | nonictal_rejected | rejected_pct |
|---|---|---|---|---|---|---|
| chb01 | 370 | 74 | 0 | 296 | 1 | 0.27 |
| chb02 | 100 | 20 | 2 | 80 | 2 | 4.00 |
| chb03 | 225 | 45 | 7 | 180 | 1 | 3.56 |
| chb04 | 210 | 42 | 5 | 168 | 5 | 4.76 |
| chb05 | 310 | 62 | 0 | 248 | 0 | 0.00 |
| chb06 | 120 | 24 | 1 | 96 | 0 | 0.83 |
| chb07 | 180 | 36 | 2 | 144 | 1 | 1.67 |
| chb08 | 485 | 97 | 4 | 388 | 3 | 1.44 |
| chb09 | 155 | 31 | 3 | 124 | 0 | 1.94 |
| chb10 | 245 | 49 | 11 | 196 | 4 | 6.12 |
| chb11 | 415 | 83 | 1 | 332 | 0 | 0.24 |
| chb12 | 615 | 123 | 1 | 492 | 7 | 1.30 |
| chb13 | 335 | 67 | 0 | 268 | 5 | 1.49 |
| chb14 | 120 | 24 | 0 | 96 | 0 | 0.00 |
| chb15 | 1070 | 214 | 0 | 856 | 1 | 0.09 |
| chb16 | 70 | 14 | 0 | 56 | 1 | 1.43 |
| chb17 | 160 | 32 | 0 | 128 | 6 | 3.75 |
| chb18 | 190 | 38 | 1 | 152 | 0 | 0.53 |
| chb19 | 135 | 27 | 3 | 108 | 0 | 2.22 |
| chb20 | 180 | 36 | 1 | 144 | 2 | 1.67 |
| chb22 | 120 | 24 | 0 | 96 | 0 | 0.00 |
| chb23 | 245 | 49 | 0 | 196 | 0 | 0.00 |
| chb24 | 325 | 65 | 0 | 260 | 0 | 0.00 |

Figure: `p4_rejection.png`.

## 9. P6: signal retention and computational cost

Source: C:\Projects\python\eeg-fusion-benchmark\data\ica_logs (2719 logs). Power kept = signal power after cleaning / before, per recording.

| pipeline | method | recordings | power_kept_median | q25 | q75 | min | share_below_0.5 | share_below_0.9 | components_removed_mean | share_none_removed |
|---|---|---|---|---|---|---|---|---|---|---|
| P6a | Infomax (P6a) | 683 | 0.787 | 0.535 | 0.900 | 0.090 | 0.220 | 0.750 | 1.485 | 0.138 |
| P6b | GEDAI auto- (P6b, used) | 683 | 0.999 | 0.719 | 0.999 | 0.041 | 0.205 | 0.277 |  |  |
| P6b-default | GEDAI default (not used) | 670 | 0.658 | 0.317 | 0.999 | 0.041 | 0.433 | 0.545 |  |  |
| P6c | AMICA (P6c) | 683 | 0.785 | 0.537 | 0.886 | 0.076 | 0.225 | 0.782 | 1.530 | 0.122 |

Computational cost (for the supplementary material; `p6_cost.csv`). Seconds are the logged processing time of each recording, summed for the total:

| pipeline | method | recordings | median_s_per_recording | mean_s_per_recording | total_compute_h | components_removed_median | components_removed_mean |
|---|---|---|---|---|---|---|---|
| P6a | Infomax (P6a) | 683 | 36.10 | 50.05 | 9.50 | 1.00 | 1.48 |
| P6b | GEDAI auto- (P6b, used) | 683 | 23.90 | 33.41 | 6.34 |  |  |
| P6b-default | GEDAI default (not used) | 670 | 21.90 | 30.07 | 5.60 |  |  |
| P6c | AMICA (P6c) | 683 | 218.40 | 356.66 | 67.67 | 1.00 | 1.53 |

Figure: `p6_signal_retention.png`.
