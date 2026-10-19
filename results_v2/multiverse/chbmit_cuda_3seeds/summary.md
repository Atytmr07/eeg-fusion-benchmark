# Multiverse analysis: chbmit_cuda

Runs found (27): loso_P0_cuda (P0, seed 0), loso_P0_r1_cuda (P0, seed 1), loso_P0_r2_cuda (P0, seed 2), loso_P1_cuda (P1, seed 0), loso_P1_r1_cuda (P1, seed 1), loso_P1_r2_cuda (P1, seed 2), loso_P2_cuda (P2, seed 0), loso_P2_r1_cuda (P2, seed 1), loso_P2_r2_cuda (P2, seed 2), loso_P3_cuda (P3, seed 0), loso_P3_r1_cuda (P3, seed 1), loso_P3_r2_cuda (P3, seed 2), loso_P4_cuda (P4, seed 0), loso_P4_r1_cuda (P4, seed 1), loso_P4_r2_cuda (P4, seed 2), loso_P5_cuda (P5, seed 0), loso_P5_r1_cuda (P5, seed 1), loso_P5_r2_cuda (P5, seed 2), loso_P6a_cuda (P6a, seed 0), loso_P6a_r1_cuda (P6a, seed 1), loso_P6a_r2_cuda (P6a, seed 2), loso_P6b_cuda (P6b, seed 0), loso_P6b_r1_cuda (P6b, seed 1), loso_P6b_r2_cuda (P6b, seed 2), loso_P6c_cuda (P6c, seed 0), loso_P6c_r1_cuda (P6c, seed 1), loso_P6c_r2_cuda (P6c, seed 2)

## 1. Performance

Means over folds per pipeline, seed and model: `performance.csv`; seed mean and SD per pipeline and model: `performance_by_pipeline.csv`. AUPRC is computed from the stored test probabilities (logvar and shallow store none).

Balanced block for the analyses below: pipelines P0, P1, P2, P3, P4, P5, P6a, P6b, P6c; seeds 0, 1, 2; 23 folds; fusion models early, late, gated, attention, score.

## Effects, separately

### Preprocessing effect (pipeline main effect, rank agreement between pipelines)

Pipeline main effect on macro F1: F(8, 176) = 0.21, p_GG = 0.912, 0.0 % of the variance; the fusion ranking agrees between runs of different pipelines with mean Kendall tau -0.02, against 0.08 between seeds of the same pipeline (D = 0.10, permutation p = 0.196).

### Seed effect (seed variance components, rank agreement between seeds)

All seed terms together hold 1.2 % of the variance of macro F1; rankings from different seeds of the same pipeline agree with mean Kendall tau 0.08 (Kendall's W over seeds, median over 9 pipelines: 0.33).

### Fusion x Preprocessing interaction (RM-ANOVA, variance share)

Fusion x Preprocessing: F(32, 704) = 1.94, p_GG = 0.034 (GG epsilon 0.36), partial eta squared 0.081; the fusion x pipeline component holds 0.1 % of the variance.

## 2. Rank stability (main analysis)

Ranking of the fusion operators by mean macro F1 per run: `rank_per_run.csv`.

| | value |
|---|---|
| mean Kendall tau, same pipeline, different seeds | 0.081 |
| mean Kendall tau, different pipelines | -0.020 |
| D = within - between | 0.101 |
| permutation p (pipeline labels permuted within seeds, 10000 permutations) | 0.1956 |

- P0: Kendall's W over 3 seeds = 0.378, mean pairwise tau = 0.067
- P1: Kendall's W over 3 seeds = 0.600, mean pairwise tau = 0.333
- P2: Kendall's W over 3 seeds = 0.267, mean pairwise tau = -0.067
- P3: Kendall's W over 3 seeds = 0.289, mean pairwise tau = -0.067
- P4: Kendall's W over 3 seeds = 0.822, mean pairwise tau = 0.600
- P5: Kendall's W over 3 seeds = 0.378, mean pairwise tau = 0.067
- P6a: Kendall's W over 3 seeds = 0.156, mean pairwise tau = -0.200
- P6b: Kendall's W over 3 seeds = 0.244, mean pairwise tau = -0.067
- P6c: Kendall's W over 3 seeds = 0.333, mean pairwise tau = 0.067

### Rank distribution

Rank of each operator in each of the 27 runs (1 = best). Not only the mean rank: the counts per rank over all runs, the median, the best and worst rank, and the share of runs the operator wins. Counts per pipeline (over seeds) and per seed (over pipelines) are in `rank_distribution.csv`; `rank_distribution.png` shows the counts and the rank map of every run.

| model | rank 1 | rank 2 | rank 3 | rank 4 | rank 5 |
|---|---|---|---|---|---|
| early | 5 | 8 | 6 | 3 | 5 |
| late | 5 | 4 | 4 | 8 | 6 |
| gated | 11 | 2 | 2 | 5 | 7 |
| attention | 5 | 7 | 6 | 5 | 4 |
| score | 1 | 6 | 9 | 6 | 5 |

| model | n_runs | mean_rank | median_rank | best_rank | worst_rank | share_first | share_last |
|---|---|---|---|---|---|---|---|
| early | 27 | 2.81 | 3.00 | 1 | 5 | 0.19 | 0.19 |
| late | 27 | 3.22 | 4.00 | 1 | 5 | 0.19 | 0.22 |
| gated | 27 | 2.81 | 3.00 | 1 | 5 | 0.41 | 0.26 |
| attention | 27 | 2.85 | 3.00 | 1 | 5 | 0.19 | 0.15 |
| score | 27 | 3.30 | 3.00 | 1 | 5 | 0.04 | 0.19 |

ICA algorithm (P6a, P6b, P6c): within tau -0.067, between tau 0.081, D -0.148, permutation p 0.9718; seed-averaged tau matrix:

| pipeline | P6a | P6b | P6c |
|---|---|---|---|
| P6a | 1.000 | 0.800 | 0.400 |
| P6b | 0.800 | 1.000 | 0.600 |
| P6c | 0.400 | 0.600 | 1.000 |

## 3. Fusion x Preprocessing

Repeated-measures ANOVA on macro F1 (seed means; folds as subjects; each effect tested against its interaction with fold; Greenhouse-Geisser):

| effect | df | df_error | F | p | gg_epsilon | p_gg | partial_eta2 |
|---|---|---|---|---|---|---|---|
| pipeline | 8 | 176 | 0.2061 | 0.9895 | 0.4242 | 0.9115 | 0.0093 |
| model | 4 | 88 | 0.2682 | 0.8977 | 0.5947 | 0.8025 | 0.0120 |
| pipeline x model | 32 | 704 | 1.9351 | 0.0017 | 0.3566 | 0.0335 | 0.0808 |

## 4. Variance components

| group | sigma2 | share_pct |
|---|---|---|
| fold (subject) | 0.015997 | 73.838621 |
| pipeline | 0.000000 | 0.000000 |
| fusion | 0.000000 | 0.000000 |
| fusion x pipeline | 0.000028 | 0.130668 |
| seed (all seed terms) | 0.000253 | 1.169932 |
| fold x (pipeline, fusion) | 0.001455 | 6.716437 |
| residual | 0.003931 | 18.144342 |

Negative raw estimates (set to 0): pipeline, model, seed, pipeline x seed, model x seed, fold x pipeline x seed

## 5. Equivalence and robust equivalence

Equivalent = corrected equivalence bound delta_min <= 0.05 macro F1 (test/train ratio 1/22). Robust equivalence = equivalent in every pipeline.

| pair | n_pipelines | equivalent_in | robust_equivalent | never_significant | max_delta_min |
|---|---|---|---|---|---|
| early-late | 9 | 6 | False | True | 0.0744 |
| early-gated | 9 | 2 | False | True | 0.0838 |
| early-attention | 9 | 6 | False | True | 0.0625 |
| early-score | 9 | 6 | False | True | 0.0679 |
| late-gated | 9 | 7 | False | True | 0.0590 |
| late-attention | 9 | 7 | False | True | 0.0530 |
| late-score | 9 | 7 | False | True | 0.0631 |
| gated-attention | 9 | 5 | False | True | 0.0559 |
| gated-score | 9 | 6 | False | True | 0.0628 |
| attention-score | 9 | 9 | True | True | 0.0490 |

Strict sensitivity analysis, margin 0.02:

| pair | n_pipelines | equivalent_in | robust_equivalent | never_significant | max_delta_min |
|---|---|---|---|---|---|
| early-late | 9 | 0 | False | True | 0.0744 |
| early-gated | 9 | 0 | False | True | 0.0838 |
| early-attention | 9 | 0 | False | True | 0.0625 |
| early-score | 9 | 0 | False | True | 0.0679 |
| late-gated | 9 | 0 | False | True | 0.0590 |
| late-attention | 9 | 1 | False | True | 0.0530 |
| late-score | 9 | 0 | False | True | 0.0631 |
| gated-attention | 9 | 0 | False | True | 0.0559 |
| gated-score | 9 | 0 | False | True | 0.0628 |
| attention-score | 9 | 0 | False | True | 0.0490 |

Both margins side by side: robustly equivalent pairs 1/10 at +-0.05 and 0/10 at +-0.02.

| pair | n_pipelines | equivalent in (+-0.05) | robust (+-0.05) | equivalent in (+-0.02) | robust (+-0.02) | max delta_min | never significant |
|---|---|---|---|---|---|---|---|
| early-late | 9 | 6 | no | 0 | no | 0.074 | yes |
| early-gated | 9 | 2 | no | 0 | no | 0.084 | yes |
| early-attention | 9 | 6 | no | 0 | no | 0.062 | yes |
| early-score | 9 | 6 | no | 0 | no | 0.068 | yes |
| late-gated | 9 | 7 | no | 0 | no | 0.059 | yes |
| late-attention | 9 | 7 | no | 1 | no | 0.053 | yes |
| late-score | 9 | 7 | no | 0 | no | 0.063 | yes |
| gated-attention | 9 | 5 | no | 0 | no | 0.056 | yes |
| gated-score | 9 | 6 | no | 0 | no | 0.063 | yes |
| attention-score | 9 | 9 | yes | 0 | no | 0.049 | yes |

## 6. Robustness and performance

| model | mean_f1 | sd_across_pipelines | cv_across_pipelines | robustness |
|---|---|---|---|---|
| early | 0.6797 | 0.0112 | 0.0164 | 0.9888 |
| late | 0.6792 | 0.0099 | 0.0145 | 0.9901 |
| gated | 0.6862 | 0.0121 | 0.0177 | 0.9879 |
| attention | 0.6807 | 0.0090 | 0.0132 | 0.9910 |
| score | 0.6764 | 0.0063 | 0.0093 | 0.9937 |

## 7. Prediction stability

Pairwise agreement and Cohen's kappa of predicted classes (p > 0.5), between pipelines (same seed) and between seeds (same pipeline):

| model | between | agreement | kappa | all_agree | pairs |
|---|---|---|---|---|---|
| attention | pipelines | 0.8846 | 0.5730 | 0.6879 | 108 |
| attention | seeds | 0.8880 | 0.5848 | 0.8320 | 27 |
| early | pipelines | 0.8738 | 0.5357 | 0.6473 | 108 |
| early | seeds | 0.8762 | 0.5439 | 0.8143 | 27 |
| gated | pipelines | 0.8796 | 0.5418 | 0.6817 | 108 |
| gated | seeds | 0.8887 | 0.5746 | 0.8331 | 27 |
| late | pipelines | 0.8750 | 0.5433 | 0.6743 | 108 |
| late | seeds | 0.8838 | 0.5757 | 0.8257 | 27 |
| raw1d | pipelines | 0.8707 | 0.5954 | 0.6513 | 108 |
| raw1d | seeds | 0.8812 | 0.6276 | 0.8218 | 27 |
| raw1d_wide | pipelines | 0.8615 | 0.5466 | 0.6258 | 108 |
| raw1d_wide | seeds | 0.8808 | 0.6086 | 0.8212 | 27 |
| score | pipelines | 0.8911 | 0.5996 | 0.7080 | 108 |
| score | seeds | 0.8984 | 0.6259 | 0.8476 | 27 |
| spec2d | pipelines | 0.8699 | 0.5273 | 0.6473 | 108 |
| spec2d | seeds | 0.8765 | 0.5498 | 0.8147 | 27 |
| spec2d_wide | pipelines | 0.8832 | 0.5478 | 0.6820 | 108 |
| spec2d_wide | seeds | 0.8903 | 0.5745 | 0.8354 | 27 |

## 8. P4: technical artefact rejection

Rejected by the rule (from meta.json of 3 P4 run(s), identical in all): 42 of 1276 ictal windows (3.3 %) and 39 of 5104 non-ictal windows (0.8 %). Rejected windows leave training and validation only; the test windows are those of every other pipeline.

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

Source: A:\Sevgi Hoca Proje20260930\eeg-fusion-benchmark\data\ica_logs (2049 logs). Power kept = signal power after cleaning / before, per recording.

| pipeline | method | recordings | power_kept_median | q25 | q75 | min | share_below_0.5 | share_below_0.9 | components_removed_mean | share_none_removed |
|---|---|---|---|---|---|---|---|---|---|---|
| P6a | Infomax (P6a) | 683 | 0.787 | 0.536 | 0.901 | 0.090 | 0.221 | 0.748 | 1.483 | 0.141 |
| P6b | GEDAI auto- (P6b, used) | 683 | 0.999 | 0.956 | 0.999 | 0.041 | 0.183 | 0.242 |  |  |
| P6c | AMICA (P6c) | 683 | 0.786 | 0.537 | 0.886 | 0.076 | 0.225 | 0.782 | 1.530 | 0.122 |

Computational cost (for the supplementary material; `p6_cost.csv`). Seconds are the logged processing time of each recording, summed for the total:

| pipeline | method | recordings | median_s_per_recording | mean_s_per_recording | total_compute_h | components_removed_median | components_removed_mean |
|---|---|---|---|---|---|---|---|
| P6a | Infomax (P6a) | 683 | 39.30 | 53.04 | 10.06 | 1.00 | 1.48 |
| P6b | GEDAI auto- (P6b, used) | 683 | 33.30 | 48.01 | 9.11 |  |  |
| P6c | AMICA (P6c) | 683 | 257.80 | 362.14 | 68.71 | 1.00 | 1.53 |

Figure: `p6_signal_retention.png`.
