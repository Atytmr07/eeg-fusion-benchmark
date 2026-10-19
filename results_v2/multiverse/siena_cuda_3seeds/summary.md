# Multiverse analysis: siena_cuda

Runs found (27): loso_P0_cuda (P0, seed 0), loso_P0_r1_cuda (P0, seed 1), loso_P0_r2_cuda (P0, seed 2), loso_P1_cuda (P1, seed 0), loso_P1_r1_cuda (P1, seed 1), loso_P1_r2_cuda (P1, seed 2), loso_P2_cuda (P2, seed 0), loso_P2_r1_cuda (P2, seed 1), loso_P2_r2_cuda (P2, seed 2), loso_P3_cuda (P3, seed 0), loso_P3_r1_cuda (P3, seed 1), loso_P3_r2_cuda (P3, seed 2), loso_P4_cuda (P4, seed 0), loso_P4_r1_cuda (P4, seed 1), loso_P4_r2_cuda (P4, seed 2), loso_P5_cuda (P5, seed 0), loso_P5_r1_cuda (P5, seed 1), loso_P5_r2_cuda (P5, seed 2), loso_P6a_cuda (P6a, seed 0), loso_P6a_r1_cuda (P6a, seed 1), loso_P6a_r2_cuda (P6a, seed 2), loso_P6b_cuda (P6b, seed 0), loso_P6b_r1_cuda (P6b, seed 1), loso_P6b_r2_cuda (P6b, seed 2), loso_P6c_cuda (P6c, seed 0), loso_P6c_r1_cuda (P6c, seed 1), loso_P6c_r2_cuda (P6c, seed 2)

## 1. Performance

Means over folds per pipeline, seed and model: `performance.csv`; seed mean and SD per pipeline and model: `performance_by_pipeline.csv`. AUPRC is computed from the stored test probabilities (logvar and shallow store none).

Balanced block for the analyses below: pipelines P0, P1, P2, P3, P4, P5, P6a, P6b, P6c; seeds 0, 1, 2; 14 folds; fusion models early, late, gated, attention, score.

## Effects, separately

### Preprocessing effect (pipeline main effect, rank agreement between pipelines)

Pipeline main effect on macro F1: F(8, 104) = 1.17, p_GG = 0.333, 0.2 % of the variance; the fusion ranking agrees between runs of different pipelines with mean Kendall tau 0.24, against 0.30 between seeds of the same pipeline (D = 0.06, permutation p = 0.209).

### Seed effect (seed variance components, rank agreement between seeds)

All seed terms together hold 4.3 % of the variance of macro F1; rankings from different seeds of the same pipeline agree with mean Kendall tau 0.30 (Kendall's W over seeds, median over 9 pipelines: 0.60).

### Fusion x Preprocessing interaction (RM-ANOVA, variance share)

Fusion x Preprocessing: F(32, 416) = 1.11, p_GG = 0.361 (GG epsilon 0.20), partial eta squared 0.079; the fusion x pipeline component holds 0.2 % of the variance.

## 2. Rank stability (main analysis)

Ranking of the fusion operators by mean macro F1 per run: `rank_per_run.csv`.

| | value |
|---|---|
| mean Kendall tau, same pipeline, different seeds | 0.304 |
| mean Kendall tau, different pipelines | 0.244 |
| D = within - between | 0.059 |
| permutation p (pipeline labels permuted within seeds, 10000 permutations) | 0.2091 |

- P0: Kendall's W over 3 seeds = 0.711, mean pairwise tau = 0.467
- P1: Kendall's W over 3 seeds = 0.422, mean pairwise tau = 0.200
- P2: Kendall's W over 3 seeds = 0.644, mean pairwise tau = 0.333
- P3: Kendall's W over 3 seeds = 0.422, mean pairwise tau = 0.200
- P4: Kendall's W over 3 seeds = 0.600, mean pairwise tau = 0.333
- P5: Kendall's W over 3 seeds = 0.622, mean pairwise tau = 0.333
- P6a: Kendall's W over 3 seeds = 0.378, mean pairwise tau = 0.067
- P6b: Kendall's W over 3 seeds = 0.556, mean pairwise tau = 0.200
- P6c: Kendall's W over 3 seeds = 0.822, mean pairwise tau = 0.600

### Rank distribution

Rank of each operator in each of the 27 runs (1 = best). Not only the mean rank: the counts per rank over all runs, the median, the best and worst rank, and the share of runs the operator wins. Counts per pipeline (over seeds) and per seed (over pipelines) are in `rank_distribution.csv`; `rank_distribution.png` shows the counts and the rank map of every run.

| model | rank 1 | rank 2 | rank 3 | rank 4 | rank 5 |
|---|---|---|---|---|---|
| early | 0 | 7 | 6 | 5 | 9 |
| late | 3 | 8 | 4 | 9 | 3 |
| gated | 4 | 4 | 7 | 1 | 11 |
| attention | 1 | 2 | 9 | 11 | 4 |
| score | 19 | 6 | 1 | 1 | 0 |

| model | n_runs | mean_rank | median_rank | best_rank | worst_rank | share_first | share_last |
|---|---|---|---|---|---|---|---|
| early | 27 | 3.59 | 4.00 | 2 | 5 | 0.00 | 0.33 |
| late | 27 | 3.04 | 3.00 | 1 | 5 | 0.11 | 0.11 |
| gated | 27 | 3.41 | 3.00 | 1 | 5 | 0.15 | 0.41 |
| attention | 27 | 3.56 | 4.00 | 1 | 5 | 0.04 | 0.15 |
| score | 27 | 1.41 | 1.00 | 1 | 4 | 0.70 | 0.00 |

ICA algorithm (P6a, P6b, P6c): within tau 0.289, between tau 0.141, D 0.148, permutation p 0.2831; seed-averaged tau matrix:

| pipeline | P6a | P6b | P6c |
|---|---|---|---|
| P6a | 1.000 | 0.200 | 0.600 |
| P6b | 0.200 | 1.000 | 0.200 |
| P6c | 0.600 | 0.200 | 1.000 |

## 3. Fusion x Preprocessing

Repeated-measures ANOVA on macro F1 (seed means; folds as subjects; each effect tested against its interaction with fold; Greenhouse-Geisser):

| effect | df | df_error | F | p | gg_epsilon | p_gg | partial_eta2 |
|---|---|---|---|---|---|---|---|
| pipeline | 8 | 104 | 1.1731 | 0.3224 | 0.3876 | 0.3327 | 0.0828 |
| model | 4 | 52 | 2.2545 | 0.0757 | 0.4267 | 0.1344 | 0.1478 |
| pipeline x model | 32 | 416 | 1.1147 | 0.3091 | 0.1981 | 0.3614 | 0.0790 |

## 4. Variance components

| group | sigma2 | share_pct |
|---|---|---|
| fold (subject) | 0.005903 | 34.665465 |
| pipeline | 0.000032 | 0.187905 |
| fusion | 0.000161 | 0.946040 |
| fusion x pipeline | 0.000033 | 0.196139 |
| seed (all seed terms) | 0.000727 | 4.271802 |
| fold x (pipeline, fusion) | 0.004003 | 23.510162 |
| residual | 0.006168 | 36.222486 |

Negative raw estimates (set to 0): seed, pipeline x seed, model x seed, pipeline x model x seed

## 5. Equivalence and robust equivalence

Equivalent = corrected equivalence bound delta_min <= 0.05 macro F1 (test/train ratio 1/13). Robust equivalence = equivalent in every pipeline.

| pair | n_pipelines | equivalent_in | robust_equivalent | never_significant | max_delta_min |
|---|---|---|---|---|---|
| early-late | 9 | 0 | False | True | 0.1116 |
| early-gated | 9 | 0 | False | True | 0.1217 |
| early-attention | 9 | 0 | False | True | 0.0939 |
| early-score | 9 | 0 | False | True | 0.1225 |
| late-gated | 9 | 2 | False | True | 0.0973 |
| late-attention | 9 | 2 | False | True | 0.0958 |
| late-score | 9 | 0 | False | True | 0.1676 |
| gated-attention | 9 | 3 | False | True | 0.0654 |
| gated-score | 9 | 0 | False | True | 0.1635 |
| attention-score | 9 | 0 | False | True | 0.1616 |

Strict sensitivity analysis, margin 0.02:

| pair | n_pipelines | equivalent_in | robust_equivalent | never_significant | max_delta_min |
|---|---|---|---|---|---|
| early-late | 9 | 0 | False | True | 0.1116 |
| early-gated | 9 | 0 | False | True | 0.1217 |
| early-attention | 9 | 0 | False | True | 0.0939 |
| early-score | 9 | 0 | False | True | 0.1225 |
| late-gated | 9 | 0 | False | True | 0.0973 |
| late-attention | 9 | 0 | False | True | 0.0958 |
| late-score | 9 | 0 | False | True | 0.1676 |
| gated-attention | 9 | 0 | False | True | 0.0654 |
| gated-score | 9 | 0 | False | True | 0.1635 |
| attention-score | 9 | 0 | False | True | 0.1616 |

Both margins side by side: robustly equivalent pairs 0/10 at +-0.05 and 0/10 at +-0.02.

| pair | n_pipelines | equivalent in (+-0.05) | robust (+-0.05) | equivalent in (+-0.02) | robust (+-0.02) | max delta_min | never significant |
|---|---|---|---|---|---|---|---|
| early-late | 9 | 0 | no | 0 | no | 0.112 | yes |
| early-gated | 9 | 0 | no | 0 | no | 0.122 | yes |
| early-attention | 9 | 0 | no | 0 | no | 0.094 | yes |
| early-score | 9 | 0 | no | 0 | no | 0.122 | yes |
| late-gated | 9 | 2 | no | 0 | no | 0.097 | yes |
| late-attention | 9 | 2 | no | 0 | no | 0.096 | yes |
| late-score | 9 | 0 | no | 0 | no | 0.168 | yes |
| gated-attention | 9 | 3 | no | 0 | no | 0.065 | yes |
| gated-score | 9 | 0 | no | 0 | no | 0.164 | yes |
| attention-score | 9 | 0 | no | 0 | no | 0.162 | yes |

## 6. Robustness and performance

| model | mean_f1 | sd_across_pipelines | cv_across_pipelines | robustness |
|---|---|---|---|---|
| early | 0.6222 | 0.0121 | 0.0194 | 0.9879 |
| late | 0.6361 | 0.0225 | 0.0355 | 0.9775 |
| gated | 0.6266 | 0.0248 | 0.0395 | 0.9752 |
| attention | 0.6264 | 0.0196 | 0.0312 | 0.9804 |
| score | 0.6644 | 0.0169 | 0.0255 | 0.9831 |

## 7. Prediction stability

Pairwise agreement and Cohen's kappa of predicted classes (p > 0.5), between pipelines (same seed) and between seeds (same pipeline):

| model | between | agreement | kappa | all_agree | pairs |
|---|---|---|---|---|---|
| attention | pipelines | 0.7919 | 0.4695 | 0.4500 | 108 |
| attention | seeds | 0.8212 | 0.5447 | 0.7318 | 27 |
| early | pipelines | 0.7549 | 0.3053 | 0.3675 | 108 |
| early | seeds | 0.7569 | 0.3114 | 0.6353 | 27 |
| gated | pipelines | 0.7971 | 0.4981 | 0.4628 | 108 |
| gated | seeds | 0.8339 | 0.5876 | 0.7509 | 27 |
| late | pipelines | 0.7944 | 0.4799 | 0.4593 | 108 |
| late | seeds | 0.8220 | 0.5504 | 0.7331 | 27 |
| raw1d | pipelines | 0.7777 | 0.4620 | 0.4165 | 108 |
| raw1d | seeds | 0.8145 | 0.5513 | 0.7218 | 27 |
| raw1d_wide | pipelines | 0.8021 | 0.5062 | 0.4657 | 108 |
| raw1d_wide | seeds | 0.8417 | 0.6028 | 0.7625 | 27 |
| score | pipelines | 0.8149 | 0.4576 | 0.5159 | 108 |
| score | seeds | 0.8431 | 0.5389 | 0.7646 | 27 |
| spec2d | pipelines | 0.8128 | 0.4560 | 0.5000 | 108 |
| spec2d | seeds | 0.8294 | 0.5022 | 0.7440 | 27 |
| spec2d_wide | pipelines | 0.8133 | 0.4726 | 0.5059 | 108 |
| spec2d_wide | seeds | 0.8415 | 0.5517 | 0.7623 | 27 |

## 8. P4: technical artefact rejection

Rejected by the rule (from meta.json of 3 P4 run(s), identical in all): 0 of 328 ictal windows (0.0 %) and 1 of 1312 non-ictal windows (0.1 %). Rejected windows leave training and validation only; the test windows are those of every other pipeline.

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

Source: A:\Sevgi Hoca Proje20260930\eeg-fusion-benchmark\data\ica_logs (20 logs). Power kept = signal power after cleaning / before, per recording.

| pipeline | method | recordings | power_kept_median | q25 | q75 | min | share_below_0.5 | share_below_0.9 | components_removed_mean | share_none_removed |
|---|---|---|---|---|---|---|---|---|---|---|
| P6a | Infomax (P6a) | 20 | 0.756 | 0.574 | 0.829 | 0.085 | 0.150 | 0.800 | 1.850 | 0.050 |

Computational cost (for the supplementary material; `p6_cost.csv`). Seconds are the logged processing time of each recording, summed for the total:

| pipeline | method | recordings | median_s_per_recording | mean_s_per_recording | total_compute_h | components_removed_median | components_removed_mean |
|---|---|---|---|---|---|---|---|
| P6a | Infomax (P6a) | 20 | 52.80 | 84.39 | 0.47 | 2.00 | 1.85 |

Figure: `p6_signal_retention.png`.
