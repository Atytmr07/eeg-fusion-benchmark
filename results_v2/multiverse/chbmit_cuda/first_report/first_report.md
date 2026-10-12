# First results: chbmit_cuda

The three items requested first: P4's rejection distribution, P6's signal retention, and the first rank stability table of the pipelines. Generated from the runs available now; rerun as more arrive (`python -m src.multiverse --dataset chbmit --first-report`).

## Runs included

45 complete run(s) (23 folds each) are included; 0 incomplete run(s) are not. `-` = not run yet.

| pipeline | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| P0 | yes | yes | yes | yes | yes |
| P1 | yes | yes | yes | yes | yes |
| P2 | yes | yes | yes | yes | yes |
| P3 | yes | yes | yes | yes | yes |
| P4 | yes | yes | yes | yes | yes |
| P5 | yes | yes | yes | yes | yes |
| P6a | yes | yes | yes | yes | yes |
| P6b | yes | yes | yes | yes | yes |
| P6c | yes | yes | yes | yes | yes |

## 1. P4: windows rejected as technical artefacts

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

![P4 rejection](p4_rejection.png)

## 2. P6: signal retention of Infomax, GEDAI and AMICA

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

![P6 signal retention](p6_signal_retention.png)

## 3. Rank stability of the fusion operators (first table)

Mean macro F1 over 23 folds and the available seeds, with the operator's rank in that pipeline (1 = best) and, where seeds disagree, the range of its single-seed ranks:

| pipeline | seeds | early | late | gated | attention | score | best | Kendall W (seeds) |
|---|---|---|---|---|---|---|---|---|
| P0 | 5 | 0.678 (3; 1-4) | 0.679 (2; 1-5) | 0.671 (4; 1-5) | 0.683 (1; 1-5) | 0.661 (5; 2-5) | attention | 0.18 |
| P1 | 5 | 0.684 (3; 1-4) | 0.665 (5; 4-5) | 0.691 (1; 1-4) | 0.687 (2; 1-3) | 0.672 (4; 3-5) | gated | 0.62 |
| P2 | 5 | 0.680 (4; 1-5) | 0.688 (1; 1-5) | 0.680 (3; 1-5) | 0.687 (2; 1-4) | 0.675 (5; 3-4) | late | 0.18 |
| P3 | 5 | 0.679 (2; 1-4) | 0.677 (4; 2-5) | 0.676 (5; 1-5) | 0.689 (1; 1-5) | 0.678 (3; 1-4) | attention | 0.14 |
| P4 | 5 | 0.688 (2; 1-3) | 0.678 (3; 1-4) | 0.694 (1; 1-3) | 0.670 (4; 1-5) | 0.660 (5; 3-5) | gated | 0.46 |
| P5 | 5 | 0.678 (5; 2-5) | 0.684 (3; 1-4) | 0.690 (2; 1-5) | 0.693 (1; 1-4) | 0.681 (4; 3-5) | attention | 0.22 |
| P6a | 5 | 0.672 (5; 3-5) | 0.679 (2; 1-5) | 0.692 (1; 1-5) | 0.677 (3; 1-5) | 0.675 (4; 2-5) | gated | 0.17 |
| P6b | 5 | 0.662 (5; 3-5) | 0.686 (2; 1-4) | 0.691 (1; 1-4) | 0.681 (4; 2-4) | 0.682 (3; 1-5) | gated | 0.27 |
| P6c | 5 | 0.671 (5; 3-5) | 0.688 (2; 1-3) | 0.688 (1; 1-4) | 0.679 (3; 1-5) | 0.674 (4; 2-5) | gated | 0.31 |

AUPRC (the second primary metric), seed mean per pipeline and operator:

| pipeline | early | late | gated | attention | score |
|---|---|---|---|---|---|
| P0 | 0.632 | 0.656 | 0.662 | 0.655 | 0.654 |
| P1 | 0.623 | 0.665 | 0.673 | 0.665 | 0.667 |
| P2 | 0.618 | 0.668 | 0.683 | 0.669 | 0.657 |
| P3 | 0.607 | 0.657 | 0.655 | 0.658 | 0.651 |
| P4 | 0.628 | 0.672 | 0.673 | 0.650 | 0.651 |
| P5 | 0.608 | 0.642 | 0.664 | 0.660 | 0.662 |
| P6a | 0.619 | 0.653 | 0.668 | 0.652 | 0.657 |
| P6b | 0.564 | 0.635 | 0.643 | 0.636 | 0.631 |
| P6c | 0.617 | 0.658 | 0.671 | 0.654 | 0.668 |

Over all included runs (rank counts, median, best and worst rank, share of runs won):

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

Kendall tau between the seed-averaged rankings of the pipelines:

| pipeline | P0 | P1 | P2 | P3 | P4 | P5 | P6a | P6b | P6c |
|---|---|---|---|---|---|---|---|---|---|
| P0 | 1.00 | 0.00 | 0.60 | 0.40 | -0.20 | 0.40 | 0.00 | -0.20 | 0.00 |
| P1 | 0.00 | 1.00 | 0.00 | 0.20 | 0.40 | 0.20 | 0.20 | 0.00 | 0.20 |
| P2 | 0.60 | 0.00 | 1.00 | 0.00 | 0.20 | 0.40 | 0.40 | 0.20 | 0.40 |
| P3 | 0.40 | 0.20 | 0.00 | 1.00 | -0.40 | -0.20 | -0.60 | -0.80 | -0.60 |
| P4 | -0.20 | 0.40 | 0.20 | -0.40 | 1.00 | 0.00 | 0.40 | 0.20 | 0.40 |
| P5 | 0.40 | 0.20 | 0.40 | -0.20 | 0.00 | 1.00 | 0.60 | 0.40 | 0.60 |
| P6a | 0.00 | 0.20 | 0.40 | -0.60 | 0.40 | 0.60 | 1.00 | 0.80 | 1.00 |
| P6b | -0.20 | 0.00 | 0.20 | -0.80 | 0.20 | 0.40 | 0.80 | 1.00 | 0.80 |
| P6c | 0.00 | 0.20 | 0.40 | -0.60 | 0.40 | 0.60 | 1.00 | 0.80 | 1.00 |

![rank distribution](rank_distribution.png)

Window-level macro F1 (seed means) is the metric here; AUPRC and the full multiverse analysis follow in `python -m src.multiverse` once the balanced block is complete. Event-based metrics are secondary (docs/EVENTS.md).
