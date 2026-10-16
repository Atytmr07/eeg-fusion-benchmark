# Controls against the fusion operators (corrected test)

Per pipeline: fold values averaged over seed sets, Nadeau-Bengio corrected t test, Holm across the 30 control-operator comparisons of the pipeline; equivalent = delta_min <= 0.05. Counts are pipelines (of 9). mean_diff = control minus operator.

## CHB-MIT, macro F1 (45 runs, 23 folds)

| control | operator | pipelines | mean_diff | control_higher_sig | control_lower_sig | equivalent | min_p_holm |
|---|---|---|---|---|---|---|---|
| raw1d | early | 9 | -0.013 | 0 | 0 | 2 | 1.000 |
| raw1d | late | 9 | -0.017 | 0 | 0 | 3 | 1.000 |
| raw1d | gated | 9 | -0.023 | 0 | 0 | 3 | 1.000 |
| raw1d | attention | 9 | -0.020 | 0 | 0 | 1 | 1.000 |
| raw1d | score | 9 | -0.010 | 0 | 0 | 8 | 1.000 |
| spec2d | early | 9 | 0.002 | 0 | 0 | 3 | 1.000 |
| spec2d | late | 9 | -0.002 | 0 | 0 | 0 | 1.000 |
| spec2d | gated | 9 | -0.007 | 0 | 0 | 1 | 1.000 |
| spec2d | attention | 9 | -0.004 | 0 | 0 | 0 | 1.000 |
| spec2d | score | 9 | 0.006 | 0 | 0 | 0 | 1.000 |
| raw1d_wide | early | 9 | -0.015 | 0 | 0 | 0 | 1.000 |
| raw1d_wide | late | 9 | -0.019 | 0 | 0 | 1 | 1.000 |
| raw1d_wide | gated | 9 | -0.024 | 0 | 0 | 1 | 1.000 |
| raw1d_wide | attention | 9 | -0.021 | 0 | 0 | 2 | 1.000 |
| raw1d_wide | score | 9 | -0.011 | 0 | 0 | 6 | 1.000 |
| spec2d_wide | early | 9 | 0.006 | 0 | 0 | 2 | 1.000 |
| spec2d_wide | late | 9 | 0.002 | 0 | 0 | 0 | 1.000 |
| spec2d_wide | gated | 9 | -0.003 | 0 | 0 | 1 | 1.000 |
| spec2d_wide | attention | 9 | 0.000 | 0 | 0 | 0 | 1.000 |
| spec2d_wide | score | 9 | 0.010 | 0 | 0 | 0 | 1.000 |
| logvar | early | 9 | -0.046 | 0 | 0 | 0 | 1.000 |
| logvar | late | 9 | -0.049 | 0 | 0 | 0 | 1.000 |
| logvar | gated | 9 | -0.055 | 0 | 0 | 0 | 1.000 |
| logvar | attention | 9 | -0.052 | 0 | 0 | 0 | 1.000 |
| logvar | score | 9 | -0.042 | 0 | 0 | 0 | 1.000 |
| shallow | early | 9 | 0.014 | 0 | 0 | 0 | 1.000 |
| shallow | late | 9 | 0.010 | 0 | 0 | 0 | 1.000 |
| shallow | gated | 9 | 0.004 | 0 | 0 | 0 | 1.000 |
| shallow | attention | 9 | 0.007 | 0 | 0 | 0 | 1.000 |
| shallow | score | 9 | 0.017 | 0 | 0 | 0 | 1.000 |

## CHB-MIT, AUPRC (45 runs, 23 folds)

| control | operator | pipelines | mean_diff | control_higher_sig | control_lower_sig | equivalent | min_p_holm |
|---|---|---|---|---|---|---|---|
| raw1d | early | 9 | -0.008 | 0 | 0 | 0 | 1.000 |
| raw1d | late | 9 | -0.052 | 0 | 0 | 0 | 1.000 |
| raw1d | gated | 9 | -0.061 | 0 | 0 | 0 | 1.000 |
| raw1d | attention | 9 | -0.051 | 0 | 0 | 0 | 1.000 |
| raw1d | score | 9 | -0.051 | 0 | 0 | 0 | 0.984 |
| spec2d | early | 9 | -0.015 | 0 | 0 | 0 | 1.000 |
| spec2d | late | 9 | -0.059 | 0 | 0 | 0 | 1.000 |
| spec2d | gated | 9 | -0.068 | 0 | 0 | 0 | 1.000 |
| spec2d | attention | 9 | -0.058 | 0 | 0 | 0 | 1.000 |
| spec2d | score | 9 | -0.058 | 0 | 0 | 0 | 1.000 |
| raw1d_wide | early | 9 | -0.008 | 0 | 0 | 0 | 1.000 |
| raw1d_wide | late | 9 | -0.051 | 0 | 0 | 0 | 1.000 |
| raw1d_wide | gated | 9 | -0.061 | 0 | 0 | 0 | 1.000 |
| raw1d_wide | attention | 9 | -0.051 | 0 | 0 | 0 | 1.000 |
| raw1d_wide | score | 9 | -0.050 | 0 | 0 | 0 | 1.000 |
| spec2d_wide | early | 9 | 0.005 | 0 | 0 | 0 | 1.000 |
| spec2d_wide | late | 9 | -0.038 | 0 | 0 | 0 | 1.000 |
| spec2d_wide | gated | 9 | -0.048 | 0 | 0 | 0 | 1.000 |
| spec2d_wide | attention | 9 | -0.038 | 0 | 0 | 0 | 1.000 |
| spec2d_wide | score | 9 | -0.037 | 0 | 0 | 0 | 1.000 |
| logvar | early | 9 | -0.035 | 0 | 0 | 0 | 1.000 |
| logvar | late | 9 | -0.079 | 0 | 0 | 0 | 1.000 |
| logvar | gated | 9 | -0.088 | 0 | 0 | 0 | 1.000 |
| logvar | attention | 9 | -0.078 | 0 | 0 | 0 | 1.000 |
| logvar | score | 9 | -0.078 | 0 | 0 | 0 | 1.000 |
| shallow | early | 9 | 0.027 | 0 | 0 | 0 | 1.000 |
| shallow | late | 9 | -0.017 | 0 | 0 | 0 | 1.000 |
| shallow | gated | 9 | -0.026 | 0 | 0 | 0 | 1.000 |
| shallow | attention | 9 | -0.016 | 0 | 0 | 0 | 1.000 |
| shallow | score | 9 | -0.016 | 0 | 0 | 0 | 1.000 |

## Siena, macro F1 (45 runs, 14 folds)

| control | operator | pipelines | mean_diff | control_higher_sig | control_lower_sig | equivalent | min_p_holm |
|---|---|---|---|---|---|---|---|
| raw1d | early | 9 | -0.008 | 0 | 0 | 0 | 1.000 |
| raw1d | late | 9 | -0.031 | 0 | 0 | 0 | 1.000 |
| raw1d | gated | 9 | -0.021 | 0 | 0 | 0 | 1.000 |
| raw1d | attention | 9 | -0.016 | 0 | 0 | 1 | 1.000 |
| raw1d | score | 9 | -0.061 | 0 | 0 | 1 | 1.000 |
| spec2d | early | 9 | 0.043 | 0 | 0 | 0 | 1.000 |
| spec2d | late | 9 | 0.020 | 0 | 0 | 0 | 1.000 |
| spec2d | gated | 9 | 0.030 | 0 | 0 | 0 | 1.000 |
| spec2d | attention | 9 | 0.035 | 0 | 0 | 0 | 1.000 |
| spec2d | score | 9 | -0.010 | 0 | 0 | 4 | 1.000 |
| raw1d_wide | early | 9 | -0.005 | 0 | 0 | 0 | 1.000 |
| raw1d_wide | late | 9 | -0.028 | 0 | 0 | 1 | 1.000 |
| raw1d_wide | gated | 9 | -0.019 | 0 | 0 | 0 | 1.000 |
| raw1d_wide | attention | 9 | -0.013 | 0 | 0 | 4 | 1.000 |
| raw1d_wide | score | 9 | -0.058 | 0 | 0 | 0 | 1.000 |
| spec2d_wide | early | 9 | 0.046 | 0 | 0 | 0 | 1.000 |
| spec2d_wide | late | 9 | 0.022 | 0 | 0 | 0 | 1.000 |
| spec2d_wide | gated | 9 | 0.032 | 0 | 0 | 1 | 1.000 |
| spec2d_wide | attention | 9 | 0.037 | 0 | 0 | 0 | 1.000 |
| spec2d_wide | score | 9 | -0.007 | 0 | 0 | 1 | 1.000 |
| logvar | early | 9 | 0.012 | 0 | 0 | 0 | 1.000 |
| logvar | late | 9 | -0.011 | 0 | 0 | 0 | 1.000 |
| logvar | gated | 9 | -0.001 | 0 | 0 | 0 | 1.000 |
| logvar | attention | 9 | 0.004 | 0 | 0 | 0 | 1.000 |
| logvar | score | 9 | -0.040 | 0 | 0 | 0 | 1.000 |
| shallow | early | 9 | 0.103 | 0 | 0 | 0 | 0.120 |
| shallow | late | 9 | 0.080 | 0 | 0 | 0 | 0.149 |
| shallow | gated | 9 | 0.090 | 0 | 0 | 0 | 0.279 |
| shallow | attention | 9 | 0.095 | 0 | 0 | 0 | 0.219 |
| shallow | score | 9 | 0.051 | 0 | 0 | 0 | 1.000 |

## Siena, AUPRC (45 runs, 14 folds)

| control | operator | pipelines | mean_diff | control_higher_sig | control_lower_sig | equivalent | min_p_holm |
|---|---|---|---|---|---|---|---|
| raw1d | early | 9 | 0.069 | 0 | 0 | 0 | 1.000 |
| raw1d | late | 9 | -0.034 | 0 | 0 | 0 | 1.000 |
| raw1d | gated | 9 | -0.026 | 0 | 0 | 0 | 1.000 |
| raw1d | attention | 9 | -0.015 | 0 | 0 | 1 | 1.000 |
| raw1d | score | 9 | -0.066 | 0 | 0 | 0 | 1.000 |
| spec2d | early | 9 | 0.080 | 0 | 0 | 0 | 1.000 |
| spec2d | late | 9 | -0.023 | 0 | 0 | 0 | 1.000 |
| spec2d | gated | 9 | -0.015 | 0 | 0 | 0 | 1.000 |
| spec2d | attention | 9 | -0.004 | 0 | 0 | 0 | 1.000 |
| spec2d | score | 9 | -0.055 | 0 | 0 | 0 | 1.000 |
| raw1d_wide | early | 9 | 0.081 | 0 | 0 | 0 | 0.315 |
| raw1d_wide | late | 9 | -0.022 | 0 | 0 | 0 | 1.000 |
| raw1d_wide | gated | 9 | -0.014 | 0 | 0 | 0 | 1.000 |
| raw1d_wide | attention | 9 | -0.003 | 0 | 0 | 1 | 1.000 |
| raw1d_wide | score | 9 | -0.054 | 0 | 0 | 0 | 1.000 |
| spec2d_wide | early | 9 | 0.085 | 0 | 0 | 0 | 0.183 |
| spec2d_wide | late | 9 | -0.018 | 0 | 0 | 0 | 1.000 |
| spec2d_wide | gated | 9 | -0.010 | 0 | 0 | 0 | 1.000 |
| spec2d_wide | attention | 9 | 0.001 | 0 | 0 | 0 | 1.000 |
| spec2d_wide | score | 9 | -0.050 | 0 | 0 | 0 | 1.000 |
| logvar | early | 9 | 0.076 | 0 | 0 | 0 | 1.000 |
| logvar | late | 9 | -0.027 | 0 | 0 | 0 | 1.000 |
| logvar | gated | 9 | -0.019 | 0 | 0 | 0 | 1.000 |
| logvar | attention | 9 | -0.008 | 0 | 0 | 0 | 1.000 |
| logvar | score | 9 | -0.059 | 0 | 0 | 0 | 1.000 |
| shallow | early | 9 | 0.170 | 0 | 0 | 0 | 0.177 |
| shallow | late | 9 | 0.068 | 0 | 0 | 0 | 1.000 |
| shallow | gated | 9 | 0.076 | 0 | 0 | 0 | 1.000 |
| shallow | attention | 9 | 0.087 | 0 | 0 | 0 | 1.000 |
| shallow | score | 9 | 0.036 | 0 | 0 | 0 | 1.000 |
