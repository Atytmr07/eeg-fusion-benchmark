# P6 reproducibility: berk vs laptop_flip

6 CHB-MIT recordings, cleaned signal of the first 10 minutes; PSD over the whole recording.

| method | recordings | identical | corr_min | corr_median | rms_diff_rel_median | rms_diff_rel_max | psd_change_db_median | psd_change_db_max | same_component_count |
|---|---|---|---|---|---|---|---|---|---|
| gedai | 6 | 0 | 0.4276 | 0.9778 | 0.214 | 0.6949 | 0.742 | 20.27 | 0 |
| infomax | 6 | 0 | 0.2094 | 1 | 1.11e-05 | 0.496 | 7.892e-05 | 18.25 | True |

## Per recording

| record | method | identical | corr_min | corr_median | rms_diff_rel | max_diff_rel | psd_change_db_median | psd_change_db_max | power_kept_a | power_kept_b | removed_a | removed_b | same_count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chb01_09.edf | gedai | False | 0.9583 | 0.9785 | 0.2119 | 0.6383 | 0.3918 | 1.859 | 0.3741 | 0.3466 | nan | nan | nan |
| chb01_25.edf | gedai | False | 0.7958 | 0.8756 | 0.6175 | 1.676 | 1.57 | 5.982 | 0.353 | 0.5701 | nan | nan | nan |
| chb02_08.edf | gedai | False | 0.4276 | 0.7888 | 0.6949 | 1.034 | 2.705 | 20.27 | 0.9994 | 0.4457 | nan | nan | nan |
| chb02_16+.edf | gedai | False | 0.9665 | 0.9798 | 0.2084 | 0.1962 | 0.1469 | 1.063 | 0.2307 | 0.2419 | nan | nan | nan |
| chb03_02.edf | gedai | False | 0.9632 | 0.9771 | 0.2161 | 0.3152 | 1.092 | 2.564 | 0.2708 | 0.259 | nan | nan | nan |
| chb07_05.edf | gedai | False | 1 | 1 | 3.88e-10 | 7.807e-09 | 8.824e-09 | 0.0002141 | 0.1288 | 0.1288 | nan | nan | nan |
| chb01_09.edf | infomax | False | 1 | 1 | 5.686e-09 | 2.311e-08 | 6.36e-07 | 0.09618 | 0.4223 | 0.4223 | [0, 2] | [0, 2] | True |
| chb01_25.edf | infomax | False | 1 | 1 | 2.22e-05 | 2.43e-05 | 0.0001572 | 1.081 | 0.5321 | 0.5321 | [0, 2, 12] | [0, 2, 12] | True |
| chb02_08.edf | infomax | False | 1 | 1 | 3.177e-10 | 7.032e-09 | 3.031e-08 | 0.006496 | 0.6935 | 0.6935 | [1, 2, 13] | [1, 2, 13] | True |
| chb02_16+.edf | infomax | False | 0.2094 | 0.981 | 0.496 | 0.3655 | 0.1584 | 18.25 | 0.5577 | 0.6464 | [0, 5, 8] | [0] | False |
| chb03_02.edf | infomax | False | 1 | 1 | 0.001023 | 0.000922 | 0.001598 | 0.7848 | 0.5319 | 0.5319 | [0] | [0] | True |
| chb07_05.edf | infomax | False | 1 | 1 | 7.863e-11 | 2.832e-09 | 1.299e-09 | 0.0005108 | 1 | 1 | [] | [] | True |

## Environments

**berk**: 192.168.1.5, arm, threads unlimited, numpy 2.5.3, scipy 1.16.1, mne 1.13.2, gedai 0.60, jamica 0.3.0, scikit-learn 1.7.2, torch 2.8.0

thread pools: 

**laptop_flip**: Lenovo-aty, Intel64 Family 6 Model 170 Stepping 4, GenuineIntel, threads unlimited, numpy 2.5.3, scipy 1.16.1, mne 1.13.2, gedai 0.60, jamica 0.3.0, scikit-learn 1.7.2, torch 2.8.0+cpu

thread pools: openblas 14 (Haswell); openblas 14 (Haswell)
