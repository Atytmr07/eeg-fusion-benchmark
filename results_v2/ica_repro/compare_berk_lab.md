# P6 reproducibility: berk vs lab

8 CHB-MIT recordings, cleaned signal of the first 10 minutes; PSD over the whole recording.

| method | recordings | identical | corr_min | corr_median | rms_diff_rel_median | rms_diff_rel_max | psd_change_db_median | psd_change_db_max | same_component_count |
|---|---|---|---|---|---|---|---|---|---|
| gedai | 8 | 0 | 1 | 1 | 9.161e-11 | 1.204e-10 | 1.309e-08 | 0.004604 | 0 |
| infomax | 8 | 0 | 1 | 1 | 6.762e-10 | 9.01e-07 | 2.09e-07 | 1.344 | True |

## Per recording

| record | method | identical | corr_min | corr_median | rms_diff_rel | max_diff_rel | psd_change_db_median | psd_change_db_max | power_kept_a | power_kept_b | removed_a | removed_b | same_count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chb01_11.edf | gedai | False | 1 | 1 | 4.397e-11 | 2.856e-09 | 8.799e-09 | 0.004207 | 0.9996 | 0.9996 | nan | nan | nan |
| chb03_08.edf | gedai | False | 1 | 1 | 1.204e-10 | 5.409e-09 | 6.94e-09 | 0.004426 | 0.9991 | 0.9991 | nan | nan | nan |
| chb05_14.edf | gedai | False | 1 | 1 | 9.899e-11 | 6.349e-09 | 1.851e-08 | 0.002978 | 0.9993 | 0.9993 | nan | nan | nan |
| chb07_10.edf | gedai | False | 1 | 1 | 6.714e-11 | 2.608e-09 | 1.237e-08 | 0.004604 | 0.9994 | 0.9994 | nan | nan | nan |
| chb08_15.edf | gedai | False | 1 | 1 | 1.042e-10 | 1.896e-08 | 4.236e-08 | 0.004507 | 0.9993 | 0.9993 | nan | nan | nan |
| chb13_03.edf | gedai | False | 1 | 1 | 8.423e-11 | 3.536e-09 | 1.38e-08 | 0.004324 | 0.9996 | 0.9996 | nan | nan | nan |
| chb17b_59.edf | gedai | False | 1 | 1 | 1.17e-10 | 4.755e-09 | 1.438e-08 | 0.003045 | 0.999 | 0.999 | nan | nan | nan |
| chb22_10.edf | gedai | False | 1 | 1 | 4.398e-11 | 2.074e-09 | 6.787e-09 | 0.002759 | 0.9995 | 0.9995 | nan | nan | nan |
| chb01_11.edf | infomax | False | 1 | 1 | 8.657e-10 | 6.128e-09 | 7.493e-08 | 0.01306 | 0.3277 | 0.3277 | [0, 1] | [0, 1] | True |
| chb03_08.edf | infomax | False | 1 | 1 | 7.645e-09 | 2.848e-08 | 1.957e-06 | 0.1025 | 0.5696 | 0.5696 | [0, 1] | [0, 1] | True |
| chb05_14.edf | infomax | False | 1 | 1 | 4.767e-10 | 1.271e-08 | 7.868e-08 | 0.01029 | 0.7747 | 0.7747 | [0] | [0] | True |
| chb07_10.edf | infomax | False | 1 | 1 | 2.639e-11 | 1.304e-09 | 4.219e-09 | 0.005245 | 1 | 1 | [] | [] | True |
| chb08_15.edf | infomax | False | 1 | 1 | 4.867e-10 | 3.796e-08 | 3.394e-07 | 0.01337 | 0.9183 | 0.9183 | [6, 14] | [6, 14] | True |
| chb13_03.edf | infomax | False | 1 | 1 | 2.198e-10 | 1.392e-08 | 1.168e-08 | 0.004466 | 0.7883 | 0.7883 | [0] | [0] | True |
| chb17b_59.edf | infomax | False | 1 | 1 | 9.01e-07 | 4.617e-07 | 1.174e-05 | 1.344 | 0.6214 | 0.6214 | [0] | [0] | True |
| chb22_10.edf | infomax | False | 1 | 1 | 8.928e-09 | 2.301e-08 | 8.126e-07 | 0.1153 | 0.2972 | 0.2972 | [0, 1] | [0, 1] | True |

## Environments

**berk**: 192.168.1.5, arm, threads unlimited, numpy 2.5.3, scipy 1.16.1, mne 1.13.2, gedai 0.60, jamica 0.3.0, scikit-learn 1.7.2, torch 2.8.0

thread pools: 

**lab**: DESKTOP-A0GHGPB, Intel64 Family 6 Model 85 Stepping 4, GenuineIntel, threads unlimited, numpy 2.5.3, scipy 1.16.1, mne 1.13.2, gedai 0.60, jamica 0.3.0, scikit-learn 1.7.2, torch 2.8.0

thread pools: openblas 24 (Haswell); openblas 24 (Haswell)
