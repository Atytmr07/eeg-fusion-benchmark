# P6 reproducibility: laptop vs lab_haswell

8 CHB-MIT recordings, cleaned signal of the first 10 minutes; PSD over the whole recording.

| method | recordings | identical | corr_min | corr_median | rms_diff_rel_median | rms_diff_rel_max | psd_change_db_median | psd_change_db_max | same_component_count |
|---|---|---|---|---|---|---|---|---|---|
| amica | 8 | 8 | 1 | 1 | 0 | 0 | 0 | 0 | True |
| gedai | 8 | 8 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| infomax | 8 | 3 | 1 | 1 | 7.569e-13 | 1.275e-08 | 0 | 0.3962 | True |

## Per recording

| record | method | identical | corr_min | corr_median | rms_diff_rel | max_diff_rel | psd_change_db_median | psd_change_db_max | power_kept_a | power_kept_b | removed_a | removed_b | same_count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chb01_11.edf | amica | True | 1 | 1 | 0 | 0 | 0 | 0 | 0.2942 | 0.2942 | [0, 2, 7] | [0, 2, 7] | True |
| chb03_08.edf | amica | True | 1 | 1 | 0 | 0 | 0 | 0 | 0.6268 | 0.6268 | [0, 1] | [0, 1] | True |
| chb05_14.edf | amica | True | 1 | 1 | 0 | 0 | 0 | 0 | 0.7702 | 0.7702 | [0] | [0] | True |
| chb07_10.edf | amica | True | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | [] | [] | True |
| chb08_15.edf | amica | True | 1 | 1 | 0 | 0 | 0 | 0 | 0.9702 | 0.9702 | [14] | [14] | True |
| chb13_03.edf | amica | True | 1 | 1 | 0 | 0 | 0 | 0 | 0.7378 | 0.7378 | [0, 6] | [0, 6] | True |
| chb17b_59.edf | amica | True | 1 | 1 | 0 | 0 | 0 | 0 | 0.5525 | 0.5525 | [0, 3] | [0, 3] | True |
| chb22_10.edf | amica | True | 1 | 1 | 0 | 0 | 0 | 0 | 0.2919 | 0.2919 | [0, 1] | [0, 1] | True |
| chb01_11.edf | gedai | True | 1 | 1 | 0 | 0 | 0 | 0 | 0.9996 | 0.9996 | nan | nan | nan |
| chb03_08.edf | gedai | True | 1 | 1 | 0 | 0 | 0 | 0 | 0.9991 | 0.9991 | nan | nan | nan |
| chb05_14.edf | gedai | True | 1 | 1 | 0 | 0 | 0 | 0 | 0.9993 | 0.9993 | nan | nan | nan |
| chb07_10.edf | gedai | True | 1 | 1 | 0 | 0 | 0 | 0 | 0.9994 | 0.9994 | nan | nan | nan |
| chb08_15.edf | gedai | True | 1 | 1 | 0 | 0 | 0 | 0 | 0.9993 | 0.9993 | nan | nan | nan |
| chb13_03.edf | gedai | True | 1 | 1 | 0 | 0 | 0 | 0 | 0.9996 | 0.9996 | nan | nan | nan |
| chb17b_59.edf | gedai | True | 1 | 1 | 0 | 0 | 0 | 0 | 0.999 | 0.999 | nan | nan | nan |
| chb22_10.edf | gedai | True | 1 | 1 | 0 | 0 | 0 | 0 | 0.9995 | 0.9995 | nan | nan | nan |
| chb01_11.edf | infomax | False | 1 | 1 | 1.18e-13 | 5.985e-12 | 0 | 0.000432 | 0.3277 | 0.3277 | [0, 1] | [0, 1] | True |
| chb03_08.edf | infomax | False | 1 | 1 | 4.798e-11 | 3.56e-09 | 7.036e-11 | 0.004296 | 0.5696 | 0.5696 | [0, 1] | [0, 1] | True |
| chb05_14.edf | infomax | False | 1 | 1 | 4.986e-12 | 3.972e-10 | 0 | 9.053e-05 | 0.7747 | 0.7747 | [0] | [0] | True |
| chb07_10.edf | infomax | True | 1 | 1 | 0 | 0 | 0 | 5.056e-12 | 1 | 1 | [] | [] | True |
| chb08_15.edf | infomax | True | 1 | 1 | 0 | 0 | 0 | 2.068e-05 | 0.9183 | 0.9183 | [6, 14] | [6, 14] | True |
| chb13_03.edf | infomax | True | 1 | 1 | 0 | 0 | 0 | 0.0002235 | 0.7883 | 0.7883 | [0] | [0] | True |
| chb17b_59.edf | infomax | False | 1 | 1 | 1.275e-08 | 4.014e-08 | 2.336e-06 | 0.3962 | 0.6214 | 0.6214 | [0] | [0] | True |
| chb22_10.edf | infomax | False | 1 | 1 | 1.396e-12 | 8.99e-11 | 0 | 4.251e-06 | 0.2972 | 0.2972 | [0, 1] | [0, 1] | True |

## Environments

**laptop**: Lenovo-aty, Intel64 Family 6 Model 170 Stepping 4, GenuineIntel, threads unlimited, numpy 2.5.3, scipy 1.16.1, mne 1.13.2, gedai 0.60, jamica 0.3.0, scikit-learn 1.7.2, torch 2.8.0+cpu

thread pools: openblas 14 (Haswell); openblas 14 (Haswell)

**lab_haswell**: DESKTOP-A0GHGPB, Intel64 Family 6 Model 85 Stepping 4, GenuineIntel, threads unlimited, numpy 2.5.3, scipy 1.16.1, mne 1.13.2, gedai 0.60, jamica 0.3.0, scikit-learn 1.7.2, torch 2.8.0

thread pools: openblas 24 (Haswell); openblas 24 (Haswell)
