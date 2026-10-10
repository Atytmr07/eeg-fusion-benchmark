# P6 reproducibility: laptop_flip vs lab_flip

6 CHB-MIT recordings, cleaned signal of the first 10 minutes; PSD over the whole recording.

| method | recordings | identical | corr_min | corr_median | rms_diff_rel_median | rms_diff_rel_max | psd_change_db_median | psd_change_db_max | same_component_count |
|---|---|---|---|---|---|---|---|---|---|
| gedai | 6 | 0 | 0.392 | 0.8821 | 0.5646 | 1.27 | 1.5 | 27.63 | 0 |
| infomax | 6 | 0 | 0.5804 | 1 | 2.534e-07 | 0.7197 | 7.102e-06 | 21.8 | True |

## Per recording

| record | method | identical | corr_min | corr_median | rms_diff_rel | max_diff_rel | psd_change_db_median | psd_change_db_max | power_kept_a | power_kept_b | removed_a | removed_b | same_count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chb01_09.edf | gedai | False | 0.392 | 0.6468 | 1.27 | 2.793 | 4.122 | 27.63 | 0.3466 | 0.9994 | nan | nan | nan |
| chb01_25.edf | gedai | False | 0.7143 | 0.8075 | 0.8262 | 1.138 | 1.286 | 22.07 | 0.5701 | 0.9993 | nan | nan | nan |
| chb02_08.edf | gedai | False | 0.4276 | 0.7888 | 0.9929 | 2.392 | 2.705 | 20.27 | 0.4457 | 0.9994 | nan | nan | nan |
| chb02_16+.edf | gedai | False | 0.9285 | 0.9566 | 0.2954 | 0.2324 | 0.3098 | 2.148 | 0.2419 | 0.2189 | nan | nan | nan |
| chb03_02.edf | gedai | False | 0.9366 | 0.9608 | 0.303 | 0.5631 | 1.715 | 3.885 | 0.259 | 0.2799 | nan | nan | nan |
| chb07_05.edf | gedai | False | 1 | 1 | 4.4e-10 | 7.807e-09 | 1.628e-09 | 9.473e-05 | 0.1288 | 0.1288 | nan | nan | nan |
| chb01_09.edf | infomax | False | 1 | 1 | 2.281e-09 | 5.779e-09 | 2.917e-07 | 0.02975 | 0.4223 | 0.4223 | [0, 2] | [0, 2] | True |
| chb01_25.edf | infomax | False | 1 | 1 | 5.045e-07 | 5.526e-07 | 1.391e-05 | 1.128 | 0.5321 | 0.5321 | [0, 2, 12] | [0, 2, 12] | True |
| chb02_08.edf | infomax | False | 1 | 1 | 5.111e-11 | 1.758e-09 | 1.14e-11 | 0.0009828 | 0.6935 | 0.6935 | [1, 2, 13] | [1, 2, 13] | True |
| chb02_16+.edf | infomax | False | 0.5804 | 0.971 | 0.7197 | 0.5052 | 0.6276 | 21.8 | 0.6464 | 1 | [0] | [] | False |
| chb03_02.edf | infomax | False | 0.8352 | 0.999 | 0.2121 | 0.3814 | 0.05918 | 10.98 | 0.5319 | 0.3745 | [0] | [0, 1] | False |
| chb07_05.edf | infomax | False | 1 | 1 | 7.105e-11 | 2.832e-09 | 6.072e-12 | 0.0002006 | 1 | 1 | [] | [] | True |

## Environments

**laptop_flip**: Lenovo-aty, Intel64 Family 6 Model 170 Stepping 4, GenuineIntel, threads unlimited, numpy 2.5.3, scipy 1.16.1, mne 1.13.2, gedai 0.60, jamica 0.3.0, scikit-learn 1.7.2, torch 2.8.0+cpu

thread pools: openblas 14 (Haswell); openblas 14 (Haswell)

**lab_flip**: DESKTOP-A0GHGPB, Intel64 Family 6 Model 85 Stepping 4, GenuineIntel, threads unlimited, numpy 2.5.3, scipy 1.16.1, mne 1.13.2, gedai 0.60, jamica 0.3.0, scikit-learn 1.7.2, torch 2.8.0

thread pools: openblas 24 (SkylakeX); openblas 24 (SkylakeX)
