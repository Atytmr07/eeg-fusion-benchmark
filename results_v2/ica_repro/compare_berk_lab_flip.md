# P6 reproducibility: berk vs lab_flip

6 CHB-MIT recordings, cleaned signal of the first 10 minutes; PSD over the whole recording.

| method | recordings | identical | corr_min | corr_median | rms_diff_rel_median | rms_diff_rel_max | psd_change_db_median | psd_change_db_max | same_component_count |
|---|---|---|---|---|---|---|---|---|---|
| gedai | 6 | 0 | 0.4274 | 0.9808 | 0.2036 | 1.212 | 0.393 | 27.1 | 0 |
| infomax | 6 | 0 | 0.2085 | 1 | 1.136e-05 | 0.9448 | 8.059e-05 | 29.42 | True |

## Per recording

| record | method | identical | corr_min | corr_median | rms_diff_rel | max_diff_rel | psd_change_db_median | psd_change_db_max | power_kept_a | power_kept_b | removed_a | removed_b | same_count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chb01_09.edf | gedai | False | 0.4274 | 0.676 | 1.212 | 2.422 | 3.423 | 27.1 | 0.3741 | 0.9994 | nan | nan | nan |
| chb01_25.edf | gedai | False | 0.5385 | 0.6699 | 1.175 | 1.986 | 3.237 | 22.33 | 0.353 | 0.9993 | nan | nan | nan |
| chb02_08.edf | gedai | False | 1 | 1 | 7.285e-11 | 3.549e-09 | 6.264e-09 | 0.002391 | 0.9994 | 0.9994 | nan | nan | nan |
| chb02_16+.edf | gedai | False | 0.9615 | 0.9779 | 0.2146 | 0.2322 | 0.175 | 1.215 | 0.2307 | 0.2189 | nan | nan | nan |
| chb03_02.edf | gedai | False | 0.9728 | 0.9837 | 0.1926 | 0.5492 | 0.611 | 2.368 | 0.2708 | 0.2799 | nan | nan | nan |
| chb07_05.edf | gedai | False | 1 | 1 | 4.443e-10 | 7.807e-09 | 9.383e-09 | 0.000211 | 0.1288 | 0.1288 | nan | nan | nan |
| chb01_09.edf | infomax | False | 1 | 1 | 5.21e-09 | 2.311e-08 | 5.759e-07 | 0.09548 | 0.4223 | 0.4223 | [0, 2] | [0, 2] | True |
| chb01_25.edf | infomax | False | 1 | 1 | 2.271e-05 | 2.485e-05 | 0.0001606 | 1.511 | 0.5321 | 0.5321 | [0, 2, 12] | [0, 2, 12] | True |
| chb02_08.edf | infomax | False | 1 | 1 | 3.158e-10 | 7.032e-09 | 3.034e-08 | 0.006531 | 0.6935 | 0.6935 | [1, 2, 13] | [1, 2, 13] | True |
| chb02_16+.edf | infomax | False | 0.2085 | 0.7566 | 0.9448 | 0.6039 | 1.435 | 29.42 | 0.5577 | 1 | [0, 5, 8] | [] | False |
| chb03_02.edf | infomax | False | 0.8348 | 0.999 | 0.2121 | 0.3818 | 0.05994 | 10.99 | 0.5319 | 0.3745 | [0] | [0, 1] | False |
| chb07_05.edf | infomax | False | 1 | 1 | 7.109e-11 | 2.832e-09 | 1.387e-09 | 0.0004913 | 1 | 1 | [] | [] | True |

## Environments

**berk**: 192.168.1.5, arm, threads unlimited, numpy 2.5.3, scipy 1.16.1, mne 1.13.2, gedai 0.60, jamica 0.3.0, scikit-learn 1.7.2, torch 2.8.0

thread pools: 

**lab_flip**: DESKTOP-A0GHGPB, Intel64 Family 6 Model 85 Stepping 4, GenuineIntel, threads unlimited, numpy 2.5.3, scipy 1.16.1, mne 1.13.2, gedai 0.60, jamica 0.3.0, scikit-learn 1.7.2, torch 2.8.0

thread pools: openblas 24 (SkylakeX); openblas 24 (SkylakeX)
