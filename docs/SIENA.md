# Siena Scalp EEG

The third corpus. Bonn is the controlled proof of concept, CHB-MIT the primary
subject-independent evaluation, and Siena an external check: a different hospital,
amplifier and montage, and adults instead of children. This document covers getting
the data into the CHB-MIT form; no model has been trained on it yet.

## 1. Source and license

| Property | Value |
|---|---|
| Source | PhysioNet, `siena-scalp-eeg` v1.0.0, https://physionet.org/content/siena-scalp-eeg/1.0.0/ (mirror used for downloads: https://physionet-open.s3.amazonaws.com/siena-scalp-eeg/1.0.0/) |
| License | Creative Commons Attribution 4.0 International |
| Cite | Detti, Vatti, Zabalo Manrique de Lara (2020), *EEG synchronization analysis for seizure prediction: a study on data of noninvasive recordings*, Processes 8(7):846; Detti (2020), PhysioNet, doi:10.13026/5d4a-j060; Goldberger et al. (2000), Circulation 101(23) |
| Patients | 14 adults (PN00, PN01, PN03, PN05, PN06, PN07, PN09, PN10, PN11, PN12, PN13, PN14, PN16, PN17), one recording session each |
| Recordings | 41 EDF files, 128 hours, 20.3 GB; every recording contains at least one seizure |
| Seizures | 47, 47.7 minutes in total, median 63 s (PN10's ten seizures last 5 to 69 s) |
| Sampling rate | 512 Hz |
| Montage | referential 10-20 with older names (T3, T4, T5, T6), 29 EEG electrodes (19 in PN10's seizure list), plus EKG and other signals |
| Annotations | `PNxx/Seizures-list-PNxx.txt`: wall-clock times (hh.mm.ss) of each recording's start and end and of each seizure's start and end |

## 2. Conversion to the CHB-MIT form (`src/siena.py`)

| Step | Setting |
|---|---|
| Channels | the 18 bipolar channels of `src/chbmit.py: TARGET_CHANNELS`, same names and order, derived by subtraction: FP1-F7 = Fp1 - F7, T7-P7 = T3 - T5, P8-O2 = T6 - O2, and so on (T7 = T3, T8 = T4, P7 = T5, P8 = T6) |
| Electrodes needed | 19: Fp1, Fp2, F3, F4, F7, F8, Fz, C3, C4, Cz, P3, P4, Pz, O1, O2, T3, T4, T5, T6; present in all 41 recordings |
| Label matching | case-insensitive, `EEG ` prefix removed (`EEG Fp2` and `EEG FP2`, `EEG Cz` and `EEG CZ` both occur) |
| Unit | microvolts; the physical dimension of every needed electrode is `uV` in all 41 headers, checked on every read |
| Resampling | 512 to 256 Hz with `scipy.signal.resample_poly(x, 1, 2, axis=-1)` on the continuous bipolar recording, before windowing |
| Preprocessing hook | `signal_fn(x, fs, key)` runs on the continuous 256 Hz bipolar signal, as in CHB-MIT, so the P0-P6 pipelines of `src/preprocess.py` apply unchanged |
| Memory | only the 19 needed electrodes are converted to float; the longest recording (PN14-3, 11.7 hours, 2.1 GB) stays within a few GB |

Windowing, labelling, subsampling (4 non-ictal per ictal window, within each subject,
seed 20260727), the leakage checks and the corpus validation are CHB-MIT's own
functions (`src/siena_corpus.py`). Each subject is its own person group: 14 LOSO folds.

## 3. Header audit

`python -m src.siena_download --audit` reads only the EDF headers (256 bytes plus 256
per signal, about 0.5 MB) and compares them with the seizure lists. Result for all 41
recordings:

| Check | Result |
|---|---|
| File size equals the size computed from the header | 41/41 |
| Sampling rate | 512 Hz in every recording and every signal |
| All 19 electrodes present | 41/41 |
| Electrode unit | `uV` everywhere |
| O1 label | `EEG O1` in every header (the lists say "Channel 5: 1") |
| List start time equals the EDF header start time | 39/41. PN14-3: list 16.17.45, header 19.17.45. PN05-3: list 06.01.23, header 06.01.13 |
| List end time equals header start + duration | 22/41. In 18 of the other 19 the list end is exactly 20 s earlier; in PN05-3 it is 10 s earlier |

**Which start time is used.** Seizure times are converted relative to the EDF header
start time, modulo 24 hours, for every recording. The header start time is by
definition the time of the first sample, and it agrees with the list for 39
recordings. For PN14-3 the header is right: its start plus the recording duration
gives the listed end time exactly, while the listed start would make the recording
three hours longer than the file. PN05-3 is not settled (Section 5).

**The 20 s end offset.** In 18 recordings the file runs exactly 20 s past the listed
registration end. The start times agree, so seizure times are unaffected; the most
likely reading is that the listed end was noted before the recording was stopped.

## 4. Problems in the seizure lists and how they are handled

All fixes are in one dictionary, `src/siena.py: CORRECTIONS`, each with its reason.
The parser is strict: a time it cannot read unambiguously raises an error instead of
being guessed.

| Problem | Where | Handling |
|---|---|---|
| Mixed time separators: `19.58.36`, `21:51:02`, `16:13.23` | several lists | `.` and `:` both accepted, also mixed |
| Space inside a time: `1 6.49.25` | PN10, seizure 7 | text replacement to `16.49.25` (matches the next two blocks and the header) |
| No space after the colon: `Registration start time:14.18.30` | PN03, PN05, PN06, PN10, PN14 | accepted |
| Misspelt file names: `PN01.edf`, `PNO6-1.edf`, `PNO6-2.edf`, `PNO6-4.edf` (letter O), `PN11-.edf` | PN01, PN06, PN11 | mapped to the names in RECORDS |
| File name and registration times before "Seizure n 1" | PN01 | sequential parser: the context applies to both seizures |
| No registration times for seizure 2 | PN12 | taken from seizure 1 of the same file; only the blocks listed in `CORRECTIONS["inherited_registration"]` may do this |
| Recordings past midnight | PN01, PN03, PN06-2, PN07, PN14-3, PN16-2 | times taken modulo 24 hours |
| Channel 5 named `1` | all lists | channels are taken from the EDF headers (`EEG O1`), never from the lists |
| Wrong registration start time | PN14-3, PN05-3 | the EDF header start time is used (Section 3) |
| Two times in one field | PN10 seizures 2 and 3 | resolved by `DECISIONS` (Section 5); any other multi-time field raises |

## 5. Decisions for the advisor

Both options of each are implemented; the active choice is the `DECISIONS` constant
at the top of `src/siena.py`, part of the corpus cache name, and recorded in the
corpus info.

| Key | Problem | Default | Alternative |
|---|---|---|---|
| `PN00_seizure3` | PN00-3: end `19.29.29`, but the recording ends at 18.57.13, a 61-minute seizure past the end of the file | `typo`: 18.29.29, a 60 s seizure (the patient's other four last 54 to 74 s) | `exclude`: the seizure and every window from its onset to the end of the recording are dropped |
| `PN10_seizure3_onset` | PN10-3: `15.43.53 (CLINICAL ONSET); 15.43.59 (ELECTRIC ONSET)` | `electric`: 15.43.59, consistent with CHB-MIT's electrographic annotations | `clinical`: 15.43.53 |
| `PN10_seizure2_end` | PN10-2: end `11.41.04 opure 11.40.43` (Italian "oppure", "or") | `first`: 11.41.04 | `second`: 11.40.43 |

Further open points found during the work:

- **PN05-3 start time.** The list says 06.01.23 and the header 06.01.13. The list's
  value fits the 20 s end-offset pattern of the other recordings; the header's does
  not. The header is used, as for every recording. The seizure lasts 30 s, so the
  10 s can move its label by one window.
- **PN10 seizure 6** gives only a clinical onset (`15.18.26 (CLINICAL ONSET)`), so the
  `electric` choice for seizure 3 cannot be applied consistently to all of PN10.
- **PN10's electrodes.** Its seizure list names 19 EEG channels and `subject_info.csv`
  20, but its EDF headers carry the same 29 EEG labels as the other patients. Only the
  19 standard electrodes are used, so this does not affect the corpus.

## 6. Running

```bash
python -m src.siena_download --audit                 # header audit, about 0.5 MB
python -m src.siena_download --subjects PN00         # one subject, about 400 MB
python -m src.siena_download                         # all 14 subjects, 20.3 GB
python -m src.verify_siena                           # integrity of what is on disk
python -m src.siena_corpus --check                   # build the cache, counts, leakage checks
```

Downloads resume where they stopped, skip complete files, and check every EDF against
`SHA256SUMS.txt`. Raw data goes to `data/siena/PNxx/`, the corpus cache to
`data/siena/_cache/`; neither is committed.

From Python, the corpus has the same interface as CHB-MIT's:

```python
from src.siena_corpus import build_corpus
from src.preprocess import PIPELINES

d = build_corpus()                                    # P0
p = PIPELINES["P1"]
d1 = build_corpus(signal_fn=p.apply_signal, signal_tag=p.signal_tag)
```

## 7. Test on PN00

Only PN00 (5 recordings, 400 MB) was downloaded for the test; the full corpus is to be
downloaded on the lab machine.

| Check | Result |
|---|---|
| Download | 5/5 recordings, SHA-256 matches `SHA256SUMS.txt` |
| `verify_siena` | 5/5 intact, 3.24 hours, 5 seizures, 325 s of seizure, no problems |
| Corpus (default decisions) | 185 windows, 37 ictal (boundary windows included), 5 seizures, 18 x 2560 |
| Bipolar derivation | identical (max difference 0.0) to an independent path through `src/chbmit.py: read_edf` on all signals |
| Seizure timing | in every recording the seizure's windows have a higher mean line length than the recording's median window (1.35 to 6.4 times) |
| Amplitude | median window standard deviation 27 uV ictal, 14 uV non-ictal |
| `PN00_seizure3 = exclude` | 150 windows, 30 ictal, 4 seizures; 174 windows from the seizure's onset to the end of PN00-3 dropped |
| Preprocessing hook (P1) | exactly the same windows as P0, different signal |
| Leakage negative control | a random window split is caught (same person and split seizures) |

PN00's seizure 3 lasts 60 s when its end is read as 18.29.29, so the patient's total
is 325 s rather than the 326 s expected from a 61 s reading. LOSO cannot be checked
on one subject; it runs once two or more subjects are present.
