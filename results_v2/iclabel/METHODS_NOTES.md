# ICLabel overlap check: methods notes for the technical report

Code: `src/iclabel_check.py` (environment `.venv-ica`, `requirements-ica.txt`).
Batch: `scripts/run_iclabel_siena.cmd`. Per patient output in `results_v2/iclabel/<patient>/`.

## Data

- Siena Scalp EEG, raw referential EDF, all EEG electrodes (29; PN10 has 20), 512 Hz.
  Old names mapped to 10-05 (T3 -> T7, T4 -> T8, T5 -> P7, T6 -> P8); standard_1005 positions.
- All recordings of each patient. Seizure times and excluded intervals as in the main
  pipeline (`src/siena.py`: `seizures_in_record`); excluded intervals count as neither
  seizure nor background.
- Whether P6 itself removed a component: `src/iclabel_p6_match.py` recomputes the P6a ICA
  (must reproduce the cache log, docs/ICA_REPRO.md) and reports, per ICLabel component,
  the share of its bipolar 0.5 to 40 Hz signal that P6 removes. Runs after the batch,
  alone on the laptop that built the P6 caches.
- CHB-MIT: not run yet. Bipolar montage; FT9-FT10 links the left and right chains, so an
  exact common average over the 18 lateral electrodes is possible where FT9-FT10 exists;
  FZ, CZ, PZ form a separate group and must be dropped. Per-recording check pending.

## Decomposition (ICLabel-conformant, separate from the P6 ICA)

| setting | value |
|---|---|
| notch | 50 and 100 Hz |
| band-pass | 1 to 100 Hz (ICLabel's training preprocessing) |
| reference | common average |
| algorithm | extended Infomax (MNE), max_iter 1000 |
| components | number of EEG electrodes in the file minus 1 (rank after average reference): 28 for PN00 (29 electrodes), 30 for PN12 and PN14 (31: P9 and P10 as well), 19 for PN10 (20) |
| fit | every 8th sample (`ICA_FIT_DECIM = 8`, as P6) |
| seed | CRC32 of the recording name |
| classifier | mne-icalabel 0.10.0 (torch backend); class = argmax of the 7 probabilities |

Note: the P6 ICA in the paper is a different decomposition (18 bipolar channels, 256 Hz,
0.5 to 40 Hz, 16 components). The results here therefore say "the P6 rule flagged this
component in this decomposition", not "P6 removed this component".

## The P6 ocular rule, applied as in the paper

- The four frontopolar bipolar derivations FP1-F7, FP1-F3, FP2-F4, FP2-F8 are formed from
  the raw referential signal (electrode difference), then band-passed 0.5 to 40 Hz with the
  P6 filter (`src/preprocess.py: bandpass`). They are added as EOG channels; they are not
  part of the ICA.
- MNE `find_bads_eog` per proxy channel, threshold |z| > 3 (`EOG_Z`); a component is
  flagged if any of the four proxies flags it. Same function and threshold as P6.

## Seizure-related activity per component

- `ictal_power_ratio`: mean power of the component time course during seizures divided by
  background (non-seizure, non-excluded) power.
- `ictal_gamma_ratio`: the same after a 30 to 100 Hz band-pass (4th-order Butterworth,
  zero phase).

## Patient selection (from subject_info.csv only, before any result)

| patient | group | rule |
|---|---|---|
| PN00 | right temporal IAS | most seizures per recording hour among 29-electrode patients |
| PN12 | left temporal IAS | same, among left-sided patients |
| PN14 | WIAS | the only WIAS patient |
| PN10 | frontal, bilateral FBTC | the only FBTC patient (20 electrodes) |
| PN17 | right temporal IAS, second | same rule among right-sided patients other than PN00 with at least 2 seizures |

## Files per patient

- `components.csv`: one row per component. Columns: record, component, iclabel (class),
  iclabel_p, p6_rule (flagged), p6_max_abs_z, explained_variance, p_brain ... p_other,
  ictal_power_ratio, ictal_gamma_ratio.
- `components_<record>.csv`: the same per recording (checkpoints).
- `summary.md`: overlap table, flagged components, class table, top ictal components.
- `topomaps_<record>.png`: all component maps with ICLabel class and probability; red
  title = flagged by the P6 rule.
- `meta.json`: settings and package versions.
