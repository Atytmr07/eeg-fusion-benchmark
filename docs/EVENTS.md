# Event-based metrics: definition fixed before the runs

Fixed on 4 October 2026, before any multiverse run, on the advisor's decision. Not to be
changed after seeing results; any later deviation is reported as such.

## Role

Window-level macro F1 and AUPRC remain the **primary** metrics. Seizure-level
sensitivity and false alarms per hour are **secondary, clinical** metrics.

- **CHB-MIT:** seizure-level sensitivity and false alarms per hour, for every pipeline,
  seed set and model.
- **Siena:** seizure-level sensitivity only. Its recordings are cut around seizures, so
  a false alarm rate per hour would not represent clinical use; it is not part of any
  comparison and this is stated as a limitation.

## What is evaluated

The window-level results use the subsampled test windows (all ictal windows and four
non-ictal windows per ictal window). Event metrics instead need the model's output on
**every** 10 s window of every recording of the held-out person:

1. **Recordings:** all recordings of the person held out in the LOSO fold that contain
   the 18-channel montage (the same 683 recordings the corpus is built from; for chb01
   and chb21 together, both cases).
2. **Windows:** 10 s, non-overlapping, from the start of each recording (the windowing of
   the corpus, `src/chbmit.py: plan_windows`, stride 10 s); an incomplete last window
   is dropped.
3. **Preprocessing:** the pipeline's own steps on the continuous recording (band-pass,
   notch, and for P6 the same per-recording decomposition, recomputed with the same
   seed), then the pipeline's window normalisation. P4's artefact rejection is a
   training step only; every window is scored at evaluation.
4. **Models:** the weights saved for that fold (`results_v2/chbmit/<run>/models/`);
   score fusion with that fold's blend weight; the classical baselines refitted on the
   fold's training persons (deterministic).
5. **Decision:** a window is positive if its predicted ictal probability is above 0.5,
   the same rule as the window-level metrics. No smoothing or other post-processing
   besides the event scoring below.

## Event scoring

SzCORE event-based scoring (Dan et al. 2024), as implemented in its reference library
`timescoring` 0.0.7 (`timescoring.scoring.EventScoring`), with the library's default
parameters:

| Parameter | Value |
|---|---|
| tolerance before a seizure's onset | 30 s |
| tolerance after a seizure's end | 60 s |
| minimum overlap for a detection | any overlap |
| events separated by less than | 90 s are merged |
| events longer than | 5 min are split |

Reference events are the annotated seizures (parsed by `src/chbmit.py: parse_summary`).
The hypothesis is the sequence of positive windows of a recording.

## Metrics

- **Seizure-level sensitivity:** detected reference seizures / reference seizures, per
  person and pooled over persons.
- **False alarms per hour:** false-positive events / recording hours, per person, on the
  true (unsubsampled) recording duration.
- Event precision and event F1 are reported alongside, not used in comparisons.

## Statistics

Per-person values (23 persons) are compared between models and pipelines with the same
corrected paired framework as the window-level metrics (`src/stats.py`, test/train ratio
1/22), as secondary results.

## Implementation status

The runs save every fold's model weights (`chbmit_run`, `models/fold<k>_<model>.pt`),
which is all the evaluation needs from training. The evaluation itself (`src/events.py`)
runs afterwards on the saved models and is not part of the training queue.

## Siena and the cross-dataset experiment: implementation notes

Added on 4 October 2026. The definition above is unchanged; this section only says how
`src/events.py` applies it to Siena and to the CHB-MIT to Siena experiment.

```bash
python -m src.events --dataset siena --pipeline P1 --repeats 0 1 2   # Siena LOSO runs
python -m src.events --cross chbmit_to_siena_P0                      # cross-dataset run
python -m src.events --dataset siena --oracle                        # scoring check
```

**Siena LOSO (`--dataset siena`).** The procedure is the same as for CHB-MIT, with
these inputs:

- **Recordings, seizures and DECISIONS** come from `src/siena.py`:
  - Recordings are listed in RECORDS.
  - Seizures come from `parse_seizure_list`, with the DECISIONS recorded in the
    corpus info.
  - `seizures_in_record` converts seizure times to seconds from the first sample,
    relative to the EDF header start time.
  - If a DECISION excludes a seizure, scoring stops at its onset. The corpus drops
    those windows too.
- **Signal.** It is read with `read_bipolar` (18 bipolar channels, 256 Hz), with the
  pipeline's `apply_signal`, as `siena_corpus` builds the corpus.
- **Folds and baselines.** The 14 folds come from `siena_corpus.build_corpus` and
  `leave_one_subject_out`. Each fold is checked against the person that the run's
  `perfold.csv` tested.
- **Output** goes to `results_v2/siena/<run>/events/`.

**What is reported for Siena.** Seizure-level sensitivity, pooled and per person, is
the main column of `summary.csv`. False alarms per hour are still computed but kept
in separate columns whose header says "not representative for Siena, not compared".
`meta.json` states the same, and no comparison uses them.

**Cross-dataset (`--cross <tag>`).** This mode reads what `src/cross_dataset.py`
saved under `results_v2/cross/<tag>/`:

- the models of its single CHB-MIT training (`models/<model>.pt`),
- the score fusion weight (`score_blend_w` in `meta.json`).

It applies them to every recording of every Siena patient. The classical baselines
are refitted as in `cross_dataset`: on the windows of the inner training and
validation split of all CHB-MIT persons, without P4's rejected windows. This needs
the CHB-MIT corpus; `--skip-baselines` exists for tests only.

The output (`results_v2/cross/<tag>/events/`) gives seizure-level sensitivity per
Siena patient, with the same false-alarm note.

**Checks (4 October 2026, on PN00, PN05 and PN11).**

- **Oracle.** With the window labels scored as predictions, all 9 seizures are
  detected and there are no false alarms (11.7 h).
- **Windows.** The windows scored here are identical to the corpus windows (maximum
  difference 0, P0 and P1).
- **Cross mode vs LOSO.** With the same models, cross mode reproduces the LOSO
  counts.
- **CHB-MIT.** On the same synthetic CHB-MIT inputs, the previous and the current
  `events.py` write byte-identical `perperson.csv` and `summary.csv` for P0, P1
  and P4.
