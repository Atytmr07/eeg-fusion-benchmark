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
