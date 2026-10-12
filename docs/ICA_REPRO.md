# Reproducibility of P6's artefact removal across machines and execution contexts

Investigated 10 October 2026 at the advisor's request, after `events.py`'s window check
found that the P6 signals recomputed on the lab PC differed from the cached signals the
models were trained on. Code: `src/ica_repro.py`. Data: `results_v2/ica_repro/`.

## Setup

| | Laptop (built the P6 caches) | Lab PC |
|---|---|---|
| CPU | Intel Core Ultra 7 155U (family 6, model 170), 14 logical CPUs | Intel Xeon (family 6, model 85), 2 sockets, 64 logical CPUs |
| Packages | numpy 2.5.3, scipy 1.16.1, mne 1.13.2, gedai 0.60, jamica 0.3.0, scikit-learn 1.7.2 | the same |
| BLAS | OpenBLAS, Haswell kernels, 14 threads | OpenBLAS, Haswell kernels, 24 threads |
| Seeds | per recording, `crc32(file name)` | the same |

Everything that a methods section would normally list (code, package versions, seeds,
parameters) was identical. Each recording is cleaned on its own (Section 3.6), so
recordings are independent.

## Tests and results

**1. Fixed sample, one process per machine** (8 CHB-MIT recordings drawn with a fixed
seed; `compare_laptop_lab.md`). AMICA and GEDAI were bit-identical on all 8. Infomax
was bit-identical on 3; on 5 the cleaned signals differed at rounding level (relative
RMS difference 1e-13 to 1e-8, correlation 1.000) with the same components removed. In
all 8 recordings GEDAI removed almost nothing (retained power 0.999), which is why it
showed no difference here (tests 3 and 4).

**2. BLAS kernel** (`compare_lab_lab_haswell.md`). Forcing OpenBLAS's Haswell kernels on
the lab PC (`OPENBLAS_CORETYPE=Haswell`) changed nothing: it already used them. The kernel
choice is not the cause.

**3. Whole corpus, from the per-recording logs** (`logs_cache_vs_lab*.csv`; the logs
written when the laptop built the caches against those written when the lab PC
recomputed the cleaning for the event metrics; 670 CHB-MIT recordings):

| Method | Different components removed | Retained power differs by > 0.001 | by > 0.1 | Largest difference |
|---|---|---|---|---|
| Extended Infomax | 17 (2.5 %) | 17 | 4 | 0.44 |
| GEDAI (conservative) | not applicable | 184 (27 %) | 49 (7 %) | 0.75 |
| AMICA | 0 | 1 | 0 | 0.003 |

GEDAI's threshold differed between the machines in 64 % of recordings and by more than a
factor of two in 154. The thresholds lie between 2e-15 and 3e-9, close to the numerical
precision of the covariances they are compared with.

**4. Recordings that differed, one process per machine** (6 recordings, Infomax and
GEDAI; `compare_laptop_flip_lab_flip.md`). Run alone on the lab PC, GEDAI gave the lab's
result, not the laptop's: for example chb01_09 kept 0.35 of its power on the laptop and
0.999 on the lab PC (correlation of the cleaned signals down to 0.39, PSD changes up to
27.6 dB). GEDAI is therefore deterministic on each machine but machine-dependent.

**5. Same machine, different execution context.** On the laptop, Infomax run alone gave
the same result twice in a row (`laptop_rep1`, `laptop_rep2`), but for 2 of 3 recordings
not the result found when the caches were built, where 6 recordings were processed in
parallel worker processes. Recomputing the whole corpus on the laptop, again in parallel
(`logs_cache_vs_laptop_rerun*.csv`, from the event metric run):

| Method | Different components removed | Retained power differs by > 0.001 |
|---|---|---|
| Extended Infomax | 8 of 670 (1.2 %) | 9 |
| GEDAI | 0 of 670 | 0 |
| AMICA (first folds) | 0 | 0; cleaned windows differ by at most 1.4 % of their amplitude |

**6. Sensitivity to a rounding-level perturbation** (`src/ica_sensitivity.py`,
`sensitivity.md`, `sensitivity.csv`). Each of 9 recordings (the 6 of test 4 and 3 that
agreed between machines) was cleaned, then cleaned again three times after multiplying
every sample by (1 + 1e-7 N(0, 1)), about the precision of the float32 signal. GEDAI
was also run on a full-rank montage (16 of the 18 channels, one channel of each closed
electrode loop dropped) to test the rank-deficiency hypothesis.

| Method | Perturbations that changed the result | Recordings affected | Largest change in retained power |
|---|---|---|---|
| Extended Infomax | 7 of 27 | 3 of 9 (chb02_16+, chb03_02, chb07_05) | 0.44 |
| GEDAI | 16 of 27 | 6 of 9, exactly the 6 that differed between machines | 0.76 |
| GEDAI, full-rank montage | 11 of 27 | 5 of 9 | 0.61 |
| AMICA (chb01_09 only; 14 min per cleaning) | 0 of 3 | 0 of 1 | same components, power kept equal to four decimals |

A change is a relative RMS difference above 1e-3 or a different set of removed
components; unchanged results differed by about 1e-7, the size of the perturbation.
Infomax's component decision changed only on recordings where it had also changed
between machines or runs. GEDAI's result moved between a few discrete solutions (for
chb02_08: 0.446 or 0.999 of the power kept); the three recordings that agreed between
machines did not change under any perturbation. The full-rank montage was not more
stable, so rank deficiency is not the cause.

**7. A third machine with a different architecture** (Apple Silicon, ARM, macOS;
the same package versions; run by H. B. Berber; `berk/`, `compare_berk_*.md`). On the 8
recordings of test 1 the Mac reproduced the laptop and the lab PC up to rounding
(relative RMS difference at most 9e-7, the same components). On the 6 recordings of
test 4 it often agreed with neither:

| Recording | GEDAI power kept: Mac / laptop / lab | Infomax components: Mac / laptop / lab |
|---|---|---|
| chb01_09 | 0.374 / 0.347 / 0.999 | [0, 2] on all three |
| chb01_25 | 0.353 / 0.570 / 0.999 | [0, 2, 12] on all three |
| chb02_08 | 0.999 / 0.446 / 0.999 | [1, 2, 13] on all three |
| chb02_16+ | 0.231 / 0.242 / 0.219 | [0, 5, 8] / [0] / [] |
| chb03_02 | 0.271 / 0.259 / 0.280 | [0] / [0] / [0, 1] |
| chb07_05 | 0.129 on all three | [] on all three |

Several of the Mac's results are ones the perturbation test found on the laptop (GEDAI
chb02_08 0.9994 and chb02_16+ 0.2307 to four decimals): a second machine produces the
same alternatives as a rounding-level change of the input.

## Conclusions

- **AMICA** reproduces across machines up to rounding: no component decision changed.
- **Extended Infomax** reproduces across machines and execution contexts except where a
  component lies near the rejection threshold (|z| = 3 against the frontal EOG proxies):
  there, rounding-level differences flip the decision. This affected 2.5 % of recordings
  between machines and 1.2 % between two runs on the same machine.
- **GEDAI** is deterministic on one machine but differs between machines in about a
  quarter of recordings, often by a large amount. Its noise threshold is estimated at the
  level of numerical precision. The rank deficiency of our bipolar montage is not the
  cause: a full-rank montage was as unstable (test 6).
- **Source.** The differences come from the methods' numerical sensitivity on particular
  recordings, not from one machine, library or setting: on those recordings a relative
  change of 1e-7 in the input changes the result as much as a change of machine does,
  three machines of two architectures gave up to three different results, and the
  recordings that are stable under perturbation agreed on all three machines (tests 6
  and 7). Machines differ in the order of floating-point operations (SIMD code paths,
  thread partitions), which changes results at the level of rounding; Infomax's |z| = 3
  rule and GEDAI's threshold turn that into a different cleaning when a recording lies
  near a decision boundary. No machine is "correct", and the instability cannot be
  removed by choosing a different machine or thread count.
- Fixing seeds and package versions is therefore not enough to reproduce ICA-based
  cleaning; the outcome also depends on rounding at decision thresholds, which differs
  between machines and, for Infomax, between execution contexts.

## Consequences for this study

- **Window-level results are unaffected.** Every P6 training and test window comes from
  one set of caches, built once on the laptop.
- **Event metrics** must clean every recording again. They were computed on the laptop:
  GEDAI's cleaning is reproduced exactly; Infomax's differs from the training-time
  cleaning in 8 of 670 recordings; AMICA's by at most 1.4 % of the signal amplitude.
  `events.py` reports this per fold (`window_check` in `events/meta.json`).
- **Published logs.** Per recording and method, the outcome of the training-time cleaning
  (`results_v2/qc/p6_recording_logs.csv`: components removed, GEDAI threshold, power
  kept; CHB-MIT and Siena) and of the event-time recleaning on the laptop
  (`results_v2/qc/p6_recording_logs_events.csv`, including 13 CHB-MIT recordings without
  selected windows that only the event metrics use). They differ only for Infomax on
  CHB-MIT: 8 recordings with different components and 2 more whose retained power
  differs by more than 0.001. A reproduction can be checked against these files without
  the cached signals.
- **Effect on the results** (`src/ica_repro_impact.py`, `results_v2/ica_repro/impact_events.md`).
  The same trained models were scored on CHB-MIT's full recordings twice: cleaned on the
  lab PC (`events_lab_FAIL/`) and on the laptop (the valid evaluation), three seed sets
  per pipeline, five operators:

  | Pipeline | Pooled sensitivity, mean / max absolute change | False alarms per hour, mean / max absolute change | Persons whose detected count changed (per operator, mean) | Kendall tau of the operator ranking by sensitivity |
  |---|---|---|---|---|
  | P6a Infomax | 0.003 / 0.005 | 0.012 / 0.037 | 0.6 | 0.95 in all three runs; the best operator changed in one run |
  | P6b GEDAI | 0.018 / 0.054 | 0.027 / 0.061 | 2.5 | 0.80, 0.60, 1.00; the best operator unchanged |
  | P6c AMICA | 0 / 0 | 0.001 / 0.003 | 0 | 1.00 |

  The ranking by false alarms per hour did not change in any run. AMICA's
  irreproducibility has no effect on the results; Infomax's flips the order of nearly
  tied operators; GEDAI's changes seizure-level sensitivity by up to 0.054 and the order
  of the operators by sensitivity, an effect of the order of the differences between
  operators. A cleaning step that does not reproduce across machines can therefore
  change a comparison of fusion operators by itself.
