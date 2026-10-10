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

## Conclusions

- **AMICA** reproduces across machines up to rounding: no component decision changed.
- **Extended Infomax** reproduces across machines and execution contexts except where a
  component lies near the rejection threshold (|z| = 3 against the frontal EOG proxies):
  there, rounding-level differences flip the decision. This affected 2.5 % of recordings
  between machines and 1.2 % between two runs on the same machine.
- **GEDAI** is deterministic on one machine but differs between machines in about a
  quarter of recordings, often by a large amount. Its noise threshold is estimated at the
  level of numerical precision; a likely contributor, not yet tested, is that our bipolar
  montage is rank deficient (18 channels, rank 16, possibly one less after the mean removal),
  so the generalised eigenvalue problem it solves is ill-conditioned.
- Fixing seeds and package versions is therefore not enough to reproduce ICA-based
  cleaning; the outcome also depends on the machine (GEDAI) or on rounding at decision
  thresholds (Infomax).

## Consequences for this study

- **Window-level results are unaffected.** Every P6 training and test window comes from
  one set of caches, built once on the laptop.
- **Event metrics** must clean every recording again. They were computed on the laptop:
  GEDAI's cleaning is reproduced exactly; Infomax's differs from the training-time
  cleaning in 8 of 670 recordings; AMICA's by at most 1.4 % of the signal amplitude.
  `events.py` reports this per fold (`window_check` in `events/meta.json`).
- **Effect on the results.** The lab PC's event metrics for P6 (computed on the
  differently cleaned signals, kept as `events_lab_FAIL/`) will be compared with the
  laptop's, as a measure of how much this irreproducibility changes seizure-level
  sensitivity and the operators' ranking.
