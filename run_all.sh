#!/usr/bin/env bash
# Bonn experiment runs, in parallel.
# 14 cores; 2 torch threads per run -> 6 runs x 2 = 12, leaving room for the system.
set -u
PY=${PY:-python}
# The thread count changes the results and is part of the run hash (see
# docs/PIPELINE.md). Runs with a different count are not comparable.
THREADS=${THREADS:-2}
cd "$(dirname "$0")"
mkdir -p logs

launch () {
  local task=$1 norm=$2 tag=$3
  OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS     "$PY" -m src.run_benchmark --task "$task" --norm "$norm" --threads "$THREADS"     > "logs/$tag.log" 2>&1 &
  echo "started $tag (pid $!)"
}

# --- Main runs -----------------------------------------------------------------
launch T1_3class              N2_z_zspec       main_T1_N2   # main comparison
launch T2_intracranial_binary N2_z_zspec       main_T2_N2   # no electrode confound
launch T1_3class              N1_z_rawspec     abl_T1_N1    # normalisation contrast
launch T3_classic_binary      N2_z_zspec       ref_T3_N2    # classic binary setup (reference)

# --- Further ablations ---------------------------------------------------------
# Run after the ones above; sharing the CPU slows all of them down.
# launch T1_3class            N0_raw_raw       abl_T1_N0
# launch T1_3class            N3_z_zspec_specz abl_T1_N3

echo "=== $(date '+%H:%M:%S') runs started, waiting ==="
wait
echo "=== $(date '+%H:%M:%S') all done ==="
