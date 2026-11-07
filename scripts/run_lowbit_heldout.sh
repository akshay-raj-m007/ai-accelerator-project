#!/usr/bin/env bash
# Experiments 1-3 of the LAMPQ + bit-fault study (see docs/EXPERIMENTS.md, docs/CODE_SUMMARY.md section 12).
#
#   exp 1  lower-bit formats (INT4 per-tensor / per-channel, FP4 scaled): creates an accuracy-vs-storage tradeoff
#   exp 2  Fisher from HELD-OUT validation images (first 2,048), validation sweep on the remaining 2,952
#   exp 3  heavier random bit-flip loads on all-E4M11 (and all-INT8 / all-INT4 for reference)
#
# usage:  export FR_ROOT=$HOME/fr_runs
#         nohup bash scripts/run_lowbit_heldout.sh > $FR_ROOT/lowbit.out 2>&1 &
#         bash scripts/run_lowbit_heldout.sh load          # one stage only: fisher | sweeps | load | compare
# Every stage is resumable: re-running skips finished work. Nothing here touches the test set except the
# `final`/`churn` phases (saved selections only) and the fault campaign (as before).
set -euo pipefail
: "${FR_ROOT:?set FR_ROOT first, e.g. export FR_ROOT=\$HOME/fr_runs}"
RUN="${RUN:-scratch}"
CK="${CK:-$FR_ROOT/checkpoints/$RUN/best.pth}"
RES="$FR_ROOT/results/$RUN"
cd "$(dirname "$0")/.."
PY="python -u"
TARGETS="14 12 10 8 6 5"
STAGES="${*:-fisher sweeps load compare}"

has() { [[ " $STAGES " == *" $1 "* ]]; }

if has fisher; then
  echo "== exp 2: held-out Fisher (first 2048 validation images) =="
  $PY scripts/fisher_sensitivity.py --checkpoint "$CK" --run-name "$RUN" --source val --calib-size 2048 --kinds true
fi

if has sweeps; then
  echo "== exp 1 + 2: held-out Fisher, 8-bit and 4-bit arms, budgets $TARGETS =="
  held="--checkpoint $CK --run-name $RUN --fisher-dir sensitivity_heldout --out-name mixed_heldout --val-skip 2048 \
        --arms int8_pc int8_pt fp8_scaled int4_pc int4_pt fp4_scaled --targets $TARGETS"
  for phase in sweep final churn; do $PY scripts/mixed_precision.py $held --phase $phase; done

  echo "== exp 1: 4-bit arms with the original (training-image) Fisher, same budgets =="
  low="--checkpoint $CK --run-name $RUN --out-name mixed_lowbit --arms int4_pc int4_pt fp4_scaled --targets $TARGETS"
  for phase in sweep final churn; do $PY scripts/mixed_precision.py $low --phase $phase; done
fi

if has load; then
  echo "== exp 3: fault-load calibration on the references (fixed k grid; no selections involved) =="
  $PY scripts/mixed_fault_campaign.py --checkpoint "$CK" --run-name "$RUN" \
      --layouts e4m11 all:int8_pc all:int4_pc --ks 16 64 256 1024 4096 --trials 40 \
      --out-dir "$RES/mixed_faults_load"
fi

if has compare; then
  echo "== selections: training-image Fisher vs held-out Fisher =="
  $PY scripts/compare_selections.py "$RES/mixed" "$RES/mixed_heldout" --out "$RES/selection_compare_8bit.md"
  $PY scripts/compare_selections.py "$RES/mixed_lowbit" "$RES/mixed_heldout" --out "$RES/selection_compare_4bit.md"
fi
echo "== done =="
