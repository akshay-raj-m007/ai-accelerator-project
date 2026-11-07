#!/usr/bin/env bash
# Plan A: all-INT8 (per-channel, per-tensor) with and without faults in the float32 scale metadata.
# Decision runs use HELD-OUT VALIDATION images (--split val --val-skip 2048: 2,952 images), not the test set.
#
#   base    E4M11 / all-INT8 pc / all-INT8 pt, scales fault-free, bit-uniform sampling, k = 256 1024 4096
#   scale   same INT layouts, scale words share the memory (physical view)
#   only    every strike hits a scale word: damage per scale hit, by bit group
#   groups  weight bit-group breakdown (exponent / sign / mantissa) at k = 1024, scales fault-free
#   compare writes scale_faults_report.md
#
# usage:  export FR_ROOT=$HOME/fr_runs
#         nohup bash scripts/run_scale_faults.sh > $FR_ROOT/scale.out 2>&1 &
#         bash scripts/run_scale_faults.sh only            # one stage
set -euo pipefail
: "${FR_ROOT:?set FR_ROOT first}"
RUN="${RUN:-scratch}"
CK="${CK:-$FR_ROOT/checkpoints/$RUN/best.pth}"
RES="$FR_ROOT/results/$RUN"
OUT="$RES/mixed_faults_scale"
cd "$(dirname "$0")/.."
STAGES="${*:-base scale only groups compare}"
TRIALS="${TRIALS:-100}"
has() { [[ " $STAGES " == *" $1 "* ]]; }
common="--checkpoint $CK --run-name $RUN --split val --val-skip 2048 --samplings bits"

if has base; then
  python -u scripts/mixed_fault_campaign.py $common --out-dir "$OUT/off" --layouts e4m11 all:int8_pc all:int8_pt \
      --groups any --ks 256 1024 4096 --trials "$TRIALS" --scale-faults off
fi
if has scale; then
  python -u scripts/mixed_fault_campaign.py $common --out-dir "$OUT/on" --layouts all:int8_pc all:int8_pt \
      --groups any --ks 256 1024 4096 --trials "$TRIALS" --scale-faults on
fi
if has only; then
  python -u scripts/mixed_fault_campaign.py $common --out-dir "$OUT/only" --layouts all:int8_pc all:int8_pt \
      --groups any exponent sign mantissa_msb mantissa_lsb --ks 1 --trials "$TRIALS" --scale-faults only
  python -u scripts/mixed_fault_campaign.py $common --out-dir "$OUT/only" --layouts all:int8_pc all:int8_pt \
      --groups any --ks 8 --trials "$TRIALS" --scale-faults only
fi
if has groups; then
  python -u scripts/mixed_fault_campaign.py $common --out-dir "$OUT/groups" --layouts e4m11 all:int8_pc all:int8_pt \
      --groups exponent sign mantissa_msb mantissa_lsb --ks 1024 --trials 50 --scale-faults off
fi
if has compare; then
  python -u scripts/compare_scale_faults.py --root "$OUT" --out "$RES/scale_faults_report.md"
fi
echo "== done =="
