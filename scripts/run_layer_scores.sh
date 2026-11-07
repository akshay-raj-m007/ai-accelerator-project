#!/usr/bin/env bash
# Plan B: per-layer scores S0 / S1 / S2 / S2n on the all-E4M11 model (held-out data only). About 1-1.5 h on an RTX 3050.
# usage: export FR_ROOT=$HOME/fr_runs; bash scripts/run_layer_scores.sh
set -euo pipefail
: "${FR_ROOT:?set FR_ROOT first}"
RUN="${RUN:-scratch}"
CK="${CK:-$FR_ROOT/checkpoints/$RUN/best.pth}"
RES="$FR_ROOT/results/$RUN"
cd "$(dirname "$0")/.."
python -u scripts/layer_fault_scores.py --checkpoint "$CK" --run-name "$RUN" \
    --fisher "$RES/sensitivity_heldout/fisher_true.pt" --groups any exponent --k "${K:-16}" --trials "${TRIALS:-100}"
echo "== done =="; echo "report: $RES/layer_scores/layer_scores.md"
