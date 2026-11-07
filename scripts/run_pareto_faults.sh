#!/usr/bin/env bash
# Experiment 4 (main comparison): accuracy under random bit flips at equal storage, with vs without Fisher.
#
# Layouts (selections from the HELD-OUT-Fisher sweep, mixed_heldout/selections.json):
#   references   e4m11, all:int4_pt, all:fp4_scaled
#   per arm (int4_pt, fp4_scaled) and budget T8 / T5 bits per weight:
#       lambda:0  Fisher only      product  sum F*dw^2      -> "with Fisher"
#       lambda:1  error only                                  -> "without Fisher"
#       random:0..2  three random feasible sets
# Fault load: k = 1024 weights struck per trial, group `any`, sampling `weights` (paper wording). k was fixed from the
# all-E4M11 reference curve only (docs/CODE_SUMMARY.md rule L7: median 88.99 %, p10 66.7 %; k=256 barely moves the
# median, k=4096 collapses it). The struck weights are identical for every layout (common random numbers).
#
# usage:  export FR_ROOT=$HOME/fr_runs
#         bash scripts/run_pareto_faults.sh pilot      # 100 trials, about 2 h
#         bash scripts/run_pareto_faults.sh full       # tops up to 500 trials (resumable), about 8 h more
set -euo pipefail
: "${FR_ROOT:?set FR_ROOT first}"
RUN="${RUN:-scratch}"
CK="${CK:-$FR_ROOT/checkpoints/$RUN/best.pth}"
RES="$FR_ROOT/results/$RUN"
cd "$(dirname "$0")/.."
STAGE="${1:-pilot}"
case "$STAGE" in pilot) TRIALS=100;; full) TRIALS=500;; *) echo "stage: pilot | full"; exit 1;; esac
K="${K:-1024}"

LAYOUTS=(e4m11 all:int4_pt all:fp4_scaled)
for arm in int4_pt fp4_scaled; do
  for t in 8 5; do
    for m in lambda:0 product lambda:1 random:0 random:1 random:2; do LAYOUTS+=("$arm:T$t:$m"); done
  done
done

python -u scripts/mixed_fault_campaign.py --checkpoint "$CK" --run-name "$RUN" \
    --sweep-dir "$RES/mixed_heldout" --out-dir "$RES/mixed_faults_pareto" \
    --layouts "${LAYOUTS[@]}" --groups any --ks "$K" --trials "$TRIALS" --samplings weights
