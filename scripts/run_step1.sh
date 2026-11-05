#!/usr/bin/env bash
# Full step-1 pipeline for the GPU machine.
#   export FR_ROOT=$HOME/fr_data      # data/, checkpoints/, results/ live here
#   cp resnet18_cifar10_checkpoint.pth $FR_ROOT/checkpoints/
#   bash scripts/run_step1.sh                   # fine-tune + scratch + reports
#   SKIP_SCRATCH=1 bash scripts/run_step1.sh    # fine-tune only
#   RESUME=1 bash scripts/run_step1.sh          # continue interrupted runs
# Extra args (e.g. --normalize none) go to every training command via TRAIN_ARGS.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${FR_ROOT:?set FR_ROOT first}"
RESUME_FLAG=""; [ "${RESUME:-0}" = "1" ] && RESUME_FLAG="--resume"
TRAIN_ARGS="${TRAIN_ARGS:-}"

run_one () {  # $1 = run name, $2 = yaml
  local name="$1" cfg="$2"
  if [ ! -f "$FR_ROOT/checkpoints/$name/last.pth" ] || [ -n "$RESUME_FLAG" ]; then
    python scripts/train_baseline.py --config "$cfg" --run-name "$name" $RESUME_FLAG $TRAIN_ARGS
  fi
  python scripts/eval_fp32.py --checkpoint "$FR_ROOT/checkpoints/$name/best.pth" --run-name "$name"
  python scripts/quantize_eval.py --checkpoint "$FR_ROOT/checkpoints/$name/best.pth" \
      --run-name "$name" --compare-modes
}

run_one finetune configs/finetune.yaml
[ "${SKIP_SCRATCH:-0}" = "1" ] || run_one scratch configs/scratch.yaml
echo "Done. Reports: $FR_ROOT/results/{finetune,scratch}/step1_report.md"
