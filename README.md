# Fault robustness of E4M11: step 1 (FP32 baseline + E4M11 PTQ)

Reproduction of the baseline of Haripriya & Tudu, *Fault Robustness of Custom Floating-Point and
Integer Formats* for **ResNet-18 / CIFAR-10**:

* (a) FP32 test accuracy and weight statistics (paper Table 3 style),
* (b) E4M11 post-training quantization: test accuracy and SQNR (paper Eq. 6).

Fault injection is **not** part of this step. `quantize_model` already returns the packed `uint16`
buffers that the fault-injection step will flip bits in.

See [docs/CODE_SUMMARY.md](docs/CODE_SUMMARY.md) for the line-by-line comparison of the old notebook with the
paper and the deviation table.

## Layout

```
src/fr/        config, data, model, e4m11, quantize, sqnr, evaluate, train, report, step1, faults,
               formats8, sensitivity, allocation, mixed, mixed_run
scripts/       train_baseline.py  eval_fp32.py  quantize_eval.py  inspect_checkpoint.py  run_step1.sh
configs/       finetune.yaml  scratch.yaml  paper_targets.yaml
tests/         pytest suite (CPU only, no dataset download)
docs/          CODE_SUMMARY.md
```

Key decisions (all justified in the code docstrings / CODE_SUMMARY):

* `e4m11.py` `mode="paper"` (default): Eq. 4 with mantissa carry + Algorithm 1 flush/clip; `mode="legacy"`
  reproduces the old notebook bit-for-bit (verified in tests against a literal copy of the notebook code).
* `mode="subnormal"` is an **exploratory, non-paper** variant (es=0 stores subnormals `m*2^-17` instead of flushing to
  zero). It exists because the paper's flush rule zeroes ~87 % of trained ResNet-18 weights (see CODE_SUMMARY/report).
  Reported next to `paper`/`legacy` via `--compare-modes`; never the default.
* **Fixed baseline for later steps: `--e4m11-mode subnormal` (bias 7)**. Accuracy 95.11 % = FP32; SQNR 69.27 dB.
  Reproduce: `python scripts/quantize_eval.py --checkpoint <best.pth> --run-name <run> --compare-modes`
  (the report lists paper, legacy and subnormal side by side). See docs/CODE_SUMMARY.md section 8.
* Default quantized scope is **Conv2d/Linear weights only** (`--quantize-scope convlinear`); `all` (also BN gamma/beta and
  the fc bias) reproduces the old notebook.
* Input normalization is never guessed. For `--init checkpoint` it is read from `checkpoint["config"]`; if it is not
  recorded, training/eval stops with a message and you must pass `--normalize none|cifar10` yourself
  (`scripts/inspect_checkpoint.py` prints the config). Checkpoints written by `train.py` record it themselves.
* 5,000 train images are held out for validation; model selection uses validation only; the test set is evaluated once at the end.

## Local smoke test (CPU, a few minutes)

```bash
pip install -r requirements.txt          # plus torch/torchvision
python -m pytest -q
python scripts/train_baseline.py --config configs/scratch.yaml --run-name smoke --root ./_smoke \
    --epochs 1 --train-subset 512 --val-size 200 --test-subset 1000 --num-workers 0 --batch-size 64 --no-amp
python scripts/quantize_eval.py --checkpoint _smoke/checkpoints/smoke/last.pth --run-name smoke --root ./_smoke \
    --compare-modes --test-subset 1000 --num-workers 0
```

## GPU machine (Linux)

### 1. Clone, venv, PyTorch for your CUDA version

```bash
git clone https://github.com/akshay-raj-m007/ai-accelerator-project.git
cd ai-accelerator-project
git checkout step1-baseline-e4m11
python3 -m venv .venv && source .venv/bin/activate

nvidia-smi          # top-right shows "CUDA Version: 12.x" = the highest CUDA runtime the driver supports
# install the wheel for that CUDA version or lower (cu118, cu121, cu124, cu126, cu128, ...):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121     # <- replace cu121
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest -q          # expect all passed
```

### 2. Working directory and the existing checkpoint (NOT in git)

All data/checkpoints/results live under `FR_ROOT` (or `--root`):

```bash
export FR_ROOT=$HOME/fr_runs            # add to ~/.bashrc if you like
mkdir -p $FR_ROOT/checkpoints
# from the machine that has the checkpoint (Windows PowerShell / Git Bash or Linux):
scp "/path/to/resnet18_cifar10_checkpoint.pth" user@gpu-host:fr_runs/checkpoints/
# or: rsync -avP resnet18_cifar10_checkpoint.pth user@gpu-host:fr_runs/checkpoints/
```

CIFAR-10 downloads automatically into `$FR_ROOT/data` on first use.

First inspect the checkpoint (needed to know its preprocessing):

```bash
python scripts/inspect_checkpoint.py $FR_ROOT/checkpoints/resnet18_cifar10_checkpoint.pth
# reproduce the old numbers (FP32 80.31 %, E4M11 79.46 %); the E4M11 line must use legacy + all:
python scripts/quantize_eval.py --checkpoint $FR_ROOT/checkpoints/resnet18_cifar10_checkpoint.pth \
    --run-name legacy --e4m11-mode legacy --quantize-scope all --normalize none
# optional diagnostic if checkpoint["config"] does not say what normalization was used:
python scripts/eval_fp32.py --checkpoint $FR_ROOT/checkpoints/resnet18_cifar10_checkpoint.pth \
    --run-name legacy --diagnose-normalization
```

### 3. Commands

```bash
# (1) fine-tune from the existing checkpoint (60 epochs, lr 0.01). If normalization is not recorded in the
#     checkpoint config the run stops; add --normalize none  (or cifar10) after looking at the config.
python scripts/train_baseline.py --config configs/finetune.yaml --run-name finetune

# (2) train from scratch (200 epochs, lr 0.1, CIFAR mean/std normalization)
python scripts/train_baseline.py --config configs/scratch.yaml --run-name scratch

# (3) FP32 baseline + E4M11 (paper and legacy encoders) on the best-on-validation checkpoint
python scripts/eval_fp32.py     --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch
python scripts/quantize_eval.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch --compare-modes
#     (use .../last.pth for the final-epoch weights)

# (4) resume after an interruption: same command + --resume (hyper-parameters come from last.pth)
python scripts/train_baseline.py --config configs/scratch.yaml --run-name scratch --resume

# (5) survive SSH disconnects
tmux new -s fr          # then run the commands above; detach with Ctrl-b d, re-attach: tmux attach -t fr
# or
nohup python scripts/train_baseline.py --config configs/scratch.yaml --run-name scratch \
    > $FR_ROOT/scratch.out 2>&1 &
tail -f $FR_ROOT/scratch.out

# everything at once (fine-tune + scratch + reports); RESUME=1 to continue, SKIP_SCRATCH=1 to skip scratch
nohup bash scripts/run_step1.sh > $FR_ROOT/step1.out 2>&1 &
```

### 4. Outputs

| Path (under `$FR_ROOT`) | Content |
|---|---|
| `checkpoints/<run>/last.pth`, `best.pth` | model, optimizer, scheduler, scaler, epoch, RNG states, config, git commit (`best` = best validation accuracy) |
| `results/<run>/train_log.csv` | epoch, lr, train loss/acc, val loss/acc, time |
| `results/<run>/train.log`, `run_meta.json` | log; config + torch/CUDA versions + GPU name + git commit |
| `results/<run>/final_metrics.json` | test accuracy of the final and best-on-val checkpoints |
| `results/<run>/fp32_report.json` | FP32 accuracy + weight stats (Conv/Linear and all parameters) |
| `results/<run>/step1_report.json` / `.md` | FP32 and E4M11 accuracy + CE, drop, SQNR (+ |w| bins, per layer), weight stats, comparison with `configs/paper_targets.yaml` |

The reports state mismatches with the paper as they are; nothing is tuned toward the targets. Small reports can be committed with
`git add -f results/<run>/step1_report.md` (`results/` is git-ignored).

## Step 2: random bit-flip fault injection (paper Sec. 4, App. B.3)

`scripts/fault_campaign.py` flips bits in the packed **E4M11 buffer** (not in FP32), decodes, and evaluates:
one experiment per (fault group, k = 1..8 flips), 100 trials each; k weight indices drawn uniformly from the whole
Conv/Linear buffer with replacement, one bit per index drawn uniformly from the group (exponent 14-11, sign 15,
mantissa MSB 10-6, mantissa LSB 5-0); the clean buffer is restored after every trial. Default format = the fixed
baseline (`subnormal`, bias 7). Resumable: re-run the same command. Day-9 style (single layer, fixed bit) is available via
`--groups bit14 --ks 100 --layer layer4.0.conv1`. Output: `results/<run>/faults_<mode>_b<bias>/{trials.csv,summary.md,summary.json}`.

```bash
python scripts/fault_campaign.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch
```

## Step 3: Fisher sensitivity and mixed E4M11 / 8-bit storage (clean metrics)

Layers with low sensitivity are stored as INT8 (per-tensor; per-channel as an ablation) or FP8 E4M3 (with subnormals;
without and with a per-tensor power-of-two scale) while the rest stay E4M11 (baseline: `subnormal`, bias 7).
Design decisions and limits: docs/CODE_SUMMARY.md section 9. Fault injection on mixed buffers is a later step.

```bash
export FR_ROOT=$HOME/fr_runs
# 1. diagonal Fisher on 1,024 training images (true + empirical), per-layer summary in the log
python scripts/fisher_sensitivity.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch
# 2. validation sweep: 4 arms x targets (14, 12, 10 bits) x [lambda 0, .25, .5, .75, 1, product, worst, 5 random]
python scripts/mixed_precision.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch
# 3. test-set evaluation of the saved selections + report (lambda was chosen on validation)
python scripts/mixed_precision.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch --phase final
```

Outputs under `$FR_ROOT/results/scratch/`: `sensitivity/fisher_{true,empirical}.pt` and `*_layers.json`;
`mixed/mixed_val.csv`, `selections.json`, `layer_table.json`, `sweep_meta.json`, `mixed_test.csv`, `mixed_report.md`.
Both phases are resumable (re-run the same command). Useful options: `--arms`, `--targets`, `--lambdas`,
`--random-draws`, `--fisher-kind empirical`, `--fisher-mode total`, `--score-norm zscore`.

### Step 3 diagnostics: prediction churn and Fisher stability

```bash
# (a) flips / KL vs the all-E4M11 model for every saved selection (needs the validation sweep; resumable)
python scripts/mixed_precision.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch --phase churn
#     -> results/scratch/mixed/mixed_churn_{val,test}.csv, mixed_churn_report.md

# (b) how stable are the Fisher layer rankings? 4,096 training + 4,096 held-out images, per-sample traces cached
python scripts/fisher_stability.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch
#     -> results/scratch/sensitivity/traces_{train,val}_true.npz, fisher_stability.md / .json
```

## Step 4 (part D): bit flips on mixed E4M11 / 8-bit storage

Same fault protocol as step 2, applied to the mixed layouts of step 3 (docs/CODE_SUMMARY.md section 11 for the bit
groups, sampling models and assumptions). Every layout has its own resumable campaign directory; comparison tables are
written to `results/<run>/mixed_faults/<sampling>/comparison.md`.

```bash
# plan only: layouts and number of evaluations
python scripts/mixed_fault_campaign.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch --dry-run

# core comparison (default): e4m11, all:int8_pc/int8_pt/fp8_scaled and, at 12 bits/weight, the saved selections
# product / lambda=0 / random 0 / random 1 / worst for int8_pc, int8_pt, fp8_scaled; group any, k = 8, 100 trials
nohup python scripts/mixed_fault_campaign.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch \
    > $FR_ROOT/mixed_faults.out 2>&1 &

# second budget (references are resumed, only the new layouts run)
python scripts/mixed_fault_campaign.py --checkpoint ... --run-name scratch --targets 10

# decomposition by paper bit group for chosen layouts
python scripts/mixed_fault_campaign.py --checkpoint ... --run-name scratch \
    --layouts e4m11 all:int8_pc int8_pc:T10:product int8_pc:T10:random:0 fp8_scaled:T12:product \
    --groups exponent sign mantissa_msb --ks 8 --trials 100

# more trials for the layouts that matter (resumes from the existing trials)
python scripts/mixed_fault_campaign.py --checkpoint ... --run-name scratch \
    --layouts e4m11 int8_pc:T10:product int8_pc:T10:random:0 --trials 500
```

## Step 5: lower-bit formats, held-out Fisher, heavier fault loads (experiments 1-3)

Details and decisions: docs/CODE_SUMMARY.md section 12. All stages are resumable.

```bash
export FR_ROOT=$HOME/fr_runs
nohup bash scripts/run_lowbit_heldout.sh > $FR_ROOT/lowbit.out 2>&1 &     # everything, about 2-3 h
bash scripts/run_lowbit_heldout.sh load                                    # one stage: fisher | sweeps | load | compare
```

Outputs under `$FR_ROOT/results/<run>/`: `sensitivity_heldout/` (Fisher on validation images 0-2047),
`mixed_heldout/` (held-out Fisher; int8_pc, int8_pt, fp8_scaled, int4_pc, int4_pt, fp4_scaled; budgets 14 12 10 8 6 5;
`mixed_report.md` starts with the frontier table), `mixed_lowbit/` (4-bit arms with the original training-image Fisher),
`mixed_faults_load/weights/comparison.md` (fault-load calibration), `selection_compare_{8,4}bit.md`.
Then `python scripts/export_results.py --fr-root $FR_ROOT --run-name scratch` and push to share.

## Step 6: Pareto under bit flips (experiment 12)

```bash
export FR_ROOT=$HOME/fr_runs
nohup bash scripts/run_pareto_faults.sh pilot > $FR_ROOT/pareto.out 2>&1 &   # 100 trials, about 2 h
bash scripts/run_pareto_faults.sh full                                        # tops up to 500 trials (resumable)
```
Output: `$FR_ROOT/results/scratch/mixed_faults_pareto/weights/comparison.md`.
