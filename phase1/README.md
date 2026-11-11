# Phase 1 - FP32 baseline intake and layer profiling (J-FAMP runbook Stages 0 and 1)

Model: CIFAR-style ResNet-18 on CIFAR-10. Reference document: `JFAMP_ResNet18_CIFAR10_Experimental_Flow.pdf` (J-FAMP
experimental runbook, version 1.0). Numbers and tables: [RESULTS.md](RESULTS.md).

## Aim
The overall project is J-FAMP, a joint, fault-aware choice of datatype (and, later, protection) for every layer under
storage, clean-accuracy and fault-reliability constraints, using a LAMPQ-inspired layer sensitivity (not an exact LAMPQ
reproduction, see runbook Sec. 12).

Phase 1 has one purpose: **establish a verified, fully documented FP32 reference and a stable layer key** that every
later stage (clean quantization, encoded-bit fault injection, J-FAMP candidate table) is measured against. The runbook's
comparability rule applies: no result is compared without the corresponding FP32 baseline and with the same split,
preprocessing and metric definitions.

Phase 1 covers exactly:

| runbook stage | content | gate |
|---|---|---|
| 0. Baseline intake | identity and provenance of the checkpoint, strict load, architecture checks, FP32 reproduction on validation and test | Gate 0 |
| 1. Layer inventory | all 21 Conv2d/Linear layers, shapes, counts, weight statistics, FP32 bytes; weight / activation distribution plots | Gate 1 |

Not part of Phase 1 (not started under the runbook protocol): Stage 2 clean quantization of the five formats, Stage 3
fault-protocol validation, Stage 4 fault screening, Stage 5 J-FAMP candidate table, Stage 6 final package.
Earlier exploratory experiments (mixed E4M11/INT storage, Fisher sensitivity, scale-metadata faults; see
`docs/EXPERIMENTS.md`) were run before the runbook arrived. They do not follow its protocol (e.g. k = 1024 faults,
scaled FP8) and are not Phase 1 results.

## Experimental setup
| item | value |
|---|---|
| Model | `fr.model.build_resnet18_cifar`: 3x3 stem convolution (stride 1, padding 1), no max-pooling (`Identity`), 10-class `fc` |
| Checkpoint | `models/scratch/best.pth` (GPU machine: `$FR_ROOT/checkpoints/scratch/best.pth`), SHA-256 `c612da6a4b20c34fd864f236a3bd5564cb3ce253c162e479bcbfa56d80aafb3f`, 89,504,508 bytes |
| Origin | trained from scratch in this project (200 epochs, SGD/Nesterov, lr 0.1, weight decay 5e-4, batch 128, AMP, seed 0, RTX 3050); **not** supplied by a teammate. The checkpoint stores training git commit `239c285` marked dirty; training ran with `deterministic=False` |
| Dataset | CIFAR-10; official 10,000-image test set; 5,000-image deterministic validation holdout from the training set (split seed 0, `fr.data.split_indices`); validation-index SHA-256 `d2e1ec6587c2c4d4018f6e02e47393b604da05c6ce4d3e2f42e51c8b9f5b7372`, train-index SHA-256 `9e71eb3375dbeffdbd893ba4d931130a04a13c6ac9d570a63ff9603c94154aa7` |
| Preprocessing (evaluation) | `ToTensor` + `Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])`, 32x32 RGB, no resize, no augmentation. Training used RandomCrop(32, padding 4) + horizontal flip |
| Evaluation | FP32, no autocast, `model.eval()`, `requires_grad=False`, `torch.inference_mode`, batch 500 |
| Quantization scope | none in Phase 1 (weights stay FP32). Later stages: weight-only; biases, BatchNorm and activations stay FP32 |
| Environment | Python 3.12.3, torch 2.5.1+cu121, CUDA 12.1, NVIDIA GeForce RTX 3050 OEM (8 GB), Linux |
| Test-set rule | the test set is used only to reproduce the baseline; nothing is selected or tuned on it. Activation plots use validation images |

## Procedure (what was run, in order)
1. `scripts/stage01_baseline.py` (inference only, about 1 min). It
   - computes the checkpoint SHA-256, size, keys and environment; loads the model with `strict=True`;
   - checks output dimension, stem, absence of max-pooling, NaN/Inf in parameters and logits;
   - evaluates FP32 on the 5,000 validation and 10,000 test images (accuracy, correct count, cross-entropy, per-class
     accuracy, 10x10 confusion matrix) and compares the test result with the runbook reference 95.11 % (9,511/10,000,
     tolerance 0.05 pp);
   - builds the layer inventory in definition order (shape, count, min/max/mean/std, |w| percentiles p50/p90/p99/p99.9,
     FP32 bytes, layer group), the storage summary and the parameter summary, and reconciles them
     (sum of layer weights = global count; bytes = 4 x count; 21 layers; 11,164,352 quantizable weights;
     11,173,962 total parameters; 44,657,408 bytes);
   - prints and stores Gate 0 and Gate 1.
2. `scripts/plot_distributions.py --act-point both`: histograms of every layer's weights (all values) and activations
   (input and output of each Conv2d/Linear layer, 512 validation images, FP32). 8 plots per page, one heading and one
   legend per plot, log y axis. Exact mean/std/min/max/zero-fraction over all values; histogram and percentiles from a
   random subsample of at most 2,000,000 values per layer.

Definitions: the confusion matrix has rows = true class, columns = predicted class. Per-class accuracy = correct /
support. "Activation input" = tensor entering the layer (post-ReLU for all but the stem, whose input is the normalized
image); "activation output" = layer output before BatchNorm.

## Reproduce
```bash
export FR_ROOT=$HOME/fr_runs
pip install matplotlib
python scripts/stage01_baseline.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch \
    --supplied-by "<name>" --source-location "trained from scratch in this project (best.pth)"
python scripts/plot_distributions.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch --act-point both
```
Code: `src/fr/baseline_profile.py`, `src/fr/dist_plots.py`; tests `tests/test_baseline_profile.py` (5 tests).

## Artifacts (runbook names; all under `reports/scratch/jfamp/`)
| runbook files | path | present after pull |
|---|---|---|
| baseline_manifest.json, baseline_metrics.json, confusion_matrix.csv, per_class_metrics.csv, checkpoint_validation.txt | `baseline/` | yes (5 of 5) |
| layer_inventory.csv, layer_inventory.json, layer_storage_summary.csv, parameter_summary.json | `layer_profile/` | yes (4 of 4) |
| weights / activations (input, output) distribution PDF + 3 PNG pages each, distribution_stats.json | `figures-baselinefp32/` | yes (13 of 13) |

## Known deviations and limits
- The checkpoint is not teammate-supplied (the runbook's starting assumption). The manifest fields `supplied_by` and
  `source_location` carry what was passed on the command line.
- The checkpoint creation date is the file modification time (2026-11-05 UTC on the GPU machine).
- Training was not bit-reproducible (`deterministic=False`); the checkpoint is therefore the fixed reference and is not
  to be retrained.
- Per-class accuracy and confusion matrix files are for the test set only; validation per-class values are in
  `baseline_metrics.json`.
- Activation statistics come from 512 validation images; they describe this sample, not the whole dataset.
- One model-dataset pair only (runbook Sec. 12).
