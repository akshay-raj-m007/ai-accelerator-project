# Phase 1 results (runbook Stages 0 and 1)

Source files: `reports/scratch/jfamp/` (evaluated 2026-11-11 06:06 UTC on the RTX 3050 machine). Setup: [README.md](README.md).

## Gates
| gate | result | basis |
|---|---|---|
| Gate 0 (baseline) | **PASS** | strict load, 10 outputs, 3x3 stride-1 stem, no max-pooling, finite parameters and logits, test accuracy equals the 95.11 % reference (difference +0.000 pp, tolerance 0.05 pp) |
| Gate 1 (inventory) | **PASS** | 21 layers, 11,164,352 quantizable weights, 11,173,962 total parameters, 44,657,408 FP32 bytes; layer sums reconcile with the global counts |

## FP32 baseline
| split | images | accuracy | correct | cross-entropy |
|---|---|---|---|---|
| test (official) | 10,000 | 95.11 % | 9,511 | 0.1855 |
| validation holdout | 5,000 | 95.48 % | 4,774 | 0.1704 |

The validation value equals the best-epoch value stored during training (95.48 %). The test cross-entropy equals the
earlier repository report (0.1855).

### Per-class accuracy (test, 1,000 images per class)
| class | accuracy | class | accuracy |
|---|---|---|---|
| airplane | 96.2 % | dog | 91.9 % |
| automobile | 97.7 % | frog | 97.4 % |
| bird | 93.5 % | horse | 96.7 % |
| cat | **88.4 %** | ship | 96.7 % |
| deer | 96.6 % | truck | 96.0 % |

Largest confusions (test images): cat predicted as dog 58, dog predicted as cat 51, truck as automobile 21, ship as
airplane 17, cat as bird 16. Full matrix: `baseline/confusion_matrix.csv`.

## Layer inventory
| group | layers | weights | share of 11,164,352 |
|---|---|---|---|
| stem (conv1) | 1 | 1,728 | 0.02 % |
| layer1 | 4 | 147,456 | 1.32 % |
| layer2 | 5 | 524,288 | 4.70 % |
| layer3 | 5 | 2,097,152 | 18.78 % |
| layer4 | 5 | 8,388,608 | 75.14 % |
| classifier (fc) | 1 | 5,120 | 0.05 % |

Non-quantized parameters (biases and BatchNorm parameters): 9,610 (11,173,962 - 11,164,352). Per-layer rows:
`layer_profile/layer_inventory.csv`.

## Weight and activation distributions (figures in `figures/`)
Read from `layer_inventory.csv` and `distribution_stats.json`:
- **Weight scale shrinks with depth.** Standard deviation: conv1 0.067, layer1 about 0.015, layer2 0.010-0.027,
  layer3 0.008-0.015, layer4 0.0014-0.0064 (layer4.1.conv2: 0.0014, |w| max 0.020). fc is larger (0.081).
- **All weights are small.** The largest |w| anywhere is 0.556 (conv1); E4M11's maximum finite value is 255.9.
- **Most weights are tiny.** Fraction of weights with |w| < 2^-7 (0.0078, the E4M11 flush threshold of the paper's
  Algorithm 1): 87.2 % of all weights; conv1 64.4 %, layer1.0.conv1 78.6 %, fc 1.5 %. This was computed from
  `models/scratch/best.pth` and is not stored in the artifacts. In conv1 and layer1.0.conv1 the median |w| is below
  5e-5. This is consistent with the earlier finding that the paper's flush-to-zero rule collapses this model to 28.44 %.
- **Activations (512 validation images, input of each layer).** The maximum is 4.11 (layer1.1.conv1); the stem input is
  the normalized image (max 2.13). Post-ReLU layers are 32-87 % zeros (layer3.1.conv2 86.7 %, layer2.0.conv1 32.5 %),
  and the fc input is 23.5 % zeros.

## Reading and limits
- The FP32 reference is reproduced exactly, so later stages can compare against 95.11 % (test) and 95.48 % (validation).
- Cat versus dog is the weakest class pair; later fault or quantization drops should be read against per-class
  accuracy as well as the global figure.
- The distribution facts are properties of this one checkpoint (one model-dataset pair, one training run).

## Next (runbook order; not started)
Stage 2: clean quantization of E4M11-SN, FP16, FP8 E4M3 (E4M3FN preferred), INT8 and INT4 from this checkpoint with
frozen quantizer settings; then Stage 3 encoded-bit smoke tests before any fault campaign.
