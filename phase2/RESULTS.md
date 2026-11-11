# Phase 2 results (runbook Stage 2)

Source: `reports/scratch/jfamp/quantization/` (GPU run 2026-11-11; validation stage first, test stage after the configuration was
frozen: frozen-config SHA-256 `387ff9741de4a14dbea7073b9a968ccca0fee13ee94c611444e71f264ae1514a`, verified by the test stage).
Setup and requirement coverage: [README.md](README.md). INT3 skipped by decision.

## Clean accuracy and loss (weights-only quantization, FP32 compute on the decoded stored weights)
| format (exact variant) | val acc (5,000) | val drop (pp) | val dCE | test acc (10,000) | test drop (pp) | test dCE |
|---|---|---|---|---|---|---|
| FP32 reference | 95.48 % (4,774) | - | - | 95.11 % (9,511) | - | - |
| E4M11-SN | 95.48 % | 0.00 | +0.00002 | 95.11 % | 0.00 | -0.000004 |
| FP16 (IEEE binary16) | 95.48 % | 0.00 | -0.00002 | 95.11 % | 0.00 | +0.000002 |
| FP8 E4M3FN (OCP, unscaled) | 95.30 % (4,765) | 0.18 | -0.0009 | 95.12 % (9,512) | -0.01 | +0.0004 |
| INT8 per-tensor | 95.36 % (4,768) | 0.12 | -0.00003 | 95.15 % (9,515) | -0.04 | +0.0005 |
| INT4 per-tensor (packed) | 94.30 % (4,715) | **1.18** | +0.0144 | 93.88 % (9,388) | **1.23** | +0.0206 |
| E4M11-FTZ (comparison only) | 28.98 % (1,449) | **66.50** | +1.678 | 28.44 % (2,844) | **66.67** | +1.663 |

Drops are FP32 minus candidate accuracy on the same images. Negative test drops (FP8 +1 image, INT8 +4 images) are within the
test-set standard error (about 0.22 pp) and are not gains.

## Round-trip error (weights, all 21 layers pooled; per-layer rows in `quantization_metrics.csv`)
| format | global SQNR | weakest layer (SQNR) | best layer (SQNR) | weights nonzero in FP32 that decode to exactly 0 | clipped / non-finite |
|---|---|---|---|---|---|
| E4M11-SN | 69.27 dB | layer4.1.conv2 (55.8) | conv1 (80.2) | 48,622 (0.44 %) | 0 / 0 |
| FP16 | 73.67 dB | layer1.0.conv1 (73.5) | conv1 (74.2) | 14,715 (0.13 %) | 0 / 0 |
| FP8 E4M3FN | 21.20 dB | layer4.1.conv2 (8.0) | fc (31.7) | 3,882,542 (34.8 %) | 0 / 0 |
| INT8 per-tensor | 32.58 dB | layer4.0.conv2 (26.8) | fc (41.3) | 752,833 (6.7 %) | 0 / 0 |
| INT4 per-tensor | 7.85 dB | layer4.0.conv2 (3.0) | fc (16.3) | 8,287,528 (74.2 %) | 0 / 0 |
| E4M11-FTZ | 6.64 dB | layer4.1.conv2 (0.0) | fc (40.5) | 9,736,012 (87.2 %) | 0 / 0 |

## Actual storage (`storage_summary.csv`)
| format | bits/weight (theoretical) | payload bytes | scale bytes | padding | total weight bytes | compression vs FP32 weights (44,657,408 B) | full model incl. FP32 biases/BN/buffers |
|---|---|---|---|---|---|---|---|
| E4M11-SN, FP16, E4M11-FTZ | 16 | 22,328,704 | 0 | 0 | 22,328,704 | 2.000x | 1.997x |
| FP8 E4M3FN | 8 | 11,164,352 | 0 | 0 | 11,164,352 | 4.000x | 3.979x |
| INT8 per-tensor | 8 | 11,164,352 | 84 (21 float32 scales) | 0 | 11,164,436 | 3.99997x | 3.979x |
| INT4 per-tensor | 4 | 5,582,176 | 84 | 0 (all counts even) | 5,582,260 | 7.99988x | 7.905x |

Non-quantized parameters stay FP32: 38,440 B (biases and BatchNorm parameters) plus 38,560 B of BatchNorm buffers.

## Reading
- **Every primary format has clean metrics and storage measured** (Gate 2's first condition). E4M11-SN and FP16 reproduce FP32
  exactly (same correct counts on validation and test). FP8 E4M3FN and INT8 lose at most 0.18 pp on validation and 0 on test.
  INT4 loses 1.18 pp (validation) and 1.23 pp (test). The runbook gives no threshold for "unacceptable" degradation, so Gate 2
  is not judged here: that needs delta_clean (to be agreed).
- **A prediction made before this run was wrong.** From the weight statistics I expected unscaled FP8 E4M3FN to lose clean
  accuracy, because 34.8 % of the weights decode to exactly zero. It lost only 0.18 pp. The weights that round to zero are the tiny
  ones (|w| below 2^-10, about 0.001). Losing them costs little accuracy here; I have not measured how much of the weight
  energy they hold, so this is an observation, not an explanation.
- **SQNR does not rank accuracy.** FP8 E4M3FN (21.2 dB) and INT8 (32.6 dB) are within 0.2 pp of each other; INT4 (7.85 dB) loses
  about 1.2 pp; E4M11-FTZ (6.64 dB, similar SQNR to INT4) collapses to 28.4 %. In FTZ, layer4.1.conv2 has 99.9 % of its weights
  flushed (layer SQNR 0.05 dB) and layer4.0.conv2 98.1 %; this is consistent with, but not proven to be, the cause of the
  collapse (no per-layer ablation was run).
- **FP16 has a higher global SQNR than E4M11-SN (73.67 vs 69.27 dB)**, unlike the paper's finding on its own model. Both give
  identical accuracy here. The weakest E4M11-SN layer is layer4.1.conv2 (55.8 dB), the layer with the smallest weights.
- **INT8 per-tensor is the cheapest format with negligible loss** (4x compression, 0.12 pp validation drop); INT4 gives 8x at
  about 1.2 pp. These are clean-accuracy facts only; fault reliability (Stages 3-4) has not been measured under this protocol.

## Checks
- 29 of 29 codec self-tests passed and are stored in the manifest (including cross-checks against `torch.float8_e4m3fn` and
  `torch.float16`).
- Global SQNR values equal the locally computed weights-only values (69.27 / 73.67 / 21.20 / 32.58 / 7.85 / 6.64 dB).
- The manifest records git commit `c05709b` with a dirty working tree on the GPU machine, plus the SHA-256 of every codec
  source file; each format has a content hash for its encoded weights.

## Limits
- `quantized_weights/` (about 94 MB) exists on the GPU machine only and is not in git; the manifest has the content hash and
  size of each file. Stage 3 must use these files and the packed INT4 bit mapping in `stage2.int4_bit_location`.
- One model-dataset pair, one checkpoint; the test set was used once per format, after freezing.
- Validation accuracy (5,000 images) has a standard error of about 0.3 pp, so differences of 0.12-0.18 pp between FP32, INT8
  and FP8 are not significant; the INT4 drop (1.18-1.23 pp, consistent on validation and test) is larger than that noise.
