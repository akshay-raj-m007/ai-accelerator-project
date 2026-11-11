# Phase 2 - Candidate quantization and clean evaluation (J-FAMP runbook Stage 2, Sections 4.1-4.3)

Reference: `JFAMP_ResNet18_CIFAR10_Experimental_Flow.pdf` Section 4. Phase 1 (Stages 0-1) is in [../phase1/](../phase1/README.md).
Status: code written and unit-tested; **GPU run and results pending** (this file is updated with results after the run).

## Aim
Produce, from the verified FP32 checkpoint, the candidate weight formats of the runbook stored as **real encoded words**,
with the quantizer configuration fixed and recorded, and measure for every format the round-trip error, clean validation
accuracy and loss, test accuracy (after freezing), and the actual storage. INT3 is skipped by decision (optional, deferred).

## Formats
| key | exact variant (name used in every table) | storage word | role |
|---|---|---|---|
| `e4m11_sn` | E4M11-SN: 1+4+11 bits, bias 7, **subnormals enabled** (`e4m11` codec, mode `subnormal`, the codec used so far) | uint16 | primary |
| `fp16` | FP16: IEEE 754 binary16 (E5M10), bias 15 | uint16 | primary |
| `fp8_e4m3fn` | FP8 **E4M3FN** (OCP): 1+4+3 bits, bias 7, no infinities, single NaN pattern S.1111.111, max finite 448, subnormals enabled, **unscaled** | uint8 | primary |
| `int8_pt` | INT8 symmetric, **per-tensor** (the paper's Eq. 3) | uint8 + float32 scale | primary |
| `int4_pt` | INT4 symmetric, **per-tensor**, **packed two codes per byte** | packed uint8 + float32 scale | primary |
| `e4m11_ftz` | E4M11-FTZ: same layout, subnormals flushed to zero (the paper's Algorithm 1; `e4m11` mode `paper`) | uint16 | comparison only |

The earlier exploratory formats (`fp8` with max 240, `fp8_scaled`, per-channel INT, FP4) are **not** used in this stage.

## Requirement coverage (runbook text -> where it is satisfied)
| runbook requirement | covered by |
|---|---|
| Per format: rounding mode, saturation/clipping, scale calculation, zero-point policy, subnormal/FTZ, NaN/Inf treatment | `quantizer_manifest.json` -> `formats.<key>.config` (`rounding`, `saturation_clipping`, `scale`, `zero_point`, `subnormal_behavior`, `nan_inf`) |
| Integer formats: symmetric/asymmetric, per-tensor/per-channel, signed range, calibration method, per layer or per channel | same `config`: `mapping` (symmetric), `granularity` (per-tensor), `signed_range`, `calibration` (max-abs, data-free), `scale_computed` (per layer) |
| FP8: distinguish E4M3FN from other variants; never write only "FP8 E4M3" | label `FP8 E4M3FN (OCP, bias 7, max 448)` in every CSV/table; `finite_nan_handling` in the config; the earlier max-240 variant is excluded |
| E4M11-SN: codec version/commit, bit layout, bias, subnormal implementation, encode/decode test results; E4M11-FTZ only as separate comparison | manifest `codec_source` (git commit and dirty flag, SHA-256 of `e4m11.py`, `formats8.py`, `stage2.py`, `sqnr.py`, the script), config (`bit_layout`, `exponent_bias`, `subnormal_behavior`), `codec_selftests` (29 tests incl. all 61,440 non-reserved E4M11-SN words round-trip, paper example 3.14 -> 0x448F, ties-to-even); FTZ has `role: comparison` |
| Record simulated vs actually packed storage | manifest `quantization_simulated_or_packed`: actual encoded words, INT4 packed; evaluation decodes them to FP32 |
| Generate each candidate from the original FP32 checkpoint | manifest `independence`; every codec call takes the original tensor; checkpoint SHA-256 is checked |
| Round trip FP32 -> encoded -> decoded: max abs error, MAE, RMSE, SQNR per layer and globally | `quantization_metrics.csv` (21 layer rows + `GLOBAL` per format; also MSE, relative L2, clipped count, zeroed-from-nonzero, decoded non-finite). Decoding uses the stored words only |
| Clean validation accuracy and loss for every candidate | `clean_accuracy.csv` (`val_*` columns, all 5,000 held-out validation images, FP32 reference row, drop in pp, delta cross-entropy) |
| Test set only after the configuration is frozen; no tuning on test | `--stage quantize` never reads the test set and writes `frozen_config_sha256`; `--stage test` refuses to run without the manifest, checks the checkpoint hash and the content hash of every stored encoded file, then adds `test_*` columns |
| Actual stored bytes incl. scales, zero-points, alignment/padding, packing overhead, codec metadata; theoretical bits separately | `storage_summary.csv`: `theoretical_bits`, `payload_bytes`, `padding_bits`, `scale_bytes`, `codec_metadata_bytes`, `total_bytes` (no zero-points exist; no codec metadata exists) |
| Metric definitions: accuracy + pp drop, cross-entropy, SQNR with signal and noise defined, reconstruction error, storage bytes, compression ratio with what is included | manifest `metric_definitions` |
| Non-quantized model storage shown separately | `storage_summary.csv`: `non_quantized_param_bytes_fp32`, `buffer_bytes`, `full_model_bytes`, `compression_ratio_full_model`; `compression_ratio_weights` excludes them (stated) |
| Stage 2 outputs: quantizer_manifest.json, quantization_metrics.csv, clean_accuracy.csv, quantized_weights/, storage_summary.csv | all five, under `results/<run>/jfamp/quantization/` |

## Decisions made here (not dictated by the runbook; stated so they can be challenged)
- **INT4 packing order:** element 2j in the low nibble of byte j, element 2j+1 in the high nibble; an odd count pads one zero
  nibble. All 21 ResNet-18 weight tensors have even counts, so no padding occurs. Stage 3 must use
  `stage2.int4_bit_location(element, bit)` for the packed bit mapping.
- **FP16 saturation:** |w| is clipped to 65,504 before the cast (no overflow to inf). No weight is near that range
  (largest |w| is 0.556).
- **FP8 E4M3FN is unscaled** (no scale is mentioned for FP8 in the runbook). The earlier exploratory scaled FP8 is excluded.
- **Gate 2 is not auto-judged:** the runbook gives no accuracy threshold for "unacceptable clean degradation", so the script
  reports drops and the reading is done after the run.
- **`quantized_weights/` is not meant for git:** about 94 MB in total and exactly regenerable from the checkpoint; the
  manifest holds a content hash per format.

## Weights-only smoke check (computed locally on the real checkpoint, no data, no accuracy; to be confirmed by the GPU run)
| format | global SQNR | weights decoded to exactly 0 although nonzero in FP32 | stored weight bytes | compression vs FP32 weights |
|---|---|---|---|---|
| E4M11-SN | 69.27 dB | 0.4 % | 22,328,704 | 2.00x |
| FP16 | 73.67 dB | 0.1 % | 22,328,704 | 2.00x |
| FP8 E4M3FN (unscaled) | 21.20 dB | 34.8 % | 11,164,352 | 4.00x |
| INT8 per-tensor | 32.58 dB | 6.7 % | 11,164,436 (incl. 84 B of scales) | 4.00x |
| INT4 per-tensor (packed) | 7.85 dB | 74.2 % | 5,582,260 (incl. 84 B of scales) | 8.00x |
| E4M11-FTZ (comparison) | 6.64 dB | 87.2 % | 22,328,704 | 2.00x |

These reproduce earlier repository numbers (E4M11-SN 69.27 dB, E4M11-FTZ 6.64 dB, INT8 per-tensor 32.58 dB).
FP16 having a higher SQNR than E4M11-SN here differs from the paper's finding (E4M11 highest SQNR); it is a property of this
checkpoint's weight scale (87 % of weights are below 2^-7, where E4M11-SN has an absolute step of 2^-17 while FP16 has
finer steps), not an error in the codecs, which are cross-checked against NumPy/torch FP16 and FP8 E4M3FN.

## Run (GPU machine)
```bash
cd ~/Music/aia/ai-accelerator-project && git pull
export FR_ROOT=$HOME/fr_runs
# 1) freeze + validation (no test image is read)
python scripts/stage2_quantize.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch --stage quantize
# 2) test set, only after step 1 (verifies the frozen hashes first)
python scripts/stage2_quantize.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch --stage test
```
Code: `src/fr/stage2.py`, `scripts/stage2_quantize.py`; tests `tests/test_stage2.py` (8 tests).
