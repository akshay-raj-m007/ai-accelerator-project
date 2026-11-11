# Phase 3 results (runbook Stage 3 - fault-injection protocol validation)

Source: `reports/scratch/jfamp/fault_injection/` (GPU run, run id `20261111T073720Z-35b172e9`, RTX 3050). Setup and requirement
coverage: [README.md](README.md). Inputs: the frozen Stage 2 encoded weights (frozen-config SHA-256 `387ff9741de4...`, checkpoint
`c612da6a...`). Model-level tests used the first 1,000 validation images.

## Gate 3
| format | role | tests passed | Gate 3 |
|---|---|---|---|
| E4M11-SN | primary | 21 / 21 | **PASS** |
| FP16 | primary | 21 / 21 | **PASS** |
| FP8 E4M3FN | primary | 21 / 21 | **PASS** |
| INT8 per-tensor | primary | 22 / 22 | **PASS** |
| INT4 per-tensor (packed) | primary | 23 / 23 | **PASS** |
| E4M11-FTZ | comparison | 21 / 21 | **PASS** |

All 497 synthetic bit cases passed (bit mapping 420, scale-bit mapping 64, special values 12, INT4 packing/padding 1). No test
failed for any format. The gate is: bit mapping, no-fault control and exact restoration pass for every encoded format.

## Evidence per runbook test (Section 5.2)
| test | result |
|---|---|
| Round-trip exactness, stable bytes | for all 6 formats, re-encoding the FP32 checkpoint reproduces the frozen Stage 2 bytes (content hash equal, e.g. INT4 `a93d0082...` both sides) and equals the manifest hash; decode of the stored words equals the independent reference decoder (0 mismatches on the sampled elements, e.g. 41,728 for INT4) |
| Bit mapping on a tiny synthetic tensor | every bit of every element: E4M11-SN 96, FP16 96, E4M11-FTZ 96, FP8 E4M3FN 48, INT8 56, INT4 28 cases, all PASS (stored word, stored byte/bit, decoded value, and no other element changed). Special values: 2 per format, all PASS. INT scale: 32 bits each for INT8 and INT4, PASS |
| INT4 packing | the padding-nibble test (20,000 random flips on a 7-element tensor) passed: elements always < 7, padding nibble never touched |
| Scope | layer names, order and element counts equal the model and the Stage 1 inventory (21 layers, 11,164,352 elements); 0 scope violations in the layer-wise and global draws; the global scope is uniform over weights (max deviation of a layer's share from its size share: 0.0012 for INT4) |
| No-fault control | zero-flip trial gives bit-equal logits and identical accuracy for every format (96.3 % E4M11-SN / FP16 / INT8, 96.0 % FP8 E4M3FN, 95.4 % INT4, 29.9 % E4M11-FTZ on the 1,000 images) |
| Restoration | 264 of 264 trials restored byte-exactly (encoded-state hash before == after); after the trials the decoded weights are bit-identical to the clean ones and the logits equal the clean logits; the FP32 source tensors were not touched |
| Flip applied to the encoded bytes | 264 of 264 trials: the hash during the fault differs from the hash before (the flip changed the stored bytes) |
| Non-finite handling | see below |

## Trial records (Section 5.3)
264 schema-complete trial records (44 per format), 0 missing fields, 264 distinct seeds, sample_count 1,000 in every record.
They cover: 1 no-fault control, 12 restoration trials (groups sign / exponent / fraction or sign / value, plus all; k = 1 and 8),
1 forced non-finite scenario, and the smoke campaign (global scope, layer scope for fc and layer4.1.conv2 at k = 8, and for INT
formats the optional scale scope). The smoke campaign validates the machinery; its accuracies are not a screening result.

## Non-finite outcomes: 9 of 264 trials flagged, none dropped
| format | scope | trials flagged | flags | why |
|---|---|---|---|---|
| FP16 | weights (forced) | 1 | weights, activations, logits, loss | exponent forced to all ones (inf/NaN) |
| FP8 E4M3FN | weights (forced) | 1 | weights, activations, logits, loss | forced to the NaN pattern S.1111.111 |
| INT8 | scales (forced) | 1 | weights, activations, logits, loss | float32 scale forced to exponent all ones |
| INT4 | scales (forced) | 1 | weights, activations, logits, loss | same |
| INT8 | scales (smoke) | 2 | **activations, logits, loss only** | scale bit 30 or sign/exponent hits |
| INT4 | scales (smoke) | 3 | **activations, logits, loss only** | scale bit 30 or sign/exponent hits |

- **A finding:** in the 5 unforced scale-scope trials, **no weight was non-finite** (`nonfinite_weight_count = 0`), yet the
  activations overflowed to inf/NaN and the loss was non-finite. A flip of the float32 scale's top exponent bit (bit 30) makes the
  scale about 10^36, and `code x scale` is still below the float32 maximum (INT8 codes reach 127, INT4 codes 7), so the
  weights stay finite while the activations overflow. A weights-only non-finite check would have missed these trials, which is
  why the runbook asks for detection in activations, logits and loss as well. The accuracies in these trials were 0.0-4.6 %
  because images with non-finite logits count as incorrect.
- **E4M11 cannot produce non-finite weights:** forcing the reserved exponent (es = 15) leaves the weight finite (read as es = 14,
  about 128x): no flag was set. In the forced fc trial, E4M11-SN went from 96.3 % / loss 0.147 to 95.3 % / 0.288 on the 1,000
  images; E4M11-FTZ stayed at 29.9 % with loss 1.83 -> 3.43.
- In the 250 unforced weight-scope trials, no random weight flip (k = 1 or 8) caused a non-finite outcome. With 3 trials per
  setting this is not evidence that none can occur.

## Stage 4 planning numbers (from `fault_protocol.json`)
- Measured mean time per trial on the RTX 3050 including hashing: 0.39 s (1,000 images).
- Layer-wise screening of the five primary formats: 756 (layer x format x group) configurations x k in {1, 8} included, 30 trials
  each = 22,680 evaluations, about **2.4 hours**. Global screening adds 18 (format x group) x 2 k x 30 = 1,080 evaluations, about
  7 minutes (my arithmetic, not in the file).
- The runbook range is 20-30 trials; 20 trials would cut the layer-wise run to about 1.6 hours.

## Reading and limits
- The injector flips bits of the stored encoded bytes, decodes only from them, and restores them exactly, for all six formats, so
  Gate 3 is met and Stage 4 screening can start.
- The runs used the code at commit `e33a700` with a dirty working tree on the GPU machine (recorded in
  `codec_fault_validation.json`); the code files are the pushed ones.
- The smoke trials (3 per setting) test the machinery only. No robustness claim follows from them.
- Distinct-element sampling differs from the paper's with-replacement draw; immaterial for k <= 8.
- INT scale faults are an optional extension beyond the runbook scope and are off in the runbook screening. Given the finding
  above, I would add a separate scale-scope screening for INT8 and INT4 in Stage 4.
- One checkpoint, base seed 0.

## Next (runbook Stage 4, not started)
Global screening at k = 1 and 8, then layer-wise screening with equal trial budgets, for sign / exponent / fraction (or value)
groups, on validation images; outputs `fault_trials.jsonl`, `fault_summary.csv`, `global_fault_summary.csv`,
`layer_fault_heatmap.csv`, `fault_campaign_manifest.json`.
