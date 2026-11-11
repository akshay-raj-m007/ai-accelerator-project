# Phase 3 - Fault-injection protocol validation (J-FAMP runbook Stage 3, Sections 5.1-5.3)

Reference: `JFAMP_ResNet18_CIFAR10_Experimental_Flow.pdf` Section 5. Previous phases: [phase1](../phase1/README.md), [phase2](../phase2/README.md).
Status: **done - Gate 3 passed for all 6 formats** (GPU run, 497 synthetic cases, 264 trial records, all restored byte-exactly). Results and analysis: [RESULTS.md](RESULTS.md).

## Aim
Prove, before any fault campaign, that the injector flips bits in the **actual encoded representation** (the stored words and
packed bytes that Stage 2 saved), that the bit maps are correct for every format, that a zero-fault trial reproduces the clean
quantized model, that every trial is restored byte-for-byte, and that non-finite outcomes are detected and kept. Gate 3: no
screening (Stage 4) before these pass for every encoded format.

## How the injector works (`src/fr/stage3.py`)
1. `EncodedStore` holds a mutable copy of the stored bytes of one format (uint16 / uint8 words, INT4 packed bytes, float32 scales).
2. A fault XOR-flips bits of those **stored bytes**. Only the affected elements are then decoded from the faulted bytes and
   written into the evaluation model. The FP32 source is never touched.
3. After the trial the stored bytes and the model values are restored. SHA-256 of the whole encoded state is recorded before
   the fault, during the fault and after restoration.

## Bit maps (bit 0 = least significant bit of the element's stored word)
| format | width | sign | exponent | fraction / value |
|---|---|---|---|---|
| E4M11-SN, E4M11-FTZ | 16 | 15 | 11-14 | fraction 0-10 |
| FP16 | 16 | 15 | 10-14 | fraction 0-9 |
| FP8 E4M3FN | 8 | 7 | 3-6 | fraction 0-2 |
| INT8 per-tensor | 8 | 7 (two's complement MSB, weight -128) | none | value 0-6 |
| INT4 per-tensor (packed) | 4 | 3 (two's complement MSB, weight -8) | none | value 0-2 |
| INT scale (float32, optional scope) | 32 | 31 | 23-30 | fraction 0-22 |

Plus the group `all` (every bit of the element). Integer formats have **no exponent group**. For two's complement, the "sign bit"
is the MSB with negative weight, so flipping it changes the code by +-128 (INT8) or +-8 (INT4), not just the sign.

INT4 packing: element `e`, bit `b` is stored in byte `e // 2`, bit `(e % 2) * 4 + b`. Element `2j` is the LOW nibble (bits 0-3)
of byte `j`, element `2j+1` the HIGH nibble (bits 4-7). An odd element count leaves one zero padding nibble that is never a
fault target (all 21 ResNet-18 tensors have even counts, so none occurs).

## Fault model (Section 5.1 -> where it is satisfied)
| runbook requirement | implementation |
|---|---|
| random **distinct** encoded weight locations in the declared scope, seed recorded per trial | `draw_weight_flips`: k distinct elements without replacement from the scope (global = all 11,164,352 weights, or one layer); seed derived per trial and stored in each record |
| float groups: sign, exponent, fraction, bit indices from the codec spec | `stage3.SPECS` (cross-checked bit by bit against an independent reference decoder) |
| integer formats: no exponent; sign and value groups; document the sign bit | groups `sign` and `value`; interpretation documented in `fault_protocol.json` |
| packed INT4: element-to-bit mapping, nibble/bit order, alignment, padding; confirm the flip hits the intended element and bit | mapping above; every element x bit of a 7-element tensor tested, plus a padding test (`fault_injection_smoke_tests.csv`) |
| k = 1 and k = 8; 20-30 trials per layer/format/group for screening, labelled screening | `fault_counts_k: [1, 8]`; Stage 4 plan 30 trials (runbook: 20-30, 20 if runtime is high); the protocol is labelled "SCREENING ... not paper-grade" |
| independent realizations; record the without-replacement policy | one seed per (base seed, format, scope, scope kind, group, k, trial); policy written in the protocol |

## Validation tests (Section 5.2 -> where it is satisfied)
| runbook test | how it is done |
|---|---|
| Round-trip exactness, stable bytes | re-encoding the FP32 checkpoint reproduces the frozen Stage 2 bytes (content hash); encoding is deterministic; decode of the stored words equals an independent pure-Python reference decoder on sampled elements; `encode(decode(w)) == w` for floats |
| Bit mapping on a tiny synthetic tensor: expected direction/magnitude or special-value behavior | every bit of every element of a synthetic tensor per format: stored word, stored byte/bit, decoded value and change ratio compared with first-principles expectations; the 5 other-element values must be unchanged. Special cases: FP16 inf/NaN, FP8 E4M3FN NaN pattern 0x7F, E4M11 reserved exponent (read as 14), subnormal vs flush-to-zero, INT sign bit. INT scale: all 32 bits |
| Scope: selected layer and element count match the trial manifest | layer names and counts equal the model and the Stage 1 inventory; 3,000+ draws in layer scope and global scope stay inside the layer, are distinct, and the global scope is uniform over weights; every trial re-checks its flips against the declared scope and raises otherwise |
| No-fault control: zero flips reproduce the clean quantized model | a full trial with k = 0: logits bit-equal to the clean model, accuracy equal, hashes equal |
| Restoration: byte-for-byte equality, hash before/after | per trial: SHA-256 of the whole encoded state before, during, after; after the trials the decoded weights are bit-identical and the logits equal the clean logits; the FP32 source tensors are hash-checked as untouched |
| Non-finite handling in weights, activations, logits, loss; never drop | forced scenarios (FP16 inf/NaN, FP8 NaN, INT scale inf/NaN) must be flagged in all four places and kept as a trial; E4M11 is shown to be unable to produce non-finite weights (reserved exponent clamps) |

Non-finite convention (declared): an image whose logits contain NaN/Inf counts as **incorrect**; `fault_loss` is `null` when any
logit is non-finite; the trial is kept with `failure = true` and an `output_flag` (`weights_nonfinite`, `activations_nonfinite`,
`logits_nonfinite`, `loss_nonfinite`).

## Trial record (Section 5.3)
Every field of the runbook schema is written: experiment_id, run_id, model, dataset, checkpoint_hash, layer_id, layer_name,
format, codec_version, fault_group, bit_index (list), fault_count, trial_index, seed, sample_count, clean_accuracy,
fault_accuracy, clean_loss, fault_loss, nonfinite_weight_count, output_flag, encoded_hash_before, encoded_hash_after_restore.
Additional fields: `encoded_hash_faulted`, `restored_exactly`, `flip_applied_to_encoded_bytes`, `failure`, `scope_kind`,
`requested_k`, and per-flip details (`layer`, `element`, `bit_index`, field, stored index, bit in the stored byte).

## Outputs (runbook names, `results/<run>/jfamp/fault_injection/`)
| file | content |
|---|---|
| `fault_protocol.json` | groups and bit maps, k values, target scope (all 21 layers with element counts), sampling policy, trial budget (with a measured time estimate for Stage 4), seeds and their derivation, non-finite and restoration policy |
| `codec_fault_validation.json` | every test per format with pass/fail, the restore hashes, Gate 3 per format |
| `fault_injection_smoke_tests.csv` | synthetic cases: known tensors, expected bit mapping, observed decoded values, result |
| `fault_injection_smoke_trials.jsonl` (supporting) | the schema-complete trial records of the validation campaign |
| `stage3_report.md` (supporting) | gate table |

## Additions beyond the PDF (improvements, all optional to use)
- **Scale scope** (`scope_kind = "scales"`): the float32 scale of an INT tensor is also stored memory. Earlier experiments
  showed a single scale hit can cost up to about 10 pp per hit in per-tensor INT8, so it is implemented, validated (all 32
  bits) and left off by default for the runbook screening.
- Hash of the **whole** encoded state each trial, plus a hash during the fault (proves the flip went into the encoded bytes).
- NaN/Inf detection in the outputs of every Conv/Linear/BatchNorm module, not only in the logits.
- Independent reference decoders (pure Python) for the expected values, so a codec bug cannot validate itself.
- A measured per-trial time and the resulting Stage 4 compute estimate in `fault_protocol.json`.

## Run (GPU machine)
```bash
cd ~/Music/aia/ai-accelerator-project && git pull
export FR_ROOT=$HOME/fr_runs
python scripts/stage3_fault_validation.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch
```
Run on 2026-11-11. Needs the frozen Stage 2 outputs on the GPU machine (`results/scratch/jfamp/quantization/`, including `quantized_weights/`).
Code: `src/fr/stage3.py`, `src/fr/stage3_validation.py`, `src/fr/stage3_protocol.py`, `scripts/stage3_fault_validation.py`;
tests `tests/test_stage3.py` (26 tests).
