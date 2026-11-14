# JFAMP — Stage 5: Protection Profiling (None / Parity / SECDED / DMR) — MobileNetV2 / CIFAR-10

Protocol Section 5.3. All numbers come from `stage_05_protection/` (checkpoint SHA-256 `268f8b52…243f9e`, frozen Stage 2 configuration `471bc480…907561`).

## 1. Objective

To characterize how four protection mechanisms behave when bit faults hit the stored, encoded weights of MobileNetV2, and what they cost in bytes. The study is done at codeword level on the real serialized payload of every layer and format. It supplies the protection behaviour and the storage overhead that a joint datatype–protection allocation needs.

Formats: FP16, BF16, E4M11-SN, FP8 E4M3FN, FP8 E5M2, INT8 (per-tensor), INT4 (per-tensor, packed nibbles). Strategies:

| Strategy | Stored bits per 32-bit word | Redundancy |
|---|---:|---|
| None | 32 | none |
| Parity | 33 | 1 even-parity bit (bit 32) |
| SECDED | 39 | extended Hamming (39,32): 6 check bits at positions 1, 2, 4, 8, 16, 32 and one overall parity bit |
| DMR | 64 | a full second copy (extension beyond the protocol table; detect-only) |

## 2. Methodology

1. The frozen Stage 2 encoded weights are loaded and their content hashes checked against the quantizer manifest.
2. Each layer's payload is serialized (little-endian) and cut into 32-bit logical words. The last word is zero-padded; padding bits are never fault targets. The float32 INT scale of a layer is one more 32-bit word (region "scale"), protected in the same way and always tested.
3. Up to 100 codewords are sampled per layer and format without replacement (all words if the layer has fewer). The same words are used for every scenario and strategy.
4. For each word every strategy really encodes the word, flips bits in the stored word, reads it back through its own decoder, and the result is compared with the original.
5. Scenarios. Primary: 1 or 2 distinct DATA bits flipped, at identical positions for every strategy. Supplementary (Parity and SECDED only): 1 or 2 distinct bits anywhere in the stored word, so the redundancy bits can be hit too.
6. Reaction policy: a word that is detected but not corrected is consumed as read, so its error stays residual.

Experimental scale:

| Item | Count |
|---|---:|
| Layers / formats / strategies | 53 / 7 / 4 |
| Codewords tested | 37,022 (36,916 payload + 106 INT scale words) |
| Raw records | 444,264 (296,176 primary + 148,088 supplementary) |
| Layers with fewer than 100 words (all their words used) | 5 layer/format cases: `features.1.conv.0.0` (72 words for FP8 and INT8, 36 for INT4) and INT4 `features.1.conv.1` (64) |

Before the campaign the script validated the codes: 13 code tests (known answer, independent reference encoder, every single error and every double error of the 39 stored bits, all 9,139 triple errors, parity and DMR behaviour) and 7 layout cross-checks (300 random layer/element/bit per format, 0 mismatches) showing that the 32-bit word view addresses the same physical bit as the Stage 4 injector. Gate: PASS.

## 3. Results (primary scenarios, payload codewords, pooled over all formats and layers)

| Mechanism | Single-bit detection | Single-bit correction | Single-bit residual | Double-bit detection | Double-bit correction | Double-bit residual | Silent corruption (1 / 2 bits) |
|---|---:|---:|---:|---:|---:|---:|---:|
| None | 0 % | 0 % | 100 % | 0 % | 0 % | 100 % | 100 % / 100 % |
| Parity | 100 % | 0 % | 100 % | 0 % | 0 % | 100 % | 0 % / 100 % |
| SECDED | 100 % | **100 %** | **0 %** | 100 % | 0 % | 100 % | 0 % / 0 % |
| DMR | 100 % | 0 % | 100 % | 100 % | 0 % | 100 % | 0 % / 0 % |

- The rates are identical for every format and layer (36,916 codewords per cell) because the behaviour of a code depends on the error pattern, not on the weight values. The sample size matters only for confirming the implementation.
- SECDED never miscorrected a double error (0 %) and flagged every double error as uncorrectable. Parity misses every double error silently. DMR detects both but cannot say which copy is right.
- The INT scale words (106 words) behave exactly like payload words.
- Residual word error rates are binary here (a word is wrong or it is not). The residual bits per word are 1 for a single fault and 2 for a double fault, except SECDED single (0).

### Supplementary: faults may hit the redundancy bits

| Mechanism | Scenario | Detection | Correction | Residual word error | Silent corruption | Mean residual data bits |
|---|---|---:|---:|---:|---:|---:|
| Parity | 1 bit anywhere (33) | 100 % | 0 % | 97.0 % | 0 % | 0.97 |
| SECDED | 1 bit anywhere (39) | 100 % | 100 % | 0 % | 0 % | 0.00 |
| Parity | 2 bits anywhere | 0 % | 0 % | 100 % | 100 % | 1.94 |
| SECDED | 2 bits anywhere | 100 % | 0 % | 97.2 % | 0 % | 1.64 |

A single flip of the parity bit is detected but leaves the data intact (3 % of the draws), and a SECDED double fault that lands on two check bits also leaves the data intact (about 2.8 % of the draws). SECDED corrects a flipped check bit as well, so its single-error guarantee holds for the whole stored word.

## 4. Storage overhead (actual bytes)

Overhead uses the real byte counts: payload padded to whole 32-bit words, the protected float32 scale word, redundancy bits packed densely and rounded up to whole bytes per layer. FP32 weight storage is 8,810,240 bytes.

| Format | Unprotected | + Parity | + SECDED | + DMR |
|---|---:|---:|---:|---:|
| FP16 / BF16 / E4M11-SN | 4,405,120 | 4,542,780 (+3.125 %) | 5,368,740 (+21.875 %) | 8,810,240 (+100 %) |
| FP8 E4M3FN / E5M2 | 2,202,560 | 2,271,391 (+3.125 %) | 2,684,371 (+21.875 %) | 4,405,120 (+100 %) |
| INT8 | 2,202,772 | 2,271,654 (+3.127 %) | 2,684,636 (+21.875 %) | 4,405,544 (+100 %) |
| INT4 | 1,101,492 | 1,135,958 (+3.129 %) | 1,342,452 (+21.876 %) | 2,202,984 (+100 %) |

The nominal overheads (3.125 %, 21.875 %, 100 %) are matched exactly for the 16-bit formats and almost exactly for the others: the redundancy-bit rounding adds no bytes there, and the INT rows differ only through the 212 bytes of scale words and the 4-byte alignment of small layers. Relative to FP32, INT4 + SECDED is 15.2 % of the FP32 bytes and FP16 + SECDED is 60.9 %. The per-layer table `protection_storage.csv` (1,484 layer × format × strategy rows) is the input for an allocation step.

## 5. Interpretation

- Parity is cheap but only detects an odd number of flips and corrects nothing.
- SECDED is the only mechanism here that removes a single-bit error, at 21.875 % extra bytes. Against a double error it only detects.
- DMR detects both single and double errors but cannot correct them without a third copy or additional information, and costs 100 %.
- With a binary residual-error measure SECDED is always the lowest-risk strategy for single faults, so an optimizer that weighs only residual risk will pick SECDED for every layer unless the storage budget binds. Differences between layers would only appear if the residual error is weighted by its effect on the model.

## 6. Artifacts (`stage_05_protection/`)

| File | Purpose |
|---|---|
| `protection_raw_records.csv.gz` | ECC raw records: one row per codeword × scenario × strategy (word, byte offset, flipped bits, element/bit/field of each flip, outcome flags, seed) |
| `protection_summary.csv` | per format and pooled ("ALL"), scenario, strategy, region: rates and overhead |
| `protection_layer_summary.csv` | the same per layer |
| `protection_storage.csv`, `protection_storage_summary.csv` | bytes per layer / per format and strategy |
| `fault_protocol.json` | codeword definition, parity placement, partial-word handling, scale protection, sampling, reaction policy, map of the protocol's required artifact names to the files |
| `ecc_validation.json`, `protection_report.json` | code self-tests and layout cross-check; run report with file hashes |

## 7. Scope and Limitations

1. Model accuracy under protected faults was not re-evaluated; Stage 4 measured accuracy for the unprotected store only. The codeword results say whether data is repaired, not what a residual error costs the network.
2. These are sampled software tests on stored words (100 codewords per layer), not an exhaustive hardware reliability measurement.
3. The primary scenarios inject only data-bit faults, as in the shared protocol. Redundancy-bit faults are covered only by the supplementary Parity and SECDED scenarios; DMR copy B and the DMR comparator are not faulted.
4. Triple and higher-order errors were not part of the campaign. The code self-test shows that SECDED never accepts a triple error silently, but it miscorrects most of them as a single error (6,332 of 9,139 patterns).
5. Storage accounting covers the stored bytes only. Decoder logic, controller, alignment to a memory interface and the protection of any other metadata are not included.
6. The decoders are not part of an inference kernel; weights are still decoded to FP32 for evaluation.
