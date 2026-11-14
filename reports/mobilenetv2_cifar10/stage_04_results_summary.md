# JFAMP — Stage 4: Fault Injection and Fault-Robustness Characterization (MobileNetV2 / CIFAR-10)

## 1. Objective

The objective of Stage 4 was to investigate the fault tolerance of different quantized weight representations in MobileNetV2 trained on CIFAR-10 (FP32 baseline 92.16 % on the 5,000 validation images).

We studied how bit flips in the stored, encoded weight bytes affect:
- Reconstructed weight values.
- Model-level prediction changes.
- Classification accuracy.
- Sensitivity of different layers and bit fields to memory faults.

The experiments support the JFAMP objective of assigning data types and selective protection to individual layers.

## 2. Stage 4B — Quantized Weight Payload Encoding

The same seven formats as Stage 2 were used (FP16, BF16, E4M11-SN, FP8 E4M3FN, FP8 E5M2, INT8, INT4, with INT scales per tensor). Only Conv2d and Linear weights (53 layers, 2,202,560 weights) are quantized. Biases, BatchNorm and activations stay FP32. The frozen Stage 2 encodings were verified by content SHA-256 before every campaign.

| Format | Payload size |
|---|---:|
| FP16 | 4,405,120 bytes |
| BF16 | 4,405,120 bytes |
| E4M11-SN | 4,405,120 bytes |
| FP8 E4M3FN | 2,202,560 bytes |
| FP8 E5M2 | 2,202,560 bytes |
| INT8 | 2,202,560 bytes (+212 bytes of scales) |
| INT4 | 1,101,280 bytes (+212 bytes of scales) |

## 3. Stage 4C — Validation of the Encoded-Payload Injector (Section 4.3 gate)

Flips are applied to the stored words and bytes, which are then decoded. Nothing is applied to FP32 or reconstructed weights.

| Check | Result |
|---|---|
| Known-bit tests: every bit of a synthetic tensor, special values, INT4 nibble order and padding | 547 cases, all PASS |
| Round trip and stable bytes against the frozen Stage 2 hashes | PASS, all formats |
| No-fault control (max logit difference, tolerance 1e-5) | 0.0 for every format and campaign |
| Byte-for-byte restoration hash after every trial | 21,280 trials, 0 failures |
| Trial scope (layer, indices, bits, flip count regenerated from the trial seed) | PASS for every trial |
| Gate | **PASS for 7 of 7 formats** |

Bit maps: FP16 sign 15 / exponent 10–14 / fraction 0–9. BF16 sign 15 / exponent 7–14 / fraction 0–6. E4M11-SN sign 15 / exponent 11–14 / fraction 0–10. E4M3FN sign 7 / exponent 3–6 / fraction 0–2. E5M2 sign 7 / exponent 2–6 / fraction 0–1. INT8 sign 7 / value 0–6. INT4 sign 3 / value 0–2, packed two per byte (even element in the low nibble).

## 4. Stage 4D — Fault Campaigns

| Parameter | Global screening | Layer-wise screening |
|---|---:|---:|
| Scope | random weights across all 53 layers | one target layer, others clean |
| Fault groups | sign / exponent / fraction (fp), sign / value (INT) | same |
| k | 1 and 8 distinct weights | 1 and 8 |
| Independent trials per configuration | 30 | 10 |
| Configurations | 38 | 2,014 |
| Total trials | 1,140 | 20,140 |
| Evaluation images | 1,000 (class-stratified, fixed) | 1,000 (same set) |

Clean quantized accuracy on those 1,000 images: FP16, BF16, E4M11-SN and E4M3FN 92.5 %; E5M2 90.3 %; INT8 92.2 %; INT4 83.8 %. Every number below is relative to the corresponding clean quantized model.

### Format-level results (layer-wise, mean over 53 layers and all fault groups)
| Format | 1-bit faults | 8-bit faults | Exponent group only, 1-bit / 8-bit |
|---|---:|---:|---:|
| FP16 | −1.326 pp | −8.427 pp | −3.944 / −24.792 pp |
| BF16 | −1.307 pp | −7.022 pp | −3.881 / −20.664 pp |
| E4M11-SN | −0.701 pp | −4.604 pp | −2.019 / −13.475 pp |
| FP8 E4M3FN | −0.586 pp | −4.473 pp | −1.734 / −13.090 pp |
| FP8 E5M2 | −1.400 pp | −8.882 pp | −4.141 / −26.191 pp |
| INT8 | +0.016 pp | −0.206 pp | n/a (sign −0.008 / −0.462) |
| INT4 | −0.083 pp | −0.645 pp | n/a (sign −0.131 / −1.110) |

FP formats average over three groups and INT over two, so compare formats mainly through the per-group columns.

### Global screening, mean (worst) accuracy change in pp, 30 trials
| Format | Group | k = 1 | k = 8 |
|---|---|---:|---:|
| FP16 | exponent | −2.30 (−68.9) | −20.35 (−81.6) |
| FP16 | fraction / sign | 0.00 / 0.00 | −0.01 / −0.02 |
| BF16 | exponent | −3.76 (−82.5) | −26.13 (−85.2) |
| BF16 | fraction / sign | 0.00 / −0.12 (−3.5) | +0.01 / +0.02 |
| E4M11-SN | exponent | −0.02 (−0.2) | −5.11 (−84.8) |
| E4M11-SN | fraction / sign | 0.00 / −0.01 | −0.01 / −0.04 |
| FP8 E4M3FN | exponent | −1.66 (−49.8) | −0.48 (−4.5) |
| FP8 E4M3FN | fraction / sign | 0.00 / 0.00 | 0.00 / +0.01 |
| FP8 E5M2 | exponent | −0.03 (−0.5) | −19.25 (−80.1) |
| FP8 E5M2 | fraction / sign | 0.00 / 0.00 | 0.00 / −0.02 |
| INT8 | sign / value | +0.02 / +0.01 | +0.09 / +0.07 |
| INT4 | sign / value | −0.01 / +0.01 | −0.01 / −0.06 (−1.8) |

Global exponent means are heavy-tailed. For example, FP16 k=1 has a mean of −2.30 pp but a 90th percentile of 0.0 pp, because one or two of the 30 trials hit a weight and collapsed the model. The means can therefore even fall as k grows (E4M3FN exponent: −1.66 at k=1, −0.48 at k=8). These are sampling effects with 30 trials, not a real inverse trend.

### Important observations

**1. Exponent bits dominate.** In every floating-point format, almost all damage comes from the exponent group. Fraction flips cost at most about 0.03 pp on average and sign flips at most about 0.5 pp, even with 8 flips.

**2. FP16, BF16 and FP8 E5M2 are the most exposed.** Their exponent flips can inflate a small weight by a very large factor. Layer-wise, 8-bit exponent faults give mean changes of −20 to −26 pp. 28 of 53 layers exceed 1 pp for E5M2 at k=1.

**3. E4M11-SN and FP8 E4M3FN are less exposed.** Their narrow exponent range bounds the inflation. They show −13 pp (8-bit exponent) and −4.5 pp (8-bit, all groups) layer-wise.

**4. INT8 is the most tolerant.** The 8-bit mean is −0.206 pp, with only 6 of 106 layer/group configurations above 1 pp. A single sign bit in one early depthwise layer is still harmful: `features.1.conv.0.0` gives −11.8 pp mean at k=8.

**5. INT4** also shows little average damage (−0.645 pp at 8-bit). The same early layers lead its list of risky layers (`features.1.conv.0.0` sign, −13.2 pp at k=8). INT4's problem is its lower clean accuracy (83.8 % on this subset), not its fault behaviour.

**6. Non-finite outcomes were kept and flagged.** There were none in the global campaign. In the layer-wise campaign 8 trials produced non-finite activations (all BF16 8-bit exponent faults) and 1 produced a non-finite loss (BF16, `classifier.1`).

### Layer sensitivity (8-bit faults, mean accuracy drop averaged over all formats and groups)
| Most vulnerable | Mean drop | Least vulnerable | Mean drop |
|---|---:|---|---:|
| `features.2.conv.2` | 18.15 pp | `features.18.0` | 0.01 pp |
| `features.4.conv.2` | 15.04 pp | `features.16.conv.1.0` | 0.29 pp |
| `features.1.conv.1` | 14.60 pp | `features.16.conv.0.0` | 0.30 pp |
| `features.3.conv.2` | 12.80 pp | `features.17.conv.1.0` | 0.36 pp |
| `features.7.conv.2` | 12.63 pp | `features.14.conv.0.0` | 0.37 pp |

Fault sensitivity concentrates in the early blocks. The late blocks 13–18 hold about 81 % of the weights but tolerate faults well, the same pattern as the Stage 3 quantization results. The layers that are risky at k=1 also differ by format (exponent group), for example `features.2.conv.2` for FP16 and E5M2, `features.1.conv.1` for BF16 and E4M11-SN, `features.3.conv.1.0` for E4M3FN.

| Layer type (8-bit, mean drop, pp) | FP16 | BF16 | E4M11-SN | E4M3FN | E5M2 | INT8 | INT4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| depthwise (17) | 5.84 | 3.50 | 4.57 | 4.66 | 5.31 | 0.50 | 1.07 |
| pointwise (34) | 9.64 | 8.78 | 4.62 | 4.47 | 10.40 | 0.04 | 0.39 |
| stem (1) | 9.08 | 11.91 | 8.60 | 5.58 | 9.52 | 1.10 | 2.83 |
| classifier (1) | 10.63 | 2.10 | 0.57 | 0.46 | 17.34 | −0.04 | −0.01 |

### Targeted follow-up (100 trials, full 5,000 validation images)

The 35 highest-risk configurations of the layer-wise screening were repeated with 100 independent trials each on all 5,000 validation images (3,500 trials; no-fault control exact for every format; 0 restoration failures; every trial scope-verified). The shortlist rule was the top 5 configurations per format by layer-wise mean accuracy drop with a drop of at least 0.5 pp. **The configurations were chosen from the screening results, so these estimates are biased toward high risk and must not be read as format averages.** All 35 shortlisted configurations turned out to be k = 8 (25 exponent, 10 sign); k = 1 configurations were not followed up.

| Format | Configs | Screening mean (10 trials) | Targeted mean (100 trials) | Trials with a drop above 60 pp | Typical per-config SEM |
|---|---:|---:|---:|---:|---:|
| FP16 (exponent) | 5 | 68.99 pp | 58.81 pp | 358 / 500 | 3.6 pp |
| BF16 (exponent) | 5 | 59.85 pp | 50.26 pp | 305 / 500 | 4.0 pp |
| E4M11-SN (exponent) | 5 | 48.81 pp | 43.38 pp | 228 / 500 | 3.1 pp |
| FP8 E4M3FN (exponent) | 5 | 48.83 pp | 46.57 pp | 250 / 500 | 3.1 pp |
| FP8 E5M2 (exponent) | 5 | 68.68 pp | 58.10 pp | 354 / 500 | 3.5 pp |
| INT8 (sign) | 5 | 5.10 pp | 6.55 pp | 0 / 500 | 0.5 pp |
| INT4 (sign) | 5 | 8.21 pp | 9.60 pp | 0 / 500 | 0.7 pp |

Observations:
1. **Winner's curse is visible.** For 24 of the 35 configurations the 10-trial screening mean is above the 100-trial mean (average 44.1 pp against 39.0 pp), mostly for the floating-point exponent groups. This is the expected regression toward the mean after selecting the worst screening results. The INT configurations moved slightly the other way (5.1 to 6.6 pp and 8.2 to 9.6 pp), so their screening was not over-pessimistic.
2. **Floating-point exponent faults are bimodal.** A trial either collapses the model to near chance (accuracy about 10 %, a drop of about 82 pp) or does little. The median and 90th percentile sit at the collapse level for most FP16, BF16 and E5M2 configurations (median about 80 pp, p90 about 82 pp), while the mean is near 50 pp. Collapse happened in 72 % (FP16), 71 % (E5M2), 61 % (BF16), 50 % (E4M3FN) and 46 % (E4M11-SN) of the trials.
3. **Layer matters even inside the shortlist.** For E4M11-SN, `features.4.conv.0.0` has a median drop of only 7.8 pp (mean 24.9 pp) while `features.1.conv.1` and `features.2.conv.2` have medians of 79.6 and 75.1 pp. For FP8 E4M3FN the same layer has a median of 22.9 pp (mean 33.1 pp) against 79.0 pp for `features.2.conv.2`. The FP16, BF16 and E5M2 shortlist entries all have medians of 79 to 82 pp.
4. **INT sign faults are graded, not collapsing.** The worst INT8 configuration is `features.1.conv.0.0` (16.4 pp mean, 59.3 pp worst); the worst INT4 configuration is the same layer (20.4 pp mean, 51.7 pp worst). No INT trial dropped by more than 60 pp.
5. **Non-finite outcomes:** 16 of the 3,500 trials produced non-finite activations, all BF16 8-bit exponent faults (9 in `features.1.conv.1`, 6 in `features.2.conv.2`, 1 in `features.4.conv.2`); 1 of them also gave non-finite logits and loss. They are kept as failed trials (no weights were non-finite).

## 5. Measurements and Metrics

Per trial: accuracy and accuracy change relative to the clean quantized model, loss, changed predictions and prediction agreement, non-finite flags (weights, activations, logits, loss), fault group, k, layer, the exact (layer, element, bit) flips, trial seed and restoration check. Per-configuration summaries give mean, median, standard deviation, 90th percentile, worst case and failure counts.

Not recorded in this campaign: correct-to-wrong and wrong-to-correct counts, and reconstruction MSE of the faulty weights. Only the net count of changed predictions was kept.

## 6. Overall Contribution of Stage 4

1. A fault-injection pipeline works directly on the stored encoded bytes of all seven formats and passes the validation gate.
2. 24,780 trials were completed (21,280 screening and validation trials plus 3,500 targeted trials) with no restoration failure and no dropped trials.
3. Sensitivity depends strongly on format and bit field. Exponent bits of FP16, BF16 and E5M2 are the dominant risk, while INT formats and fraction/sign bits are tolerant.
4. Sensitivity also depends on layer, with the early blocks most exposed.
5. Clean quantization quality does not predict fault tolerance. E5M2 and FP16 are close on clean accuracy but behave very differently under exponent faults.

## 7. Scope and Limitations

- Screening used a fixed 1,000-image validation subset, so one image is 0.1 pp. The targeted 100-trial follow-up used all 5,000 images but only the 35 configurations selected from the screening results (all k = 8), so it is biased toward high risk and does not cover k = 1.
- Layer-wise results use 10 trials per configuration. Heavy-tailed outcomes make single configurations noisy, and only aggregates should be compared.
- The INT per-tensor scale words were not faulted. A flipped scale would affect a whole layer and is a separate risk.
- ECC and parity were not evaluated. They need the codeword layout, scale protection and reaction policy from the shared protocol.
- The experiments are software fault injection with weights decoded to FP32. No native low-precision hardware was used and no area, timing or energy was measured.
