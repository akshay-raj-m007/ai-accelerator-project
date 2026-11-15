# MobileNetV2 / CIFAR-10: Stage 6, joint fault-aware datatype-protection optimisation (results)

Structure follows the 6A-6D write-up of the shared protocol run. Everything below is read from the committed result files; nothing was re-run for this document.
Result folders: `stage_05_protection/` (6A), `stage_06_fisher/` (6B-6D, main result), `stage_06_optimization/` (comparison variant and the scale-word campaign), `figures/` (plots).

## Final result in one table
| | FP32 | **Final assignment A000 (Fisher Q)** |
|---|---:|---:|
| Test accuracy (10,000 images, read once for this variant) | 91.69 % | **91.62 %** (drop 0.07 pp) |
| Validation accuracy (2,952 images not used by the Fisher) | 91.87 % | 91.80 % (drop 0.07 pp) |
| Weight storage (53 Conv/Linear tensors, 2,202,560 weights) | 8,810,240 B (32 bit/weight) | **1,650,940 B (5.996 bit/weight)** incl. scales and protection; 81.3 % smaller |
| 8 random weight-bit flips, validation (300 trials, 1,000 images) | - | mean drop 0.009 pp, p90 0.0, worst 0.3 pp, 0 failures |
| 8 random stored-bit flips incl. check bits and copies (300 trials) | - | mean 0.010 pp, worst 0.2 pp, 0 failures |
| 1 scale-word bit flip (300 trials) | - | mean 0.0 pp, worst 0.0 pp |
| 8 random weight-bit flips, **test** (100 trials, 10,000 images) | - | mean -0.015 pp, worst 0.01 pp |
| 1 scale-word bit flip, **test** (100 trials) | - | 0.0 pp |
| Restoration | - | byte-exact after every trial |

Constraints (declared before the run, the ResNet-18 optimiser's agreed values; a protocol continuation, not an exact numeric reproduction): B = 8 bit/weight (quantized Conv/Linear weights only),
delta_clean = 0.5 pp, tau_fault = 0.5 pp (8 random weight bits), tau_scale = 1.0 pp, lambda = 1 with min-max normalisation. All four are met with margin.

## 6A - Protection profiling (protocol Section 5.3; `stage_05_protection/`)
Four mechanisms on the serialized little-endian payload of the frozen encodings, at 32-bit codeword level, for all seven formats: **none**, **parity** (1 even-parity bit per word, 33 stored bits),
**SECDED** (extended Hamming(39,32), 39 stored bits), **DMR** (two copies, detect-only). The float32 scale word of an INT layer is one more protected word. 100 sampled codewords per layer and scenario
(36,916 codewords per scenario and strategy over 53 layers x 7 formats). Detected-but-uncorrected words are consumed as read, so the error stays residual.

| scenario | strategy | detected | corrected | residual word error | miscorrected |
|---|---|---:|---:|---:|---:|
| 1 data bit | none | 0 | 0 | 100 % | 0 |
| 1 data bit | parity | 100 % | 0 | 100 % | 0 |
| 1 data bit | SECDED | 100 % | **100 %** | **0** | 0 |
| 1 data bit | DMR | 100 % | 0 | 100 % | 0 |
| 2 data bits | parity | 0 | 0 | 100 % | 0 |
| 2 data bits | SECDED | 100 % | 0 (detected) | 100 % | 0 |
| 2 data bits | DMR | 100 % | 0 | 100 % | 0 |

Actual storage overhead (measured bytes, not nominal): parity 3.1 %, SECDED 21.9 %, DMR 100 %. Plot: `figures/stage5_protection/mobilenetv2_cifar10_s5_1_protection_outcomes_and_overhead.png`.
Additional option implemented for INT layers: **scale TMR** (the scale stored 3x, bit-wise majority on read; +8 bytes per layer). It protects the scale word only, not the integer codes.

Scale-word campaign (`stage_06_optimization/scale_faults/`, all 32 bits of every INT layer scale): every one of the 106 INT layer/format cases has a mean drop above 1 pp (mean over layers 12.0 pp for INT8,
11.3 pp for INT4). INT layers are therefore inadmissible without SECDED or scale TMR (`figures/stage4/mobilenetv2_cifar10_3_global_screening_scales.png`).

## 6B - Joint candidate generation (`stage_06_fisher/jfamp_candidate_table.csv`)
Candidates = layer x datatype x protection: 53 layers x 7 datatypes x {none, parity, SECDED, DMR} = 1,484, plus scale TMR for the INT formats (2 x 53 = 106) = **1,590 candidates** (30 options per layer).
Per candidate:
- **Q** (`Q_fisher_val_exact`) = sum_i F_i (w_i - w_hat_i)^2, F = exact-expectation diagonal Fisher of the FP32 model on validation images 0..2047 (20,479 backward passes), w_hat = decoded frozen Stage 2 encoding.
  This is the ResNet-18 optimiser's primary Q. The measured one-layer-at-a-time CE change of Stage 3 is kept in the table (`Q_ce_loss_change`) and was the Q of the comparison variant.
- **R** = 8 x (n_layer / N) x per-flip drop (Stage 4 layer-wise group drops) x the measured single-bit residual rate of the protection (none 1, parity 1, DMR 1, scale TMR 1, SECDED 0). A separate
  **R_scale** (scale-word risk, never merged into R); options with R_scale > tau_scale are inadmissible.
- **S** = actual stored bytes (payload + scales + alignment + redundancy), reconciled against the assembled model for every assignment.

Fisher Q against the measured one-layer-at-a-time loss change (Spearman over 53 layers): INT4 +0.84, E4M11 +0.61, FP8 E5M2 +0.55, FP16 +0.44, BF16 +0.42, FP8 E4M3FN +0.40, INT8 +0.12. Sum of 0.5 Q for INT4 0.334 vs measured
0.249 (`figures/stage6/mobilenetv2_cifar10_s6_5_Q_validation_predicted_vs_measured.png`). Only INT4 carries a clear signal; the 16-bit formats and INT8 are below the noise of a 5,000-image evaluation (the same finding as for ResNet-18).

## 6C - Constrained joint optimisation (`stage_06_fisher/`)
Binary x[l,d,p], exactly one option per layer, minimise sum x (Qn + lambda Rn) (global min-max normalisation) subject to sum x S <= B and the predicted sum of R <= tau_fault; options with R_scale > tau_scale
excluded. Solved exactly with `scipy.optimize.milp` (HiGHS). Sweep: 8 lambdas x 9 budgets x 5 tau x 2 normalisations x 2 modes = 1,440 problems, all optimal, **201 distinct assignments**.
The assignments selected at the declared point and at the grid points shown below protect all 53 layers: unprotected INT is inadmissible (scale-word limit), and protected options are not penalised enough by the objective to be dropped. At the declared point the ILP optimum on the
layer-additive estimates is A002 (7.98 bit/weight); see "Selection and its limits".

## 6D - Final model reconstruction and validation (`stage_06_fisher/`)
All 201 assignments plus 11 uniform references were assembled (each layer decoded from its own frozen encoding, protection applied), stored bytes checked against the table, and evaluated on validation
images 2,048-4,999 (2,952 images, disjoint from the Fisher images). 33 assignments (Pareto shortlist plus references) got 100 fault trials per distribution on a class-stratified 1,000-image subset.
The constraint grid (168 settings of delta_clean, tau_fault, B, tau_scale) selected, among assignments that pass the MEASURED limits, the lowest objective: **A000** at the declared point (10 feasible).
Other grid points (delta 0.5, tau 0.5, tau_scale 1): B = 5 bit -> A007 (5.00 bit, clean drop 0.24 pp); B = 4.8 bit -> A006 (4.80 bit, 0.14 pp).
The assignment was frozen (`assignment.json`) before the test set was read; then 300 more validation trials per distribution and one test evaluation (`final_test_evaluation.json`).
Predicted vs measured: weight-fault drop Spearman +0.62 (mean predicted 0.93 pp, measured 1.01 pp over the 33); clean accuracy loss +0.51; clean loss change only +0.10 (`figures/stage6/mobilenetv2_cifar10_s6_8_predicted_vs_measured_assignments.png`).

### Final allocation (A000)
Datatype mix: 23 layers E4M11-SN + SECDED, 24 layers INT8 + scale TMR, 6 layers INT4 + scale TMR. Payload 1,616,000 B, scales 120 B, redundancy 34,820 B.

| # | layer | weights | datatype | protection | stored bytes |
|--:|---|--:|---|---|--:|
| 0 | features.0.0 | 864 | E4M11-SN | SECDED | 2,106 |
| 1 | features.1.conv.0.0 | 288 | E4M11-SN | SECDED | 702 |
| 2 | features.1.conv.1 | 512 | E4M11-SN | SECDED | 1,248 |
| 3 | features.2.conv.0.0 | 1,536 | E4M11-SN | SECDED | 3,744 |
| 4 | features.2.conv.1.0 | 864 | E4M11-SN | SECDED | 2,106 |
| 5 | features.2.conv.2 | 2,304 | E4M11-SN | SECDED | 5,616 |
| 6 | features.3.conv.0.0 | 3,456 | E4M11-SN | SECDED | 8,424 |
| 7 | features.3.conv.1.0 | 1,296 | E4M11-SN | SECDED | 3,159 |
| 8 | features.3.conv.2 | 3,456 | E4M11-SN | SECDED | 8,424 |
| 9 | features.4.conv.0.0 | 3,456 | E4M11-SN | SECDED | 8,424 |
| 10 | features.4.conv.1.0 | 1,296 | E4M11-SN | SECDED | 3,159 |
| 11 | features.4.conv.2 | 4,608 | E4M11-SN | SECDED | 11,232 |
| 12 | features.5.conv.0.0 | 6,144 | E4M11-SN | SECDED | 14,976 |
| 13 | features.5.conv.1.0 | 1,728 | E4M11-SN | SECDED | 4,212 |
| 14 | features.5.conv.2 | 6,144 | E4M11-SN | SECDED | 14,976 |
| 15 | features.6.conv.0.0 | 6,144 | E4M11-SN | SECDED | 14,976 |
| 16 | features.6.conv.1.0 | 1,728 | E4M11-SN | SECDED | 4,212 |
| 17 | features.6.conv.2 | 6,144 | E4M11-SN | SECDED | 14,976 |
| 18 | features.7.conv.0.0 | 6,144 | E4M11-SN | SECDED | 14,976 |
| 19 | features.7.conv.1.0 | 1,728 | E4M11-SN | SECDED | 4,212 |
| 20 | features.7.conv.2 | 12,288 | E4M11-SN | SECDED | 29,952 |
| 21 | features.8.conv.0.0 | 24,576 | INT8 | scale TMR | 24,588 |
| 22 | features.8.conv.1.0 | 3,456 | E4M11-SN | SECDED | 8,424 |
| 23 | features.8.conv.2 | 24,576 | INT8 | scale TMR | 24,588 |
| 24 | features.9.conv.0.0 | 24,576 | INT8 | scale TMR | 24,588 |
| 25 | features.9.conv.1.0 | 3,456 | INT8 | scale TMR | 3,468 |
| 26 | features.9.conv.2 | 24,576 | INT8 | scale TMR | 24,588 |
| 27 | features.10.conv.0.0 | 24,576 | INT8 | scale TMR | 24,588 |
| 28 | features.10.conv.1.0 | 3,456 | INT8 | scale TMR | 3,468 |
| 29 | features.10.conv.2 | 24,576 | INT8 | scale TMR | 24,588 |
| 30 | features.11.conv.0.0 | 24,576 | INT8 | scale TMR | 24,588 |
| 31 | features.11.conv.1.0 | 3,456 | E4M11-SN | SECDED | 8,424 |
| 32 | features.11.conv.2 | 36,864 | INT8 | scale TMR | 36,876 |
| 33 | features.12.conv.0.0 | 55,296 | INT8 | scale TMR | 55,308 |
| 34 | features.12.conv.1.0 | 5,184 | INT8 | scale TMR | 5,196 |
| 35 | features.12.conv.2 | 55,296 | INT8 | scale TMR | 55,308 |
| 36 | features.13.conv.0.0 | 55,296 | INT8 | scale TMR | 55,308 |
| 37 | features.13.conv.1.0 | 5,184 | INT8 | scale TMR | 5,196 |
| 38 | features.13.conv.2 | 55,296 | INT8 | scale TMR | 55,308 |
| 39 | features.14.conv.0.0 | 55,296 | INT8 | scale TMR | 55,308 |
| 40 | features.14.conv.1.0 | 5,184 | INT8 | scale TMR | 5,196 |
| 41 | features.14.conv.2 | 92,160 | INT8 | scale TMR | 92,172 |
| 42 | features.15.conv.0.0 | 153,600 | INT4 | scale TMR | 76,812 |
| 43 | features.15.conv.1.0 | 8,640 | INT8 | scale TMR | 8,652 |
| 44 | features.15.conv.2 | 153,600 | INT4 | scale TMR | 76,812 |
| 45 | features.16.conv.0.0 | 153,600 | INT4 | scale TMR | 76,812 |
| 46 | features.16.conv.1.0 | 8,640 | INT8 | scale TMR | 8,652 |
| 47 | features.16.conv.2 | 153,600 | INT4 | scale TMR | 76,812 |
| 48 | features.17.conv.0.0 | 153,600 | INT8 | scale TMR | 153,612 |
| 49 | features.17.conv.1.0 | 8,640 | INT8 | scale TMR | 8,652 |
| 50 | features.17.conv.2 | 307,200 | INT4 | scale TMR | 153,612 |
| 51 | features.18.0 | 409,600 | INT4 | scale TMR | 204,812 |
| 52 | classifier.1 | 12,800 | INT8 | scale TMR | 12,812 |

## Comparison with the unassigned (uniform, unprotected) models
Every layer in one format, no protection, same assembled-model protocol (validation images 2,048-4,999 for clean accuracy; 100 trials on the 1,000-image subset). FP32 = 91.87 %.

| model | bit/weight | stored bytes | clean val. acc. | drop vs FP32 | 8 weight-bit flips, mean drop | 1 scale-bit flip, mean drop |
|---|--:|--:|--:|--:|--:|--:|
| FP32 | 32 | 8,810,240 | 91.87 % | - | - | - |
| FP16 | 16 | 4,405,120 | 91.87 % | 0.00 pp | 4.52 pp | - |
| BF16 | 16 | 4,405,120 | 91.77 % | 0.10 pp | 16.25 pp | - |
| E4M11-SN | 16 | 4,405,120 | 91.87 % | 0.00 pp | 0.83 pp | - |
| FP8 E4M3FN | 8 | 2,202,560 | 91.73 % | 0.14 pp | 0.98 pp | - |
| FP8 E5M2 | 8 | 2,202,560 | 90.65 % | 1.22 pp | 8.47 pp | - |
| INT8 | 8.00 | 2,202,772 | 91.87 % | 0.00 pp | 0.01 pp | **13.6 pp** |
| INT4 | 4.00 | 1,101,492 | 83.88 % | 7.99 pp | 0.08 pp | **8.6 pp** |
| *references with protection* | | | | | | |
| INT8 + scale TMR | 8.00 | 2,203,196 | 91.87 % | 0.00 pp | -0.002 pp | 0.0 pp |
| INT4 + scale TMR | 4.00 | 1,101,916 | 83.88 % | 7.99 pp | 0.06 pp | 0.0 pp |
| INT8 + SECDED | 9.75 | 2,684,636 | 91.87 % | 0.00 pp | 0.0 pp | 0.0 pp |
| FP8 E4M3FN + SECDED | 9.75 | 2,684,371 | 91.73 % | 0.14 pp | 0.0 pp | - |
| **A000 (assigned, all protected)** | **6.00** | **1,650,940** | **91.80 %** | **0.07 pp** | **0.005 pp** | **0.0 pp** |

Reading: no single uniform format passes everything. FP16, BF16, FP8 and E4M11-SN break the 0.5 pp fault limit; unprotected INT breaks the scale-word limit; INT4 breaks the clean limit; the protected uniform
INT8 / FP8 references pass but cost 8.0-9.75 bit/weight, 33-63 % more than A000. A000 is smaller than every uniform model that meets all limits (`figures/stage6/mobilenetv2_cifar10_s6_11_assigned_vs_uniform.png`).
The official test accuracy of the uniform formats was not measured (`stage2_quantize.py --stage test` has not been run for MobileNetV2), so this comparison is on validation data only.

## Comparison variant: measured Q (`stage_06_optimization/`)
The first complete run used the measured one-layer-at-a-time CE change as Q (all 5,000 validation images for the clean check). Its frozen assignment (also called A000 in its own folder; a different assignment) used 5.854 bit/weight,
clean validation drop 0.04 pp, **test 91.25 % (drop 0.44 pp)**, fault drops about 0. Allocation: 18 INT8, 16 INT4, 8 FP8 E5M2, 6 FP8 E4M3FN, 4 E4M11-SN, 1 FP16, all 53 layers protected (git tag `mobilenetv2-measuredQ-stage6`).
The two variants use different validation images, so their validation numbers are not directly comparable. The Fisher variant is reported as the main result because it is the ResNet-18 method;
this choice does not rest on the test numbers.

## Selection and its limits (read before quoting the numbers)
- **Test set read twice.** Once per variant, each after its assignment was frozen from validation data. Neither variant was tuned on test results.
- **The ILP optimum was not the selected assignment.** At the declared point the ILP's own optimum (A002, 7.98 bit/weight, objective 0.0013) was measured clean (drop -0.03 pp) but is not on the fault-trial
  shortlist, so it is not eligible under the rule "lowest objective among assignments that pass the MEASURED limits". A000 (objective 0.0096) is the lowest-objective assignment with complete fault trials.
  Whether A002 would pass the fault limits was not measured.
- **Estimates are approximate.** Layer-additive Q and R predict the assembled models only moderately (Spearman 0.10-0.62 above); decisions use the measured end-to-end values.
- **Statistical depth.** Fault trials: 100 per distribution during selection, 300 for the frozen assignment on validation, 100 on test; weight-fault means below about 0.05 pp cannot be ranked and rare tail events
  are under-resolved. The fault subset (1,000 images) is a subset of the images used for the clean check.
- **Constants.** B, delta_clean, tau_fault, tau_scale and lambda are the ResNet-18 optimiser's code defaults; not an exact numeric reproduction and not necessarily the values another CIFAR-100 run uses.
- **Protection scope.** Parity, SECDED, DMR operate at 32-bit codeword level on the stored payload, simulated by decoder logic (redundancy bits are not stored arrays); scale TMR protects only the scale word.
  Two flips in one SECDED word are detected, not corrected. Software simulation only: inference decodes to FP32; no claim about hardware fault rates, area, timing or energy.
- **Scope.** One checkpoint (FP32 92.16 % validation / 91.69 % test), one seed, weights only (biases, BatchNorm and activations stay FP32).

## Figures (`figures/`, produced by `scripts/plot_mobilenetv2_summary.py` from the committed CSV/JSON files)
Same design as the ResNet-18 figures (`plot_stage4_summary.py` look for `stage4/`, `plot_stage5_summary.py` look for `stage5_protection/` and `stage6/`; BF16 and FP8 E5M2, which ResNet-18 did not have, take the next colours of the same palettes).
- `stage4/`: 1 format overview, 2 global screening (weights), 3 global screening (scale words), 4 layer heatmaps (53 layers, value in every cell), 5 quality vs robustness.
- `stage5_protection/`: s5_1 protection outcomes and overhead.
- `stage6/`: s6_1 overview, s6_2 layer sensitivity profile, s6_3 Q heatmap, s6_4 R heatmaps (weights and scale words), s6_5 Q predicted vs measured, s6_6 selected assignments per layer,
  s6_7 storage / accuracy / fault trade-off, s6_8 predicted vs measured assignments, s6_9 fault-trial distributions of A000, s6_10 lambda sweep, s6_11 assigned vs uniform, s6_12 test accuracy of both variants.
Differences that follow from the data: Stage 4 of MobileNetV2 used the protocol's sign / exponent / fraction groups (30 global, 10 layer-wise trials) and has no "any bit" group, so figures 2, 4 and 5 show the groups
(figure 5 pools them bit-weighted); the R heatmap shows 8 x the group-weighted per-flip drop. Not reproduced from the ResNet-18 set: weight / activation / logit distributions, activation growth and feature-map figures
(they need per-trial tensors that the MobileNetV2 runs did not save) and the datatype-only vs protected pairs figure.
