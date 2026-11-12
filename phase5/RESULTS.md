# Phase 5 results - J-FAMP datatype-only optimisation, end-to-end check (2026-11-12)

**DATATYPE-ONLY (protection `p = none`; nothing implemented).** Files: `reports/scratch/jfamp/optimization/`. Q = exact-expectation Fisher on validation images 0..2047 (primary).
Clean metrics: validation images 2,048..4,999 (2,952 images, FP32 95.29 %). Fault trials: first 1,000 validation images, 50 trials per distribution (`weights_k8`, `scales_k1`),
27 assignments (22 from the optimiser's two non-dominated fronts + 5 uniform references), all 2,350 trials restored byte-exactly, no incomplete cells. 256 assignments measured for clean metrics.
Nothing here uses the test set. Layer-additive scores are approximations; every number below is a measurement of the assembled model.

## 1. Clean accuracy: the Q-based optimisation works
- Predicted sum Q vs measured clean accuracy drop over the 256 assembled assignments: Spearman +0.87.
- Measured clean drop of the 256 assignments: -0.03 .. +1.22 pp (median +0.20). Uniform: E4M11-SN 0.00, FP16 0.00, FP8 +0.24, INT8 +0.17, INT4 +1.22 pp.
- Cross-entropy is not a usable check (Spearman of predicted vs measured dCE = -0.45): INT quantization lowers the cross-entropy of late layers (calibration, see Q_ANALYSIS.md).

## 2. Fault risk, weights only (`weights_k8`: 8 distinct weights over the whole model, bit uniform over the weight's own encoding)
| model | avg bits | clean drop (pp) | mean drop (pp) | p90 | worst |
|---|---|---|---|---|---|
| uniform E4M11-SN | 16 | 0.00 | 0.012 | 0.0 | 0.6 |
| uniform FP16 | 16 | 0.00 | **4.82** (SE 2.5) | 6.7 | **89.0** |
| uniform FP8 E4M3FN | 8 | +0.24 | -0.03 | 0.0 | 0.3 |
| uniform INT8 | 8 | +0.17 | 0.000 | 0.0 | 0.0 |
| uniform INT4 | 4 | +1.22 | 0.014 | 0.0 | 0.2 |
| A044 (17 INT8, 2 FP8, 1 E4M11, 1 INT4) | 7.58 | +0.14 | 0.016 | 0.1 | 0.1 |
Weight-bit faults are harmless for every format except FP16 (exponent flips; one trial lost 89 pp) and rare tail events in a few mixed assignments (worst 4.7 - 15 pp in A018, A029, A063, A194, each in 2 - 6 % of trials).
With 50 trials most means are indistinguishable from 0 (SE 0.002 - 0.36 pp); these are screening results.

## 3. INT scale words (`scales_k1`, extension, one float32 scale, one of its 32 bits)
Every assignment with INT layers loses on average 8 - 30 pp per flipped scale word (uniform INT8 15.8, INT4 21.4; SE 3 - 6 pp; bimodal: most flips are harmless, exponent flips collapse the model, failure rate 2 - 6 %).
This is the dominant weakness of every INT assignment and it is not visible in `weights_k8`. It concerns 21 words of 4 bytes (84 bytes, 0.0015 % of the storage).
Whether it matters depends on the fault model: under uniform random flips over all stored bits a scale word is hit with probability about 6e-5 per 8-flip trial, under targeted or structure-aware faults it is the critical word.
This is exactly where a protection option would apply (nothing implemented yet).

## 4. Predicted (layer-additive) vs measured
| quantity | Spearman | note |
|---|---|---|
| predicted sum Q vs measured clean drop (256) | +0.87 | usable |
| optimiser's mean layer risk vs measured `weights_k8` drop (27) | +0.38 | mismatch of definition, see below |
| exposure-weighted per-flip estimate `8 sum_l (n_l/N) R_k1(l,d)` vs measured (27) | +0.54 | same magnitude as the measurement where it is non-trivial: A018 0.23 vs 0.23, A029 0.21 vs 0.26, A063 0.22 vs 0.18, FP16 9.5 vs 4.8 |
| predicted per-INT-layer scale risk vs measured `scales_k1` (20) | +0.09 | mean absolute difference 4.6 pp; Stage 4 per-layer scale risks come from 20 bimodal trials (SE about 10 pp each) and do not discriminate layers |
The optimiser's R is the unweighted mean over layers of a single-layer k = 8 experiment (8 flips concentrated in one layer). The assembled model receives its 8 flips spread over all weights in proportion to layer size, so the
proxy overstates tiny risks by 5 - 6 x (e.g. 1.4 - 1.5 predicted vs 0.0 - 0.3 measured for FP8/E4M11 mixes) and ranks them by the wrong exposure. Per runbook 7.2 the layer estimates should be updated and the optimisation repeated.
Proposed update (not done yet, needs your approval): `R_exposure(l,d) = 8 (n_l/N) R_k1(l,d)` (additive over layers, same unit as the end-to-end metric).

## 5. Constraint grid and the provisional selection
`constraint_grid_selection.csv` (delta_clean in {0.25, 0.5, 1.0} pp, tau_fault in {0.25 .. 5} pp, B in {5, 6, 8, 10, 12, 16} bits, metric `weights_k8`): every cell is feasible, and 60 of 90 cells select A044 (7.58 bits, clean drop +0.14 pp,
`weights_k8` 0.016 pp), 20 select A020 (4.83 bits), 10 select A048 (5.68 bits). With the weights-only fault metric the constraints do not bind, so the grid cannot choose among assignments; the ranking comes from the objective alone.
No final assignment is frozen (`assignment.json` is still the optimiser's provisional point; `--finalize` needs agreed delta_clean, tau_fault and B), and the test set has not been used.

## 6. What can and cannot be claimed
- Can: a datatype-only mixed-precision assignment (e.g. A044) stores the quantized weights at 7.58 bits per weight (47 % of E4M11-SN) with +0.14 pp clean drop and no measurable weight-bit fault penalty, measured on the assembled model.
- Can: FP16 is the least robust 16-bit option; E4M11-SN is robust at 16 bits.
- Cannot: any statement about protection; any statement that the INT assignments are fault-safe (scale-word exposure); layer-level fault rankings from 20-trial cells.
- Limits: 50 trials per cell, one checkpoint, 1,000 fault images, Fisher from 2,048 images, R from Stage 4 screening, no test-set evaluation yet.
