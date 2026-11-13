# Phase 5 results, round 2 (2026-11-13): declared fault distribution, exposure-weighted R, measured protection `scale_tmr`

Data: `reports/scratch/jfamp/optimization/` (GPU run of 2026-11-12). Decisions that define the setting: [DECISIONS.md](DECISIONS.md). Round 1: [RESULTS.md](RESULTS.md).
All numbers are VALIDATION data (clean: images 2,048-4,999 = 2,952 images; faults: first 1,000 images, 100 trials per distribution). The test set has not been read. No assignment is frozen yet.

## 1. The protection `scale_tmr` works as designed (measured)
| check | result |
|---|---|
| stored-byte check, every single flipped bit (any copy, any of 32 bits, 21 layers, INT8 and INT4) | 2,016 / 2,016 corrected for each format |
| two flips in two different copies | 62,496 / 64,512 corrected; the 2,016 uncorrected are exactly the same-bit-position cases (documented limit) |
| layer-wise measured trials (20 per layer and format, one flipped scale bit, 1,000 images) | protected: mean 0.000 pp, worst 0.000 pp in all 42 cells; paired unprotected control: INT8 12.9 pp, INT4 15.1 pp mean |
| stored cost | +8 bytes per protected layer (168 bytes for 21 layers); the end-to-end byte count agrees with the candidate table |

## 2. Clean accuracy (198 assembled assignments, FP32 = 95.29 % on the same 2,952 images)
- Predicted sum Q vs measured clean accuracy drop: **Spearman +0.91** (round 1: +0.87). Q ranks assignments by their accuracy loss.
- Predicted clean cross-entropy change vs measured: Spearman -0.47, i.e. NOT usable. Many assignments *lower* the cross-entropy a little (calibration effect, see Q_ANALYSIS.md), and
  for the five uniform formats the rank correlation is exactly 0.000 (E4M11, FP16, INT8, FP8 all sit at the numerical noise floor, only INT4 is separated). The printed
  "Spearman +0.000" in the candidate-table step is therefore correct, not a bug. Statement to use: Q is a good ranking of accuracy loss, not an absolute prediction of loss.

## 3. Fault results on 31 assembled assignments (`weights_k8` primary, `scales_k1` second)
Selected rows (full table: `end_to_end_assignments.csv`). "Protected layers" = INT layers with `scale_tmr`.
| assignment | avg bits | protected layers | clean drop (pp) | `weights_k8` mean / p90 / worst (pp) | `scales_k1` mean / worst (pp) |
|---|---|---|---|---|---|
| uniform INT4 | 4.00 | 0 | 1.22 | 0.010 / 0.0 / 0.2 | 24.6 / 95.4 |
| uniform INT4 + TMR | 4.00 | 21 | 1.22 | 0.003 / 0.0 / 0.1 | **0.0 / 0.0** |
| A156 (16 INT8+T, 5 INT4+T) | 4.83 | 21 | **0.24** | 0.002 / 0.0 / 0.1 | 0.0 / 0.0 |
| A117 (joint) vs A059 (datatype-only), same layer formats | 5.78 | 16 / 0 | 0.17 | 0.014 / 0.006 | 0.0 / 8.8 (worst 96.5) |
| **A128** (19 INT8+T, 1 INT4+T, 1 E4M11) | 7.58 | 20 | 0.135 | 0.009 / 0.0 / 0.8 | **0.0 / 0.0** |
| A071 (same layer formats as A128, no protection) | 7.58 | 0 | 0.135 | 0.003 / 0.0 / 0.1 | 11.4 / 96.3 |
| uniform INT8 / INT8 + TMR | 8.00 | 0 / 21 | 0.169 | 0.001 / 0.001 | 15.7 (worst 96.3) / 0.0 |
| uniform FP16 | 16.0 | n/a | 0.00 | **3.94** / 5.6 / **89.0** | n/a |
| uniform E4M11-SN | 16.0 | n/a | 0.00 | 0.055 / 0.0 / 2.2 | n/a |
| A114 (11 INT8+T, 10 E4M11) | 9.30 | 11 | 0.00 | 0.967 / 0.0 / **85.2** | 0.0 / 0.0 |

Reading:
- **Weight-bit faults (primary)**: for 28 of 31 assignments the mean drop is below 0.05 pp and the p90 is 0 or 0.1 pp. FP16 is the exception (exponent flips), and A114 / A170 contain rare
  large events (one trial of 85 pp, one of 10 pp, one of 3 pp; the 85 pp trial includes an E4M11 bit-14 flip in layer 3 and several high INT8 bits; the log does not isolate the culprit flip).
  With 100 trials these rare events are seen 0 - 2 times, so the mean under-resolves them.
- **Scale words (second metric)**: every unprotected INT assignment has a mean drop of 7 - 25 pp per flipped scale word (worst case 87 - 96 pp, i.e. the model is destroyed). With `scale_tmr` the
  measured value is 0.0 pp for all protected assignments. This is the effect of the protection; it is measured end to end, not assumed.
- Cost of the protection: 8 bytes per protected layer. A128 and A071 have the same layer formats, so the same clean accuracy; their average bits differ only in the fifth decimal (7.57752 vs 7.57741).

## 4. Predicted (layer-additive) vs measured
| quantity | result |
|---|---|
| predicted sum Q vs measured clean drop (198) | Spearman +0.91 |
| predicted `sum R_exposure` vs measured `weights_k8` mean (31) | Spearman +0.35 (0.28 without FP16); mean predicted 0.28 pp vs measured 0.17 pp |
| predicted scale-word risk vs measured `scales_k1` (protected and unprotected assignments) | Spearman +0.92, but this is mostly protected (0) versus unprotected (~10 pp) |
Meaning: R_exposure over-predicts for FP16 / FP8 (7.8 vs 3.9 pp, 0.26 vs -0.006 pp), and under-predicts rare large events in mixed assignments (A114: 0.05 predicted, 0.97 measured, caused by one
trial). Over all 31 assignments the prediction is in the right order of magnitude; the ranking among the many assignments whose measured mean is below 0.05 pp is noise, not information.
Honest limit: with 100 trials and per-flip drops quantised to 0.1 pp (1,000 images), `weights_k8` cannot distinguish assignments below about 0.05 pp.

## 5. Constraint grid and the selection (primary point and variants)
`constraint_grid_selection.csv` (delta_clean in {0.25, 0.5, 1.0}, tau_fault in {0.25, 0.5, 1, 2}, B in {5, 6, 8, 10, 12, 16}, tau_scale none or 1 pp): best MEASURED-feasible assignment.
| setting | selected | avg bits | clean drop (pp) | `weights_k8` (pp) | `scales_k1` (pp) |
|---|---|---|---|---|---|
| **primary** delta 0.5, tau_fault 0.5, B 8, tau_scale 1 | **A128** | 7.58 | 0.135 | 0.009 | 0.0 |
| same limits, DATATYPE-ONLY (no scale constraint, no protection) | A071 | 7.58 | 0.135 | 0.003 | **11.4 (worst 96.3)** |
| aggressive variant: delta 0.5, B 5, tau_scale 1 | A156 | 4.83 | 0.237 | 0.002 | 0.0 |
| B 6 | A117 | 5.78 | 0.169 | 0.014 | 0.0 |
| B 10 | U_INT8_TMR | 8.00 | 0.169 | 0.001 | 0.0 |
| B 12 / 16 | A123 | 11.28 | 0.068 | 0.000 | 0.0 |
The primary operating point is feasible and the selection is stable between delta_clean 0.5 and 1.0 (the clean constraint is not what decides). The weight-fault constraint (0.5 pp) binds only for FP16,
A114 and the rare-event assignments; for the others it is satisfied by a large margin. The scale-word constraint is the one that decides between protected and unprotected.

### What the optimiser does and does not do (read before choosing)
The objective minimises `Q_norm + lambda R_norm`; storage enters only as the budget `B`. At B = 8 the optimiser therefore returns the best-quality assignment that fits in 8 bits (7.58 bits, almost uniform
INT8 with one E4M11 layer and one INT4 layer), not the smallest one. It saves 5 % of the stored bytes against uniform INT8 + TMR with +0.03 pp less clean drop (one image out of 2,952, within noise).
The larger compression results are at tighter budgets: **A156 stores 4.83 bits per weight with 0.24 pp clean drop and zero measured scale-word and weight-fault penalty**, versus uniform INT4 + TMR at
4.00 bits with 1.22 pp. A claim of compression should therefore be made at the B = 5 point (aggressive variant), and the B = 8 point should be described as "no loss relative to INT8 at 5 % less storage".

## 6. What can and cannot be claimed
- Can: Q (exact validation Fisher) ranks assembled mixed-precision models by their clean accuracy loss (Spearman +0.91 over 198 assignments).
- Can: the implemented protection `scale_tmr` removes the measured scale-word failure of INT layers (0.0 pp in all 42 layer cells and in all 14 protected assignments of the shortlist), costs 8 bytes per layer,
  and corrects every single flipped bit (exhaustively verified); two flips in the same bit position of two copies are not corrected.
- Can: a measured joint datatype-protection assignment (A128; or A156 at 4.83 bits) meets delta_clean 0.5 pp, tau_fault 0.5 pp and the scale-word limit on validation data.
- Cannot: any protection other than `scale_tmr` (integer codes and E4M11 / FP exponent bits are unprotected); exponent-bit faults of FP16 / E4M11 are a residual tail risk; the weight-fault metric is
  too coarse (100 trials, 1,000 images) to rank assignments below 0.05 pp; one checkpoint; test accuracy is reported in section 7.

## 7. Frozen assignment A128 and the single test-set evaluation (2026-11-13)
Frozen at the primary operating point (delta_clean 0.5 pp, tau_fault 0.5 pp, B 8 bits, tau_scale 1 pp, lambda 1, min-max) from validation measurements only; `--finalize` was run once.
Assignment: 19 layers INT8 + `scale_tmr`, layer4.1.conv2 INT4 + `scale_tmr`, layer3.1.conv2 E4M11-SN (no protection; it has no scale word). Stored 10,574,768 bytes of weights (7.5775 bits per weight,
including 160 bytes of protection overhead for 20 layers), 10,651,768 bytes including the fixed parameters.
| measurement | result |
|---|---|
| clean validation (2,952 images) | 95.16 % vs FP32 95.29 %: drop 0.135 pp |
| **official test set (10,000 images, read once)** | **95.20 % vs FP32 95.11 %: +0.09 pp (9 images), no loss** |
| validation `weights_k8`, 300 trials, 1,000 images | mean drop 0.008 pp, median 0, p90 0, worst 0.6 pp; 13 of 300 trials show any drop |
| validation `scales_k1`, 300 trials | 0.000 pp in every trial |
| test set `weights_k8`, 100 trials, 10,000 images (same frozen assignment, no re-selection) | mean 0.005 pp, p90 0, worst 0.33 pp; 0 failures, 0 non-finite weights / activations / logits / loss |
| test set `scales_k1`, 100 trials | 0.000 pp in every trial |
| restoration | every trial restored the stored bytes exactly (SHA-256 check) |
Reading:
- The validation drop (-0.135 pp) and the test change (+0.09 pp) are both within about 0.15 pp of FP32, i.e. a few images: the correct statement is "no measurable accuracy loss", not an improvement.
- The scale-word result is a guarantee check, not sampling evidence: with 20 protected layers every single flipped scale bit is corrected (exhaustive verification, section 1), so 0.0 pp in 400 trials is the expected outcome.
- Residual risk, as in section 6: integer codes and the E4M11-SN layer have no protection; the 13 of 300 validation trials with a drop (worst 0.6 pp) come from high-bit flips of those.
- The constraints were checked on measured values (delta 0.135 <= 0.5, tau_fault 0.008 <= 0.5, tau_scale 0.0 <= 1.0, 7.58 <= 8 bits).
Files: `assignment.json` (frozen), `final_validation_fault_trials.jsonl`, `final_test_evaluation.json`, `stage5_final.log`.

## 8. Remaining (Stage 6)
Figures regenerated from round-2 / final data (the current Stage 5 PNGs use round-1 data), validation-vs-test comparison table, `final_summary.md`, rebuild script. No further selection is allowed on the test set.
