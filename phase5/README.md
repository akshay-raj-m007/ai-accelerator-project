# Phase 5 - J-FAMP optimisation inputs and end-to-end check (runbook Stage 5, Sections 7.1-7.2)

Reference: `JFAMP_ResNet18_CIFAR10_Experimental_Flow.pdf` Section 7 and the J-FAMP objective. Previous phases: [phase1](../phase1/README.md), [phase2](../phase2/README.md),
[phase3](../phase3/README.md), [phase4](../phase4/README.md). Decisions that fix the open points: [DECISIONS.md](DECISIONS.md). Analysis of the quantization penalty: [Q_ANALYSIS.md](Q_ANALYSIS.md).

Status: **round 1** (datatype-only, first R definition) is complete and in `reports/scratch/jfamp/optimization_round1/` ([RESULTS.md](RESULTS.md)). **Round 2** (declared fault distribution,
exposure-weighted R, scale-word second metric, measured protection `scale_tmr`, agreed constraints) was run on the GPU machine (2026-11-12) and is in `reports/scratch/jfamp/optimization/` ([RESULTS_ROUND2.md](RESULTS_ROUND2.md)). Frozen assignment A128 and the single test-set evaluation are done (RESULTS_ROUND2.md section 7). Stage 6 (figures, final summary) remains.

**Labelling rule.** The `p = none` subset of the candidate table is the DATATYPE-ONLY reliability-aware analysis. A result is called a measured joint datatype-protection result only if the
frozen assignment contains the implemented and measured protection `scale_tmr` (INT scale word stored 3x, bit-wise majority vote). No other protection is claimed.

## The formulation (as specified)
Minimise `sum_{l,d,p} x[l,d,p] (Q_norm(l,d) + lambda R_norm(l,d,p))` subject to: (A) one (datatype, protection) per layer, (B) `sum x S <= B`,
(C) `A_clean(x) >= A_FP32 - delta_clean`, (D) `R_fault(x) <= tau_fault`, (E) x binary.

| quantity | what we use (round 2) | where |
|---|---|---|
| Q(l,d) | **PRIMARY** `Q_fisher_val_exact`: `sum_i F_i (w_i - w^_i)^2`, F = diagonal Fisher of the FP32 model, expectation over labels computed EXACTLY, on held-out validation images 0..2047 (`scripts/fisher_exact.py`); w^ = decoded frozen Stage 2 encoding. ALTERNATIVE (labelled) `Q_fisher_train_sampled` (training-image sampled-label Fisher). `0.5 Q` = predicted clean cross-entropy increase | `stage5.py`, `stage5_candidate_table.py` |
| R(l,d,p) | **PRIMARY** `R_exposure_k8_pp = 8 (n_l/N) max(0, mean(k=1 drop, k=8 drop / 8))` from the Stage 4 layer-wise "any bit" drops: additive over layers, `sum_l R` predicts the end-to-end mean accuracy drop under the declared distribution (random flips over all stored weight bits, 8 distinct weights). Round-1 column `R_any_k8_mean_drop_pp` kept | `stage5.py`, `stage5_candidate_table.py` |
| R_scale(l,d,p) | **SECOND metric**, never merged into R: mean drop for one flipped bit of the INT scale word. Unprotected: Stage 4 values (pooled per format `R_scale_pooled_format_pp` for the optimiser); `scale_tmr`: MEASURED on the protected store (`stage5_tmr_layerwise.py`) | table columns `R_scale_*` |
| S(l,d,p) | actual stored bytes: payload + float32 scale(s) + protection overhead (`scale_tmr`: +8 bytes per INT layer) + padding (none) | `stage5_candidate_table.py`, checked against the assembled model |
| A_clean(x) | MEASURED on the assembled model, validation images 2,048..4,999 (2,952 images), DISJOINT from the 2,048 images behind Q; A_FP32 on the same images | `stage5_end_to_end.py` |
| R_fault(x) | MEASURED: mean accuracy drop of the assembled model under the declared distribution (`weights_k8`, first 1,000 validation images); `scales_k1` reported as the second metric | `stage5_end_to_end.py` |

Inside the ILP: `sum S <= B`, `sum_l R_exposure <= tau_fault` (predicted), and in the joint analysis options with scale-word risk above `tau_scale` are inadmissible. Constraints C and D are
checked on the ASSEMBLED model, and the final assignment is the measured-feasible one with the lowest objective.

## Normalisation and sweep
Global over all candidates: `minmax` -> `(v - min)/(max - min)`, `rank` -> average rank / (n - 1). Both are swept, lambda in {0, 0.1, 0.25, 0.5, 1, 2, 4, 10},
B in average bits/weight {4.5, 5, 6, 7, 8, 10, 12, 16}, tau_fault in {none, 2, 1, 0.5, 0.25} pp, for the datatype-only and the joint analysis. Solver: `scipy.optimize.milp` (HiGHS), exact;
checked against brute force in `tests/test_stage5.py`.

## Pipeline and outputs (`results/<run>/jfamp/optimization/`)
| step | script | outputs |
|---|---|---|
| 0 exact validation Fisher (once; GPU, minutes) | `scripts/fisher_exact.py` | `results/<run>/sensitivity_heldout/fisher_exact_val.pt` |
| 1 measure the protection (GPU, ~10 min) | `scripts/stage5_tmr_layerwise.py` | `tmr_layerwise_summary.csv`, `tmr_layerwise_trials.jsonl`, `tmr_exhaustive_check.json` (all single flips corrected; double flips corrected unless same bit position), `tmr_manifest.json` |
| 2 candidate table | `scripts/stage5_candidate_table.py` | `jfamp_candidate_table.csv` (147 rows = 21 layers x 7 options incl. per-layer measured clean metric), `candidate_table_meta.json`, `q_validation.json` |
| 2b layer-wise Q validation (once; copy `q_layerwise_validation.*` from round 1 if unchanged) | `scripts/stage5_q_validation.py` | `q_layerwise_validation.csv/.json` |
| 3 optimiser (CPU) | `scripts/stage5_optimize.py` | `normalization_lambda_sweep.csv`, `distinct_assignments.csv`, predicted `assignment.json`, `optimization_report.md` |
| 4 end-to-end (GPU) | `scripts/stage5_end_to_end.py` | `end_to_end_clean.csv`, `end_to_end_fault_trials.jsonl`, `end_to_end_assignments.csv`, `end_to_end_assignment_metrics.json` (predicted vs measured), `constraint_grid_selection.csv`, `end_to_end_manifest.json` |
| 5 freeze + test (once) | `stage5_end_to_end.py --finalize --final-delta-clean 0.5 --final-tau-fault 0.5 --final-avg-bits 8 --final-tau-scale 1.0` | final `assignment.json`, `final_validation_fault_trials.jsonl` (300 trials per distribution), `final_test_evaluation.json` (official test set, ONE frozen assignment) |

The test set is not read before step 5. Refinement (runbook 7.2): round 1 -> round 2 changed R because predicted and measured fault drops disagreed; the predicted-vs-measured comparison is repeated
in `end_to_end_assignment_metrics.json`.

## Limits (to state in any report)
Layer-additive estimates are approximations; Fisher from 2,048 validation images; R from 20 trials per layer cell and format (screening) and 100 end-to-end trials per distribution
(300 for the frozen assignment); fault trials on the first 1,000 validation images; `scale_tmr` protects only the scale word (not the integer codes) and fails only if the same bit position
is flipped in two copies; one checkpoint; no INT3.

Code: `src/fr/stage5.py`, `src/fr/stage5_e2e.py`, `src/fr/protect.py`; tests `tests/test_stage5.py`, `tests/test_stage5_e2e.py`, `tests/test_protect.py`.
