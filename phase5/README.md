# Phase 5 - J-FAMP optimisation inputs and end-to-end check (runbook Stage 5, Sections 7.1-7.2)

Reference: `JFAMP_ResNet18_CIFAR10_Experimental_Flow.pdf` Section 7 and the J-FAMP objective. Previous phases: [phase1](../phase1/README.md), [phase2](../phase2/README.md),
[phase3](../phase3/README.md), [phase4](../phase4/README.md).
Status: code written, unit-tested and dry-run locally; **GPU runs and results pending**.

**This is a DATATYPE-ONLY, reliability-aware analysis (protection `p = none`).** No protection mechanism (duplication, ECC, ...) is implemented or claimed, so no
result here is a measured joint datatype-protection result. The candidate table has `protection = none` only.

## The formulation (as specified)
Minimise `sum_{l,d,p} x[l,d,p] (Q_norm(l,d) + lambda R_norm(l,d,p))` subject to: (A) one (datatype, protection) per layer, (B) `sum x S <= B`,
(C) `A_clean(x) >= A_FP32 - delta_clean`, (D) `R_fault(x) <= tau_fault`, (E) x binary.

| quantity | what we use | where |
|---|---|---|
| Q(l,d) | Fisher-weighted reconstruction penalty `sum_i F_i (w_i - w^_i)^2`, F = diagonal true Fisher of the FP32 model on 1,024 TRAINING-split images, w^ = decoded frozen Stage 2 encoding. `0.5 Q` = predicted clean cross-entropy increase | `stage5.py`, `stage5_candidate_table.py` |
| R(l,d,p=none) | Stage 4 layer-wise mean accuracy drop (pp), group `all` (any bit of the stored weight), k = 8, 20 trials x 1,000 validation images; k = 1, p90, worst, failure rate, exponent-group and (INT) scale-word risk kept as extra columns | `stage5_candidate_table.py` |
| S(l,d,p) | actual stored bytes: payload + float32 scale (INT) + padding (none occurs) + protection overhead (0) | `stage5_candidate_table.py` |
| A_clean(x) | MEASURED: clean accuracy of the assembled mixed-datatype model on all 5,000 validation images | `stage5_end_to_end.py` |
| R_fault(x) | MEASURED: mean accuracy drop of the assembled model under random bit flips in its stored bytes (declared distributions below) | `stage5_end_to_end.py` |

Constraints C and D are therefore checked on END-TO-END measurements, not on layer-additive proxies. In the ILP the proxies are used only as an initial guide:
a mean-layer-risk limit `tau`, a predicted-clean-CE limit `eps` and the storage limit `B`; the sweep varies them.

## Normalisation and lambda sweep
Global over all 105 candidates: `minmax` -> `(v - min)/(max - min)`, and `rank` -> average rank / (n - 1). Both are swept. lambda in {0, 0.1, 0.25, 0.5, 1, 2, 4, 10},
B in average bits/weight {4.5, 5, 6, 7, 8, 10, 12, 16}, tau in {none, 5, 2, 1, 0.5} pp, eps in {none, 0.002, 0.005, 0.01}, risk variant `weights` (R as defined) and
`plus_scale` (R plus the Stage 4 INT scale-word k = 1 risk, a sensitivity check that unions two different fault processes). Solver: `scipy.optimize.milp` (HiGHS), exact; checked
against brute force on small problems in `tests/test_stage5.py`.

## Declared end-to-end fault distributions (assembled model, weights only)
- `weights_k8`: 8 distinct weights drawn uniformly from ALL quantizable weights of the assembled model; one bit per weight uniform over ALL bits of that weight's own encoding
  (16, 8 or 4 bits by layer). This is the Stage 4 global "any bit, k = 8" distribution on a mixed model. Primary R_fault(x).
- `scales_k1` (extension, not in the paper): one INT scale word, one of its 32 bits; defined only if some layer is INT.
Flips go into the stored bytes (Stage 3 rules): decode only affected elements, byte-exact restore, SHA-256 of the whole encoded state before / during / after, non-finite outcomes kept.

## Pipeline and outputs (`results/<run>/jfamp/optimization/`)
| step | script | outputs |
|---|---|---|
| 1 candidate table (GPU machine: needs Fisher + frozen encodings) | `scripts/stage5_candidate_table.py` | `jfamp_candidate_table.csv` (105 rows), `candidate_table_meta.json` (definitions, input hashes), `q_validation.json` (does sum 0.5 Q predict the measured clean CE change of the uniform formats?) |
| 2 optimiser (CPU) | `scripts/stage5_optimize.py` | `normalization_lambda_sweep.csv`, `distinct_assignments.csv`, provisional `assignment.json`, `optimization_report.md` |
| 3 end-to-end (GPU) | `scripts/stage5_end_to_end.py` | `end_to_end_clean.csv`, `end_to_end_fault_trials.jsonl` (raw, resumable), `end_to_end_assignments.csv`, `end_to_end_assignment_metrics.json` (predicted vs measured, Spearman), `constraint_grid_selection.csv`, `end_to_end_manifest.json` |
| 4 freeze (only when the constraints are agreed) | `stage5_end_to_end.py --finalize --final-delta-clean .. --final-tau-fault .. --final-avg-bits ..` | final `assignment.json` (measured metrics), `final_test_evaluation.json` (official test set, ONE frozen assignment) |

The test set is not read until step 4. The refinement loop of runbook 7.2 is: predicted (layer-additive) vs measured (assembled) are compared in
`end_to_end_assignment_metrics.json`, and the constraint grid picks only MEASURED-feasible assignments; if the proxies mis-rank, re-run the optimiser with tighter proxy limits.

## Limits (to state in any report)
Datatype-only; layer-additive estimates are approximations; Fisher from 1,024 training images; R from 20 trials per layer cell (screening); end-to-end fault trials limited in number
(`--trials`, default 50) and measured only for a shortlist; INT scale-word risk is not part of the primary R; clean accuracy per layer was not measured (format-level and end-to-end only).

Code: `src/fr/stage5.py`, `src/fr/stage5_e2e.py`; tests `tests/test_stage5.py`, `tests/test_stage5_e2e.py`.
