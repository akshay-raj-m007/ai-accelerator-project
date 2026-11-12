# Phase 5 decisions (agreed 2026-11-12) and how they are implemented

These four decisions fix the open points of the Stage 5 formulation. They were taken before looking at any test-set result; the test set is read only by the final
`--finalize` step for one frozen assignment.

## 1. Declared fault distribution
**Random bit flips over all stored weight bits of the assembled model, 8 distinct weights per trial (primary), 1 flip as the secondary case.** It is the Stage 4 global
"any bit, k = 8" screening and the paper's fault model (flips in the quantized weight representation; biases, BatchNorm and activations are not faulted).
- Primary end-to-end metric `R_fault(x)` = mean accuracy drop (pp) of the assembled model under this distribution (`weights_k8`, first 1,000 validation images).
- Layer-level `R(l,d,p)` is defined to match it: `R_exposure = 8 * (n_l / N) * max(0, per-flip drop)`, with the per-flip drop = mean of the Stage 4 k = 1 estimate and the k = 8
  estimate / 8 (clipped at 0, negative values are noise). It is additive over layers and in the unit of the end-to-end metric: `sum_l R_exposure` predicts `R_fault(x)`.
- Why this estimator: the round-1 proxy (unweighted mean over layers of the single-layer k = 8 drop) ranked the 27 measured assignments with Spearman +0.38 and overstated small risks
  5 - 6x; the pooled exposure estimate ranks them with +0.54 (k = 1 alone: +0.55) and has the smaller total error (see `phase5/RESULTS.md`, round 1). The choice was made on
  validation measurements, not on test data.

## 2. Scale words
**Not part of the primary R.** The scale-word fault (one flipped bit of the float32 scale of an INT layer) is a **second, mandatory metric** (`scales_k1`: one flip uniform over
all stored scale bits of the model), always reported next to the primary one and never merged into it.
- Reason: under uniform random flips a scale word is hit with probability about 6e-5 per 8-flip trial, but one flipped exponent bit collapses the network (8 - 30 pp on average).
- Handled by decision 4: in the joint analysis an option whose scale-word risk exceeds `tau_scale` (1 pp per flipped scale word) is inadmissible, i.e. INT layers must be protected.

## 3. Constraints (primary operating point)
| quantity | value |
|---|---|
| `delta_clean` | 0.5 pp below FP32 (measured on the assembled model, validation images 2,048 - 4,999) |
| `tau_fault` | 0.5 pp mean accuracy drop under the declared weight-fault distribution (predicted by `sum R_exposure` inside the ILP, then checked on the assembled model) |
| `B` | 8 bits per quantized weight on average (stored bytes including scales and protection overhead) |
| `lambda` | 1, min-max normalisation (rank normalisation and the lambda sweep are reported) |
| `tau_scale` | 1 pp per flipped scale word (joint analysis; follows from decisions 2 and 4; my addition to make the protection effective, not a separately agreed number) |
The final assignment is chosen among the optimiser's solutions as the one that is MEASURED-feasible under these limits with the lowest objective. A looser variant (`delta_clean` 1.0 pp,
5 bits) is reported as a labelled variant only.

## 4. Protection: `scale_tmr`
Implemented (`src/fr/protect.py`), tested (`tests/test_protect.py`) and measured (`scripts/stage5_tmr_layerwise.py`, `end_to_end`):
- the float32 scale of an INT layer is stored **three times** (+8 bytes per protected layer, 168 bytes for 21 layers, 4-byte aligned, no padding); on every read the effective scale is the
  **bit-wise majority** of the three stored words;
- every single flipped bit (3 x 32 positions per layer) is corrected - exhaustively verified on the stored bytes; two flips in two different copies are corrected unless they hit the same bit
  position (documented limit, reported by the exhaustive check);
- the integer weight codes are not protected by this option (their flips are the primary R);
- the candidate table has the options `INT8`, `INT4` with `p = none` and `p = scale_tmr`; the datatype-only analysis is the `p = none` subset. Protected results are reported as measured joint
  datatype-protection results; nothing else is claimed.
