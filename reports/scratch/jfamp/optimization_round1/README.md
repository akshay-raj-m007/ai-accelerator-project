# Round 1 (datatype-only, p = none): primary Q = exact validation Fisher, R = unweighted mean over layers of the single-layer k = 8 drop

First complete Stage 5 pass (2026-11-12). Candidate table (105 rows), 5,120 optimiser problems, 251 distinct assignments, end-to-end measurements of 256 assemblies (clean accuracy) and
27 (bit-flip trials, 50 per distribution), Q validation (`q_validation.json`, `q_layerwise_validation.*`). Its R definition did not match the declared end-to-end fault distribution
(Spearman +0.38 against the measured weight-fault drops); the decisions in `phase5/DECISIONS.md` replaced it. Round 2 (declared distribution, exposure-weighted R, scale-word metric,
protection `scale_tmr`) is in `../optimization/`. Kept unchanged for comparison and as the "before" side of the runbook's refinement step.
