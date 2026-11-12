# ALTERNATIVE analysis: Q from the training-image Fisher (first Stage 5 run, 2026-11-11)

These files were produced with Q = sum F (w - w_hat)^2 where F is the sampled-label true Fisher on 1,024 TRAINING images (column `Q_fisher_weighted`).
That Fisher is too small for this network (saturated on its training images, sampled-label estimator undercounts) - see `phase5/Q_ANALYSIS.md`. The PRIMARY Q (exact validation Fisher) was used from round 1 on; it is in
`../optimization_round1/` (round 1) and `../optimization/` (round 2) - exact-expectation Fisher on held-out validation images 0..2047. Everything here is kept for comparison, labelled as the alternative.
No end-to-end measurements exist for this alternative.
