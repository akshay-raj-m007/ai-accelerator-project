# Why the Fisher-weighted Q is so small (diagnosis, 2026-11-11)

Candidate table: Q = sum_i F_ii (w_i - w^_i)^2 with the sampled-label true Fisher on 1,024 TRAINING images. Check (q_validation.json): for INT4, 1/2 sum Q = 2.7e-6 but the measured
clean CE change is +0.0144 (5,000 validation images): about 5,000 x too small. Diagnostic scripts: `scripts/stage5_q_diagnosis.py` (500 validation images, CPU; result
`reports/scratch/jfamp/optimization/q_diagnosis_cpu500.json`) and `scripts/stage5_exact_fisher_check.py` (128 images per set). These are diagnostics on small subsets, not runbook outputs.

## Factorisation: per layer Q_l is almost exactly trace(F_l) x mean(d^2)
Q_l / (trace_l x mse_l) = 0.98 - 1.0 for most layers, 1.51 overall for INT4 (conv1 / layer1: 2.4 - 3.4, because rounding to zero correlates the error with the weights). So the
diagonal structure is not the problem. Both factors are small in absolute terms; the question is which one is wrong for the purpose.

## The Fisher is small because of the data it is computed on, and because of label sampling
| quantity | train images | validation images |
|---|---|---|
| accuracy | 100 % | 96.8 % |
| mean max-probability | 0.99914 | 0.97996 |
| mean (1 - p_max)^2 | 1.8e-6 | 5.4e-3 (about 3,000 x) |
| fraction with p_max > 0.999 | 80 % | 59 % |
| stored true-Fisher trace (sampled labels) | 0.158 (1,024 imgs) | 2,842 (2,048 imgs) -> 18,000 x |
| exact-expectation Fisher trace (sum over all 10 labels, no sampling), 128 imgs | 21.8 | 2,446 -> 112 x |

1. Data: the network is saturated on the training images, so gradients of the log-likelihood are tiny there; the evaluation (validation) distribution is far less saturated.
2. Monte-Carlo label sampling: with y ~ p(y|x) almost every sample is the arg-max class (small gradient); the expectation is carried by rare labels, which a 1,024-sample draw seldom hits.
   The stored train trace (0.158) is about 140 x below the exact expectation (21.8, 128 images), so the stored value is dominated by chance. On validation images the estimator is fine
   (2,842 sampled vs 2,446 exact).

## Decomposition of the INT4 gap (500 validation images, |d|/|w| = 0.405)
actual dL(t=1) = +0.0214; g.d = -0.0242 (a descent component: rounding shrinks weights); 1/2 d^T H d (exact Hessian-vector product) = +0.0652; second-order Taylor = +0.0410;
1/2 d^T diag(F_val) d = +0.0322; 1/2 d^T diag(F_train) d (256-image sampled Fisher) = +1.45e-4; exact-expectation Fisher: train 2.95e-4, validation 5.19e-2.
Gap table-to-measured (5,250 x) = 108 x (sampling undercount: 2.95e-4 / 2.74e-6) x 176 x (train -> validation distribution: 5.19e-2 / 2.95e-4), offset by 0.28 (measured / 1/2 dFd_val,
higher-order and first-order terms). So the right-hand side Sum F d^2 is the right size when F is taken on the evaluation distribution (0.052 vs measured 0.0144 - 0.021, within a factor 2.5 - 3.6).
Scaling the perturbation: dL(0.25) = -0.0039, dL(0.5) = -0.0018, dL(1) = +0.0214, not proportional to t^2: INT4 is outside the quadratic regime (relative perturbation 40 %).

## The 8-bit and 16-bit formats: the signal is below the evaluation noise
For a finite set of N images, g.d is a random variable with standard deviation about sqrt(d^T F d / N) (proportional to |d|), while the expected change 1/2 d^T F d is proportional to |d|^2.
Signal / noise = 1/2 sqrt(N d^T F d): INT4 about 11 (detectable), FP8 about 1.5, INT8 about 0.6 (N = 5,000, validation Fisher). Measured: INT8 -3e-5, FP8 -9e-4, i.e. noise; the 500-image
run shows the same (INT8 dL = -8.3e-4 with g.d = -8.6e-4 versus 1/2 dFd = +9.5e-5). A second-order score cannot be validated against these measurements, for any Fisher.

## Consequences
- Multiplying Q by a constant (or any positive affine map) changes nothing: min-max and rank normalisation are invariant to it, and lambda already weights Q against R.
- Rank normalisation is invariant to any monotone transform of Q (log, power); min-max is not.
- A different Fisher source changes the layer profile of Q, but across datatypes Q is dominated by d^2 (INT4 >> FP8 >> INT8 >> FP16 > E4M11); after min-max normalisation only INT4 (and slightly FP8) carry a non-negligible Q.

## Result of the primary (validation exact-Fisher) Q on the GPU run
Files: `reports/scratch/jfamp/optimization/` (`q_validation.json`, `q_layerwise_validation.json/.csv`, `fisher_exact_val_*`). Fisher: 2,048 validation images (indices 0..2047), 20,480 backward passes,
128 s, neglected probability mass <= 1e-8 per image. Layer-wise check on validation images 2,048..4,999 (2,952 images, disjoint from the Fisher images).
- Scale fixed: INT4 sum 1/2 Q = 0.0589 vs measured all-layers change +0.0158 (3.7 x over; before: 2.7e-6, 5,000 x under). The sum of the 21 single-layer measurements is +0.0199, so the
  clean change is close to additive (all-layers / sum = 0.79).
- INT4 ranking over the 21 layers: Spearman +0.713 (training-Fisher Q: +0.708). The layers with the largest predicted change agree in magnitude (conv1 0.0149 predicted / 0.0099 measured; layer1.0.conv1
  0.0116 / 0.0100; layer2.0.downsample 0.0048 / 0.0041; layer1.0.conv2 0.0049 / 0.0031).
- The two Q columns are almost the same ranking: Spearman 0.993 over the 105 candidates, 0.94 - 0.98 per datatype, constant ratio of about 20,600. Because the optimiser normalises Q, the change of Fisher source
  alters the optimisation very little (251 distinct assignments, 188 shared with the training-Q run, same provisional assignment). The gain is the absolute scale (loss units), not a different decision.
- Cross-entropy is confounded by calibration: several late layers show a clearly NEGATIVE single-layer change under INT4 (layer4.0.conv2 -0.018, layer4.1.conv2 -0.005), i.e. quantization shrinks the
  over-confident logits and lowers the validation cross-entropy. The first-order term g.d is sign-indefinite and no positive-semidefinite score can predict it; the clean constraint is therefore on accuracy,
  measured end to end. Per-layer measured/predicted ratios are only meaningful where the predicted change is above about 1e-3.
- FP8, INT8, FP16, E4M11-SN: measured changes are within noise (|dCE| <= 1e-3); Spearman over layers 0.04 (FP8), -0.11 (INT8), 0.36 (FP16), 0.18 (E4M11-SN), no information.
