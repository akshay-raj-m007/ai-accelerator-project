# Per-layer scores on the all-E4M11 model (Plan B)

* S1 measured on 2952 held-out validation images; clean accuracy 95.29 %; k = 16 bits struck in one layer only, 100 trials per layer, sampling `bits`
* Fisher: first 2,048 validation images (held out from S1). Definitions of S0, S1, S2, S2n: `src/fr/layer_scores.py`

## Group `any`

| layer | weights | S0 tr(F)/N | S2 pred. | S2n (no F) | S1 dCE mean | S1 dCE median | S1 KL mean | S1 dAcc mean (pp) | S1 dAcc p90 | non-finite |
|---|---|---|---|---|---|---|---|---|---|---|
| conv1.weight | 1728 | 0.0334 | 1.353 | 9.335 | +0.9683 | +0.0043 | 0.9914 | +16.79 | +77.07 | 0 |
| layer1.0.conv1.weight | 36864 | 0.003809 | 0.01466 | 0.5372 | +0.0567 | +0.0000 | 0.05943 | +1.28 | +0.34 | 0 |
| layer1.0.conv2.weight | 36864 | 0.003446 | 0.006493 | 0.6196 | +0.0769 | +0.0006 | 0.08227 | +1.41 | +1.47 | 0 |
| layer1.1.conv1.weight | 36864 | 0.003554 | 0.009316 | 0.5965 | +0.2642 | +0.0001 | 0.2726 | +3.81 | +2.59 | 0 |
| layer1.1.conv2.weight | 36864 | 0.002348 | 0.002552 | 0.6149 | +0.0927 | +0.0003 | 0.09705 | +1.88 | +1.09 | 0 |
| layer2.0.conv1.weight | 73728 | 0.002311 | 0.00166 | 0.5572 | +0.0518 | +0.0004 | 0.0578 | +1.20 | +2.02 | 0 |
| layer2.0.conv2.weight | 147456 | 0.001444 | 0.0008637 | 0.4805 | +0.0587 | +0.0002 | 0.06375 | +1.31 | +1.97 | 0 |
| layer2.0.downsample.0.weight | 8192 | 0.008532 | 0.05495 | 1.536 | +0.4298 | +0.0012 | 0.4435 | +7.99 | +24.60 | 0 |
| layer2.1.conv1.weight | 147456 | 0.001017 | 0.0005566 | 0.3897 | +0.0494 | +0.0001 | 0.05273 | +1.15 | +2.17 | 0 |
| layer2.1.conv2.weight | 147456 | 0.0006415 | 0.0003286 | 0.3422 | +0.0269 | +0.0001 | 0.02827 | +0.64 | +0.49 | 0 |
| layer3.0.conv1.weight | 294912 | 0.0009229 | 0.0004742 | 0.414 | +0.0419 | +0.0001 | 0.04579 | +0.94 | +1.43 | 0 |
| layer3.0.conv2.weight | 589824 | 0.0007038 | 0.0003131 | 0.3712 | +0.0203 | +0.0002 | 0.02378 | +0.62 | +1.06 | 0 |
| layer3.0.downsample.0.weight | 32768 | 0.00132 | 0.001196 | 0.5849 | +0.0454 | +0.0002 | 0.05137 | +1.24 | +1.95 | 0 |
| layer3.1.conv1.weight | 589824 | 0.0006189 | 0.0002738 | 0.3448 | +0.0334 | +0.0001 | 0.03736 | +0.92 | +1.33 | 0 |
| layer3.1.conv2.weight | 589824 | 0.0003011 | 0.0001091 | 0.2616 | +0.0042 | +0.0001 | 0.005325 | +0.21 | +0.51 | 0 |
| layer4.0.conv1.weight | 1179648 | 0.0001156 | 2.718e-05 | 0.2113 | +0.0002 | -0.0000 | 0.0009103 | +0.07 | +0.24 | 0 |
| layer4.0.conv2.weight | 2359296 | 3.863e-05 | 7.103e-06 | 0.164 | -0.0002 | -0.0000 | 0.0002527 | +0.05 | +0.14 | 0 |
| layer4.0.downsample.0.weight | 131072 | 9.145e-05 | 2.563e-05 | 0.228 | +0.0000 | -0.0000 | 0.0009436 | +0.09 | +0.27 | 0 |
| layer4.1.conv1.weight | 2359296 | 2.846e-05 | 5.023e-06 | 0.1518 | +0.0001 | -0.0000 | 0.00013 | +0.03 | +0.10 | 0 |
| layer4.1.conv2.weight | 2359296 | 8.815e-06 | 1.331e-06 | 0.1418 | +0.0000 | +0.0000 | 1.776e-05 | +0.02 | +0.04 | 0 |
| fc.weight | 5120 | 0.000147 | 0.005479 | 13.43 | +0.2655 | +0.0017 | 0.2627 | +2.57 | +12.01 | 0 |

## Group `exponent`

| layer | weights | S0 tr(F)/N | S2 pred. | S2n (no F) | S1 dCE mean | S1 dCE median | S1 KL mean | S1 dAcc mean (pp) | S1 dAcc p90 | non-finite |
|---|---|---|---|---|---|---|---|---|---|---|
| conv1.weight | 1728 | 0.0334 | 5.41 | 37.34 | +3.0258 | +3.5866 | 3.091 | +53.68 | +84.74 | 0 |
| layer1.0.conv1.weight | 36864 | 0.003809 | 0.05865 | 2.149 | +0.4001 | +0.0063 | 0.4146 | +7.39 | +26.84 | 0 |
| layer1.0.conv2.weight | 36864 | 0.003446 | 0.02597 | 2.478 | +0.7253 | +0.0580 | 0.7515 | +13.20 | +62.41 | 0 |
| layer1.1.conv1.weight | 36864 | 0.003554 | 0.03726 | 2.386 | +0.7825 | +0.0575 | 0.8083 | +13.65 | +62.46 | 0 |
| layer1.1.conv2.weight | 36864 | 0.002348 | 0.01021 | 2.459 | +0.2653 | +0.0222 | 0.2806 | +5.32 | +15.07 | 0 |
| layer2.0.conv1.weight | 73728 | 0.002311 | 0.006639 | 2.229 | +0.3465 | +0.0478 | 0.369 | +7.47 | +25.21 | 0 |
| layer2.0.conv2.weight | 147456 | 0.001444 | 0.003455 | 1.922 | +0.1668 | +0.0445 | 0.1839 | +3.72 | +6.02 | 0 |
| layer2.0.downsample.0.weight | 8192 | 0.008532 | 0.2198 | 6.146 | +2.0672 | +1.0283 | 2.118 | +35.12 | +84.44 | 0 |
| layer2.1.conv1.weight | 147456 | 0.001017 | 0.002226 | 1.559 | +0.2991 | +0.0568 | 0.3194 | +6.41 | +25.54 | 0 |
| layer2.1.conv2.weight | 147456 | 0.0006415 | 0.001314 | 1.369 | +0.0358 | +0.0135 | 0.04339 | +0.98 | +2.34 | 0 |
| layer3.0.conv1.weight | 294912 | 0.0009229 | 0.001897 | 1.656 | +0.1106 | +0.0256 | 0.1245 | +2.58 | +5.35 | 0 |
| layer3.0.conv2.weight | 589824 | 0.0007038 | 0.001252 | 1.485 | +0.1146 | +0.0421 | 0.1312 | +2.98 | +9.17 | 0 |
| layer3.0.downsample.0.weight | 32768 | 0.00132 | 0.004784 | 2.339 | +0.4181 | +0.0751 | 0.4507 | +9.80 | +31.11 | 0 |
| layer3.1.conv1.weight | 589824 | 0.0006189 | 0.001095 | 1.379 | +0.1219 | +0.0600 | 0.1364 | +3.10 | +7.54 | 0 |
| layer3.1.conv2.weight | 589824 | 0.0003011 | 0.0004362 | 1.046 | +0.0134 | +0.0048 | 0.01903 | +0.58 | +1.37 | 0 |
| layer4.0.conv1.weight | 1179648 | 0.0001156 | 0.0001087 | 0.8452 | +0.0014 | +0.0005 | 0.00342 | +0.22 | +0.41 | 0 |
| layer4.0.conv2.weight | 2359296 | 3.863e-05 | 2.841e-05 | 0.656 | -0.0007 | -0.0003 | 0.0008966 | +0.14 | +0.30 | 0 |
| layer4.0.downsample.0.weight | 131072 | 9.145e-05 | 0.0001025 | 0.912 | -0.0001 | -0.0003 | 0.002848 | +0.18 | +0.34 | 0 |
| layer4.1.conv1.weight | 2359296 | 2.846e-05 | 2.009e-05 | 0.6072 | +0.0008 | +0.0000 | 0.001071 | +0.11 | +0.24 | 0 |
| layer4.1.conv2.weight | 2359296 | 8.815e-06 | 5.324e-06 | 0.5672 | +0.0000 | -0.0000 | 5.767e-05 | +0.04 | +0.10 | 0 |
| fc.weight | 5120 | 0.000147 | 0.02192 | 53.7 | +1.5157 | +0.4846 | 1.493 | +11.85 | +28.15 | 0 |

## Spearman rank correlation of each score with the measured damage S1 (21 layers)

| group | S1 target | score | Spearman |
|---|---|---|---|
| any | S1 d_ce_mean | S0 trace per parameter | +0.848 |
| any | S1 d_ce_mean | S0 trace (sum) | -0.053 |
| any | S1 d_ce_mean | S2 Fisher-predicted | +0.951 |
| any | S1 d_ce_mean | S2n no Fisher | +0.952 |
| any | S1 kl_mean | S0 trace per parameter | +0.873 |
| any | S1 kl_mean | S0 trace (sum) | -0.048 |
| any | S1 kl_mean | S2 Fisher-predicted | +0.962 |
| any | S1 kl_mean | S2n no Fisher | +0.960 |
| any | S1 d_acc_mean | S0 trace per parameter | +0.871 |
| any | S1 d_acc_mean | S0 trace (sum) | -0.078 |
| any | S1 d_acc_mean | S2 Fisher-predicted | +0.964 |
| any | S1 d_acc_mean | S2n no Fisher | +0.968 |
| exponent | S1 d_ce_mean | S0 trace per parameter | +0.834 |
| exponent | S1 d_ce_mean | S0 trace (sum) | -0.082 |
| exponent | S1 d_ce_mean | S2 Fisher-predicted | +0.947 |
| exponent | S1 d_ce_mean | S2n no Fisher | +0.944 |
| exponent | S1 kl_mean | S0 trace per parameter | +0.843 |
| exponent | S1 kl_mean | S0 trace (sum) | -0.081 |
| exponent | S1 kl_mean | S2 Fisher-predicted | +0.956 |
| exponent | S1 kl_mean | S2n no Fisher | +0.956 |
| exponent | S1 d_acc_mean | S0 trace per parameter | +0.869 |
| exponent | S1 d_acc_mean | S0 trace (sum) | -0.047 |
| exponent | S1 d_acc_mean | S2 Fisher-predicted | +0.955 |
| exponent | S1 d_acc_mean | S2n no Fisher | +0.948 |

## Fisher of the FP32 model vs the E4M11-quantized model (layer traces)

* Spearman +1.0000; ratio E4M11/FP32 per layer: min 0.996, median 1.000, max 1.001
