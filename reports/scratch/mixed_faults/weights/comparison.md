# Fault robustness of mixed E4M11 / 8-bit storage

* reference layout: `e4m11` (clean accuracy 95.11%); 'drop vs E4M11' always uses that value
* scale metadata of INT8 / scaled-FP8 layers is assumed fault-free; bit groups per format and the two sampling models are defined in `fr/faults_mixed.py` (project extension, not the paper)
* mean accuracy of faulty runs is heavy-tailed (rare catastrophic trials): read the median, p10 and bootstrap interval next to the mean

## group `any`, sampling `weights`, k = 8

| layout | bits/weight | clean acc | mean acc | 95% CI of mean | median | p10 | drop vs E4M11 (pp) | median KL | mean KL | non-finite | 8-bit hits/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e4m11 | 16.00 | 95.11 | 95.11 | [95.10, 95.12] | 95.11 | 95.11 | -0.00 | 4.99e-07 | 3.61e-04 | 0 | 0.00 |
| fp8_scaled_T12_product | 12.00 | 95.11 | 95.11 | [95.10, 95.12] | 95.11 | 95.11 | 0.00 | 1.83e-07 | 3.55e-04 | 0 | 4.12 |
| int8_pc_T12_product | 12.00 | 95.12 | 95.11 | [95.10, 95.12] | 95.12 | 95.11 | -0.00 | 1.09e-07 | 3.55e-04 | 0 | 4.12 |
| int8_pt_T12_product | 12.00 | 95.11 | 95.11 | [95.10, 95.12] | 95.11 | 95.11 | -0.00 | 1.98e-07 | 3.55e-04 | 0 | 4.12 |
| fp8_scaled_T12_lambda_0 | 11.98 | 95.13 | 95.14 | [95.13, 95.14] | 95.13 | 95.13 | -0.03 | 2.25e-07 | 4.80e-05 | 0 | 4.15 |
| int8_pc_T12_lambda_0 | 11.98 | 95.12 | 95.12 | [95.12, 95.12] | 95.12 | 95.11 | -0.01 | 1.61e-07 | 4.81e-05 | 0 | 4.15 |
| int8_pt_T12_lambda_0 | 11.98 | 95.11 | 95.12 | [95.11, 95.12] | 95.11 | 95.11 | -0.01 | 2.42e-07 | 4.88e-05 | 0 | 4.15 |
| fp8_scaled_T12_random_1 | 11.81 | 95.13 | 95.13 | [95.12, 95.14] | 95.13 | 95.13 | -0.02 | 2.07e-07 | 3.54e-04 | 0 | 4.39 |
| int8_pc_T12_random_1 | 11.81 | 95.14 | 95.14 | [95.12, 95.14] | 95.14 | 95.14 | -0.03 | 1.91e-07 | 3.47e-04 | 0 | 4.39 |
| int8_pt_T12_random_1 | 11.81 | 95.13 | 95.13 | [95.11, 95.13] | 95.13 | 95.12 | -0.02 | 3.17e-07 | 3.52e-04 | 0 | 4.39 |
| fp8_scaled_T12_random_0 | 11.81 | 95.18 | 95.18 | [95.18, 95.18] | 95.18 | 95.18 | -0.07 | 1.25e-07 | 3.33e-05 | 0 | 4.14 |
| int8_pc_T12_random_0 | 11.81 | 95.12 | 95.12 | [95.12, 95.13] | 95.12 | 95.12 | -0.01 | 1.11e-07 | 3.31e-05 | 0 | 4.14 |
| int8_pt_T12_random_0 | 11.81 | 95.14 | 95.14 | [95.14, 95.14] | 95.14 | 95.14 | -0.03 | 1.91e-07 | 3.35e-05 | 0 | 4.14 |
| fp8_scaled_T12_worst | 11.80 | 95.16 | 95.16 | [95.16, 95.16] | 95.16 | 95.15 | -0.05 | 2.44e-07 | 7.14e-06 | 0 | 4.08 |
| int8_pc_T12_worst | 11.80 | 95.17 | 95.17 | [95.16, 95.17] | 95.17 | 95.16 | -0.06 | 1.70e-07 | 6.52e-06 | 0 | 4.08 |
| int8_pt_T12_worst | 11.80 | 95.15 | 95.15 | [95.15, 95.15] | 95.15 | 95.14 | -0.04 | 3.93e-07 | 7.28e-06 | 0 | 4.08 |
| all_fp8_scaled | 8.00 | 95.18 | 95.18 | [95.18, 95.18] | 95.18 | 95.18 | -0.07 | 1.03e-07 | 1.09e-06 | 0 | 8.00 |
| all_int8_pc | 8.00 | 95.14 | 95.14 | [95.14, 95.14] | 95.14 | 95.13 | -0.03 | 3.78e-08 | 4.39e-07 | 0 | 8.00 |
| all_int8_pt | 8.00 | 95.15 | 95.15 | [95.15, 95.15] | 95.15 | 95.15 | -0.04 | 2.00e-07 | 1.27e-06 | 0 | 8.00 |

Paired difference in accuracy vs `e4m11` (same struck weights per trial; negative = worse than the reference):

| layout | trials | mean diff (pp) | s.e. | trials worse / equal / better |
|---|---|---|---|---|
| fp8_scaled_T12_product | 100 | -0.002 | 0.001 | 20 / 72 / 8 |
| int8_pc_T12_product | 100 | +0.002 | 0.001 | 12 / 50 / 38 |
| int8_pt_T12_product | 100 | +0.000 | 0.001 | 7 / 81 / 12 |
| fp8_scaled_T12_lambda_0 | 100 | +0.023 | 0.005 | 4 / 5 / 91 |
| int8_pc_T12_lambda_0 | 100 | +0.010 | 0.005 | 10 / 37 / 53 |
| int8_pt_T12_lambda_0 | 100 | +0.005 | 0.005 | 19 / 56 / 25 |
| fp8_scaled_T12_random_1 | 100 | +0.017 | 0.001 | 7 / 4 / 89 |
| int8_pc_T12_random_1 | 100 | +0.025 | 0.001 | 5 / 5 / 90 |
| int8_pt_T12_random_1 | 100 | +0.014 | 0.001 | 9 / 8 / 83 |
| fp8_scaled_T12_random_0 | 100 | +0.069 | 0.006 | 1 / 1 / 98 |
| int8_pc_T12_random_0 | 100 | +0.012 | 0.005 | 10 / 9 / 81 |
| int8_pt_T12_random_0 | 100 | +0.029 | 0.005 | 5 / 4 / 91 |
| fp8_scaled_T12_worst | 100 | +0.048 | 0.006 | 5 / 1 / 94 |
| int8_pc_T12_worst | 100 | +0.055 | 0.006 | 5 / 1 / 94 |
| int8_pt_T12_worst | 100 | +0.040 | 0.006 | 5 / 2 / 93 |
| all_fp8_scaled | 100 | +0.067 | 0.005 | 2 / 1 / 97 |
| all_int8_pc | 100 | +0.029 | 0.006 | 7 / 3 / 90 |
| all_int8_pt | 100 | +0.041 | 0.006 | 5 / 4 / 91 |
