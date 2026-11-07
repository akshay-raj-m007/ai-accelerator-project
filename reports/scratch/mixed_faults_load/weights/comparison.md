# Fault robustness of mixed E4M11 / 8-bit storage

* reference layout: `e4m11` (clean accuracy 95.11%); 'drop vs E4M11' always uses that value
* scale metadata of INT8 / scaled-FP8 layers is assumed fault-free; bit groups per format and the two sampling models are defined in `fr/faults_mixed.py` (project extension, not the paper)
* mean accuracy of faulty runs is heavy-tailed (rare catastrophic trials): read the median, p10 and bootstrap interval next to the mean

## group `any`, sampling `weights`, k = 16

| layout | bits/weight | clean acc | mean acc | 95% CI of mean | median | p10 | drop vs E4M11 (pp) | median KL | mean KL | non-finite | 8-bit hits/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e4m11 | 16.00 | 95.11 | 95.09 | [95.03, 95.12] | 95.11 | 95.11 | 0.02 | 5.13e-06 | 1.92e-03 | 0 | 0.00 |
| all_int8_pc | 8.00 | 95.14 | 95.14 | [95.14, 95.14] | 95.14 | 95.13 | -0.03 | 3.81e-07 | 8.27e-07 | 0 | 16.00 |
| all_int4_pc | 4.00 | 94.99 | 94.98 | [94.98, 94.99] | 94.99 | 94.97 | 0.13 | 1.41e-06 | 2.47e-06 | 0 | 16.00 |

Paired difference in accuracy vs `e4m11` (same struck weights per trial; negative = worse than the reference):

| layout | trials | mean diff (pp) | s.e. | trials worse / equal / better |
|---|---|---|---|---|
| all_int8_pc | 40 | +0.051 | 0.027 | 3 / 2 / 35 |
| all_int4_pc | 40 | -0.106 | 0.027 | 38 / 0 / 2 |

## group `any`, sampling `weights`, k = 64

| layout | bits/weight | clean acc | mean acc | 95% CI of mean | median | p10 | drop vs E4M11 (pp) | median KL | mean KL | non-finite | 8-bit hits/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e4m11 | 16.00 | 95.11 | 94.99 | [94.89, 95.08] | 95.12 | 94.74 | 0.12 | 6.40e-04 | 7.32e-03 | 0 | 0.00 |
| all_int8_pc | 8.00 | 95.14 | 95.15 | [95.14, 95.15] | 95.15 | 95.13 | -0.04 | 1.92e-06 | 5.63e-06 | 0 | 64.00 |
| all_int4_pc | 4.00 | 94.99 | 94.98 | [94.98, 94.98] | 94.98 | 94.96 | 0.13 | 6.62e-06 | 3.43e-05 | 0 | 64.00 |

Paired difference in accuracy vs `e4m11` (same struck weights per trial; negative = worse than the reference):

| layout | trials | mean diff (pp) | s.e. | trials worse / equal / better |
|---|---|---|---|---|
| all_int8_pc | 40 | +0.153 | 0.051 | 9 / 2 / 29 |
| all_int4_pc | 40 | -0.016 | 0.052 | 32 / 0 / 8 |

## group `any`, sampling `weights`, k = 256

| layout | bits/weight | clean acc | mean acc | 95% CI of mean | median | p10 | drop vs E4M11 (pp) | median KL | mean KL | non-finite | 8-bit hits/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e4m11 | 16.00 | 95.11 | 91.77 | [89.27, 93.70] | 94.55 | 86.55 | 3.34 | 3.31e-02 | 1.68e-01 | 0 | 0.00 |
| all_int8_pc | 8.00 | 95.14 | 95.15 | [95.14, 95.15] | 95.15 | 95.13 | -0.04 | 1.63e-05 | 1.80e-05 | 0 | 256.00 |
| all_int4_pc | 4.00 | 94.99 | 94.98 | [94.97, 94.98] | 94.98 | 94.95 | 0.13 | 4.28e-05 | 5.28e-05 | 0 | 256.00 |

Paired difference in accuracy vs `e4m11` (same struck weights per trial; negative = worse than the reference):

| layout | trials | mean diff (pp) | s.e. | trials worse / equal / better |
|---|---|---|---|---|
| all_int8_pc | 40 | +3.383 | 1.144 | 2 / 2 / 36 |
| all_int4_pc | 40 | +3.210 | 1.144 | 11 / 0 / 29 |

## group `any`, sampling `weights`, k = 1024

| layout | bits/weight | clean acc | mean acc | 95% CI of mean | median | p10 | drop vs E4M11 (pp) | median KL | mean KL | non-finite | 8-bit hits/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e4m11 | 16.00 | 95.11 | 80.27 | [74.70, 85.08] | 88.99 | 66.69 | 14.84 | 2.84e-01 | 6.77e-01 | 0 | 0.00 |
| all_int8_pc | 8.00 | 95.14 | 95.15 | [95.14, 95.15] | 95.14 | 95.12 | -0.04 | 6.16e-05 | 1.26e-04 | 0 | 1024.00 |
| all_int4_pc | 4.00 | 94.99 | 94.97 | [94.96, 94.98] | 94.97 | 94.94 | 0.14 | 1.96e-04 | 3.26e-04 | 0 | 1024.00 |

Paired difference in accuracy vs `e4m11` (same struck weights per trial; negative = worse than the reference):

| layout | trials | mean diff (pp) | s.e. | trials worse / equal / better |
|---|---|---|---|---|
| all_int8_pc | 40 | +14.875 | 2.722 | 0 / 0 / 40 |
| all_int4_pc | 40 | +14.699 | 2.720 | 0 / 0 / 40 |

## group `any`, sampling `weights`, k = 4096

| layout | bits/weight | clean acc | mean acc | 95% CI of mean | median | p10 | drop vs E4M11 (pp) | median KL | mean KL | non-finite | 8-bit hits/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e4m11 | 16.00 | 95.11 | 27.17 | [22.60, 32.22] | 22.35 | 11.05 | 67.94 | 3.12e+00 | 4.05e+00 | 0 | 0.00 |
| all_int8_pc | 8.00 | 95.14 | 95.16 | [95.15, 95.16] | 95.16 | 95.11 | -0.05 | 3.36e-04 | 4.17e-04 | 0 | 4096.00 |
| all_int4_pc | 4.00 | 94.99 | 94.95 | [94.93, 94.96] | 94.95 | 94.89 | 0.16 | 1.04e-03 | 1.48e-03 | 0 | 4096.00 |

Paired difference in accuracy vs `e4m11` (same struck weights per trial; negative = worse than the reference):

| layout | trials | mean diff (pp) | s.e. | trials worse / equal / better |
|---|---|---|---|---|
| all_int8_pc | 40 | +67.981 | 2.523 | 0 / 0 / 40 |
| all_int4_pc | 40 | +67.772 | 2.523 | 0 / 0 / 40 |
