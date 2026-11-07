# Fault robustness of mixed E4M11 / 8-bit storage

* reference layout: `e4m11` (clean accuracy 95.29%); 'drop vs E4M11' always uses that value
* scale metadata of INT8 / scaled-FP8 layers is assumed fault-free; bit groups per format and the two sampling models are defined in `fr/faults_mixed.py` (project extension, not the paper)
* mean accuracy of faulty runs is heavy-tailed (rare catastrophic trials): read the median, p10 and bootstrap interval next to the mean

## group `any`, sampling `bits`, k = 256

| layout | bits/weight | clean acc | mean acc | 95% CI of mean | median | p10 | drop vs E4M11 (pp) | median KL | mean KL | non-finite | 8-bit hits/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e4m11 | 16.00 | 95.29 | 92.79 | [92.12, 93.40] | 94.14 | 87.62 | 2.50 | 4.84e-02 | 1.08e-01 | 0 | 0.00 |
| all_int8_pc | 8.00 | 95.16 | 95.17 | [95.17, 95.18] | 95.19 | 95.12 | 0.12 | 1.22e-05 | 2.51e-05 | 0 | 256.00 |
| all_int8_pt | 8.00 | 95.12 | 95.12 | [95.11, 95.13] | 95.12 | 95.05 | 0.17 | 4.98e-05 | 9.64e-05 | 0 | 256.00 |

## group `any`, sampling `bits`, k = 1024

| layout | bits/weight | clean acc | mean acc | 95% CI of mean | median | p10 | drop vs E4M11 (pp) | median KL | mean KL | non-finite | 8-bit hits/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e4m11 | 16.00 | 95.29 | 82.33 | [79.73, 84.77] | 86.64 | 60.68 | 12.96 | 3.83e-01 | 5.97e-01 | 0 | 0.00 |
| all_int8_pc | 8.00 | 95.16 | 95.18 | [95.16, 95.19] | 95.19 | 95.12 | 0.12 | 7.31e-05 | 3.77e-04 | 0 | 1024.00 |
| all_int8_pt | 8.00 | 95.12 | 95.11 | [95.09, 95.13] | 95.11 | 95.02 | 0.18 | 2.73e-04 | 6.40e-04 | 0 | 1024.00 |

## group `any`, sampling `bits`, k = 4096

| layout | bits/weight | clean acc | mean acc | 95% CI of mean | median | p10 | drop vs E4M11 (pp) | median KL | mean KL | non-finite | 8-bit hits/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e4m11 | 16.00 | 95.29 | 27.97 | [25.09, 31.09] | 24.51 | 11.64 | 67.32 | 3.02e+00 | 5.24e+00 | 0 | 0.00 |
| all_int8_pc | 8.00 | 95.16 | 95.13 | [95.11, 95.14] | 95.12 | 95.05 | 0.16 | 3.39e-04 | 5.31e-04 | 0 | 4096.00 |
| all_int8_pt | 8.00 | 95.12 | 95.06 | [95.04, 95.08] | 95.09 | 94.88 | 0.23 | 1.28e-03 | 1.81e-03 | 0 | 4096.00 |
