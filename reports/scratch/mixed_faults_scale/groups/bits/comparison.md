# Fault robustness of mixed E4M11 / 8-bit storage

* reference layout: `e4m11` (clean accuracy 95.29%); 'drop vs E4M11' always uses that value
* scale metadata of INT8 / scaled-FP8 layers is assumed fault-free; bit groups per format and the two sampling models are defined in `fr/faults_mixed.py` (project extension, not the paper)
* mean accuracy of faulty runs is heavy-tailed (rare catastrophic trials): read the median, p10 and bootstrap interval next to the mean

## group `exponent`, sampling `bits`, k = 1024

| layout | bits/weight | clean acc | mean acc | 95% CI of mean | median | p10 | drop vs E4M11 (pp) | median KL | mean KL | non-finite | 8-bit hits/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e4m11 | 16.00 | 95.29 | 27.16 | [23.55, 30.91] | 25.49 | 11.99 | 68.13 | 3.19e+00 | 4.56e+00 | 0 | 0.00 |

Not applicable (the layout has no `exponent` bits): all_int8_pc, all_int8_pt

## group `mantissa_lsb`, sampling `bits`, k = 1024

| layout | bits/weight | clean acc | mean acc | 95% CI of mean | median | p10 | drop vs E4M11 (pp) | median KL | mean KL | non-finite | 8-bit hits/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e4m11 | 16.00 | 95.29 | 95.29 | [95.29, 95.29] | 95.29 | 95.29 | -0.00 | 1.84e-08 | 1.90e-08 | 0 | 0.00 |
| all_int8_pc | 8.00 | 95.16 | 95.16 | [95.16, 95.16] | 95.16 | 95.16 | 0.13 | 2.84e-07 | 3.26e-07 | 0 | 1024.00 |
| all_int8_pt | 8.00 | 95.12 | 95.12 | [95.11, 95.12] | 95.12 | 95.09 | 0.18 | 8.98e-07 | 1.25e-06 | 0 | 1024.00 |

## group `mantissa_msb`, sampling `bits`, k = 1024

| layout | bits/weight | clean acc | mean acc | 95% CI of mean | median | p10 | drop vs E4M11 (pp) | median KL | mean KL | non-finite | 8-bit hits/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e4m11 | 16.00 | 95.29 | 95.28 | [95.28, 95.29] | 95.29 | 95.26 | 0.01 | 2.38e-06 | 3.99e-06 | 0 | 0.00 |
| all_int8_pc | 8.00 | 95.16 | 95.16 | [95.15, 95.17] | 95.16 | 95.12 | 0.13 | 4.21e-05 | 5.32e-05 | 0 | 1024.00 |
| all_int8_pt | 8.00 | 95.12 | 95.11 | [95.09, 95.12] | 95.09 | 95.05 | 0.18 | 1.30e-04 | 1.98e-04 | 0 | 1024.00 |

## group `sign`, sampling `bits`, k = 1024

| layout | bits/weight | clean acc | mean acc | 95% CI of mean | median | p10 | drop vs E4M11 (pp) | median KL | mean KL | non-finite | 8-bit hits/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e4m11 | 16.00 | 95.29 | 95.16 | [95.14, 95.18] | 95.19 | 95.05 | 0.13 | 1.18e-04 | 3.87e-04 | 0 | 0.00 |
| all_int8_pc | 8.00 | 95.16 | 95.11 | [95.08, 95.13] | 95.11 | 94.95 | 0.19 | 5.20e-04 | 8.71e-04 | 0 | 1024.00 |
| all_int8_pt | 8.00 | 95.12 | 95.02 | [94.98, 95.05] | 95.05 | 94.82 | 0.28 | 2.09e-03 | 3.43e-03 | 0 | 1024.00 |
