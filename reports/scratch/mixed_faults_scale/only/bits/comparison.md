# Fault robustness of mixed E4M11 / 8-bit storage

* reference layout: `e4m11` (clean accuracy 95.16%); 'drop vs E4M11' always uses that value
* scale metadata of INT layers is struck as well (float32 words; modes: only); bit groups per format and the two sampling models are defined in `fr/faults_mixed.py` (project extension, not the paper)
* mean accuracy of faulty runs is heavy-tailed (rare catastrophic trials): read the median, p10 and bootstrap interval next to the mean

## group `any`, sampling `bits`, k = 1

| layout | bits/weight | clean acc | mean acc | 95% CI of mean | median | p10 | drop vs E4M11 (pp) | median KL | mean KL | non-finite | 8-bit hits/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all_int8_pc | 8.00 | 95.16 | 94.20 | [93.09, 95.02] | 95.16 | 95.08 | 0.95 | 4.09e-10 | 1.02e+34 | 1 | 1.00 |
| all_int8_pt | 8.00 | 95.12 | 85.40 | [80.35, 89.97] | 95.12 | 37.40 | 9.76 | 3.82e-06 | 6.66e-01 | 2 | 1.00 |

## group `any`, sampling `bits`, k = 8

| layout | bits/weight | clean acc | mean acc | 95% CI of mean | median | p10 | drop vs E4M11 (pp) | median KL | mean KL | non-finite | 8-bit hits/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all_int8_pc | 8.00 | 95.16 | 71.71 | [64.78, 78.16] | 95.00 | 11.62 | 23.45 | 1.48e-03 | 1.47e+35 | 14 | 8.00 |
| all_int8_pt | 8.00 | 95.12 | 34.42 | [27.60, 41.66] | 10.76 | 9.58 | 60.73 | 2.81e+00 | 1.56e+30 | 24 | 8.00 |

## group `exponent`, sampling `bits`, k = 1

| layout | bits/weight | clean acc | mean acc | 95% CI of mean | median | p10 | drop vs E4M11 (pp) | median KL | mean KL | non-finite | 8-bit hits/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all_int8_pc | 8.00 | 95.16 | 81.32 | [75.73, 86.74] | 95.16 | 18.15 | 13.84 | 1.69e-04 | 6.87e+34 | 8 | 1.00 |
| all_int8_pt | 8.00 | 95.12 | 50.98 | [44.46, 57.55] | 40.04 | 9.69 | 44.18 | 1.95e+00 | 3.42e+00 | 11 | 1.00 |

## group `mantissa_lsb`, sampling `bits`, k = 1

| layout | bits/weight | clean acc | mean acc | 95% CI of mean | median | p10 | drop vs E4M11 (pp) | median KL | mean KL | non-finite | 8-bit hits/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all_int8_pc | 8.00 | 95.16 | 95.16 | [95.16, 95.16] | 95.16 | 95.16 | -0.00 | 0.00e+00 | 4.13e-10 | 0 | 1.00 |
| all_int8_pt | 8.00 | 95.12 | 95.12 | [95.12, 95.12] | 95.12 | 95.12 | 0.03 | 0.00e+00 | 5.19e-09 | 0 | 1.00 |

## group `mantissa_msb`, sampling `bits`, k = 1

| layout | bits/weight | clean acc | mean acc | 95% CI of mean | median | p10 | drop vs E4M11 (pp) | median KL | mean KL | non-finite | 8-bit hits/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all_int8_pc | 8.00 | 95.16 | 95.16 | [95.16, 95.16] | 95.16 | 95.16 | -0.00 | 5.97e-09 | 2.34e-06 | 0 | 1.00 |
| all_int8_pt | 8.00 | 95.12 | 95.10 | [95.06, 95.12] | 95.12 | 95.02 | 0.06 | 6.49e-06 | 1.76e-03 | 0 | 1.00 |

## group `sign`, sampling `bits`, k = 1

| layout | bits/weight | clean acc | mean acc | 95% CI of mean | median | p10 | drop vs E4M11 (pp) | median KL | mean KL | non-finite | 8-bit hits/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all_int8_pc | 8.00 | 95.16 | 95.00 | [94.91, 95.07] | 95.16 | 94.81 | 0.16 | 4.69e-04 | 7.07e-03 | 0 | 1.00 |
| all_int8_pt | 8.00 | 95.12 | 17.13 | [12.94, 21.83] | 9.86 | 0.06 | 78.02 | 5.15e+00 | 5.35e+00 | 0 | 1.00 |
