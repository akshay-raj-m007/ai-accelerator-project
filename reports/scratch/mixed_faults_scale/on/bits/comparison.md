# Fault robustness of mixed E4M11 / 8-bit storage

* reference layout: `e4m11` (clean accuracy 95.16%); 'drop vs E4M11' always uses that value
* scale metadata of INT layers is struck as well (float32 words; modes: on); bit groups per format and the two sampling models are defined in `fr/faults_mixed.py` (project extension, not the paper)
* mean accuracy of faulty runs is heavy-tailed (rare catastrophic trials): read the median, p10 and bootstrap interval next to the mean

## group `any`, sampling `bits`, k = 256

| layout | bits/weight | clean acc | mean acc | 95% CI of mean | median | p10 | drop vs E4M11 (pp) | median KL | mean KL | non-finite | 8-bit hits/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all_int8_pc | 8.00 | 95.16 | 94.26 | [92.54, 95.17] | 95.19 | 95.12 | 0.90 | 1.68e-05 | 6.40e-02 | 0 | 256.00 |
| all_int8_pt | 8.00 | 95.12 | 95.12 | [95.11, 95.13] | 95.12 | 95.05 | 0.03 | 4.98e-05 | 9.64e-05 | 0 | 256.00 |

## group `any`, sampling `bits`, k = 1024

| layout | bits/weight | clean acc | mean acc | 95% CI of mean | median | p10 | drop vs E4M11 (pp) | median KL | mean KL | non-finite | 8-bit hits/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all_int8_pc | 8.00 | 95.16 | 90.21 | [86.16, 93.44] | 95.16 | 94.60 | 4.95 | 8.86e-05 | 7.44e+34 | 3 | 1024.00 |
| all_int8_pt | 8.00 | 95.12 | 95.11 | [95.09, 95.13] | 95.11 | 95.02 | 0.05 | 2.73e-04 | 6.40e-04 | 0 | 1024.00 |

## group `any`, sampling `bits`, k = 4096

| layout | bits/weight | clean acc | mean acc | 95% CI of mean | median | p10 | drop vs E4M11 (pp) | median KL | mean KL | non-finite | 8-bit hits/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all_int8_pc | 8.00 | 95.16 | 71.57 | [64.84, 78.21] | 94.99 | 10.23 | 23.59 | 1.08e-03 | 1.82e+35 | 12 | 4096.00 |
| all_int8_pt | 8.00 | 95.12 | 93.83 | [91.73, 95.06] | 95.07 | 94.85 | 1.33 | 1.28e-03 | 1.74e-02 | 1 | 4096.00 |
