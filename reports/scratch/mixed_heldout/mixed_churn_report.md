# Prediction churn vs the all-E4M11 model

* flips = images whose predicted class differs from the all-E4M11 model; c2w / w2c = right->wrong / wrong->right; KL = mean KL(softmax(E4M11) || softmax(config)) in nats.
* Lower is closer to the E4M11 model. `random` rows: mean +/- std over the draws. The sets and budgets are the ones from the accuracy report.

## val

| reference | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|
| fp32 | 0 | 0 | 0 | 4.50e-07 | 0.0006 |
| int8_pc_all | 5 | 4 | 0 | 5.63e-05 | 0.0075 |
| int8_pt_all | 7 | 6 | 1 | 1.73e-04 | 0.0124 |
| fp8_scaled_all | 13 | 7 | 4 | 9.98e-04 | 0.0310 |
| int4_pc_all | 37 | 23 | 10 | 1.25e-02 | 0.1317 |
| int4_pt_all | 94 | 59 | 23 | 5.54e-02 | 0.4263 |
| fp4_scaled_all | 114 | 70 | 30 | 6.93e-02 | 0.3537 |

### int8_pc, target <= 14 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.75 | 4 | 0 | 0 | 0 | 3.44e-06 | 0.0016 |
| Fisher only (lambda=0) | 0 | 3 | 0 | 0 | 0 | 4.41e-06 | 0.0017 |
| error only (lambda=1) | 1 | 4 | 0 | 0 | 0 | 3.44e-06 | 0.0016 |
| product |  | 5 | 0 | 0 | 0 | 2.05e-06 | 0.0013 |
| worst |  | 17 | 4 | 3 | 0 | 5.61e-05 | 0.0073 |
| random (n=5) | - | - | 1 +/- 1 | - | - | 2.07e-05 +/- 1.3e-05 | - |

### int8_pc, target <= 12 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.5 | 4 | 0 | 0 | 0 | 5.83e-06 | 0.0024 |
| Fisher only (lambda=0) | 0 | 4 | 0 | 0 | 0 | 3.85e-06 | 0.0022 |
| error only (lambda=1) | 1 | 4 | 0 | 0 | 0 | 5.83e-06 | 0.0024 |
| product |  | 5 | 1 | 1 | 0 | 1.86e-06 | 0.0017 |
| worst |  | 18 | 4 | 3 | 0 | 5.60e-05 | 0.0073 |
| random (n=5) | - | - | 1 +/- 1 | - | - | 2.51e-05 +/- 1.4e-05 | - |

### int8_pc, target <= 10 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 5 | 1 | 1 | 0 | 4.38e-07 | 0.0015 |
| Fisher only (lambda=0) | 0 | 5 | 1 | 1 | 0 | 4.38e-07 | 0.0015 |
| error only (lambda=1) | 1 | 5 | 1 | 1 | 0 | 4.38e-07 | 0.0015 |
| product |  | 5 | 1 | 1 | 0 | 4.38e-07 | 0.0015 |
| worst |  | 19 | 5 | 4 | 0 | 5.62e-05 | 0.0075 |
| random (n=5) | - | - | 2 +/- 1 | - | - | 3.49e-05 +/- 1.3e-05 | - |

### int8_pc, target <= 8 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 21 | 5 | 4 | 0 | 5.63e-05 | 0.0075 |
| Fisher only (lambda=0) | 0 | 21 | 5 | 4 | 0 | 5.63e-05 | 0.0075 |
| error only (lambda=1) | 1 | 21 | 5 | 4 | 0 | 5.63e-05 | 0.0075 |
| product |  | 21 | 5 | 4 | 0 | 5.63e-05 | 0.0075 |
| worst |  | 21 | 5 | 4 | 0 | 5.63e-05 | 0.0075 |
| random (n=5) | - | - | 5 +/- 0 | - | - | 5.63e-05 +/- 0.0e+00 | - |

### int8_pc, target <= 6 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|

### int8_pc, target <= 5 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|

### int8_pt, target <= 14 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 1 | 1 | 0 | 1.55e-05 | 0.0034 |
| Fisher only (lambda=0) | 0 | 3 | 1 | 1 | 0 | 1.55e-05 | 0.0034 |
| error only (lambda=1) | 1 | 3 | 1 | 1 | 0 | 1.84e-05 | 0.0035 |
| product |  | 5 | 0 | 0 | 0 | 8.15e-06 | 0.0028 |
| worst |  | 17 | 7 | 6 | 1 | 1.66e-04 | 0.0115 |
| random (n=5) | - | - | 3 +/- 2 | - | - | 5.33e-05 +/- 2.0e-05 | - |

### int8_pt, target <= 12 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.25 | 4 | 1 | 1 | 0 | 1.99e-05 | 0.0047 |
| Fisher only (lambda=0) | 0 | 4 | 1 | 1 | 0 | 1.58e-05 | 0.0045 |
| error only (lambda=1) | 1 | 5 | 2 | 2 | 0 | 2.74e-05 | 0.0051 |
| product |  | 5 | 1 | 1 | 0 | 6.59e-06 | 0.0037 |
| worst |  | 18 | 6 | 6 | 0 | 1.75e-04 | 0.0121 |
| random (n=5) | - | - | 2 +/- 2 | - | - | 6.25e-05 +/- 3.1e-05 | - |

### int8_pt, target <= 10 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 5 | 0 | 0 | 0 | 3.37e-06 | 0.0040 |
| Fisher only (lambda=0) | 0 | 5 | 0 | 0 | 0 | 3.37e-06 | 0.0040 |
| error only (lambda=1) | 1 | 6 | 1 | 1 | 0 | 2.33e-05 | 0.0055 |
| product |  | 5 | 0 | 0 | 0 | 3.37e-06 | 0.0040 |
| worst |  | 19 | 6 | 6 | 0 | 1.72e-04 | 0.0124 |
| random (n=5) | - | - | 4 +/- 2 | - | - | 9.52e-05 +/- 3.5e-05 | - |

### int8_pt, target <= 8 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 21 | 7 | 6 | 1 | 1.73e-04 | 0.0124 |
| Fisher only (lambda=0) | 0 | 21 | 7 | 6 | 1 | 1.73e-04 | 0.0124 |
| error only (lambda=1) | 1 | 21 | 7 | 6 | 1 | 1.73e-04 | 0.0124 |
| product |  | 21 | 7 | 6 | 1 | 1.73e-04 | 0.0124 |
| worst |  | 21 | 7 | 6 | 1 | 1.73e-04 | 0.0124 |
| random (n=5) | - | - | 7 +/- 0 | - | - | 1.73e-04 +/- 0.0e+00 | - |

### int8_pt, target <= 6 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|

### int8_pt, target <= 5 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|

### fp8_scaled, target <= 14 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 2 | 2 | 0 | 2.73e-05 | 0.0051 |
| Fisher only (lambda=0) | 0 | 3 | 2 | 2 | 0 | 2.73e-05 | 0.0051 |
| error only (lambda=1) | 1 | 3 | 2 | 2 | 0 | 2.85e-05 | 0.0057 |
| product |  | 5 | 2 | 2 | 0 | 3.25e-05 | 0.0054 |
| worst |  | 17 | 12 | 6 | 4 | 9.97e-04 | 0.0307 |
| random (n=5) | - | - | 8 +/- 3 | - | - | 5.35e-04 +/- 3.7e-04 | - |

### fp8_scaled, target <= 12 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 4 | 1 | 1 | 0 | 2.84e-05 | 0.0058 |
| Fisher only (lambda=0) | 0 | 4 | 1 | 1 | 0 | 2.84e-05 | 0.0058 |
| error only (lambda=1) | 1 | 4 | 2 | 2 | 0 | 3.87e-05 | 0.0069 |
| product |  | 5 | 1 | 1 | 0 | 1.52e-05 | 0.0050 |
| worst |  | 18 | 14 | 8 | 4 | 9.70e-04 | 0.0306 |
| random (n=5) | - | - | 6 +/- 4 | - | - | 3.48e-04 +/- 2.7e-04 | - |

### fp8_scaled, target <= 10 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.5 | 5 | 0 | 0 | 0 | 8.43e-06 | 0.0048 |
| Fisher only (lambda=0) | 0 | 5 | 0 | 0 | 0 | 4.56e-06 | 0.0045 |
| error only (lambda=1) | 1 | 5 | 0 | 0 | 0 | 8.43e-06 | 0.0048 |
| product |  | 5 | 0 | 0 | 0 | 4.56e-06 | 0.0045 |
| worst |  | 19 | 13 | 7 | 4 | 9.96e-04 | 0.0308 |
| random (n=5) | - | - | 9 +/- 3 | - | - | 6.30e-04 +/- 3.3e-04 | - |

### fp8_scaled, target <= 8 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 21 | 13 | 7 | 4 | 9.98e-04 | 0.0310 |
| Fisher only (lambda=0) | 0 | 21 | 13 | 7 | 4 | 9.98e-04 | 0.0310 |
| error only (lambda=1) | 1 | 21 | 13 | 7 | 4 | 9.98e-04 | 0.0310 |
| product |  | 21 | 13 | 7 | 4 | 9.98e-04 | 0.0310 |
| worst |  | 21 | 13 | 7 | 4 | 9.98e-04 | 0.0310 |
| random (n=5) | - | - | 13 +/- 0 | - | - | 9.98e-04 +/- 0.0e+00 | - |

### fp8_scaled, target <= 6 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|

### fp8_scaled, target <= 5 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|

### int4_pc, target <= 14 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 7 | 7 | 0 | 3.49e-04 | 0.0263 |
| Fisher only (lambda=0) | 0 | 3 | 7 | 7 | 0 | 3.49e-04 | 0.0263 |
| error only (lambda=1) | 1 | 3 | 13 | 10 | 2 | 1.19e-03 | 0.0343 |
| product |  | 3 | 7 | 7 | 0 | 3.49e-04 | 0.0263 |
| worst |  | 13 | 43 | 26 | 13 | 1.16e-02 | 0.1221 |
| random (n=5) | - | - | 35 +/- 3 | - | - | 9.09e-03 +/- 1.5e-03 | - |

### int4_pc, target <= 12 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 13 | 8 | 2 | 1.01e-03 | 0.0315 |
| Fisher only (lambda=0) | 0 | 3 | 13 | 8 | 2 | 1.01e-03 | 0.0315 |
| error only (lambda=1) | 1 | 4 | 16 | 11 | 1 | 8.81e-04 | 0.0310 |
| product |  | 4 | 9 | 7 | 1 | 3.90e-04 | 0.0250 |
| worst |  | 16 | 40 | 25 | 11 | 1.26e-02 | 0.1274 |
| random (n=5) | - | - | 32 +/- 7 | - | - | 8.22e-03 +/- 2.9e-03 | - |

### int4_pc, target <= 10 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 4 | 13 | 8 | 3 | 1.04e-03 | 0.0405 |
| Fisher only (lambda=0) | 0 | 4 | 13 | 8 | 3 | 1.04e-03 | 0.0405 |
| error only (lambda=1) | 1 | 4 | 19 | 11 | 4 | 2.04e-03 | 0.0435 |
| product |  | 5 | 9 | 7 | 2 | 6.19e-04 | 0.0371 |
| worst |  | 18 | 41 | 25 | 11 | 1.23e-02 | 0.1267 |
| random (n=5) | - | - | 28 +/- 6 | - | - | 5.99e-03 +/- 1.9e-03 | - |

### int4_pc, target <= 8 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 4 | 7 | 7 | 0 | 2.93e-04 | 0.0324 |
| Fisher only (lambda=0) | 0 | 4 | 7 | 7 | 0 | 2.93e-04 | 0.0324 |
| error only (lambda=1) | 1 | 4 | 15 | 12 | 1 | 1.26e-03 | 0.0427 |
| product |  | 4 | 7 | 7 | 0 | 2.93e-04 | 0.0324 |
| worst |  | 19 | 40 | 24 | 12 | 1.23e-02 | 0.1295 |
| random (n=5) | - | - | 31 +/- 5 | - | - | 8.36e-03 +/- 2.2e-03 | - |

### int4_pc, target <= 6 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.75 | 6 | 17 | 12 | 4 | 2.16e-03 | 0.0536 |
| Fisher only (lambda=0) | 0 | 6 | 12 | 10 | 2 | 1.14e-03 | 0.0466 |
| error only (lambda=1) | 1 | 6 | 17 | 12 | 4 | 2.16e-03 | 0.0536 |
| product |  | 9 | 8 | 6 | 1 | 1.24e-03 | 0.0478 |
| worst |  | 18 | 39 | 24 | 11 | 1.22e-02 | 0.1291 |
| random (n=5) | - | - | 30 +/- 9 | - | - | 7.00e-03 +/- 3.1e-03 | - |

### int4_pc, target <= 5 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 8 | 16 | 10 | 4 | 2.90e-03 | 0.0616 |
| Fisher only (lambda=0) | 0 | 8 | 16 | 10 | 4 | 2.90e-03 | 0.0616 |
| error only (lambda=1) | 1 | 8 | 16 | 10 | 4 | 2.90e-03 | 0.0616 |
| product |  | 9 | 18 | 11 | 5 | 2.93e-03 | 0.0608 |
| worst |  | 20 | 41 | 25 | 11 | 1.23e-02 | 0.1299 |
| random (n=5) | - | - | 33 +/- 2 | - | - | 9.04e-03 +/- 1.7e-03 | - |

### int4_pt, target <= 14 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.25 | 3 | 20 | 11 | 4 | 4.13e-03 | 0.1020 |
| Fisher only (lambda=0) | 0 | 3 | 19 | 13 | 3 | 3.25e-03 | 0.1002 |
| error only (lambda=1) | 1 | 4 | 32 | 24 | 6 | 7.69e-03 | 0.0904 |
| product |  | 3 | 19 | 13 | 3 | 3.25e-03 | 0.1002 |
| worst |  | 13 | 99 | 63 | 29 | 5.63e-02 | 0.2344 |
| random (n=5) | - | - | 78 +/- 9 | - | - | 4.05e-02 +/- 5.3e-03 | - |

### int4_pt, target <= 12 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 31 | 16 | 7 | 7.04e-03 | 0.2360 |
| Fisher only (lambda=0) | 0 | 3 | 31 | 16 | 7 | 7.04e-03 | 0.2360 |
| error only (lambda=1) | 1 | 5 | 22 | 13 | 4 | 6.39e-03 | 0.2008 |
| product |  | 4 | 25 | 14 | 6 | 4.17e-03 | 0.2314 |
| worst |  | 16 | 106 | 69 | 28 | 5.84e-02 | 0.2648 |
| random (n=5) | - | - | 72 +/- 17 | - | - | 3.56e-02 +/- 1.3e-02 | - |

### int4_pt, target <= 10 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.25 | 4 | 35 | 16 | 12 | 8.32e-03 | 0.2583 |
| Fisher only (lambda=0) | 0 | 4 | 35 | 18 | 10 | 7.69e-03 | 0.2671 |
| error only (lambda=1) | 1 | 5 | 23 | 14 | 3 | 6.17e-03 | 0.2622 |
| product |  | 5 | 25 | 16 | 7 | 2.91e-03 | 0.2521 |
| worst |  | 18 | 104 | 69 | 24 | 5.67e-02 | 0.4098 |
| random (n=5) | - | - | 60 +/- 18 | - | - | 2.74e-02 +/- 1.1e-02 | - |

### int4_pt, target <= 8 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.75 | 4 | 24 | 12 | 8 | 7.11e-03 | 0.2498 |
| Fisher only (lambda=0) | 0 | 4 | 18 | 12 | 3 | 5.03e-03 | 0.2471 |
| error only (lambda=1) | 1 | 4 | 24 | 12 | 8 | 7.11e-03 | 0.2498 |
| product |  | 4 | 18 | 12 | 3 | 5.03e-03 | 0.2471 |
| worst |  | 19 | 98 | 62 | 25 | 5.48e-02 | 0.3472 |
| random (n=5) | - | - | 76 +/- 10 | - | - | 3.84e-02 +/- 5.5e-03 | - |

### int4_pt, target <= 6 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 6 | 29 | 16 | 7 | 1.01e-02 | 0.2967 |
| Fisher only (lambda=0) | 0 | 6 | 29 | 16 | 7 | 1.01e-02 | 0.2967 |
| error only (lambda=1) | 1 | 6 | 28 | 17 | 5 | 1.18e-02 | 0.2988 |
| product |  | 6 | 29 | 16 | 7 | 1.01e-02 | 0.2967 |
| worst |  | 18 | 100 | 65 | 25 | 5.29e-02 | 0.3803 |
| random (n=5) | - | - | 67 +/- 15 | - | - | 3.26e-02 +/- 1.2e-02 | - |

### int4_pt, target <= 5 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.75 | 9 | 36 | 21 | 9 | 1.42e-02 | 0.3074 |
| Fisher only (lambda=0) | 0 | 8 | 44 | 29 | 9 | 1.51e-02 | 0.3089 |
| error only (lambda=1) | 1 | 9 | 36 | 21 | 9 | 1.42e-02 | 0.3074 |
| product |  | 9 | 45 | 27 | 10 | 1.43e-02 | 0.3201 |
| worst |  | 20 | 93 | 60 | 22 | 5.47e-02 | 0.4234 |
| random (n=5) | - | - | 75 +/- 10 | - | - | 3.89e-02 +/- 8.1e-03 | - |

### fp4_scaled, target <= 14 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 14 | 8 | 4 | 9.47e-04 | 0.0471 |
| Fisher only (lambda=0) | 0 | 3 | 14 | 8 | 4 | 9.47e-04 | 0.0471 |
| error only (lambda=1) | 1 | 4 | 27 | 14 | 9 | 3.13e-03 | 0.0554 |
| product |  | 3 | 14 | 8 | 4 | 9.47e-04 | 0.0471 |
| worst |  | 13 | 99 | 60 | 27 | 6.69e-02 | 0.2386 |
| random (n=5) | - | - | 79 +/- 10 | - | - | 4.16e-02 +/- 1.1e-02 | - |

### fp4_scaled, target <= 12 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 24 | 9 | 9 | 4.09e-03 | 0.0939 |
| Fisher only (lambda=0) | 0 | 3 | 24 | 9 | 9 | 4.09e-03 | 0.0939 |
| error only (lambda=1) | 1 | 5 | 31 | 18 | 7 | 3.54e-03 | 0.0836 |
| product |  | 4 | 12 | 8 | 2 | 8.69e-04 | 0.0774 |
| worst |  | 16 | 106 | 63 | 28 | 7.38e-02 | 0.2550 |
| random (n=5) | - | - | 65 +/- 24 | - | - | 3.34e-02 +/- 1.7e-02 | - |

### fp4_scaled, target <= 10 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 4 | 25 | 11 | 9 | 4.46e-03 | 0.1346 |
| Fisher only (lambda=0) | 0 | 4 | 25 | 11 | 9 | 4.46e-03 | 0.1346 |
| error only (lambda=1) | 1 | 5 | 22 | 12 | 5 | 3.09e-03 | 0.1260 |
| product |  | 5 | 15 | 10 | 3 | 1.22e-03 | 0.1258 |
| worst |  | 18 | 111 | 66 | 31 | 6.77e-02 | 0.3661 |
| random (n=5) | - | - | 56 +/- 12 | - | - | 2.34e-02 +/- 9.1e-03 | - |

### fp4_scaled, target <= 8 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 4 | 17 | 9 | 6 | 3.81e-03 | 0.1652 |
| Fisher only (lambda=0) | 0 | 4 | 17 | 9 | 6 | 3.81e-03 | 0.1652 |
| error only (lambda=1) | 1 | 4 | 18 | 11 | 5 | 3.97e-03 | 0.1643 |
| product |  | 4 | 17 | 9 | 6 | 3.81e-03 | 0.1652 |
| worst |  | 19 | 121 | 74 | 31 | 7.13e-02 | 0.3421 |
| random (n=5) | - | - | 72 +/- 13 | - | - | 3.50e-02 +/- 1.3e-02 | - |

### fp4_scaled, target <= 6 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 6 | 25 | 14 | 8 | 4.70e-03 | 0.1753 |
| Fisher only (lambda=0) | 0 | 6 | 25 | 14 | 8 | 4.70e-03 | 0.1753 |
| error only (lambda=1) | 1 | 6 | 25 | 14 | 7 | 4.91e-03 | 0.1751 |
| product |  | 6 | 25 | 14 | 8 | 4.70e-03 | 0.1753 |
| worst |  | 18 | 121 | 73 | 30 | 7.12e-02 | 0.3460 |
| random (n=5) | - | - | 67 +/- 26 | - | - | 3.27e-02 +/- 1.9e-02 | - |

### fp4_scaled, target <= 5 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 8 | 32 | 11 | 12 | 8.08e-03 | 0.1849 |
| Fisher only (lambda=0) | 0 | 8 | 32 | 11 | 12 | 8.08e-03 | 0.1849 |
| error only (lambda=1) | 1 | 9 | 30 | 18 | 7 | 6.88e-03 | 0.1833 |
| product |  | 9 | 33 | 20 | 9 | 6.50e-03 | 0.1887 |
| worst |  | 20 | 113 | 68 | 31 | 6.74e-02 | 0.3497 |
| random (n=5) | - | - | 76 +/- 21 | - | - | 3.89e-02 +/- 1.4e-02 | - |

### Summary (val)

* lambda0: KL below the random-draw mean in 27 of 30 (arm, target) cells
* product: KL below the random-draw mean in 27 of 30 (arm, target) cells
* Spearman(KL, weights switched to 8 bit) = 0.06; Spearman(KL, -SQNR) = 0.94 (over 360 configurations)

## test

| reference | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|
| fp32 | 0 | 0 | 0 | 3.91e-07 | 0.0006 |
| int8_pc_all | 8 | 2 | 5 | 5.23e-05 | 0.0074 |
| int8_pt_all | 12 | 2 | 6 | 1.59e-04 | 0.0122 |
| fp8_scaled_all | 25 | 7 | 14 | 1.04e-03 | 0.0310 |
| int4_pc_all | 142 | 65 | 53 | 1.51e-02 | 0.1318 |
| int4_pt_all | 351 | 212 | 89 | 5.81e-02 | 0.4233 |
| fp4_scaled_all | 368 | 217 | 84 | 6.90e-02 | 0.3496 |

### int8_pc, target <= 14 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.75 | 4 | 1 | 0 | 0 | 3.10e-06 | 0.0016 |
| Fisher only (lambda=0) | 0 | 3 | 4 | 1 | 2 | 3.98e-06 | 0.0017 |
| error only (lambda=1) | 1 | 4 | 1 | 0 | 0 | 3.10e-06 | 0.0016 |
| product |  | 5 | 0 | 0 | 0 | 1.78e-06 | 0.0013 |
| worst |  | 17 | 8 | 2 | 5 | 5.18e-05 | 0.0073 |
| random (n=5) | - | - | 4 +/- 1 | - | - | 2.04e-05 +/- 1.2e-05 | - |

### int8_pc, target <= 12 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.5 | 4 | 3 | 1 | 1 | 6.27e-06 | 0.0024 |
| Fisher only (lambda=0) | 0 | 4 | 4 | 1 | 2 | 4.17e-06 | 0.0022 |
| error only (lambda=1) | 1 | 4 | 3 | 1 | 1 | 6.27e-06 | 0.0024 |
| product |  | 5 | 2 | 0 | 1 | 1.80e-06 | 0.0017 |
| worst |  | 18 | 7 | 0 | 6 | 5.15e-05 | 0.0073 |
| random (n=5) | - | - | 4 +/- 3 | - | - | 2.27e-05 +/- 1.2e-05 | - |

### int8_pc, target <= 10 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 5 | 0 | 0 | 0 | 4.24e-07 | 0.0015 |
| Fisher only (lambda=0) | 0 | 5 | 0 | 0 | 0 | 4.24e-07 | 0.0015 |
| error only (lambda=1) | 1 | 5 | 0 | 0 | 0 | 4.24e-07 | 0.0015 |
| product |  | 5 | 0 | 0 | 0 | 4.24e-07 | 0.0015 |
| worst |  | 19 | 7 | 2 | 4 | 5.22e-05 | 0.0074 |
| random (n=5) | - | - | 3 +/- 1 | - | - | 3.19e-05 +/- 1.2e-05 | - |

### int8_pc, target <= 8 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 21 | 8 | 2 | 5 | 5.23e-05 | 0.0074 |
| Fisher only (lambda=0) | 0 | 21 | 8 | 2 | 5 | 5.23e-05 | 0.0074 |
| error only (lambda=1) | 1 | 21 | 8 | 2 | 5 | 5.23e-05 | 0.0074 |
| product |  | 21 | 8 | 2 | 5 | 5.23e-05 | 0.0074 |
| worst |  | 21 | 8 | 2 | 5 | 5.23e-05 | 0.0074 |
| random (n=5) | - | - | 8 +/- 0 | - | - | 5.23e-05 +/- 6.8e-21 | - |

### int8_pc, target <= 6 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|

### int8_pc, target <= 5 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|

### int8_pt, target <= 14 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 3 | 0 | 2 | 1.30e-05 | 0.0034 |
| Fisher only (lambda=0) | 0 | 3 | 3 | 0 | 2 | 1.30e-05 | 0.0034 |
| error only (lambda=1) | 1 | 3 | 5 | 2 | 2 | 1.33e-05 | 0.0034 |
| product |  | 5 | 2 | 0 | 1 | 8.03e-06 | 0.0028 |
| worst |  | 17 | 15 | 3 | 8 | 1.52e-04 | 0.0114 |
| random (n=5) | - | - | 7 +/- 3 | - | - | 5.11e-05 +/- 2.3e-05 | - |

### int8_pt, target <= 12 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.25 | 4 | 6 | 2 | 3 | 1.84e-05 | 0.0046 |
| Fisher only (lambda=0) | 0 | 4 | 5 | 2 | 2 | 1.26e-05 | 0.0044 |
| error only (lambda=1) | 1 | 5 | 2 | 0 | 1 | 2.15e-05 | 0.0049 |
| product |  | 5 | 0 | 0 | 0 | 5.97e-06 | 0.0037 |
| worst |  | 18 | 12 | 2 | 6 | 1.58e-04 | 0.0120 |
| random (n=5) | - | - | 6 +/- 4 | - | - | 5.19e-05 +/- 2.2e-05 | - |

### int8_pt, target <= 10 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 5 | 1 | 0 | 0 | 3.03e-06 | 0.0040 |
| Fisher only (lambda=0) | 0 | 5 | 1 | 0 | 0 | 3.03e-06 | 0.0040 |
| error only (lambda=1) | 1 | 6 | 1 | 0 | 0 | 1.96e-05 | 0.0054 |
| product |  | 5 | 1 | 0 | 0 | 3.03e-06 | 0.0040 |
| worst |  | 19 | 13 | 2 | 7 | 1.59e-04 | 0.0122 |
| random (n=5) | - | - | 7 +/- 3 | - | - | 8.65e-05 +/- 3.1e-05 | - |

### int8_pt, target <= 8 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 21 | 12 | 2 | 6 | 1.59e-04 | 0.0122 |
| Fisher only (lambda=0) | 0 | 21 | 12 | 2 | 6 | 1.59e-04 | 0.0122 |
| error only (lambda=1) | 1 | 21 | 12 | 2 | 6 | 1.59e-04 | 0.0122 |
| product |  | 21 | 12 | 2 | 6 | 1.59e-04 | 0.0122 |
| worst |  | 21 | 12 | 2 | 6 | 1.59e-04 | 0.0122 |
| random (n=5) | - | - | 12 +/- 0 | - | - | 1.59e-04 +/- 0.0e+00 | - |

### int8_pt, target <= 6 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|

### int8_pt, target <= 5 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|

### fp8_scaled, target <= 14 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 3 | 0 | 3 | 2.63e-05 | 0.0050 |
| Fisher only (lambda=0) | 0 | 3 | 3 | 0 | 3 | 2.63e-05 | 0.0050 |
| error only (lambda=1) | 1 | 3 | 3 | 0 | 3 | 2.78e-05 | 0.0057 |
| product |  | 5 | 4 | 0 | 2 | 3.14e-05 | 0.0053 |
| worst |  | 17 | 24 | 7 | 13 | 1.04e-03 | 0.0306 |
| random (n=5) | - | - | 18 +/- 7 | - | - | 5.60e-04 +/- 4.0e-04 | - |

### fp8_scaled, target <= 12 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 4 | 4 | 1 | 3 | 2.60e-05 | 0.0058 |
| Fisher only (lambda=0) | 0 | 4 | 4 | 1 | 3 | 2.60e-05 | 0.0058 |
| error only (lambda=1) | 1 | 4 | 4 | 0 | 3 | 3.96e-05 | 0.0069 |
| product |  | 5 | 2 | 0 | 0 | 1.43e-05 | 0.0050 |
| worst |  | 18 | 25 | 8 | 13 | 1.03e-03 | 0.0306 |
| random (n=5) | - | - | 12 +/- 6 | - | - | 3.35e-04 +/- 3.1e-04 | - |

### fp8_scaled, target <= 10 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.5 | 5 | 2 | 0 | 1 | 8.98e-06 | 0.0047 |
| Fisher only (lambda=0) | 0 | 5 | 0 | 0 | 0 | 4.19e-06 | 0.0044 |
| error only (lambda=1) | 1 | 5 | 2 | 0 | 1 | 8.98e-06 | 0.0047 |
| product |  | 5 | 0 | 0 | 0 | 4.19e-06 | 0.0044 |
| worst |  | 19 | 25 | 7 | 14 | 1.04e-03 | 0.0307 |
| random (n=5) | - | - | 20 +/- 8 | - | - | 6.58e-04 +/- 3.8e-04 | - |

### fp8_scaled, target <= 8 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 21 | 25 | 7 | 14 | 1.04e-03 | 0.0310 |
| Fisher only (lambda=0) | 0 | 21 | 25 | 7 | 14 | 1.04e-03 | 0.0310 |
| error only (lambda=1) | 1 | 21 | 25 | 7 | 14 | 1.04e-03 | 0.0310 |
| product |  | 21 | 25 | 7 | 14 | 1.04e-03 | 0.0310 |
| worst |  | 21 | 25 | 7 | 14 | 1.04e-03 | 0.0310 |
| random (n=5) | - | - | 25 +/- 0 | - | - | 1.04e-03 +/- 0.0e+00 | - |

### fp8_scaled, target <= 6 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|

### fp8_scaled, target <= 5 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|

### int4_pc, target <= 14 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 21 | 4 | 11 | 3.58e-04 | 0.0261 |
| Fisher only (lambda=0) | 0 | 3 | 21 | 4 | 11 | 3.58e-04 | 0.0261 |
| error only (lambda=1) | 1 | 3 | 35 | 12 | 18 | 1.15e-03 | 0.0340 |
| product |  | 3 | 21 | 4 | 11 | 3.58e-04 | 0.0261 |
| worst |  | 13 | 140 | 68 | 47 | 1.34e-02 | 0.1217 |
| random (n=5) | - | - | 120 +/- 14 | - | - | 1.07e-02 +/- 2.0e-03 | - |

### int4_pc, target <= 12 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 27 | 7 | 17 | 8.40e-04 | 0.0312 |
| Fisher only (lambda=0) | 0 | 3 | 27 | 7 | 17 | 8.40e-04 | 0.0312 |
| error only (lambda=1) | 1 | 4 | 26 | 7 | 14 | 7.30e-04 | 0.0304 |
| product |  | 4 | 16 | 4 | 8 | 3.65e-04 | 0.0247 |
| worst |  | 16 | 137 | 64 | 51 | 1.50e-02 | 0.1273 |
| random (n=5) | - | - | 106 +/- 22 | - | - | 9.20e-03 +/- 3.5e-03 | - |

### int4_pc, target <= 10 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 4 | 32 | 8 | 18 | 1.05e-03 | 0.0404 |
| Fisher only (lambda=0) | 0 | 4 | 32 | 8 | 18 | 1.05e-03 | 0.0404 |
| error only (lambda=1) | 1 | 4 | 44 | 14 | 22 | 1.86e-03 | 0.0428 |
| product |  | 5 | 23 | 5 | 15 | 5.66e-04 | 0.0368 |
| worst |  | 18 | 137 | 62 | 54 | 1.45e-02 | 0.1268 |
| random (n=5) | - | - | 89 +/- 18 | - | - | 6.35e-03 +/- 2.5e-03 | - |

### int4_pc, target <= 8 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 4 | 18 | 5 | 9 | 3.09e-04 | 0.0322 |
| Fisher only (lambda=0) | 0 | 4 | 18 | 5 | 9 | 3.09e-04 | 0.0322 |
| error only (lambda=1) | 1 | 4 | 31 | 9 | 17 | 1.15e-03 | 0.0424 |
| product |  | 4 | 18 | 5 | 9 | 3.09e-04 | 0.0322 |
| worst |  | 19 | 144 | 66 | 54 | 1.51e-02 | 0.1296 |
| random (n=5) | - | - | 107 +/- 17 | - | - | 9.27e-03 +/- 2.5e-03 | - |

### int4_pc, target <= 6 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.75 | 6 | 49 | 22 | 19 | 2.23e-03 | 0.0535 |
| Fisher only (lambda=0) | 0 | 6 | 35 | 14 | 16 | 1.19e-03 | 0.0464 |
| error only (lambda=1) | 1 | 6 | 49 | 22 | 19 | 2.23e-03 | 0.0535 |
| product |  | 9 | 37 | 11 | 20 | 1.32e-03 | 0.0475 |
| worst |  | 18 | 141 | 66 | 51 | 1.48e-02 | 0.1293 |
| random (n=5) | - | - | 92 +/- 30 | - | - | 7.45e-03 +/- 3.7e-03 | - |

### int4_pc, target <= 5 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 8 | 56 | 18 | 28 | 3.25e-03 | 0.0616 |
| Fisher only (lambda=0) | 0 | 8 | 56 | 18 | 28 | 3.25e-03 | 0.0616 |
| error only (lambda=1) | 1 | 8 | 56 | 18 | 28 | 3.25e-03 | 0.0616 |
| product |  | 9 | 54 | 21 | 26 | 2.90e-03 | 0.0606 |
| worst |  | 20 | 137 | 62 | 53 | 1.45e-02 | 0.1300 |
| random (n=5) | - | - | 114 +/- 20 | - | - | 1.05e-02 +/- 2.1e-03 | - |

### int4_pt, target <= 14 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.25 | 3 | 78 | 35 | 27 | 4.30e-03 | 0.1006 |
| Fisher only (lambda=0) | 0 | 3 | 59 | 25 | 26 | 2.89e-03 | 0.0980 |
| error only (lambda=1) | 1 | 4 | 111 | 49 | 46 | 8.79e-03 | 0.0905 |
| product |  | 3 | 59 | 25 | 26 | 2.89e-03 | 0.0980 |
| worst |  | 13 | 322 | 192 | 91 | 6.08e-02 | 0.2304 |
| random (n=5) | - | - | 269 +/- 27 | - | - | 4.37e-02 +/- 6.6e-03 | - |

### int4_pt, target <= 12 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 92 | 43 | 42 | 6.29e-03 | 0.2330 |
| Fisher only (lambda=0) | 0 | 3 | 92 | 43 | 42 | 6.29e-03 | 0.2330 |
| error only (lambda=1) | 1 | 5 | 106 | 44 | 45 | 7.12e-03 | 0.2001 |
| product |  | 4 | 64 | 29 | 25 | 3.68e-03 | 0.2291 |
| worst |  | 16 | 343 | 204 | 100 | 6.45e-02 | 0.2609 |
| random (n=5) | - | - | 224 +/- 59 | - | - | 3.39e-02 +/- 1.2e-02 | - |

### int4_pt, target <= 10 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.25 | 4 | 88 | 47 | 33 | 7.15e-03 | 0.2551 |
| Fisher only (lambda=0) | 0 | 4 | 89 | 40 | 41 | 6.77e-03 | 0.2636 |
| error only (lambda=1) | 1 | 5 | 93 | 49 | 33 | 7.06e-03 | 0.2608 |
| product |  | 5 | 69 | 30 | 29 | 2.90e-03 | 0.2508 |
| worst |  | 18 | 349 | 216 | 92 | 5.97e-02 | 0.4088 |
| random (n=5) | - | - | 210 +/- 59 | - | - | 2.94e-02 +/- 1.3e-02 | - |

### int4_pt, target <= 8 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.75 | 4 | 90 | 47 | 28 | 6.83e-03 | 0.2493 |
| Fisher only (lambda=0) | 0 | 4 | 67 | 31 | 28 | 5.11e-03 | 0.2453 |
| error only (lambda=1) | 1 | 4 | 90 | 47 | 28 | 6.83e-03 | 0.2493 |
| product |  | 4 | 67 | 31 | 28 | 5.11e-03 | 0.2453 |
| worst |  | 19 | 346 | 209 | 94 | 5.80e-02 | 0.3452 |
| random (n=5) | - | - | 244 +/- 31 | - | - | 3.80e-02 +/- 6.4e-03 | - |

### int4_pt, target <= 6 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 6 | 120 | 59 | 41 | 1.08e-02 | 0.2946 |
| Fisher only (lambda=0) | 0 | 6 | 120 | 59 | 41 | 1.08e-02 | 0.2946 |
| error only (lambda=1) | 1 | 6 | 135 | 69 | 43 | 1.25e-02 | 0.2975 |
| product |  | 6 | 120 | 59 | 41 | 1.08e-02 | 0.2946 |
| worst |  | 18 | 326 | 195 | 92 | 5.56e-02 | 0.3770 |
| random (n=5) | - | - | 226 +/- 52 | - | - | 3.32e-02 +/- 1.2e-02 | - |

### int4_pt, target <= 5 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.75 | 9 | 156 | 82 | 49 | 1.54e-02 | 0.3055 |
| Fisher only (lambda=0) | 0 | 8 | 156 | 82 | 52 | 1.60e-02 | 0.3069 |
| error only (lambda=1) | 1 | 9 | 156 | 82 | 49 | 1.54e-02 | 0.3055 |
| product |  | 9 | 153 | 75 | 50 | 1.55e-02 | 0.3187 |
| worst |  | 20 | 345 | 213 | 86 | 5.74e-02 | 0.4210 |
| random (n=5) | - | - | 257 +/- 43 | - | - | 3.89e-02 +/- 9.8e-03 | - |

### fp4_scaled, target <= 14 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 33 | 9 | 16 | 8.85e-04 | 0.0467 |
| Fisher only (lambda=0) | 0 | 3 | 33 | 9 | 16 | 8.85e-04 | 0.0467 |
| error only (lambda=1) | 1 | 4 | 63 | 20 | 33 | 3.38e-03 | 0.0550 |
| product |  | 3 | 33 | 9 | 16 | 8.85e-04 | 0.0467 |
| worst |  | 13 | 347 | 203 | 91 | 6.50e-02 | 0.2321 |
| random (n=5) | - | - | 253 +/- 37 | - | - | 4.00e-02 +/- 1.1e-02 | - |

### fp4_scaled, target <= 12 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 60 | 27 | 26 | 3.75e-03 | 0.0939 |
| Fisher only (lambda=0) | 0 | 3 | 60 | 27 | 26 | 3.75e-03 | 0.0939 |
| error only (lambda=1) | 1 | 5 | 65 | 29 | 24 | 3.77e-03 | 0.0833 |
| product |  | 4 | 31 | 12 | 13 | 8.55e-04 | 0.0769 |
| worst |  | 16 | 374 | 225 | 91 | 7.13e-02 | 0.2490 |
| random (n=5) | - | - | 228 +/- 75 | - | - | 3.42e-02 +/- 1.6e-02 | - |

### fp4_scaled, target <= 10 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 4 | 59 | 28 | 24 | 4.03e-03 | 0.1338 |
| Fisher only (lambda=0) | 0 | 4 | 59 | 28 | 24 | 4.03e-03 | 0.1338 |
| error only (lambda=1) | 1 | 5 | 72 | 32 | 30 | 3.38e-03 | 0.1261 |
| product |  | 5 | 42 | 11 | 21 | 1.24e-03 | 0.1246 |
| worst |  | 18 | 367 | 213 | 88 | 6.73e-02 | 0.3620 |
| random (n=5) | - | - | 187 +/- 39 | - | - | 2.20e-02 +/- 7.8e-03 | - |

### fp4_scaled, target <= 8 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 4 | 48 | 16 | 23 | 3.61e-03 | 0.1640 |
| Fisher only (lambda=0) | 0 | 4 | 48 | 16 | 23 | 3.61e-03 | 0.1640 |
| error only (lambda=1) | 1 | 4 | 61 | 22 | 26 | 3.90e-03 | 0.1631 |
| product |  | 4 | 48 | 16 | 23 | 3.61e-03 | 0.1640 |
| worst |  | 19 | 378 | 233 | 82 | 7.05e-02 | 0.3374 |
| random (n=5) | - | - | 235 +/- 46 | - | - | 3.47e-02 +/- 1.4e-02 | - |

### fp4_scaled, target <= 6 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 6 | 56 | 23 | 23 | 4.51e-03 | 0.1745 |
| Fisher only (lambda=0) | 0 | 6 | 56 | 23 | 23 | 4.51e-03 | 0.1745 |
| error only (lambda=1) | 1 | 6 | 69 | 28 | 26 | 4.86e-03 | 0.1741 |
| product |  | 6 | 56 | 23 | 23 | 4.51e-03 | 0.1745 |
| worst |  | 18 | 372 | 225 | 85 | 6.96e-02 | 0.3411 |
| random (n=5) | - | - | 219 +/- 87 | - | - | 3.19e-02 +/- 2.0e-02 | - |

### fp4_scaled, target <= 5 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 8 | 97 | 40 | 41 | 8.42e-03 | 0.1848 |
| Fisher only (lambda=0) | 0 | 8 | 97 | 40 | 41 | 8.42e-03 | 0.1848 |
| error only (lambda=1) | 1 | 9 | 80 | 30 | 36 | 7.06e-03 | 0.1818 |
| product |  | 9 | 87 | 39 | 36 | 6.68e-03 | 0.1877 |
| worst |  | 20 | 368 | 216 | 88 | 6.76e-02 | 0.3456 |
| random (n=5) | - | - | 262 +/- 56 | - | - | 3.94e-02 +/- 1.3e-02 | - |

### Summary (test)

* lambda0: KL below the random-draw mean in 27 of 30 (arm, target) cells
* product: KL below the random-draw mean in 27 of 30 (arm, target) cells
* Spearman(KL, weights switched to 8 bit) = 0.05; Spearman(KL, -SQNR) = 0.93 (over 279 configurations)
