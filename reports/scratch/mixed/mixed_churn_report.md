# Prediction churn vs the all-E4M11 model

* flips = images whose predicted class differs from the all-E4M11 model; c2w / w2c = right->wrong / wrong->right; KL = mean KL(softmax(E4M11) || softmax(config)) in nats.
* Lower is closer to the E4M11 model. `random` rows: mean +/- std over the draws. The sets and budgets are the ones from the accuracy report.

## val

| reference | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|
| fp32 | 0 | 0 | 0 | 4.54e-07 | 0.0006 |
| int8_pt_all | 8 | 7 | 1 | 1.89e-04 | 0.0123 |
| int8_pc_all | 6 | 5 | 0 | 5.78e-05 | 0.0074 |
| fp8_all | 23 | 13 | 4 | 1.34e-03 | 0.0682 |
| fp8_scaled_all | 22 | 11 | 7 | 1.07e-03 | 0.0310 |

### int8_pt, target <= 14 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 2 | 1 | 0 | 1.43e-05 | 0.0033 |
| Fisher only (lambda=0) | 0 | 3 | 2 | 1 | 0 | 1.43e-05 | 0.0033 |
| error only (lambda=1) | 1 | 3 | 2 | 1 | 0 | 1.65e-05 | 0.0035 |
| product |  | 5 | 0 | 0 | 0 | 8.27e-06 | 0.0028 |
| worst |  | 17 | 8 | 7 | 1 | 1.83e-04 | 0.0114 |
| random (n=5) | - | - | 4 +/- 2 | - | - | 5.50e-05 +/- 2.2e-05 | - |

### int8_pt, target <= 12 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.5 | 4 | 2 | 1 | 0 | 2.27e-05 | 0.0046 |
| Fisher only (lambda=0) | 0 | 4 | 2 | 1 | 0 | 1.53e-05 | 0.0044 |
| error only (lambda=1) | 1 | 5 | 2 | 2 | 0 | 2.66e-05 | 0.0050 |
| product |  | 5 | 1 | 1 | 0 | 6.13e-06 | 0.0036 |
| worst |  | 18 | 8 | 7 | 1 | 1.90e-04 | 0.0120 |
| random (n=5) | - | - | 3 +/- 2 | - | - | 6.22e-05 +/- 3.1e-05 | - |

### int8_pt, target <= 10 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 5 | 0 | 0 | 0 | 3.37e-06 | 0.0040 |
| Fisher only (lambda=0) | 0 | 5 | 0 | 0 | 0 | 3.37e-06 | 0.0040 |
| error only (lambda=1) | 1 | 6 | 2 | 1 | 0 | 2.41e-05 | 0.0054 |
| product |  | 5 | 0 | 0 | 0 | 3.37e-06 | 0.0040 |
| worst |  | 19 | 7 | 7 | 0 | 1.88e-04 | 0.0123 |
| random (n=5) | - | - | 5 +/- 2 | - | - | 9.97e-05 +/- 3.7e-05 | - |

### int8_pc, target <= 14 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.75 | 4 | 0 | 0 | 0 | 3.15e-06 | 0.0016 |
| Fisher only (lambda=0) | 0 | 3 | 0 | 0 | 0 | 4.20e-06 | 0.0017 |
| error only (lambda=1) | 1 | 4 | 0 | 0 | 0 | 3.15e-06 | 0.0016 |
| product |  | 5 | 0 | 0 | 0 | 2.14e-06 | 0.0013 |
| worst |  | 17 | 5 | 4 | 0 | 5.83e-05 | 0.0072 |
| random (n=5) | - | - | 2 +/- 2 | - | - | 2.14e-05 +/- 1.4e-05 | - |

### int8_pc, target <= 12 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.75 | 4 | 0 | 0 | 0 | 5.83e-06 | 0.0024 |
| Fisher only (lambda=0) | 0 | 4 | 0 | 0 | 0 | 3.79e-06 | 0.0021 |
| error only (lambda=1) | 1 | 4 | 0 | 0 | 0 | 5.83e-06 | 0.0024 |
| product |  | 5 | 1 | 1 | 0 | 1.80e-06 | 0.0017 |
| worst |  | 18 | 5 | 4 | 0 | 5.72e-05 | 0.0072 |
| random (n=5) | - | - | 2 +/- 1 | - | - | 2.49e-05 +/- 1.4e-05 | - |

### int8_pc, target <= 10 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 5 | 1 | 1 | 0 | 4.25e-07 | 0.0015 |
| Fisher only (lambda=0) | 0 | 5 | 1 | 1 | 0 | 4.25e-07 | 0.0015 |
| error only (lambda=1) | 1 | 5 | 1 | 1 | 0 | 4.25e-07 | 0.0015 |
| product |  | 5 | 1 | 1 | 0 | 4.25e-07 | 0.0015 |
| worst |  | 19 | 6 | 5 | 0 | 5.76e-05 | 0.0074 |
| random (n=5) | - | - | 3 +/- 1 | - | - | 3.65e-05 +/- 1.5e-05 | - |

### fp8, target <= 14 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 7 | 3 | 1 | 1.53e-04 | 0.0590 |
| Fisher only (lambda=0) | 0 | 3 | 7 | 3 | 1 | 1.53e-04 | 0.0590 |
| error only (lambda=1) | 1 | 3 | 10 | 8 | 2 | 1.22e-04 | 0.0133 |
| product |  | 5 | 4 | 3 | 0 | 1.08e-04 | 0.0572 |
| worst |  | 17 | 24 | 13 | 5 | 1.24e-03 | 0.0338 |
| random (n=5) | - | - | 16 +/- 6 | - | - | 6.63e-04 +/- 3.9e-04 | - |

### fp8, target <= 12 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.5 | 4 | 3 | 2 | 1 | 1.30e-04 | 0.0215 |
| Fisher only (lambda=0) | 0 | 4 | 5 | 3 | 1 | 1.66e-04 | 0.0607 |
| error only (lambda=1) | 1 | 4 | 7 | 5 | 1 | 1.70e-04 | 0.0215 |
| product |  | 5 | 5 | 2 | 2 | 1.25e-04 | 0.0597 |
| worst |  | 18 | 21 | 11 | 5 | 1.23e-03 | 0.0360 |
| random (n=5) | - | - | 11 +/- 3 | - | - | 5.35e-04 +/- 3.1e-04 | - |

### fp8, target <= 10 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 5 | 4 | 2 | 0 | 9.05e-05 | 0.0597 |
| Fisher only (lambda=0) | 0 | 5 | 4 | 2 | 0 | 9.05e-05 | 0.0597 |
| error only (lambda=1) | 1 | 5 | 6 | 4 | 1 | 1.37e-04 | 0.0599 |
| product |  | 5 | 4 | 2 | 0 | 9.05e-05 | 0.0597 |
| worst |  | 19 | 23 | 13 | 5 | 1.30e-03 | 0.0409 |
| random (n=5) | - | - | 18 +/- 6 | - | - | 8.88e-04 +/- 3.3e-04 | - |

### fp8_scaled, target <= 14 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 2 | 2 | 0 | 2.61e-05 | 0.0050 |
| Fisher only (lambda=0) | 0 | 3 | 2 | 2 | 0 | 2.61e-05 | 0.0050 |
| error only (lambda=1) | 1 | 3 | 2 | 2 | 0 | 2.71e-05 | 0.0056 |
| product |  | 4 | 2 | 1 | 1 | 2.95e-05 | 0.0054 |
| worst |  | 17 | 21 | 10 | 7 | 1.07e-03 | 0.0306 |
| random (n=5) | - | - | 13 +/- 6 | - | - | 5.81e-04 +/- 4.1e-04 | - |

### fp8_scaled, target <= 12 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 4 | 2 | 1 | 1 | 3.01e-05 | 0.0057 |
| Fisher only (lambda=0) | 0 | 4 | 2 | 1 | 1 | 3.01e-05 | 0.0057 |
| error only (lambda=1) | 1 | 4 | 3 | 2 | 1 | 3.81e-05 | 0.0068 |
| product |  | 5 | 1 | 1 | 0 | 1.42e-05 | 0.0050 |
| worst |  | 18 | 22 | 11 | 8 | 1.05e-03 | 0.0305 |
| random (n=5) | - | - | 10 +/- 6 | - | - | 3.63e-04 +/- 3.0e-04 | - |

### fp8_scaled, target <= 10 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.5 | 5 | 0 | 0 | 0 | 8.00e-06 | 0.0048 |
| Fisher only (lambda=0) | 0 | 5 | 1 | 0 | 0 | 4.29e-06 | 0.0045 |
| error only (lambda=1) | 1 | 5 | 0 | 0 | 0 | 8.00e-06 | 0.0048 |
| product |  | 5 | 1 | 0 | 0 | 4.29e-06 | 0.0045 |
| worst |  | 19 | 22 | 11 | 7 | 1.06e-03 | 0.0307 |
| random (n=5) | - | - | 15 +/- 6 | - | - | 6.73e-04 +/- 3.6e-04 | - |

### Summary (val)

* lambda0: KL below the random-draw mean in 12 of 12 (arm, target) cells
* product: KL below the random-draw mean in 12 of 12 (arm, target) cells
* Spearman(KL, weights switched to 8 bit) = 0.15; Spearman(KL, -SQNR) = 0.81 (over 144 configurations)

## test

| reference | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|
| fp32 | 0 | 0 | 0 | 3.91e-07 | 0.0006 |
| int8_pt_all | 12 | 2 | 6 | 1.59e-04 | 0.0122 |
| int8_pc_all | 8 | 2 | 5 | 5.23e-05 | 0.0074 |
| fp8_all | 32 | 13 | 14 | 1.32e-03 | 0.0682 |
| fp8_scaled_all | 25 | 7 | 14 | 1.04e-03 | 0.0310 |

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
| lambda (best on val acc) | 0.5 | 4 | 3 | 0 | 2 | 1.72e-05 | 0.0046 |
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
| lambda (best on val acc) | 0.75 | 4 | 3 | 1 | 1 | 6.27e-06 | 0.0024 |
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

### fp8, target <= 14 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 6 | 1 | 5 | 1.29e-04 | 0.0591 |
| Fisher only (lambda=0) | 0 | 3 | 6 | 1 | 5 | 1.29e-04 | 0.0591 |
| error only (lambda=1) | 1 | 3 | 13 | 3 | 7 | 1.17e-04 | 0.0135 |
| product |  | 5 | 5 | 0 | 5 | 9.85e-05 | 0.0574 |
| worst |  | 17 | 30 | 12 | 12 | 1.24e-03 | 0.0336 |
| random (n=5) | - | - | 20 +/- 5 | - | - | 6.42e-04 +/- 3.8e-04 | - |

### fp8, target <= 12 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.5 | 4 | 10 | 3 | 4 | 1.09e-04 | 0.0215 |
| Fisher only (lambda=0) | 0 | 4 | 7 | 3 | 3 | 1.36e-04 | 0.0608 |
| error only (lambda=1) | 1 | 4 | 13 | 5 | 5 | 1.42e-04 | 0.0214 |
| product |  | 5 | 4 | 2 | 2 | 1.06e-04 | 0.0600 |
| worst |  | 18 | 31 | 11 | 15 | 1.29e-03 | 0.0358 |
| random (n=5) | - | - | 17 +/- 6 | - | - | 4.81e-04 +/- 3.0e-04 | - |

### fp8, target <= 10 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 5 | 5 | 3 | 1 | 9.36e-05 | 0.0599 |
| Fisher only (lambda=0) | 0 | 5 | 5 | 3 | 1 | 9.36e-05 | 0.0599 |
| error only (lambda=1) | 1 | 5 | 10 | 3 | 5 | 1.45e-04 | 0.0602 |
| product |  | 5 | 5 | 3 | 1 | 9.36e-05 | 0.0599 |
| worst |  | 19 | 33 | 13 | 15 | 1.34e-03 | 0.0406 |
| random (n=5) | - | - | 22 +/- 8 | - | - | 8.45e-04 +/- 3.4e-04 | - |

### fp8_scaled, target <= 14 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 3 | 0 | 3 | 2.63e-05 | 0.0050 |
| Fisher only (lambda=0) | 0 | 3 | 3 | 0 | 3 | 2.63e-05 | 0.0050 |
| error only (lambda=1) | 1 | 3 | 3 | 0 | 3 | 2.78e-05 | 0.0057 |
| product |  | 4 | 3 | 0 | 2 | 2.57e-05 | 0.0054 |
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

### Summary (test)

* lambda0: KL below the random-draw mean in 12 of 12 (arm, target) cells
* product: KL below the random-draw mean in 12 of 12 (arm, target) cells
* Spearman(KL, weights switched to 8 bit) = 0.21; Spearman(KL, -SQNR) = 0.82 (over 113 configurations)
