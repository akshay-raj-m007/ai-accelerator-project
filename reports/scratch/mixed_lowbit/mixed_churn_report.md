# Prediction churn vs the all-E4M11 model

* flips = images whose predicted class differs from the all-E4M11 model; c2w / w2c = right->wrong / wrong->right; KL = mean KL(softmax(E4M11) || softmax(config)) in nats.
* Lower is closer to the E4M11 model. `random` rows: mean +/- std over the draws. The sets and budgets are the ones from the accuracy report.

## val

| reference | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|
| fp32 | 0 | 0 | 0 | 4.54e-07 | 0.0006 |
| int4_pc_all | 78 | 46 | 22 | 1.31e-02 | 0.1303 |
| int4_pt_all | 170 | 105 | 46 | 5.54e-02 | 0.4254 |
| fp4_scaled_all | 191 | 118 | 52 | 6.70e-02 | 0.3507 |

### int4_pc, target <= 14 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 10 | 7 | 1 | 3.64e-04 | 0.0260 |
| Fisher only (lambda=0) | 0 | 3 | 10 | 7 | 1 | 3.64e-04 | 0.0260 |
| error only (lambda=1) | 1 | 3 | 20 | 15 | 3 | 1.14e-03 | 0.0336 |
| product |  | 3 | 10 | 7 | 1 | 3.64e-04 | 0.0260 |
| worst |  | 13 | 78 | 48 | 23 | 1.18e-02 | 0.1210 |
| random (n=5) | - | - | 63 +/- 7 | - | - | 9.41e-03 +/- 1.5e-03 | - |

### int4_pc, target <= 12 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 20 | 11 | 6 | 9.70e-04 | 0.0311 |
| Fisher only (lambda=0) | 0 | 3 | 20 | 11 | 6 | 9.70e-04 | 0.0311 |
| error only (lambda=1) | 1 | 4 | 23 | 16 | 3 | 8.05e-04 | 0.0304 |
| product |  | 4 | 14 | 8 | 4 | 3.93e-04 | 0.0247 |
| worst |  | 16 | 80 | 48 | 21 | 1.31e-02 | 0.1260 |
| random (n=5) | - | - | 55 +/- 15 | - | - | 8.28e-03 +/- 3.0e-03 | - |

### int4_pc, target <= 10 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.5 | 4 | 22 | 13 | 6 | 1.04e-03 | 0.0357 |
| Fisher only (lambda=0) | 0 | 4 | 27 | 16 | 7 | 1.07e-03 | 0.0401 |
| error only (lambda=1) | 1 | 4 | 33 | 21 | 8 | 2.02e-03 | 0.0426 |
| product |  | 5 | 13 | 7 | 4 | 6.15e-04 | 0.0367 |
| worst |  | 18 | 81 | 49 | 23 | 1.27e-02 | 0.1256 |
| random (n=5) | - | - | 51 +/- 10 | - | - | 6.17e-03 +/- 2.1e-03 | - |

### int4_pc, target <= 8 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 4 | 9 | 7 | 1 | 3.01e-04 | 0.0322 |
| Fisher only (lambda=0) | 0 | 4 | 9 | 7 | 1 | 3.01e-04 | 0.0322 |
| error only (lambda=1) | 1 | 4 | 22 | 18 | 1 | 1.18e-03 | 0.0419 |
| product |  | 4 | 9 | 7 | 1 | 3.01e-04 | 0.0322 |
| worst |  | 19 | 82 | 47 | 25 | 1.29e-02 | 0.1283 |
| random (n=5) | - | - | 57 +/- 11 | - | - | 8.59e-03 +/- 2.2e-03 | - |

### int4_pc, target <= 6 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 6 | 27 | 17 | 7 | 1.39e-03 | 0.0463 |
| Fisher only (lambda=0) | 0 | 6 | 27 | 17 | 7 | 1.39e-03 | 0.0463 |
| error only (lambda=1) | 1 | 6 | 32 | 20 | 8 | 2.37e-03 | 0.0530 |
| product |  | 9 | 19 | 10 | 6 | 1.25e-03 | 0.0472 |
| worst |  | 18 | 78 | 46 | 23 | 1.27e-02 | 0.1279 |
| random (n=5) | - | - | 54 +/- 15 | - | - | 7.11e-03 +/- 3.1e-03 | - |

### int4_pc, target <= 5 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 8 | 38 | 21 | 10 | 3.40e-03 | 0.0611 |
| Fisher only (lambda=0) | 0 | 8 | 38 | 21 | 10 | 3.40e-03 | 0.0611 |
| error only (lambda=1) | 1 | 8 | 38 | 21 | 10 | 3.40e-03 | 0.0611 |
| product |  | 9 | 37 | 21 | 10 | 3.25e-03 | 0.0602 |
| worst |  | 20 | 78 | 47 | 22 | 1.26e-02 | 0.1287 |
| random (n=5) | - | - | 62 +/- 4 | - | - | 9.55e-03 +/- 1.8e-03 | - |

### int4_pt, target <= 14 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 28 | 18 | 5 | 3.24e-03 | 0.0989 |
| Fisher only (lambda=0) | 0 | 3 | 28 | 18 | 5 | 3.24e-03 | 0.0989 |
| error only (lambda=1) | 1 | 4 | 64 | 42 | 17 | 8.34e-03 | 0.0897 |
| product |  | 3 | 28 | 18 | 5 | 3.24e-03 | 0.0989 |
| worst |  | 13 | 167 | 104 | 48 | 5.57e-02 | 0.2319 |
| random (n=5) | - | - | 134 +/- 12 | - | - | 4.03e-02 +/- 4.9e-03 | - |

### int4_pt, target <= 12 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 49 | 24 | 14 | 7.08e-03 | 0.2343 |
| Fisher only (lambda=0) | 0 | 3 | 49 | 24 | 14 | 7.08e-03 | 0.2343 |
| error only (lambda=1) | 1 | 5 | 48 | 28 | 12 | 6.91e-03 | 0.2006 |
| product |  | 4 | 36 | 19 | 9 | 3.82e-03 | 0.2297 |
| worst |  | 16 | 179 | 111 | 50 | 5.84e-02 | 0.2623 |
| random (n=5) | - | - | 121 +/- 27 | - | - | 3.47e-02 +/- 1.3e-02 | - |

### int4_pt, target <= 10 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 4 | 57 | 29 | 19 | 7.64e-03 | 0.2655 |
| Fisher only (lambda=0) | 0 | 4 | 57 | 29 | 19 | 7.64e-03 | 0.2655 |
| error only (lambda=1) | 1 | 5 | 45 | 27 | 8 | 6.31e-03 | 0.2612 |
| product |  | 5 | 37 | 21 | 13 | 2.89e-03 | 0.2516 |
| worst |  | 18 | 176 | 113 | 43 | 5.57e-02 | 0.4082 |
| random (n=5) | - | - | 109 +/- 27 | - | - | 2.83e-02 +/- 1.1e-02 | - |

### int4_pt, target <= 8 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.5 | 4 | 45 | 24 | 16 | 6.59e-03 | 0.2487 |
| Fisher only (lambda=0) | 0 | 4 | 35 | 20 | 11 | 4.96e-03 | 0.2462 |
| error only (lambda=1) | 1 | 4 | 45 | 24 | 16 | 6.59e-03 | 0.2487 |
| product |  | 4 | 35 | 20 | 11 | 4.96e-03 | 0.2462 |
| worst |  | 19 | 177 | 111 | 47 | 5.59e-02 | 0.3461 |
| random (n=5) | - | - | 122 +/- 15 | - | - | 3.71e-02 +/- 4.5e-03 | - |

### int4_pt, target <= 6 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 6 | 53 | 28 | 17 | 1.06e-02 | 0.2960 |
| Fisher only (lambda=0) | 0 | 6 | 53 | 28 | 17 | 1.06e-02 | 0.2960 |
| error only (lambda=1) | 1 | 6 | 57 | 30 | 17 | 1.16e-02 | 0.2977 |
| product |  | 6 | 53 | 28 | 17 | 1.06e-02 | 0.2960 |
| worst |  | 18 | 174 | 109 | 45 | 5.32e-02 | 0.3795 |
| random (n=5) | - | - | 117 +/- 25 | - | - | 3.23e-02 +/- 1.1e-02 | - |

### int4_pt, target <= 5 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.75 | 9 | 73 | 41 | 22 | 1.43e-02 | 0.3063 |
| Fisher only (lambda=0) | 0 | 8 | 81 | 48 | 23 | 1.61e-02 | 0.3081 |
| error only (lambda=1) | 1 | 9 | 73 | 41 | 22 | 1.43e-02 | 0.3063 |
| product |  | 9 | 79 | 46 | 22 | 1.49e-02 | 0.3191 |
| worst |  | 20 | 167 | 104 | 44 | 5.39e-02 | 0.4220 |
| random (n=5) | - | - | 132 +/- 18 | - | - | 3.89e-02 +/- 8.0e-03 | - |

### fp4_scaled, target <= 14 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 20 | 10 | 6 | 1.01e-03 | 0.0467 |
| Fisher only (lambda=0) | 0 | 3 | 20 | 10 | 6 | 1.01e-03 | 0.0467 |
| error only (lambda=1) | 1 | 4 | 38 | 22 | 11 | 3.24e-03 | 0.0548 |
| product |  | 3 | 20 | 10 | 6 | 1.01e-03 | 0.0467 |
| worst |  | 13 | 173 | 104 | 47 | 6.42e-02 | 0.2353 |
| random (n=5) | - | - | 141 +/- 20 | - | - | 4.07e-02 +/- 1.1e-02 | - |

### fp4_scaled, target <= 12 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 3 | 39 | 15 | 15 | 3.84e-03 | 0.0932 |
| Fisher only (lambda=0) | 0 | 3 | 39 | 15 | 15 | 3.84e-03 | 0.0932 |
| error only (lambda=1) | 1 | 5 | 41 | 23 | 11 | 3.39e-03 | 0.0830 |
| product |  | 4 | 20 | 12 | 4 | 8.79e-04 | 0.0771 |
| worst |  | 16 | 182 | 110 | 48 | 7.04e-02 | 0.2510 |
| random (n=5) | - | - | 116 +/- 40 | - | - | 3.35e-02 +/- 1.6e-02 | - |

### fp4_scaled, target <= 10 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 4 | 37 | 17 | 14 | 4.23e-03 | 0.1337 |
| Fisher only (lambda=0) | 0 | 4 | 37 | 17 | 14 | 4.23e-03 | 0.1337 |
| error only (lambda=1) | 1 | 5 | 36 | 20 | 11 | 3.24e-03 | 0.1255 |
| product |  | 5 | 23 | 13 | 6 | 1.31e-03 | 0.1252 |
| worst |  | 18 | 187 | 114 | 51 | 6.57e-02 | 0.3635 |
| random (n=5) | - | - | 96 +/- 22 | - | - | 2.29e-02 +/- 9.1e-03 | - |

### fp4_scaled, target <= 8 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 4 | 36 | 19 | 13 | 3.71e-03 | 0.1638 |
| Fisher only (lambda=0) | 0 | 4 | 36 | 19 | 13 | 3.71e-03 | 0.1638 |
| error only (lambda=1) | 1 | 4 | 33 | 20 | 10 | 3.96e-03 | 0.1631 |
| product |  | 4 | 36 | 19 | 13 | 3.71e-03 | 0.1638 |
| worst |  | 19 | 199 | 121 | 54 | 6.90e-02 | 0.3386 |
| random (n=5) | - | - | 127 +/- 24 | - | - | 3.49e-02 +/- 1.3e-02 | - |

### fp4_scaled, target <= 6 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 6 | 44 | 24 | 16 | 4.76e-03 | 0.1738 |
| Fisher only (lambda=0) | 0 | 6 | 44 | 24 | 16 | 4.76e-03 | 0.1738 |
| error only (lambda=1) | 1 | 6 | 48 | 26 | 17 | 5.06e-03 | 0.1736 |
| product |  | 6 | 44 | 24 | 16 | 4.76e-03 | 0.1738 |
| worst |  | 18 | 199 | 121 | 52 | 6.88e-02 | 0.3428 |
| random (n=5) | - | - | 118 +/- 42 | - | - | 3.19e-02 +/- 1.9e-02 | - |

### fp4_scaled, target <= 5 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0 | 8 | 60 | 27 | 23 | 8.29e-03 | 0.1835 |
| Fisher only (lambda=0) | 0 | 8 | 60 | 27 | 23 | 8.29e-03 | 0.1835 |
| error only (lambda=1) | 1 | 9 | 55 | 33 | 16 | 6.88e-03 | 0.1810 |
| product |  | 9 | 55 | 34 | 16 | 6.72e-03 | 0.1871 |
| worst |  | 20 | 191 | 116 | 53 | 6.54e-02 | 0.3470 |
| random (n=5) | - | - | 135 +/- 32 | - | - | 3.88e-02 +/- 1.4e-02 | - |

### Summary (val)

* lambda0: KL below the random-draw mean in 18 of 18 (arm, target) cells
* product: KL below the random-draw mean in 18 of 18 (arm, target) cells
* Spearman(KL, weights switched to 8 bit) = 0.18; Spearman(KL, -SQNR) = 0.78 (over 216 configurations)

## test

| reference | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|
| fp32 | 0 | 0 | 0 | 3.91e-07 | 0.0006 |
| int4_pc_all | 142 | 65 | 53 | 1.51e-02 | 0.1318 |
| int4_pt_all | 351 | 212 | 89 | 5.81e-02 | 0.4233 |
| fp4_scaled_all | 368 | 217 | 84 | 6.90e-02 | 0.3496 |

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
| lambda (best on val acc) | 0.5 | 4 | 32 | 9 | 18 | 1.03e-03 | 0.0361 |
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
| lambda (best on val acc) | 0 | 6 | 35 | 14 | 16 | 1.19e-03 | 0.0464 |
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
| lambda (best on val acc) | 0 | 3 | 59 | 25 | 26 | 2.89e-03 | 0.0980 |
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
| lambda (best on val acc) | 0 | 4 | 89 | 40 | 41 | 6.77e-03 | 0.2636 |
| Fisher only (lambda=0) | 0 | 4 | 89 | 40 | 41 | 6.77e-03 | 0.2636 |
| error only (lambda=1) | 1 | 5 | 93 | 49 | 33 | 7.06e-03 | 0.2608 |
| product |  | 5 | 69 | 30 | 29 | 2.90e-03 | 0.2508 |
| worst |  | 18 | 349 | 216 | 92 | 5.97e-02 | 0.4088 |
| random (n=5) | - | - | 210 +/- 59 | - | - | 2.94e-02 +/- 1.3e-02 | - |

### int4_pt, target <= 8 bits/weight

| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |
|---|---|---|---|---|---|---|---|
| lambda (best on val acc) | 0.5 | 4 | 90 | 47 | 28 | 6.83e-03 | 0.2493 |
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

* lambda0: KL below the random-draw mean in 18 of 18 (arm, target) cells
* product: KL below the random-draw mean in 18 of 18 (arm, target) cells
* Spearman(KL, weights switched to 8 bit) = 0.11; Spearman(KL, -SQNR) = 0.76 (over 165 configurations)
