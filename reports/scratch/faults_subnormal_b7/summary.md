# Fault-injection summary

* format: E4M11 mode `subnormal`, bias 7, all-ones decode `clamp`
* clean E4M11 accuracy: 95.11% on 10000 images; FP32: 95.11%
* trials per cell: 100; seed 0; fault index range [0, 11164352] (whole buffer)

| group | k | trials | mean acc % | std | min | max | drop vs clean E4M11 (pp) |
|---|---|---|---|---|---|---|---|
| exponent | 1 | 100 | 95.10 | 0.14 | 93.80 | 95.22 | 0.01 |
| exponent | 2 | 100 | 94.83 | 1.62 | 82.88 | 95.20 | 0.28 |
| exponent | 3 | 100 | 94.66 | 4.39 | 51.00 | 95.20 | 0.45 |
| exponent | 4 | 100 | 93.97 | 8.40 | 13.44 | 95.21 | 1.14 |
| exponent | 5 | 100 | 94.81 | 2.49 | 70.23 | 95.24 | 0.30 |
| exponent | 6 | 100 | 94.91 | 1.17 | 83.82 | 95.23 | 0.20 |
| exponent | 7 | 100 | 94.95 | 0.90 | 87.84 | 95.18 | 0.16 |
| exponent | 8 | 100 | 94.38 | 3.71 | 65.52 | 95.20 | 0.73 |
| mantissa_lsb | 1 | 100 | 95.11 | 0.00 | 95.11 | 95.11 | -0.00 |
| mantissa_lsb | 2 | 100 | 95.11 | 0.00 | 95.11 | 95.11 | -0.00 |
| mantissa_lsb | 3 | 100 | 95.11 | 0.00 | 95.11 | 95.11 | -0.00 |
| mantissa_lsb | 4 | 100 | 95.11 | 0.00 | 95.11 | 95.11 | -0.00 |
| mantissa_lsb | 5 | 100 | 95.11 | 0.00 | 95.11 | 95.11 | -0.00 |
| mantissa_lsb | 6 | 100 | 95.11 | 0.00 | 95.11 | 95.11 | -0.00 |
| mantissa_lsb | 7 | 100 | 95.11 | 0.00 | 95.11 | 95.11 | -0.00 |
| mantissa_lsb | 8 | 100 | 95.11 | 0.00 | 95.11 | 95.11 | -0.00 |
| mantissa_msb | 1 | 100 | 95.11 | 0.00 | 95.11 | 95.11 | -0.00 |
| mantissa_msb | 2 | 100 | 95.11 | 0.00 | 95.11 | 95.11 | -0.00 |
| mantissa_msb | 3 | 100 | 95.11 | 0.00 | 95.11 | 95.11 | -0.00 |
| mantissa_msb | 4 | 100 | 95.11 | 0.00 | 95.11 | 95.11 | -0.00 |
| mantissa_msb | 5 | 100 | 95.11 | 0.00 | 95.11 | 95.11 | -0.00 |
| mantissa_msb | 6 | 100 | 95.11 | 0.00 | 95.11 | 95.11 | -0.00 |
| mantissa_msb | 7 | 100 | 95.11 | 0.00 | 95.11 | 95.11 | -0.00 |
| mantissa_msb | 8 | 100 | 95.11 | 0.00 | 95.11 | 95.11 | -0.00 |
| sign | 1 | 100 | 95.11 | 0.01 | 95.11 | 95.18 | -0.00 |
| sign | 2 | 100 | 95.11 | 0.00 | 95.11 | 95.12 | -0.00 |
| sign | 3 | 100 | 95.11 | 0.00 | 95.11 | 95.12 | -0.00 |
| sign | 4 | 100 | 95.11 | 0.00 | 95.11 | 95.12 | -0.00 |
| sign | 5 | 100 | 95.11 | 0.00 | 95.10 | 95.12 | -0.00 |
| sign | 6 | 100 | 95.11 | 0.01 | 95.11 | 95.16 | -0.00 |
| sign | 7 | 100 | 95.11 | 0.00 | 95.10 | 95.12 | -0.00 |
| sign | 8 | 100 | 95.11 | 0.00 | 95.10 | 95.12 | -0.00 |

Paper reference at k=8 (ResNet-18, CIFAR-10; baseline of 'drop' not stated in the paper):

| group | paper | ours (drop vs clean E4M11) | source |
|---|---|---|---|
| exponent | 1.1 % | 0.73 pp | Sec 5.3, p.5: 'At 8 bit-flips its mean accuracy drop is 1.1% for ResNet-18' (E4M11, exponent bits) |
| sign | < 0.3 % | -0.00 pp | Sec 5.4, p.6: 'Sign-bit faults produce mean accuracy drops below 0.3% for all floating-point formats ... at 8 bit-flips' |
| mantissa_msb | < 0.05 % | -0.00 pp | Sec 5.4, p.6: 'Mantissa MSB faults produce mean accuracy drops below 0.05% on CIFAR-10' |
| mantissa_lsb | < 0.01 % | -0.00 pp | Sec 5.4, p.6: 'mean accuracy drops below 0.01% across all formats, models, and datasets' |
