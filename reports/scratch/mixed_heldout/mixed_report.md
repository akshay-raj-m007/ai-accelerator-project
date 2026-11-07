# Mixed-precision (E4M11 + low-bit) results

* Fisher: `true`, 2048 calibration images (the first 2048 validation images, which the sweep skips (held out)); score normalization `rank`, Fisher `per_param`; E4M11 baseline `subnormal` bias 7.
* Selection and lambda choice use the validation split (2952 images, standard error about 0.40 pp at 95% accuracy): differences below ~0.8 pp are not significant.
* Test accuracy is reported for the saved selections only; lambda was chosen on validation.
* `random` rows: mean +/- std over draws; `worst` = feasible set with the largest total damage (sanity envelope). avg bits = payload bits per weight; +meta also counts scale metadata.

## References

| config | val acc % | test acc % | SQNR dB | avg bits |
|---|---|---|---|---|
| fp32 | 95.29 | 95.11 | - | 32.00 |
| e4m11_all | 95.29 | 95.11 | 69.27 | 16.00 |
| int8_pc_all | 95.16 | 95.14 | 39.55 | 8.00 |
| int8_pt_all | 95.12 | 95.15 | 32.58 | 8.00 |
| fp8_scaled_all | 95.19 | 95.18 | 31.55 | 8.00 |
| int4_pc_all | 94.85 | 94.99 | 14.40 | 4.00 |
| int4_pt_all | 94.07 | 93.88 | 7.85 | 4.00 |
| fp4_scaled_all | 93.94 | 93.78 | 9.80 | 4.00 |

## Frontier: accuracy % at each storage budget (test; `v` = validation only)

Fisher-based: Fisher only (lambda=0), product. Without Fisher: error only (lambda=1), random (mean +/- std of the draws), worst (largest damage). `uniform` = every layer in the low-bit format.

| arm | budget | avg bits | Fisher only | product | error only | random | worst | uniform (all layers) |
|---|---|---|---|---|---|---|---|---|
| int8_pc | 14 | 13.99 | 95.12 | 95.11 | 95.11 | 95.13 +/- 0.01 | 95.14 | 95.14 |
| int8_pc | 12 | 11.98 | 95.12 | 95.12 | 95.11 | 95.13 +/- 0.01 | 95.17 | 95.14 |
| int8_pc | 10 | 9.99 | 95.11 | 95.11 | 95.11 | 95.13 +/- 0.02 | 95.13 | 95.14 |
| int8_pc | 8 | 8.00 | 95.14 | 95.14 | 95.14 | 95.14 +/- 0.00 | 95.14 | 95.14 |
| int8_pt | 14 | 13.99 | 95.13 | 95.12 | 95.11 | 95.15 +/- 0.02 | 95.16 | 95.15 |
| int8_pt | 12 | 11.98 | 95.11 | 95.11 | 95.12 | 95.13 +/- 0.01 | 95.15 | 95.15 |
| int8_pt | 10 | 9.99 | 95.11 | 95.11 | 95.11 | 95.15 +/- 0.01 | 95.16 | 95.15 |
| int8_pt | 8 | 8.00 | 95.15 | 95.15 | 95.15 | 95.15 +/- 0.00 | 95.15 | 95.15 |
| fp8_scaled | 14 | 13.99 | 95.14 | 95.13 | 95.14 | 95.19 +/- 0.02 | 95.17 | 95.18 |
| fp8_scaled | 12 | 11.98 | 95.13 | 95.11 | 95.14 | 95.16 +/- 0.02 | 95.16 | 95.18 |
| fp8_scaled | 10 | 9.99 | 95.11 | 95.11 | 95.12 | 95.16 +/- 0.03 | 95.18 | 95.18 |
| fp8_scaled | 8 | 8.00 | 95.18 | 95.18 | 95.18 | 95.18 +/- 0.00 | 95.18 | 95.18 |
| int4_pc | 14 | 13.96 | 95.18 | 95.18 | 95.17 | 95.02 +/- 0.06 | 94.90 | 94.99 |
| int4_pc | 12 | 11.88 | 95.21 | 95.15 | 95.18 | 95.04 +/- 0.14 | 94.98 | 94.99 |
| int4_pc | 10 | 9.98 | 95.21 | 95.21 | 95.19 | 95.12 +/- 0.04 | 95.03 | 94.99 |
| int4_pc | 8 | 7.76 | 95.15 | 95.15 | 95.19 | 95.11 +/- 0.09 | 94.99 | 94.99 |
| int4_pc | 6 | 5.86 | 95.13 | 95.20 | 95.08 | 95.08 +/- 0.10 | 94.96 | 94.99 |
| int4_pc | 5 | 4.91 | 95.21 | 95.16 | 95.21 | 95.07 +/- 0.10 | 95.02 | 94.99 |
| int4_pt | 14 | 13.96 | 95.12 | 95.12 | 95.08 | 94.40 +/- 0.14 | 94.10 | 93.88 |
| int4_pt | 12 | 11.88 | 95.10 | 95.07 | 95.12 | 94.52 +/- 0.23 | 94.07 | 93.88 |
| int4_pt | 10 | 9.98 | 95.12 | 95.10 | 94.95 | 94.60 +/- 0.22 | 93.87 | 93.88 |
| int4_pt | 8 | 7.76 | 95.08 | 95.08 | 94.92 | 94.45 +/- 0.17 | 93.96 | 93.88 |
| int4_pt | 6 | 5.86 | 94.93 | 94.93 | 94.85 | 94.49 +/- 0.22 | 94.08 | 93.88 |
| int4_pt | 5 | 4.91 | 94.81 | 94.86 | 94.78 | 94.41 +/- 0.24 | 93.84 | 93.88 |
| fp4_scaled | 14 | 13.96 | 95.18 | 95.18 | 95.24 | 94.50 +/- 0.27 | 93.99 | 93.78 |
| fp4_scaled | 12 | 11.88 | 95.10 | 95.12 | 95.06 | 94.57 +/- 0.33 | 93.77 | 93.78 |
| fp4_scaled | 10 | 9.98 | 95.07 | 95.21 | 95.09 | 94.71 +/- 0.13 | 93.86 | 93.78 |
| fp4_scaled | 8 | 7.76 | 95.18 | 95.18 | 95.15 | 94.54 +/- 0.28 | 93.60 | 93.78 |
| fp4_scaled | 6 | 5.86 | 95.11 | 95.11 | 95.09 | 94.55 +/- 0.45 | 93.71 | 93.78 |
| fp4_scaled | 5 | 4.91 | 95.12 | 95.08 | 95.17 | 94.42 +/- 0.36 | 93.83 | 93.78 |

## int8_pc, target <= 14 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0.75 | 13.98 | 13.99 | 4 | 95.29 | 95.11 | +0.00 | 48.61 |
| Fisher only (lambda=0) | 0 | 13.99 | 13.99 | 3 | 95.29 | 95.12 | +0.01 | 48.12 |
| error only (lambda=1) | 1 | 13.98 | 13.99 | 4 | 95.29 | 95.11 | +0.00 | 48.61 |
| product (sum F*dw^2) |  | 13.98 | 13.99 | 5 | 95.29 | 95.11 | +0.00 | 49.62 |
| worst (max damage) |  | 13.92 | 13.92 | 17 | 95.19 | 95.14 | +0.03 | 40.58 |
| random (n=5) | - | 13.97 | - | - | 95.26 +/- 0.04 | 95.13 +/- 0.01 | +0.02 | - |

lambda chosen on validation: 0.75; 8-bit layers: layer3.0.conv1.weight, layer3.0.downsample.0.weight, layer4.0.downsample.0.weight, layer4.1.conv2.weight

## int8_pc, target <= 12 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0.5 | 11.98 | 11.99 | 4 | 95.29 | 95.11 | +0.00 | 44.35 |
| Fisher only (lambda=0) | 0 | 11.98 | 11.99 | 4 | 95.29 | 95.12 | +0.01 | 45.77 |
| error only (lambda=1) | 1 | 11.98 | 11.99 | 4 | 95.29 | 95.11 | +0.00 | 44.35 |
| product (sum F*dw^2) |  | 12.00 | 12.00 | 5 | 95.26 | 95.12 | +0.01 | 46.88 |
| worst (max damage) |  | 11.80 | 11.81 | 18 | 95.19 | 95.17 | +0.06 | 40.23 |
| random (n=5) | - | 11.86 | - | - | 95.25 +/- 0.03 | 95.13 +/- 0.01 | +0.02 | - |

lambda chosen on validation: 0.5; 8-bit layers: layer3.0.conv1.weight, layer3.0.conv2.weight, layer4.0.conv2.weight, layer4.1.conv2.weight

## int8_pc, target <= 10 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 9.99 | 10.00 | 5 | 95.26 | 95.11 | +0.00 | 46.13 |
| Fisher only (lambda=0) | 0 | 9.99 | 10.00 | 5 | 95.26 | 95.11 | +0.00 | 46.13 |
| error only (lambda=1) | 1 | 9.99 | 10.00 | 5 | 95.26 | 95.11 | +0.00 | 46.13 |
| product (sum F*dw^2) |  | 9.99 | 10.00 | 5 | 95.26 | 95.11 | +0.00 | 46.13 |
| worst (max damage) |  | 9.78 | 9.80 | 19 | 95.16 | 95.13 | +0.02 | 39.65 |
| random (n=5) | - | 9.86 | - | - | 95.24 +/- 0.03 | 95.13 +/- 0.02 | +0.02 | - |

lambda chosen on validation: 0; 8-bit layers: layer4.0.conv1.weight, layer4.0.conv2.weight, layer4.0.downsample.0.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## int8_pc, target <= 8 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 8.00 | 8.01 | 21 | 95.16 | 95.14 | +0.03 | 39.55 |
| Fisher only (lambda=0) | 0 | 8.00 | 8.01 | 21 | 95.16 | 95.14 | +0.03 | 39.55 |
| error only (lambda=1) | 1 | 8.00 | 8.01 | 21 | 95.16 | 95.14 | +0.03 | 39.55 |
| product (sum F*dw^2) |  | 8.00 | 8.01 | 21 | 95.16 | 95.14 | +0.03 | 39.55 |
| worst (max damage) |  | 8.00 | 8.01 | 21 | 95.16 | 95.14 | +0.03 | 39.55 |
| random (n=5) | - | 8.00 | - | - | 95.16 +/- 0.00 | 95.14 +/- 0.00 | +0.03 | - |

lambda chosen on validation: 0; 8-bit layers: conv1.weight, layer1.0.conv1.weight, layer1.0.conv2.weight, layer1.1.conv1.weight, layer1.1.conv2.weight, layer2.0.conv1.weight, layer2.0.conv2.weight, layer2.0.downsample.0.weight, layer2.1.conv1.weight, layer2.1.conv2.weight, layer3.0.conv1.weight, layer3.0.conv2.weight, layer3.0.downsample.0.weight, layer3.1.conv1.weight, layer3.1.conv2.weight, layer4.0.conv1.weight, layer4.0.conv2.weight, layer4.0.downsample.0.weight, layer4.1.conv1.weight, layer4.1.conv2.weight, fc.weight

## int8_pt, target <= 14 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 13.99 | 13.99 | 3 | 95.26 | 95.13 | +0.02 | 40.74 |
| Fisher only (lambda=0) | 0 | 13.99 | 13.99 | 3 | 95.26 | 95.13 | +0.02 | 40.74 |
| error only (lambda=1) | 1 | 13.99 | 13.99 | 3 | 95.26 | 95.11 | +0.00 | 41.03 |
| product (sum F*dw^2) |  | 13.98 | 13.98 | 5 | 95.29 | 95.12 | +0.01 | 42.79 |
| worst (max damage) |  | 13.92 | 13.92 | 17 | 95.12 | 95.16 | +0.05 | 34.81 |
| random (n=5) | - | 13.97 | - | - | 95.24 +/- 0.03 | 95.15 +/- 0.02 | +0.04 | - |

lambda chosen on validation: 0; 8-bit layers: layer2.1.conv2.weight, layer3.0.conv1.weight, layer4.1.conv2.weight

## int8_pt, target <= 12 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0.25 | 11.98 | 11.98 | 4 | 95.26 | 95.12 | +0.01 | 38.13 |
| Fisher only (lambda=0) | 0 | 11.98 | 11.98 | 4 | 95.26 | 95.11 | +0.00 | 38.14 |
| error only (lambda=1) | 1 | 11.98 | 11.98 | 5 | 95.22 | 95.12 | +0.01 | 38.68 |
| product (sum F*dw^2) |  | 12.00 | 12.00 | 5 | 95.26 | 95.11 | +0.00 | 39.21 |
| worst (max damage) |  | 11.80 | 11.80 | 18 | 95.09 | 95.15 | +0.04 | 33.42 |
| random (n=5) | - | 11.86 | - | - | 95.22 +/- 0.07 | 95.13 +/- 0.01 | +0.02 | - |

lambda chosen on validation: 0.25; 8-bit layers: layer3.0.conv1.weight, layer3.1.conv1.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## int8_pt, target <= 10 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 9.99 | 9.99 | 5 | 95.29 | 95.11 | +0.00 | 36.43 |
| Fisher only (lambda=0) | 0 | 9.99 | 9.99 | 5 | 95.29 | 95.11 | +0.00 | 36.43 |
| error only (lambda=1) | 1 | 9.98 | 9.98 | 6 | 95.26 | 95.11 | +0.00 | 36.03 |
| product (sum F*dw^2) |  | 9.99 | 9.99 | 5 | 95.29 | 95.11 | +0.00 | 36.43 |
| worst (max damage) |  | 9.78 | 9.78 | 19 | 95.09 | 95.16 | +0.05 | 32.71 |
| random (n=5) | - | 9.86 | - | - | 95.20 +/- 0.03 | 95.15 +/- 0.01 | +0.04 | - |

lambda chosen on validation: 0; 8-bit layers: layer4.0.conv1.weight, layer4.0.conv2.weight, layer4.0.downsample.0.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## int8_pt, target <= 8 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 8.00 | 8.00 | 21 | 95.12 | 95.15 | +0.04 | 32.58 |
| Fisher only (lambda=0) | 0 | 8.00 | 8.00 | 21 | 95.12 | 95.15 | +0.04 | 32.58 |
| error only (lambda=1) | 1 | 8.00 | 8.00 | 21 | 95.12 | 95.15 | +0.04 | 32.58 |
| product (sum F*dw^2) |  | 8.00 | 8.00 | 21 | 95.12 | 95.15 | +0.04 | 32.58 |
| worst (max damage) |  | 8.00 | 8.00 | 21 | 95.12 | 95.15 | +0.04 | 32.58 |
| random (n=5) | - | 8.00 | - | - | 95.12 +/- 0.00 | 95.15 +/- 0.00 | +0.04 | - |

lambda chosen on validation: 0; 8-bit layers: conv1.weight, layer1.0.conv1.weight, layer1.0.conv2.weight, layer1.1.conv1.weight, layer1.1.conv2.weight, layer2.0.conv1.weight, layer2.0.conv2.weight, layer2.0.downsample.0.weight, layer2.1.conv1.weight, layer2.1.conv2.weight, layer3.0.conv1.weight, layer3.0.conv2.weight, layer3.0.downsample.0.weight, layer3.1.conv1.weight, layer3.1.conv2.weight, layer4.0.conv1.weight, layer4.0.conv2.weight, layer4.0.downsample.0.weight, layer4.1.conv1.weight, layer4.1.conv2.weight, fc.weight

## fp8_scaled, target <= 14 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 13.99 | 13.99 | 3 | 95.22 | 95.14 | +0.03 | 40.16 |
| Fisher only (lambda=0) | 0 | 13.99 | 13.99 | 3 | 95.22 | 95.14 | +0.03 | 40.16 |
| error only (lambda=1) | 1 | 13.99 | 13.99 | 3 | 95.22 | 95.14 | +0.03 | 39.73 |
| product (sum F*dw^2) |  | 13.98 | 13.98 | 5 | 95.22 | 95.13 | +0.02 | 40.96 |
| worst (max damage) |  | 13.92 | 13.92 | 17 | 95.22 | 95.17 | +0.06 | 32.28 |
| random (n=5) | - | 13.97 | - | - | 95.22 +/- 0.04 | 95.19 +/- 0.02 | +0.08 | - |

lambda chosen on validation: 0; 8-bit layers: layer2.1.conv2.weight, layer3.0.conv1.weight, layer4.1.conv2.weight

## fp8_scaled, target <= 12 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 11.98 | 11.98 | 4 | 95.26 | 95.13 | +0.02 | 38.43 |
| Fisher only (lambda=0) | 0 | 11.98 | 11.98 | 4 | 95.26 | 95.13 | +0.02 | 38.43 |
| error only (lambda=1) | 1 | 11.98 | 11.98 | 4 | 95.22 | 95.14 | +0.03 | 37.29 |
| product (sum F*dw^2) |  | 12.00 | 12.00 | 5 | 95.26 | 95.11 | +0.00 | 39.61 |
| worst (max damage) |  | 11.80 | 11.80 | 18 | 95.16 | 95.16 | +0.05 | 32.06 |
| random (n=5) | - | 11.86 | - | - | 95.20 +/- 0.05 | 95.16 +/- 0.02 | +0.05 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.0.conv1.weight, layer3.1.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## fp8_scaled, target <= 10 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0.5 | 9.98 | 9.98 | 5 | 95.29 | 95.12 | +0.01 | 38.78 |
| Fisher only (lambda=0) | 0 | 9.99 | 9.99 | 5 | 95.29 | 95.11 | +0.00 | 39.33 |
| error only (lambda=1) | 1 | 9.98 | 9.98 | 5 | 95.29 | 95.12 | +0.01 | 38.78 |
| product (sum F*dw^2) |  | 9.99 | 9.99 | 5 | 95.29 | 95.11 | +0.00 | 39.33 |
| worst (max damage) |  | 9.78 | 9.78 | 19 | 95.19 | 95.18 | +0.07 | 31.64 |
| random (n=5) | - | 9.86 | - | - | 95.22 +/- 0.07 | 95.16 +/- 0.03 | +0.05 | - |

lambda chosen on validation: 0.5; 8-bit layers: layer2.1.conv2.weight, layer4.0.conv1.weight, layer4.0.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## fp8_scaled, target <= 8 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 8.00 | 8.00 | 21 | 95.19 | 95.18 | +0.07 | 31.55 |
| Fisher only (lambda=0) | 0 | 8.00 | 8.00 | 21 | 95.19 | 95.18 | +0.07 | 31.55 |
| error only (lambda=1) | 1 | 8.00 | 8.00 | 21 | 95.19 | 95.18 | +0.07 | 31.55 |
| product (sum F*dw^2) |  | 8.00 | 8.00 | 21 | 95.19 | 95.18 | +0.07 | 31.55 |
| worst (max damage) |  | 8.00 | 8.00 | 21 | 95.19 | 95.18 | +0.07 | 31.55 |
| random (n=5) | - | 8.00 | - | - | 95.19 +/- 0.00 | 95.18 +/- 0.00 | +0.07 | - |

lambda chosen on validation: 0; 8-bit layers: conv1.weight, layer1.0.conv1.weight, layer1.0.conv2.weight, layer1.1.conv1.weight, layer1.1.conv2.weight, layer2.0.conv1.weight, layer2.0.conv2.weight, layer2.0.downsample.0.weight, layer2.1.conv1.weight, layer2.1.conv2.weight, layer3.0.conv1.weight, layer3.0.conv2.weight, layer3.0.downsample.0.weight, layer3.1.conv1.weight, layer3.1.conv2.weight, layer4.0.conv1.weight, layer4.0.conv2.weight, layer4.0.downsample.0.weight, layer4.1.conv1.weight, layer4.1.conv2.weight, fc.weight

## int4_pc, target <= 14 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 13.96 | 13.96 | 3 | 95.05 | 95.18 | +0.07 | 21.66 |
| Fisher only (lambda=0) | 0 | 13.96 | 13.96 | 3 | 95.05 | 95.18 | +0.07 | 21.66 |
| error only (lambda=1) | 1 | 13.96 | 13.96 | 3 | 95.02 | 95.17 | +0.06 | 20.22 |
| product (sum F*dw^2) |  | 13.96 | 13.96 | 3 | 95.05 | 95.18 | +0.07 | 21.66 |
| worst (max damage) |  | 13.97 | 13.97 | 13 | 94.85 | 94.90 | -0.21 | 16.45 |
| random (n=5) | - | 13.93 | - | - | 94.94 +/- 0.08 | 95.02 +/- 0.06 | -0.09 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.1.conv2.weight, layer4.0.conv1.weight, layer4.0.downsample.0.weight

## int4_pc, target <= 12 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 11.88 | 11.88 | 3 | 95.09 | 95.21 | +0.10 | 21.41 |
| Fisher only (lambda=0) | 0 | 11.88 | 11.88 | 3 | 95.09 | 95.21 | +0.10 | 21.41 |
| error only (lambda=1) | 1 | 11.98 | 11.98 | 4 | 94.95 | 95.18 | +0.07 | 22.93 |
| product (sum F*dw^2) |  | 11.90 | 11.90 | 4 | 95.09 | 95.15 | +0.04 | 22.78 |
| worst (max damage) |  | 11.91 | 11.91 | 16 | 94.82 | 94.98 | -0.13 | 15.16 |
| random (n=5) | - | 11.90 | - | - | 94.91 +/- 0.13 | 95.04 +/- 0.14 | -0.07 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.0.conv1.weight, layer4.0.conv1.weight, layer4.1.conv2.weight

## int4_pc, target <= 10 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 9.98 | 9.98 | 4 | 95.12 | 95.21 | +0.10 | 20.61 |
| Fisher only (lambda=0) | 0 | 9.98 | 9.98 | 4 | 95.12 | 95.21 | +0.10 | 20.61 |
| error only (lambda=1) | 1 | 9.98 | 9.98 | 4 | 95.05 | 95.19 | +0.08 | 19.20 |
| product (sum F*dw^2) |  | 9.99 | 10.00 | 5 | 95.12 | 95.21 | +0.10 | 21.72 |
| worst (max damage) |  | 9.71 | 9.72 | 18 | 94.82 | 95.03 | -0.08 | 15.09 |
| random (n=5) | - | 9.78 | - | - | 94.99 +/- 0.12 | 95.12 +/- 0.04 | +0.01 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.0.conv1.weight, layer3.1.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## int4_pc, target <= 8 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 7.76 | 7.76 | 4 | 95.05 | 95.15 | +0.04 | 21.22 |
| Fisher only (lambda=0) | 0 | 7.76 | 7.76 | 4 | 95.05 | 95.15 | +0.04 | 21.22 |
| error only (lambda=1) | 1 | 7.76 | 7.76 | 4 | 94.92 | 95.19 | +0.08 | 19.89 |
| product (sum F*dw^2) |  | 7.76 | 7.76 | 4 | 95.05 | 95.15 | +0.04 | 21.22 |
| worst (max damage) |  | 7.80 | 7.81 | 19 | 94.88 | 94.99 | -0.12 | 14.89 |
| random (n=5) | - | 7.88 | - | - | 94.94 +/- 0.09 | 95.11 +/- 0.09 | -0.00 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.1.conv2.weight, layer4.0.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## int4_pc, target <= 6 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0.75 | 5.86 | 5.86 | 6 | 95.02 | 95.08 | -0.03 | 17.16 |
| Fisher only (lambda=0) | 0 | 5.86 | 5.86 | 6 | 95.02 | 95.13 | +0.02 | 17.82 |
| error only (lambda=1) | 1 | 5.86 | 5.86 | 6 | 95.02 | 95.08 | -0.03 | 17.16 |
| product (sum F*dw^2) |  | 6.00 | 6.01 | 9 | 95.12 | 95.20 | +0.09 | 18.55 |
| worst (max damage) |  | 5.57 | 5.58 | 18 | 94.85 | 94.96 | -0.15 | 15.02 |
| random (n=5) | - | 5.76 | - | - | 95.10 +/- 0.10 | 95.08 +/- 0.10 | -0.03 | - |

lambda chosen on validation: 0.75; 8-bit layers: layer3.0.conv2.weight, layer3.1.conv1.weight, layer4.0.conv1.weight, layer4.0.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## int4_pc, target <= 5 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 4.91 | 4.91 | 8 | 95.09 | 95.21 | +0.10 | 15.88 |
| Fisher only (lambda=0) | 0 | 4.91 | 4.91 | 8 | 95.09 | 95.21 | +0.10 | 15.88 |
| error only (lambda=1) | 1 | 4.91 | 4.91 | 8 | 95.09 | 95.21 | +0.10 | 15.88 |
| product (sum F*dw^2) |  | 4.92 | 4.93 | 9 | 95.09 | 95.16 | +0.05 | 16.23 |
| worst (max damage) |  | 4.63 | 4.65 | 20 | 94.82 | 95.02 | -0.09 | 14.80 |
| random (n=5) | - | 4.75 | - | - | 94.95 +/- 0.11 | 95.07 +/- 0.10 | -0.04 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.0.conv1.weight, layer3.0.conv2.weight, layer3.1.conv1.weight, layer3.1.conv2.weight, layer4.0.conv1.weight, layer4.0.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## int4_pt, target <= 14 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0.25 | 13.96 | 13.96 | 3 | 95.05 | 95.03 | -0.08 | 13.73 |
| Fisher only (lambda=0) | 0 | 13.96 | 13.96 | 3 | 94.95 | 95.12 | +0.01 | 13.79 |
| error only (lambda=1) | 1 | 13.94 | 13.94 | 4 | 94.68 | 95.08 | -0.03 | 13.00 |
| product (sum F*dw^2) |  | 13.96 | 13.96 | 3 | 94.95 | 95.12 | +0.01 | 13.79 |
| worst (max damage) |  | 13.97 | 13.97 | 13 | 94.14 | 94.10 | -1.01 | 11.01 |
| random (n=5) | - | 13.93 | - | - | 94.35 +/- 0.23 | 94.40 +/- 0.14 | -0.71 | - |

lambda chosen on validation: 0.25; 8-bit layers: layer3.1.conv1.weight, layer4.0.conv1.weight, layer4.0.downsample.0.weight

## int4_pt, target <= 12 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 11.88 | 11.88 | 3 | 94.99 | 95.10 | -0.01 | 13.29 |
| Fisher only (lambda=0) | 0 | 11.88 | 11.88 | 3 | 94.99 | 95.10 | -0.01 | 13.29 |
| error only (lambda=1) | 1 | 12.00 | 12.00 | 5 | 94.99 | 95.12 | +0.01 | 14.01 |
| product (sum F*dw^2) |  | 11.90 | 11.90 | 4 | 95.02 | 95.07 | -0.04 | 14.48 |
| worst (max damage) |  | 11.91 | 11.91 | 16 | 93.90 | 94.07 | -1.04 | 9.10 |
| random (n=5) | - | 11.90 | - | - | 94.47 +/- 0.24 | 94.52 +/- 0.23 | -0.59 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.0.conv1.weight, layer4.0.conv1.weight, layer4.1.conv2.weight

## int4_pt, target <= 10 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0.25 | 9.98 | 9.98 | 4 | 95.16 | 94.97 | -0.14 | 13.38 |
| Fisher only (lambda=0) | 0 | 9.98 | 9.98 | 4 | 95.02 | 95.12 | +0.01 | 13.45 |
| error only (lambda=1) | 1 | 9.98 | 9.98 | 5 | 94.92 | 94.95 | -0.16 | 14.01 |
| product (sum F*dw^2) |  | 9.99 | 9.99 | 5 | 94.99 | 95.10 | -0.01 | 14.69 |
| worst (max damage) |  | 9.71 | 9.71 | 18 | 93.77 | 93.87 | -1.24 | 8.62 |
| random (n=5) | - | 9.78 | - | - | 94.62 +/- 0.25 | 94.60 +/- 0.22 | -0.51 | - |

lambda chosen on validation: 0.25; 8-bit layers: layer3.0.conv1.weight, layer3.1.conv1.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## int4_pt, target <= 8 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0.75 | 7.76 | 7.76 | 4 | 95.16 | 94.92 | -0.19 | 12.99 |
| Fisher only (lambda=0) | 0 | 7.76 | 7.76 | 4 | 94.99 | 95.08 | -0.03 | 13.18 |
| error only (lambda=1) | 1 | 7.76 | 7.76 | 4 | 95.16 | 94.92 | -0.19 | 12.99 |
| product (sum F*dw^2) |  | 7.76 | 7.76 | 4 | 94.99 | 95.08 | -0.03 | 13.18 |
| worst (max damage) |  | 7.80 | 7.80 | 19 | 94.04 | 93.96 | -1.15 | 8.67 |
| random (n=5) | - | 7.88 | - | - | 94.46 +/- 0.22 | 94.45 +/- 0.17 | -0.66 | - |

lambda chosen on validation: 0.75; 8-bit layers: layer3.0.conv2.weight, layer4.0.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## int4_pt, target <= 6 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 5.86 | 5.86 | 6 | 94.99 | 94.93 | -0.18 | 10.52 |
| Fisher only (lambda=0) | 0 | 5.86 | 5.86 | 6 | 94.99 | 94.93 | -0.18 | 10.52 |
| error only (lambda=1) | 1 | 5.86 | 5.86 | 6 | 94.88 | 94.85 | -0.26 | 10.42 |
| product (sum F*dw^2) |  | 5.86 | 5.86 | 6 | 94.99 | 94.93 | -0.18 | 10.52 |
| worst (max damage) |  | 5.57 | 5.57 | 18 | 93.94 | 94.08 | -1.03 | 8.82 |
| random (n=5) | - | 5.76 | - | - | 94.51 +/- 0.24 | 94.49 +/- 0.22 | -0.62 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.1.conv1.weight, layer3.1.conv2.weight, layer4.0.conv1.weight, layer4.0.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## int4_pt, target <= 5 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0.75 | 4.98 | 4.98 | 9 | 94.88 | 94.78 | -0.33 | 9.50 |
| Fisher only (lambda=0) | 0 | 4.91 | 4.91 | 8 | 94.61 | 94.81 | -0.30 | 9.07 |
| error only (lambda=1) | 1 | 4.98 | 4.98 | 9 | 94.88 | 94.78 | -0.33 | 9.50 |
| product (sum F*dw^2) |  | 4.92 | 4.92 | 9 | 94.72 | 94.86 | -0.25 | 9.48 |
| worst (max damage) |  | 4.63 | 4.63 | 20 | 94.00 | 93.84 | -1.27 | 8.25 |
| random (n=5) | - | 4.75 | - | - | 94.40 +/- 0.19 | 94.41 +/- 0.24 | -0.70 | - |

lambda chosen on validation: 0.75; 8-bit layers: layer2.0.conv1.weight, layer2.1.conv1.weight, layer3.0.conv2.weight, layer3.1.conv1.weight, layer3.1.conv2.weight, layer4.0.conv1.weight, layer4.0.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## fp4_scaled, target <= 14 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 13.96 | 13.96 | 3 | 95.16 | 95.18 | +0.07 | 16.83 |
| Fisher only (lambda=0) | 0 | 13.96 | 13.96 | 3 | 95.16 | 95.18 | +0.07 | 16.83 |
| error only (lambda=1) | 1 | 13.94 | 13.94 | 4 | 95.12 | 95.24 | +0.13 | 16.73 |
| product (sum F*dw^2) |  | 13.96 | 13.96 | 3 | 95.16 | 95.18 | +0.07 | 16.83 |
| worst (max damage) |  | 13.97 | 13.97 | 13 | 94.17 | 93.99 | -1.12 | 13.06 |
| random (n=5) | - | 13.93 | - | - | 94.44 +/- 0.12 | 94.50 +/- 0.27 | -0.61 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.1.conv2.weight, layer4.0.conv1.weight, layer4.0.downsample.0.weight

## fp4_scaled, target <= 12 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 11.88 | 11.88 | 3 | 95.29 | 95.10 | -0.01 | 15.65 |
| Fisher only (lambda=0) | 0 | 11.88 | 11.88 | 3 | 95.29 | 95.10 | -0.01 | 15.65 |
| error only (lambda=1) | 1 | 12.00 | 12.00 | 5 | 94.92 | 95.06 | -0.05 | 17.54 |
| product (sum F*dw^2) |  | 11.90 | 11.90 | 4 | 95.09 | 95.12 | +0.01 | 17.58 |
| worst (max damage) |  | 11.91 | 11.91 | 16 | 94.11 | 93.77 | -1.34 | 11.25 |
| random (n=5) | - | 11.90 | - | - | 94.67 +/- 0.26 | 94.57 +/- 0.33 | -0.54 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.0.conv1.weight, layer4.0.conv1.weight, layer4.1.conv2.weight

## fp4_scaled, target <= 10 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 9.98 | 9.98 | 4 | 95.22 | 95.07 | -0.04 | 15.80 |
| Fisher only (lambda=0) | 0 | 9.98 | 9.98 | 4 | 95.22 | 95.07 | -0.04 | 15.80 |
| error only (lambda=1) | 1 | 9.98 | 9.98 | 5 | 95.05 | 95.09 | -0.02 | 16.66 |
| product (sum F*dw^2) |  | 9.99 | 9.99 | 5 | 95.05 | 95.21 | +0.10 | 17.83 |
| worst (max damage) |  | 9.71 | 9.71 | 18 | 94.11 | 93.86 | -1.25 | 10.39 |
| random (n=5) | - | 9.78 | - | - | 94.72 +/- 0.24 | 94.71 +/- 0.13 | -0.40 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.0.conv1.weight, layer3.1.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## fp4_scaled, target <= 8 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 7.76 | 7.76 | 4 | 95.19 | 95.18 | +0.07 | 14.81 |
| Fisher only (lambda=0) | 0 | 7.76 | 7.76 | 4 | 95.19 | 95.18 | +0.07 | 14.81 |
| error only (lambda=1) | 1 | 7.76 | 7.76 | 4 | 95.09 | 95.15 | +0.04 | 14.79 |
| product (sum F*dw^2) |  | 7.76 | 7.76 | 4 | 95.19 | 95.18 | +0.07 | 14.81 |
| worst (max damage) |  | 7.80 | 7.80 | 19 | 93.83 | 93.60 | -1.51 | 10.44 |
| random (n=5) | - | 7.88 | - | - | 94.59 +/- 0.19 | 94.54 +/- 0.28 | -0.57 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.1.conv2.weight, layer4.0.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## fp4_scaled, target <= 6 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 5.86 | 5.86 | 6 | 95.09 | 95.11 | +0.00 | 12.80 |
| Fisher only (lambda=0) | 0 | 5.86 | 5.86 | 6 | 95.09 | 95.11 | +0.00 | 12.80 |
| error only (lambda=1) | 1 | 5.86 | 5.86 | 6 | 95.05 | 95.09 | -0.02 | 12.79 |
| product (sum F*dw^2) |  | 5.86 | 5.86 | 6 | 95.09 | 95.11 | +0.00 | 12.80 |
| worst (max damage) |  | 5.57 | 5.57 | 18 | 93.83 | 93.71 | -1.40 | 10.52 |
| random (n=5) | - | 5.76 | - | - | 94.61 +/- 0.25 | 94.55 +/- 0.45 | -0.56 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.1.conv1.weight, layer3.1.conv2.weight, layer4.0.conv1.weight, layer4.0.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## fp4_scaled, target <= 5 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 4.91 | 4.91 | 8 | 95.33 | 95.12 | +0.01 | 11.43 |
| Fisher only (lambda=0) | 0 | 4.91 | 4.91 | 8 | 95.33 | 95.12 | +0.01 | 11.43 |
| error only (lambda=1) | 1 | 4.98 | 4.98 | 9 | 94.92 | 95.17 | +0.06 | 11.95 |
| product (sum F*dw^2) |  | 4.92 | 4.92 | 9 | 94.92 | 95.08 | -0.03 | 12.07 |
| worst (max damage) |  | 4.63 | 4.63 | 20 | 94.04 | 93.83 | -1.28 | 10.08 |
| random (n=5) | - | 4.75 | - | - | 94.55 +/- 0.30 | 94.42 +/- 0.36 | -0.69 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.0.conv1.weight, layer3.0.conv2.weight, layer3.1.conv1.weight, layer3.1.conv2.weight, layer4.0.conv1.weight, layer4.0.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight
