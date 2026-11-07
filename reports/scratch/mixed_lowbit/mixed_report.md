# Mixed-precision (E4M11 + low-bit) results

* Fisher: `true`, 1024 calibration images (the training split); score normalization `rank`, Fisher `per_param`; E4M11 baseline `subnormal` bias 7.
* Selection and lambda choice use the validation split (5000 images, standard error about 0.31 pp at 95% accuracy): differences below ~0.6 pp are not significant.
* Test accuracy is reported for the saved selections only; lambda was chosen on validation.
* `random` rows: mean +/- std over draws; `worst` = feasible set with the largest total damage (sanity envelope). avg bits = payload bits per weight; +meta also counts scale metadata.

## References

| config | val acc % | test acc % | SQNR dB | avg bits |
|---|---|---|---|---|
| fp32 | 95.48 | 95.11 | - | 32.00 |
| e4m11_all | 95.48 | 95.11 | 69.27 | 16.00 |
| int4_pc_all | 95.00 | 94.99 | 14.40 | 4.00 |
| int4_pt_all | 94.30 | 93.88 | 7.85 | 4.00 |
| fp4_scaled_all | 94.16 | 93.78 | 9.80 | 4.00 |

## Frontier: accuracy % at each storage budget (test; `v` = validation only)

Fisher-based: Fisher only (lambda=0), product. Without Fisher: error only (lambda=1), random (mean +/- std of the draws), worst (largest damage). `uniform` = every layer in the low-bit format.

| arm | budget | avg bits | Fisher only | product | error only | random | worst | uniform (all layers) |
|---|---|---|---|---|---|---|---|---|
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

## int4_pc, target <= 14 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 13.96 | 13.96 | 3 | 95.36 | 95.18 | +0.07 | 21.66 |
| Fisher only (lambda=0) | 0 | 13.96 | 13.96 | 3 | 95.36 | 95.18 | +0.07 | 21.66 |
| error only (lambda=1) | 1 | 13.96 | 13.96 | 3 | 95.24 | 95.17 | +0.06 | 20.22 |
| product (sum F*dw^2) |  | 13.96 | 13.96 | 3 | 95.36 | 95.18 | +0.07 | 21.66 |
| worst (max damage) |  | 13.97 | 13.97 | 13 | 94.98 | 94.90 | -0.21 | 16.45 |
| random (n=5) | - | 13.93 | - | - | 95.14 +/- 0.09 | 95.02 +/- 0.06 | -0.09 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.1.conv2.weight, layer4.0.conv1.weight, layer4.0.downsample.0.weight

## int4_pc, target <= 12 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 11.88 | 11.88 | 3 | 95.38 | 95.21 | +0.10 | 21.41 |
| Fisher only (lambda=0) | 0 | 11.88 | 11.88 | 3 | 95.38 | 95.21 | +0.10 | 21.41 |
| error only (lambda=1) | 1 | 11.98 | 11.98 | 4 | 95.22 | 95.18 | +0.07 | 22.93 |
| product (sum F*dw^2) |  | 11.90 | 11.90 | 4 | 95.40 | 95.15 | +0.04 | 22.78 |
| worst (max damage) |  | 11.91 | 11.91 | 16 | 94.94 | 94.98 | -0.13 | 15.16 |
| random (n=5) | - | 11.90 | - | - | 95.13 +/- 0.10 | 95.04 +/- 0.14 | -0.07 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.0.conv1.weight, layer4.0.conv1.weight, layer4.1.conv2.weight

## int4_pc, target <= 10 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0.5 | 9.98 | 9.98 | 4 | 95.34 | 95.20 | +0.09 | 20.31 |
| Fisher only (lambda=0) | 0 | 9.98 | 9.98 | 4 | 95.30 | 95.21 | +0.10 | 20.61 |
| error only (lambda=1) | 1 | 9.98 | 9.98 | 4 | 95.22 | 95.19 | +0.08 | 19.20 |
| product (sum F*dw^2) |  | 9.99 | 10.00 | 5 | 95.42 | 95.21 | +0.10 | 21.72 |
| worst (max damage) |  | 9.71 | 9.72 | 18 | 94.96 | 95.03 | -0.08 | 15.09 |
| random (n=5) | - | 9.78 | - | - | 95.18 +/- 0.10 | 95.12 +/- 0.04 | +0.01 | - |

lambda chosen on validation: 0.5; 8-bit layers: layer3.0.conv1.weight, layer3.1.conv2.weight, layer4.0.conv2.weight, layer4.1.conv2.weight

## int4_pc, target <= 8 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 7.76 | 7.76 | 4 | 95.36 | 95.15 | +0.04 | 21.22 |
| Fisher only (lambda=0) | 0 | 7.76 | 7.76 | 4 | 95.36 | 95.15 | +0.04 | 21.22 |
| error only (lambda=1) | 1 | 7.76 | 7.76 | 4 | 95.14 | 95.19 | +0.08 | 19.89 |
| product (sum F*dw^2) |  | 7.76 | 7.76 | 4 | 95.36 | 95.15 | +0.04 | 21.22 |
| worst (max damage) |  | 7.80 | 7.81 | 19 | 95.04 | 94.99 | -0.12 | 14.89 |
| random (n=5) | - | 7.88 | - | - | 95.15 +/- 0.08 | 95.11 +/- 0.09 | -0.00 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.1.conv2.weight, layer4.0.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## int4_pc, target <= 6 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 5.86 | 5.86 | 6 | 95.28 | 95.13 | +0.02 | 17.82 |
| Fisher only (lambda=0) | 0 | 5.86 | 5.86 | 6 | 95.28 | 95.13 | +0.02 | 17.82 |
| error only (lambda=1) | 1 | 5.86 | 5.86 | 6 | 95.24 | 95.08 | -0.03 | 17.16 |
| product (sum F*dw^2) |  | 6.00 | 6.01 | 9 | 95.40 | 95.20 | +0.09 | 18.55 |
| worst (max damage) |  | 5.57 | 5.58 | 18 | 95.02 | 94.96 | -0.15 | 15.02 |
| random (n=5) | - | 5.76 | - | - | 95.24 +/- 0.13 | 95.08 +/- 0.10 | -0.03 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.1.conv1.weight, layer3.1.conv2.weight, layer4.0.conv1.weight, layer4.0.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## int4_pc, target <= 5 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 4.91 | 4.91 | 8 | 95.26 | 95.21 | +0.10 | 15.88 |
| Fisher only (lambda=0) | 0 | 4.91 | 4.91 | 8 | 95.26 | 95.21 | +0.10 | 15.88 |
| error only (lambda=1) | 1 | 4.91 | 4.91 | 8 | 95.26 | 95.21 | +0.10 | 15.88 |
| product (sum F*dw^2) |  | 4.92 | 4.93 | 9 | 95.26 | 95.16 | +0.05 | 16.23 |
| worst (max damage) |  | 4.63 | 4.65 | 20 | 94.98 | 95.02 | -0.09 | 14.80 |
| random (n=5) | - | 4.75 | - | - | 95.18 +/- 0.10 | 95.07 +/- 0.10 | -0.04 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.0.conv1.weight, layer3.0.conv2.weight, layer3.1.conv1.weight, layer3.1.conv2.weight, layer4.0.conv1.weight, layer4.0.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## int4_pt, target <= 14 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 13.96 | 13.96 | 3 | 95.22 | 95.12 | +0.01 | 13.79 |
| Fisher only (lambda=0) | 0 | 13.96 | 13.96 | 3 | 95.22 | 95.12 | +0.01 | 13.79 |
| error only (lambda=1) | 1 | 13.94 | 13.94 | 4 | 94.98 | 95.08 | -0.03 | 13.00 |
| product (sum F*dw^2) |  | 13.96 | 13.96 | 3 | 95.22 | 95.12 | +0.01 | 13.79 |
| worst (max damage) |  | 13.97 | 13.97 | 13 | 94.36 | 94.10 | -1.01 | 11.01 |
| random (n=5) | - | 13.93 | - | - | 94.73 +/- 0.16 | 94.40 +/- 0.14 | -0.71 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.1.conv2.weight, layer4.0.conv1.weight, layer4.0.downsample.0.weight

## int4_pt, target <= 12 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 11.88 | 11.88 | 3 | 95.28 | 95.10 | -0.01 | 13.29 |
| Fisher only (lambda=0) | 0 | 11.88 | 11.88 | 3 | 95.28 | 95.10 | -0.01 | 13.29 |
| error only (lambda=1) | 1 | 12.00 | 12.00 | 5 | 95.16 | 95.12 | +0.01 | 14.01 |
| product (sum F*dw^2) |  | 11.90 | 11.90 | 4 | 95.28 | 95.07 | -0.04 | 14.48 |
| worst (max damage) |  | 11.91 | 11.91 | 16 | 94.26 | 94.07 | -1.04 | 9.10 |
| random (n=5) | - | 11.90 | - | - | 94.80 +/- 0.16 | 94.52 +/- 0.23 | -0.59 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.0.conv1.weight, layer4.0.conv1.weight, layer4.1.conv2.weight

## int4_pt, target <= 10 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 9.98 | 9.98 | 4 | 95.28 | 95.12 | +0.01 | 13.45 |
| Fisher only (lambda=0) | 0 | 9.98 | 9.98 | 4 | 95.28 | 95.12 | +0.01 | 13.45 |
| error only (lambda=1) | 1 | 9.98 | 9.98 | 5 | 95.10 | 94.95 | -0.16 | 14.01 |
| product (sum F*dw^2) |  | 9.99 | 9.99 | 5 | 95.32 | 95.10 | -0.01 | 14.69 |
| worst (max damage) |  | 9.71 | 9.71 | 18 | 94.08 | 93.87 | -1.24 | 8.62 |
| random (n=5) | - | 9.78 | - | - | 94.88 +/- 0.16 | 94.60 +/- 0.22 | -0.51 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.0.conv1.weight, layer3.1.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## int4_pt, target <= 8 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0.5 | 7.76 | 7.76 | 4 | 95.32 | 94.92 | -0.19 | 12.99 |
| Fisher only (lambda=0) | 0 | 7.76 | 7.76 | 4 | 95.30 | 95.08 | -0.03 | 13.18 |
| error only (lambda=1) | 1 | 7.76 | 7.76 | 4 | 95.32 | 94.92 | -0.19 | 12.99 |
| product (sum F*dw^2) |  | 7.76 | 7.76 | 4 | 95.30 | 95.08 | -0.03 | 13.18 |
| worst (max damage) |  | 7.80 | 7.80 | 19 | 94.20 | 93.96 | -1.15 | 8.67 |
| random (n=5) | - | 7.88 | - | - | 94.79 +/- 0.16 | 94.45 +/- 0.17 | -0.66 | - |

lambda chosen on validation: 0.5; 8-bit layers: layer3.0.conv2.weight, layer4.0.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## int4_pt, target <= 6 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 5.86 | 5.86 | 6 | 95.26 | 94.93 | -0.18 | 10.52 |
| Fisher only (lambda=0) | 0 | 5.86 | 5.86 | 6 | 95.26 | 94.93 | -0.18 | 10.52 |
| error only (lambda=1) | 1 | 5.86 | 5.86 | 6 | 95.22 | 94.85 | -0.26 | 10.42 |
| product (sum F*dw^2) |  | 5.86 | 5.86 | 6 | 95.26 | 94.93 | -0.18 | 10.52 |
| worst (max damage) |  | 5.57 | 5.57 | 18 | 94.20 | 94.08 | -1.03 | 8.82 |
| random (n=5) | - | 5.76 | - | - | 94.84 +/- 0.23 | 94.49 +/- 0.22 | -0.62 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.1.conv1.weight, layer3.1.conv2.weight, layer4.0.conv1.weight, layer4.0.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## int4_pt, target <= 5 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0.75 | 4.98 | 4.98 | 9 | 95.10 | 94.78 | -0.33 | 9.50 |
| Fisher only (lambda=0) | 0 | 4.91 | 4.91 | 8 | 94.98 | 94.81 | -0.30 | 9.07 |
| error only (lambda=1) | 1 | 4.98 | 4.98 | 9 | 95.10 | 94.78 | -0.33 | 9.50 |
| product (sum F*dw^2) |  | 4.92 | 4.92 | 9 | 95.00 | 94.86 | -0.25 | 9.48 |
| worst (max damage) |  | 4.63 | 4.63 | 20 | 94.28 | 93.84 | -1.27 | 8.25 |
| random (n=5) | - | 4.75 | - | - | 94.69 +/- 0.17 | 94.41 +/- 0.24 | -0.70 | - |

lambda chosen on validation: 0.75; 8-bit layers: layer2.0.conv1.weight, layer2.1.conv1.weight, layer3.0.conv2.weight, layer3.1.conv1.weight, layer3.1.conv2.weight, layer4.0.conv1.weight, layer4.0.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## fp4_scaled, target <= 14 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 13.96 | 13.96 | 3 | 95.40 | 95.18 | +0.07 | 16.83 |
| Fisher only (lambda=0) | 0 | 13.96 | 13.96 | 3 | 95.40 | 95.18 | +0.07 | 16.83 |
| error only (lambda=1) | 1 | 13.94 | 13.94 | 4 | 95.26 | 95.24 | +0.13 | 16.73 |
| product (sum F*dw^2) |  | 13.96 | 13.96 | 3 | 95.40 | 95.18 | +0.07 | 16.83 |
| worst (max damage) |  | 13.97 | 13.97 | 13 | 94.34 | 93.99 | -1.12 | 13.06 |
| random (n=5) | - | 13.93 | - | - | 94.67 +/- 0.16 | 94.50 +/- 0.27 | -0.61 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.1.conv2.weight, layer4.0.conv1.weight, layer4.0.downsample.0.weight

## fp4_scaled, target <= 12 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 11.88 | 11.88 | 3 | 95.48 | 95.10 | -0.01 | 15.65 |
| Fisher only (lambda=0) | 0 | 11.88 | 11.88 | 3 | 95.48 | 95.10 | -0.01 | 15.65 |
| error only (lambda=1) | 1 | 12.00 | 12.00 | 5 | 95.24 | 95.06 | -0.05 | 17.54 |
| product (sum F*dw^2) |  | 11.90 | 11.90 | 4 | 95.32 | 95.12 | +0.01 | 17.58 |
| worst (max damage) |  | 11.91 | 11.91 | 16 | 94.24 | 93.77 | -1.34 | 11.25 |
| random (n=5) | - | 11.90 | - | - | 94.90 +/- 0.27 | 94.57 +/- 0.33 | -0.54 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.0.conv1.weight, layer4.0.conv1.weight, layer4.1.conv2.weight

## fp4_scaled, target <= 10 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 9.98 | 9.98 | 4 | 95.42 | 95.07 | -0.04 | 15.80 |
| Fisher only (lambda=0) | 0 | 9.98 | 9.98 | 4 | 95.42 | 95.07 | -0.04 | 15.80 |
| error only (lambda=1) | 1 | 9.98 | 9.98 | 5 | 95.30 | 95.09 | -0.02 | 16.66 |
| product (sum F*dw^2) |  | 9.99 | 9.99 | 5 | 95.34 | 95.21 | +0.10 | 17.83 |
| worst (max damage) |  | 9.71 | 9.71 | 18 | 94.22 | 93.86 | -1.25 | 10.39 |
| random (n=5) | - | 9.78 | - | - | 94.96 +/- 0.12 | 94.71 +/- 0.13 | -0.40 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.0.conv1.weight, layer3.1.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## fp4_scaled, target <= 8 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 7.76 | 7.76 | 4 | 95.36 | 95.18 | +0.07 | 14.81 |
| Fisher only (lambda=0) | 0 | 7.76 | 7.76 | 4 | 95.36 | 95.18 | +0.07 | 14.81 |
| error only (lambda=1) | 1 | 7.76 | 7.76 | 4 | 95.28 | 95.15 | +0.04 | 14.79 |
| product (sum F*dw^2) |  | 7.76 | 7.76 | 4 | 95.36 | 95.18 | +0.07 | 14.81 |
| worst (max damage) |  | 7.80 | 7.80 | 19 | 94.14 | 93.60 | -1.51 | 10.44 |
| random (n=5) | - | 7.88 | - | - | 94.83 +/- 0.16 | 94.54 +/- 0.28 | -0.57 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.1.conv2.weight, layer4.0.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## fp4_scaled, target <= 6 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 5.86 | 5.86 | 6 | 95.32 | 95.11 | +0.00 | 12.80 |
| Fisher only (lambda=0) | 0 | 5.86 | 5.86 | 6 | 95.32 | 95.11 | +0.00 | 12.80 |
| error only (lambda=1) | 1 | 5.86 | 5.86 | 6 | 95.30 | 95.09 | -0.02 | 12.79 |
| product (sum F*dw^2) |  | 5.86 | 5.86 | 6 | 95.32 | 95.11 | +0.00 | 12.80 |
| worst (max damage) |  | 5.57 | 5.57 | 18 | 94.10 | 93.71 | -1.40 | 10.52 |
| random (n=5) | - | 5.76 | - | - | 94.86 +/- 0.26 | 94.55 +/- 0.45 | -0.56 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.1.conv1.weight, layer3.1.conv2.weight, layer4.0.conv1.weight, layer4.0.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## fp4_scaled, target <= 5 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 4.91 | 4.91 | 8 | 95.40 | 95.12 | +0.01 | 11.43 |
| Fisher only (lambda=0) | 0 | 4.91 | 4.91 | 8 | 95.40 | 95.12 | +0.01 | 11.43 |
| error only (lambda=1) | 1 | 4.98 | 4.98 | 9 | 95.14 | 95.17 | +0.06 | 11.95 |
| product (sum F*dw^2) |  | 4.92 | 4.92 | 9 | 95.12 | 95.08 | -0.03 | 12.07 |
| worst (max damage) |  | 4.63 | 4.63 | 20 | 94.22 | 93.83 | -1.28 | 10.08 |
| random (n=5) | - | 4.75 | - | - | 94.77 +/- 0.22 | 94.42 +/- 0.36 | -0.69 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.0.conv1.weight, layer3.0.conv2.weight, layer3.1.conv1.weight, layer3.1.conv2.weight, layer4.0.conv1.weight, layer4.0.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight
