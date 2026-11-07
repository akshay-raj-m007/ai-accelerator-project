# Mixed-precision (E4M11 + 8-bit) results

* Fisher: `true`, 1024 calibration images (training split); score normalization `rank`, Fisher `per_param`; E4M11 baseline `subnormal` bias 7.
* Selection and lambda choice use the validation split (5000 images, standard error about 0.31 pp at 95% accuracy): differences below ~0.6 pp are not significant.
* Test accuracy is reported for the saved selections only; lambda was chosen on validation.
* `random` rows: mean +/- std over draws; `worst` = feasible set with the largest total damage (sanity envelope). avg bits = payload bits per weight; +meta also counts scale metadata.

## References

| config | val acc % | test acc % | SQNR dB | avg bits |
|---|---|---|---|---|
| fp32 | 95.48 | 95.11 | - | 32.00 |
| e4m11_all | 95.48 | 95.11 | 69.27 | 16.00 |
| int8_pt_all | 95.36 | 95.15 | 32.58 | 8.00 |
| int8_pc_all | 95.38 | 95.14 | 39.55 | 8.00 |
| fp8_all | 95.30 | 95.12 | 21.20 | 8.00 |
| fp8_scaled_all | 95.40 | 95.18 | 31.55 | 8.00 |

## int8_pt, target <= 14 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 13.99 | 13.99 | 3 | 95.46 | 95.13 | +0.02 | 40.74 |
| Fisher only (lambda=0) | 0 | 13.99 | 13.99 | 3 | 95.46 | 95.13 | +0.02 | 40.74 |
| error only (lambda=1) | 1 | 13.99 | 13.99 | 3 | 95.46 | 95.11 | +0.00 | 41.03 |
| product (sum F*dw^2) |  | 13.98 | 13.98 | 5 | 95.48 | 95.12 | +0.01 | 42.79 |
| worst (max damage) |  | 13.92 | 13.92 | 17 | 95.36 | 95.16 | +0.05 | 34.81 |
| random (n=5) | - | 13.97 | - | - | 95.45 +/- 0.02 | 95.15 +/- 0.02 | +0.04 | - |

lambda chosen on validation: 0; 8-bit layers: layer2.1.conv2.weight, layer3.0.conv1.weight, layer4.1.conv2.weight

## int8_pt, target <= 12 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0.5 | 11.98 | 11.98 | 4 | 95.46 | 95.13 | +0.02 | 37.99 |
| Fisher only (lambda=0) | 0 | 11.98 | 11.98 | 4 | 95.46 | 95.11 | +0.00 | 38.14 |
| error only (lambda=1) | 1 | 11.98 | 11.98 | 5 | 95.44 | 95.12 | +0.01 | 38.68 |
| product (sum F*dw^2) |  | 12.00 | 12.00 | 5 | 95.46 | 95.11 | +0.00 | 39.21 |
| worst (max damage) |  | 11.80 | 11.80 | 18 | 95.36 | 95.15 | +0.04 | 33.42 |
| random (n=5) | - | 11.86 | - | - | 95.44 +/- 0.05 | 95.13 +/- 0.01 | +0.02 | - |

lambda chosen on validation: 0.5; 8-bit layers: layer3.0.conv1.weight, layer3.0.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## int8_pt, target <= 10 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 9.99 | 9.99 | 5 | 95.48 | 95.11 | +0.00 | 36.43 |
| Fisher only (lambda=0) | 0 | 9.99 | 9.99 | 5 | 95.48 | 95.11 | +0.00 | 36.43 |
| error only (lambda=1) | 1 | 9.98 | 9.98 | 6 | 95.46 | 95.11 | +0.00 | 36.03 |
| product (sum F*dw^2) |  | 9.99 | 9.99 | 5 | 95.48 | 95.11 | +0.00 | 36.43 |
| worst (max damage) |  | 9.78 | 9.78 | 19 | 95.34 | 95.16 | +0.05 | 32.71 |
| random (n=5) | - | 9.86 | - | - | 95.42 +/- 0.02 | 95.15 +/- 0.01 | +0.04 | - |

lambda chosen on validation: 0; 8-bit layers: layer4.0.conv1.weight, layer4.0.conv2.weight, layer4.0.downsample.0.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## int8_pc, target <= 14 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0.75 | 13.98 | 13.99 | 4 | 95.48 | 95.11 | +0.00 | 48.61 |
| Fisher only (lambda=0) | 0 | 13.99 | 13.99 | 3 | 95.48 | 95.12 | +0.01 | 48.12 |
| error only (lambda=1) | 1 | 13.98 | 13.99 | 4 | 95.48 | 95.11 | +0.00 | 48.61 |
| product (sum F*dw^2) |  | 13.98 | 13.99 | 5 | 95.48 | 95.11 | +0.00 | 49.62 |
| worst (max damage) |  | 13.92 | 13.92 | 17 | 95.40 | 95.14 | +0.03 | 40.58 |
| random (n=5) | - | 13.97 | - | - | 95.46 +/- 0.03 | 95.13 +/- 0.01 | +0.02 | - |

lambda chosen on validation: 0.75; 8-bit layers: layer3.0.conv1.weight, layer3.0.downsample.0.weight, layer4.0.downsample.0.weight, layer4.1.conv2.weight

## int8_pc, target <= 12 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0.75 | 11.98 | 11.99 | 4 | 95.48 | 95.11 | +0.00 | 44.35 |
| Fisher only (lambda=0) | 0 | 11.98 | 11.99 | 4 | 95.48 | 95.12 | +0.01 | 45.77 |
| error only (lambda=1) | 1 | 11.98 | 11.99 | 4 | 95.48 | 95.11 | +0.00 | 44.35 |
| product (sum F*dw^2) |  | 12.00 | 12.00 | 5 | 95.46 | 95.12 | +0.01 | 46.88 |
| worst (max damage) |  | 11.80 | 11.81 | 18 | 95.40 | 95.17 | +0.06 | 40.23 |
| random (n=5) | - | 11.86 | - | - | 95.45 +/- 0.02 | 95.13 +/- 0.01 | +0.02 | - |

lambda chosen on validation: 0.75; 8-bit layers: layer3.0.conv1.weight, layer3.0.conv2.weight, layer4.0.conv2.weight, layer4.1.conv2.weight

## int8_pc, target <= 10 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 9.99 | 10.00 | 5 | 95.46 | 95.11 | +0.00 | 46.13 |
| Fisher only (lambda=0) | 0 | 9.99 | 10.00 | 5 | 95.46 | 95.11 | +0.00 | 46.13 |
| error only (lambda=1) | 1 | 9.99 | 10.00 | 5 | 95.46 | 95.11 | +0.00 | 46.13 |
| product (sum F*dw^2) |  | 9.99 | 10.00 | 5 | 95.46 | 95.11 | +0.00 | 46.13 |
| worst (max damage) |  | 9.78 | 9.80 | 19 | 95.38 | 95.13 | +0.02 | 39.65 |
| random (n=5) | - | 9.86 | - | - | 95.45 +/- 0.02 | 95.13 +/- 0.02 | +0.02 | - |

lambda chosen on validation: 0; 8-bit layers: layer4.0.conv1.weight, layer4.0.conv2.weight, layer4.0.downsample.0.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## fp8, target <= 14 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 13.99 | 13.99 | 3 | 95.44 | 95.15 | +0.04 | 27.44 |
| Fisher only (lambda=0) | 0 | 13.99 | 13.99 | 3 | 95.44 | 95.15 | +0.04 | 27.44 |
| error only (lambda=1) | 1 | 13.99 | 13.99 | 3 | 95.36 | 95.15 | +0.04 | 27.14 |
| product (sum F*dw^2) |  | 13.98 | 13.98 | 5 | 95.42 | 95.16 | +0.05 | 27.42 |
| worst (max damage) |  | 13.92 | 13.92 | 17 | 95.32 | 95.11 | +0.00 | 26.75 |
| random (n=5) | - | 13.97 | - | - | 95.42 +/- 0.09 | 95.18 +/- 0.03 | +0.07 | - |

lambda chosen on validation: 0; 8-bit layers: layer2.1.conv2.weight, layer3.0.conv1.weight, layer4.1.conv2.weight

## fp8, target <= 12 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0.5 | 11.98 | 11.98 | 4 | 95.46 | 95.12 | +0.01 | 24.20 |
| Fisher only (lambda=0) | 0 | 11.98 | 11.98 | 4 | 95.44 | 95.11 | +0.00 | 24.35 |
| error only (lambda=1) | 1 | 11.98 | 11.98 | 4 | 95.40 | 95.11 | +0.00 | 24.19 |
| product (sum F*dw^2) |  | 12.00 | 12.00 | 5 | 95.48 | 95.11 | +0.00 | 24.37 |
| worst (max damage) |  | 11.80 | 11.80 | 18 | 95.36 | 95.15 | +0.04 | 23.85 |
| random (n=5) | - | 11.86 | - | - | 95.41 +/- 0.02 | 95.13 +/- 0.02 | +0.02 | - |

lambda chosen on validation: 0.5; 8-bit layers: layer3.0.conv1.weight, layer3.1.conv2.weight, layer4.0.conv2.weight, layer4.1.conv1.weight

## fp8, target <= 10 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 9.99 | 9.99 | 5 | 95.44 | 95.09 | -0.02 | 22.55 |
| Fisher only (lambda=0) | 0 | 9.99 | 9.99 | 5 | 95.44 | 95.09 | -0.02 | 22.55 |
| error only (lambda=1) | 1 | 9.98 | 9.98 | 5 | 95.42 | 95.13 | +0.02 | 22.53 |
| product (sum F*dw^2) |  | 9.99 | 9.99 | 5 | 95.44 | 95.09 | -0.02 | 22.55 |
| worst (max damage) |  | 9.78 | 9.78 | 19 | 95.32 | 95.13 | +0.02 | 22.21 |
| random (n=5) | - | 9.86 | - | - | 95.38 +/- 0.06 | 95.12 +/- 0.01 | +0.01 | - |

lambda chosen on validation: 0; 8-bit layers: layer4.0.conv1.weight, layer4.0.conv2.weight, layer4.0.downsample.0.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## fp8_scaled, target <= 14 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 13.99 | 13.99 | 3 | 95.44 | 95.14 | +0.03 | 40.16 |
| Fisher only (lambda=0) | 0 | 13.99 | 13.99 | 3 | 95.44 | 95.14 | +0.03 | 40.16 |
| error only (lambda=1) | 1 | 13.99 | 13.99 | 3 | 95.44 | 95.14 | +0.03 | 39.73 |
| product (sum F*dw^2) |  | 13.98 | 13.98 | 4 | 95.48 | 95.13 | +0.02 | 40.34 |
| worst (max damage) |  | 13.92 | 13.92 | 17 | 95.42 | 95.17 | +0.06 | 32.28 |
| random (n=5) | - | 13.97 | - | - | 95.42 +/- 0.02 | 95.19 +/- 0.02 | +0.08 | - |

lambda chosen on validation: 0; 8-bit layers: layer2.1.conv2.weight, layer3.0.conv1.weight, layer4.1.conv2.weight

## fp8_scaled, target <= 12 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0 | 11.98 | 11.98 | 4 | 95.48 | 95.13 | +0.02 | 38.43 |
| Fisher only (lambda=0) | 0 | 11.98 | 11.98 | 4 | 95.48 | 95.13 | +0.02 | 38.43 |
| error only (lambda=1) | 1 | 11.98 | 11.98 | 4 | 95.46 | 95.14 | +0.03 | 37.29 |
| product (sum F*dw^2) |  | 12.00 | 12.00 | 5 | 95.46 | 95.11 | +0.00 | 39.61 |
| worst (max damage) |  | 11.80 | 11.80 | 18 | 95.42 | 95.16 | +0.05 | 32.06 |
| random (n=5) | - | 11.86 | - | - | 95.43 +/- 0.05 | 95.16 +/- 0.02 | +0.05 | - |

lambda chosen on validation: 0; 8-bit layers: layer3.0.conv1.weight, layer3.1.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight

## fp8_scaled, target <= 10 bits/weight

| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |
|---|---|---|---|---|---|---|---|---|
| lambda (best on val) | 0.5 | 9.98 | 9.98 | 5 | 95.48 | 95.12 | +0.01 | 38.78 |
| Fisher only (lambda=0) | 0 | 9.99 | 9.99 | 5 | 95.48 | 95.11 | +0.00 | 39.33 |
| error only (lambda=1) | 1 | 9.98 | 9.98 | 5 | 95.48 | 95.12 | +0.01 | 38.78 |
| product (sum F*dw^2) |  | 9.99 | 9.99 | 5 | 95.48 | 95.11 | +0.00 | 39.33 |
| worst (max damage) |  | 9.78 | 9.78 | 19 | 95.40 | 95.18 | +0.07 | 31.64 |
| random (n=5) | - | 9.86 | - | - | 95.42 +/- 0.03 | 95.16 +/- 0.03 | +0.05 | - |

lambda chosen on validation: 0.5; 8-bit layers: layer2.1.conv2.weight, layer4.0.conv1.weight, layer4.0.conv2.weight, layer4.1.conv1.weight, layer4.1.conv2.weight
