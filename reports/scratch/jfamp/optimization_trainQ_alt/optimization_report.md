# Stage 5 optimisation (datatype-only, p = none)

5120 problems solved (3776 optimal), 250 distinct optimal assignments; full sweep in `normalization_lambda_sweep.csv`.

## Selected operating point (provisional placeholder)

B = 8.0 avg bits/weight, lambda = 1.0, normalisation minmax, risk variant weights, tau = None, eps = None.

| layer | datatype | Q | R (pp) | bytes |
|---|---|---|---|---|
| conv1 | INT8 per-tensor | 4.22e-09 | 0.42 | 1,732 |
| layer1.0.conv1 | INT8 per-tensor | 3.64e-09 | 0.02 | 36,868 |
| layer1.0.conv2 | INT8 per-tensor | 1.47e-09 | 0.03 | 36,868 |
| layer1.1.conv1 | INT8 per-tensor | 6.88e-10 | -0.00 | 36,868 |
| layer1.1.conv2 | INT8 per-tensor | 2.19e-10 | 0.01 | 36,868 |
| layer2.0.conv1 | INT8 per-tensor | 4.68e-10 | 0.01 | 73,732 |
| layer2.0.conv2 | INT8 per-tensor | 5.83e-10 | 0.00 | 147,460 |
| layer2.0.downsample.0 | INT8 per-tensor | 1.1e-09 | 0.07 | 8,196 |
| layer2.1.conv1 | INT8 per-tensor | 2.55e-10 | 0.00 | 147,460 |
| layer2.1.conv2 | INT8 per-tensor | 2.49e-10 | 0.00 | 147,460 |
| layer3.0.conv1 | INT8 per-tensor | 1.11e-09 | 0.00 | 294,916 |
| layer3.0.conv2 | INT8 per-tensor | 8.65e-10 | 0.00 | 589,828 |
| layer3.0.downsample.0 | INT8 per-tensor | 2.12e-10 | 0.02 | 32,772 |
| layer3.1.conv1 | INT8 per-tensor | 7.25e-10 | 0.00 | 589,828 |
| layer3.1.conv2 | E4M11-SN | 6.18e-14 | 0.00 | 1,179,648 |
| layer4.0.conv1 | INT8 per-tensor | 3.48e-10 | 0.00 | 1,179,652 |
| layer4.0.conv2 | INT8 per-tensor | 1.85e-10 | 0.00 | 2,359,300 |
| layer4.0.downsample.0 | FP8 E4M3FN | 3.82e-10 | -0.07 | 131,072 |
| layer4.1.conv1 | INT8 per-tensor | 6.3e-11 | 0.00 | 2,359,300 |
| layer4.1.conv2 | INT4 per-tensor | 1.66e-09 | 0.00 | 1,179,652 |
| fc | INT8 per-tensor | 5.74e-11 | 0.00 | 5,124 |

Totals (layer-additive prediction): storage 10,574,604 bytes (7.58 bits/weight), predicted clean CE increase 0.00000, mean layer risk 0.02 pp. Not measured end to end yet.
