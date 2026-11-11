# Stage 5 optimisation (datatype-only, p = none)

1280 problems solved (944 optimal), 251 distinct optimal assignments; full sweep in `normalization_lambda_sweep.csv`.

## Selected operating point (provisional placeholder)

B = 8.0 avg bits/weight, lambda = 1.0, normalisation minmax, risk variant weights, tau = None, eps = None.

| layer | datatype | Q | R (pp) | bytes |
|---|---|---|---|---|
| conv1 | INT8 per-tensor | 8.02e-05 | 0.42 | 1,732 |
| layer1.0.conv1 | INT8 per-tensor | 7.49e-05 | 0.02 | 36,868 |
| layer1.0.conv2 | INT8 per-tensor | 3.05e-05 | 0.03 | 36,868 |
| layer1.1.conv1 | INT8 per-tensor | 1.8e-05 | -0.00 | 36,868 |
| layer1.1.conv2 | INT8 per-tensor | 5.14e-06 | 0.01 | 36,868 |
| layer2.0.conv1 | INT8 per-tensor | 1.14e-05 | 0.01 | 73,732 |
| layer2.0.conv2 | INT8 per-tensor | 1.56e-05 | 0.00 | 147,460 |
| layer2.0.downsample.0 | INT8 per-tensor | 3.01e-05 | 0.07 | 8,196 |
| layer2.1.conv1 | INT8 per-tensor | 6.04e-06 | 0.00 | 147,460 |
| layer2.1.conv2 | INT8 per-tensor | 5.76e-06 | 0.00 | 147,460 |
| layer3.0.conv1 | INT8 per-tensor | 2.61e-05 | 0.00 | 294,916 |
| layer3.0.conv2 | INT8 per-tensor | 1.75e-05 | 0.00 | 589,828 |
| layer3.0.downsample.0 | INT8 per-tensor | 4.62e-06 | 0.02 | 32,772 |
| layer3.1.conv1 | INT8 per-tensor | 1.37e-05 | 0.00 | 589,828 |
| layer3.1.conv2 | E4M11-SN | 9.58e-10 | 0.00 | 1,179,648 |
| layer4.0.conv1 | INT8 per-tensor | 4.54e-06 | 0.00 | 1,179,652 |
| layer4.0.conv2 | INT8 per-tensor | 1.64e-06 | 0.00 | 2,359,300 |
| layer4.0.downsample.0 | FP8 E4M3FN | 3.98e-06 | -0.07 | 131,072 |
| layer4.1.conv1 | INT8 per-tensor | 5.57e-07 | 0.00 | 2,359,300 |
| layer4.1.conv2 | INT4 per-tensor | 1.41e-05 | 0.00 | 1,179,652 |
| fc | INT8 per-tensor | 3.67e-07 | 0.00 | 5,124 |

Totals (layer-additive prediction): storage 10,574,604 bytes (7.58 bits/weight), predicted clean CE increase 0.00018, mean layer risk 0.02 pp. Not measured end to end yet.
