# Stage 5 optimisation

1280 problems solved (1280 optimal), 191 distinct optimal assignments; full sweep in `normalization_lambda_sweep.csv`.

## Predicted selection at the agreed operating point (joint)

B = 8.0 avg bits/weight, lambda = 1.0, normalisation minmax, tau_fault = 0.5 pp (predicted), tau_scale = 1.0.

| layer | datatype | protection | Q | R exposure (pp) | scale-word risk (pp) | bytes |
|---|---|---|---|---|---|---|
| conv1 | E4M11-SN | none | 1e-08 | 0.001 | 0.00 | 3,456 |
| layer1.0.conv1 | INT8 per-tensor | scale_tmr | 7.49e-05 | 0.000 | 0.00 | 36,876 |
| layer1.0.conv2 | INT8 per-tensor | scale_tmr | 3.05e-05 | 0.000 | 0.00 | 36,876 |
| layer1.1.conv1 | INT8 per-tensor | scale_tmr | 1.8e-05 | 0.000 | 0.00 | 36,876 |
| layer1.1.conv2 | INT8 per-tensor | scale_tmr | 5.14e-06 | 0.000 | 0.00 | 36,876 |
| layer2.0.conv1 | E4M11-SN | none | 1.15e-09 | 0.000 | 0.00 | 147,456 |
| layer2.0.conv2 | INT8 per-tensor | scale_tmr | 1.56e-05 | 0.000 | 0.00 | 147,468 |
| layer2.0.downsample.0 | INT8 per-tensor | scale_tmr | 3.01e-05 | 0.000 | 0.00 | 8,204 |
| layer2.1.conv1 | INT8 per-tensor | scale_tmr | 6.04e-06 | 0.000 | 0.00 | 147,468 |
| layer2.1.conv2 | INT8 per-tensor | scale_tmr | 5.76e-06 | 0.000 | 0.00 | 147,468 |
| layer3.0.conv1 | INT8 per-tensor | scale_tmr | 2.61e-05 | 0.000 | 0.00 | 294,924 |
| layer3.0.conv2 | INT8 per-tensor | scale_tmr | 1.75e-05 | 0.000 | 0.00 | 589,836 |
| layer3.0.downsample.0 | INT8 per-tensor | scale_tmr | 4.62e-06 | 0.000 | 0.00 | 32,780 |
| layer3.1.conv1 | INT8 per-tensor | scale_tmr | 1.37e-05 | 0.000 | 0.00 | 589,836 |
| layer3.1.conv2 | E4M11-SN | none | 9.58e-10 | 0.000 | 0.00 | 1,179,648 |
| layer4.0.conv1 | INT8 per-tensor | scale_tmr | 4.54e-06 | 0.000 | 0.00 | 1,179,660 |
| layer4.0.conv2 | INT8 per-tensor | scale_tmr | 1.64e-06 | 0.000 | 0.00 | 2,359,308 |
| layer4.0.downsample.0 | INT8 per-tensor | scale_tmr | 2.43e-07 | 0.000 | 0.00 | 131,084 |
| layer4.1.conv1 | INT8 per-tensor | scale_tmr | 5.57e-07 | 0.000 | 0.00 | 2,359,308 |
| layer4.1.conv2 | INT4 per-tensor | scale_tmr | 1.41e-05 | 0.000 | 0.00 | 1,179,660 |
| fc | INT8 per-tensor | scale_tmr | 3.67e-07 | 0.000 | 0.00 | 5,132 |

Totals (layer-additive prediction): storage 10,650,200 bytes (7.63 bits/weight), predicted weight-fault drop 0.002 pp, largest scale-word risk 0.00 pp. Not measured end to end yet.
