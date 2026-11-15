# Stage 6 optimisation (part 1, predicted)

1440 problems solved (1440 optimal), 201 distinct optimal assignments; full sweep in `normalization_lambda_sweep.csv`.

Declared operating point: B = 8.0 bits/weight (2,202,560 bytes, quantized weights only), lambda = 1.0, min-max, tau_fault = 0.5 pp (predicted), tau_scale = 1.0 pp, delta_clean = 0.5 pp (measured end to end).

| layer | datatype | protection | Q (CE) | R exposure (pp) | scale-word risk (pp) | bytes |
|---|---|---|---|---|---|---|
| features.0.0 | E4M11-SN | secded | 2.42e-08 | 0.000 | 0.00 | 2,106 |
| features.1.conv.0.0 | E4M11-SN | secded | 5.21e-08 | 0.000 | 0.00 | 702 |
| features.1.conv.1 | E4M11-SN | secded | 2.78e-08 | 0.000 | 0.00 | 1,248 |
| features.2.conv.0.0 | E4M11-SN | secded | 7.38e-09 | 0.000 | 0.00 | 3,744 |
| features.2.conv.1.0 | E4M11-SN | secded | 4.99e-08 | 0.000 | 0.00 | 2,106 |
| features.2.conv.2 | E4M11-SN | secded | 3.04e-08 | 0.000 | 0.00 | 5,616 |
| features.3.conv.0.0 | E4M11-SN | secded | 3.03e-09 | 0.000 | 0.00 | 8,424 |
| features.3.conv.1.0 | E4M11-SN | secded | 2.11e-08 | 0.000 | 0.00 | 3,159 |
| features.3.conv.2 | E4M11-SN | secded | 7.41e-09 | 0.000 | 0.00 | 8,424 |
| features.4.conv.0.0 | E4M11-SN | secded | 1.49e-08 | 0.000 | 0.00 | 8,424 |
| features.4.conv.1.0 | E4M11-SN | secded | 5.88e-09 | 0.000 | 0.00 | 3,159 |
| features.4.conv.2 | E4M11-SN | secded | 2.5e-08 | 0.000 | 0.00 | 11,232 |
| features.5.conv.0.0 | E4M11-SN | secded | 1.96e-09 | 0.000 | 0.00 | 14,976 |
| features.5.conv.1.0 | E4M11-SN | secded | 1.04e-08 | 0.000 | 0.00 | 4,212 |
| features.5.conv.2 | E4M11-SN | secded | 3.13e-09 | 0.000 | 0.00 | 14,976 |
| features.6.conv.0.0 | E4M11-SN | secded | 1.99e-09 | 0.000 | 0.00 | 14,976 |
| features.6.conv.1.0 | E4M11-SN | secded | 8.6e-09 | 0.000 | 0.00 | 4,212 |
| features.6.conv.2 | E4M11-SN | secded | 2.91e-09 | 0.000 | 0.00 | 14,976 |
| features.7.conv.0.0 | E4M11-SN | secded | 1.24e-08 | 0.000 | 0.00 | 14,976 |
| features.7.conv.1.0 | E4M11-SN | secded | 4.77e-09 | 0.000 | 0.00 | 4,212 |
| features.7.conv.2 | E4M11-SN | secded | 9.2e-09 | 0.000 | 0.00 | 29,952 |
| features.8.conv.0.0 | INT8 per-tensor | scale_tmr | 6.44e-06 | 0.000 | 0.00 | 24,588 |
| features.8.conv.1.0 | E4M11-SN | secded | 2.32e-09 | 0.000 | 0.00 | 8,424 |
| features.8.conv.2 | INT8 per-tensor | scale_tmr | 1.03e-05 | 0.000 | 0.00 | 24,588 |
| features.9.conv.0.0 | INT8 per-tensor | scale_tmr | 6.28e-06 | 0.000 | 0.00 | 24,588 |
| features.9.conv.1.0 | E4M11-SN | secded | 1.96e-09 | 0.000 | 0.00 | 8,424 |
| features.9.conv.2 | INT8 per-tensor | scale_tmr | 6.36e-06 | 0.000 | 0.00 | 24,588 |
| features.10.conv.0.0 | INT8 per-tensor | scale_tmr | 5.21e-06 | 0.000 | 0.00 | 24,588 |
| features.10.conv.1.0 | E4M11-SN | secded | 8.87e-10 | 0.000 | 0.00 | 8,424 |
| features.10.conv.2 | INT8 per-tensor | scale_tmr | 4.86e-06 | 0.000 | 0.00 | 24,588 |
| features.11.conv.0.0 | INT8 per-tensor | scale_tmr | 2.26e-05 | 0.000 | 0.00 | 24,588 |
| features.11.conv.1.0 | E4M11-SN | secded | 9.38e-09 | 0.000 | 0.00 | 8,424 |
| features.11.conv.2 | E4M11-SN | secded | 3.01e-09 | 0.000 | 0.00 | 89,856 |
| features.12.conv.0.0 | INT8 per-tensor | scale_tmr | 4.92e-06 | 0.000 | 0.00 | 55,308 |
| features.12.conv.1.0 | E4M11-SN | secded | 1.8e-09 | 0.000 | 0.00 | 12,636 |
| features.12.conv.2 | INT8 per-tensor | scale_tmr | 7.7e-06 | 0.000 | 0.00 | 55,308 |
| features.13.conv.0.0 | INT8 per-tensor | scale_tmr | 5.65e-06 | 0.000 | 0.00 | 55,308 |
| features.13.conv.1.0 | E4M11-SN | secded | 1.17e-09 | 0.000 | 0.00 | 12,636 |
| features.13.conv.2 | INT8 per-tensor | scale_tmr | 4.86e-06 | 0.000 | 0.00 | 55,308 |
| features.14.conv.0.0 | INT8 per-tensor | scale_tmr | 1.39e-05 | 0.000 | 0.00 | 55,308 |
| features.14.conv.1.0 | E4M11-SN | secded | 1.56e-09 | 0.000 | 0.00 | 12,636 |
| features.14.conv.2 | INT8 per-tensor | scale_tmr | 1.24e-05 | 0.000 | 0.00 | 92,172 |
| features.15.conv.0.0 | INT8 per-tensor | scale_tmr | 5.25e-07 | 0.000 | 0.00 | 153,612 |
| features.15.conv.1.0 | INT8 per-tensor | scale_tmr | 7.71e-07 | 0.000 | 0.00 | 8,652 |
| features.15.conv.2 | INT8 per-tensor | scale_tmr | 8.45e-07 | 0.000 | 0.00 | 153,612 |
| features.16.conv.0.0 | INT8 per-tensor | scale_tmr | 2.52e-07 | 0.000 | 0.00 | 153,612 |
| features.16.conv.1.0 | INT8 per-tensor | scale_tmr | 2.14e-07 | 0.000 | 0.00 | 8,652 |
| features.16.conv.2 | INT8 per-tensor | scale_tmr | 2.29e-07 | 0.000 | 0.00 | 153,612 |
| features.17.conv.0.0 | INT8 per-tensor | scale_tmr | 2.77e-06 | 0.000 | 0.00 | 153,612 |
| features.17.conv.1.0 | INT8 per-tensor | scale_tmr | 9.65e-07 | 0.000 | 0.00 | 8,652 |
| features.17.conv.2 | INT8 per-tensor | scale_tmr | 7.31e-07 | 0.000 | 0.00 | 307,212 |
| features.18.0 | INT4 per-tensor | scale_tmr | 1.16e-05 | 0.000 | 0.00 | 204,812 |
| classifier.1 | INT8 per-tensor | scale_tmr | 3.01e-07 | 0.000 | 0.00 | 12,812 |

Totals (layer-additive prediction): 2,196,952 bytes (7.980 bits/weight), predicted CE increase 0.0001, predicted weight-fault drop 0.000 pp, largest scale-word risk 0.00 pp.
