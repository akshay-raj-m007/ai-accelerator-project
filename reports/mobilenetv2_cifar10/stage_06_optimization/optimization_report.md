# Stage 6 optimisation (part 1, predicted)

1440 problems solved (1440 optimal), 162 distinct optimal assignments; full sweep in `normalization_lambda_sweep.csv`.

Declared operating point: B = 8.0 bits/weight (2,202,560 bytes, quantized weights only), lambda = 1.0, min-max, tau_fault = 0.5 pp (predicted), tau_scale = 1.0 pp, delta_clean = 0.5 pp (measured end to end).

| layer | datatype | protection | Q (CE) | R exposure (pp) | scale-word risk (pp) | bytes |
|---|---|---|---|---|---|---|
| features.0.0 | INT8 per-tensor | secded | 0 | 0.000 | 0.00 | 1,058 |
| features.1.conv.0.0 | E4M11-SN | secded | 4.17e-07 | 0.000 | 0.00 | 702 |
| features.1.conv.1 | INT8 per-tensor | secded | 0 | 0.000 | 0.00 | 629 |
| features.2.conv.0.0 | FP16 | secded | 0 | 0.000 | 0.00 | 3,744 |
| features.2.conv.1.0 | E4M11-SN | secded | 9.36e-06 | 0.000 | 0.00 | 2,106 |
| features.2.conv.2 | INT8 per-tensor | secded | 0 | 0.000 | 0.00 | 2,813 |
| features.3.conv.0.0 | fp8_e5m2 | secded | 0 | 0.000 | 0.00 | 4,212 |
| features.3.conv.1.0 | INT8 per-tensor | secded | 0 | 0.000 | 0.00 | 1,585 |
| features.3.conv.2 | fp8_e5m2 | secded | 0 | 0.000 | 0.00 | 4,212 |
| features.4.conv.0.0 | E4M11-SN | secded | 1.53e-05 | 0.000 | 0.00 | 8,424 |
| features.4.conv.1.0 | E4M11-SN | secded | 7.21e-06 | 0.000 | 0.00 | 3,159 |
| features.4.conv.2 | FP8 E4M3FN | secded | 0 | 0.000 | 0.00 | 5,616 |
| features.5.conv.0.0 | INT4 per-tensor | secded | 0 | 0.000 | 0.00 | 3,749 |
| features.5.conv.1.0 | INT8 per-tensor | secded | 0 | 0.000 | 0.00 | 2,111 |
| features.5.conv.2 | fp8_e5m2 | secded | 0 | 0.000 | 0.00 | 7,488 |
| features.6.conv.0.0 | INT8 per-tensor | scale_tmr | 0 | 0.000 | 0.00 | 6,156 |
| features.6.conv.1.0 | INT8 per-tensor | scale_tmr | 0 | 0.000 | 0.00 | 1,740 |
| features.6.conv.2 | FP8 E4M3FN | secded | 0 | 0.000 | 0.00 | 7,488 |
| features.7.conv.0.0 | INT8 per-tensor | scale_tmr | 0 | 0.000 | 0.00 | 6,156 |
| features.7.conv.1.0 | fp8_e5m2 | secded | 0 | 0.000 | 0.00 | 2,106 |
| features.7.conv.2 | INT8 per-tensor | scale_tmr | 0 | 0.000 | 0.00 | 12,300 |
| features.8.conv.0.0 | INT4 per-tensor | secded | 0 | 0.000 | 0.00 | 14,981 |
| features.8.conv.1.0 | INT4 per-tensor | secded | 0 | 0.000 | 0.00 | 2,111 |
| features.8.conv.2 | INT4 per-tensor | secded | 0 | 0.000 | 0.00 | 14,981 |
| features.9.conv.0.0 | fp8_e5m2 | secded | 0 | 0.000 | 0.00 | 29,952 |
| features.9.conv.1.0 | FP8 E4M3FN | secded | 0 | 0.000 | 0.00 | 4,212 |
| features.9.conv.2 | INT8 per-tensor | scale_tmr | 0 | 0.000 | 0.00 | 24,588 |
| features.10.conv.0.0 | INT8 per-tensor | scale_tmr | 0 | 0.000 | 0.00 | 24,588 |
| features.10.conv.1.0 | INT8 per-tensor | scale_tmr | 0 | 0.000 | 0.00 | 3,468 |
| features.10.conv.2 | fp8_e5m2 | secded | 0 | 0.000 | 0.00 | 29,952 |
| features.11.conv.0.0 | FP8 E4M3FN | secded | 0 | 0.000 | 0.00 | 29,952 |
| features.11.conv.1.0 | fp8_e5m2 | secded | 0 | 0.000 | 0.00 | 4,212 |
| features.11.conv.2 | FP8 E4M3FN | secded | 0 | 0.000 | 0.00 | 44,928 |
| features.12.conv.0.0 | INT8 per-tensor | scale_tmr | 0 | 0.000 | 0.00 | 55,308 |
| features.12.conv.1.0 | fp8_e5m2 | secded | 0 | 0.000 | 0.00 | 6,318 |
| features.12.conv.2 | INT8 per-tensor | scale_tmr | 0 | 0.000 | 0.00 | 55,308 |
| features.13.conv.0.0 | INT8 per-tensor | scale_tmr | 0 | 0.000 | 0.00 | 55,308 |
| features.13.conv.1.0 | INT4 per-tensor | secded | 0 | 0.000 | 0.00 | 3,164 |
| features.13.conv.2 | INT4 per-tensor | secded | 0 | 0.000 | 0.00 | 33,701 |
| features.14.conv.0.0 | INT8 per-tensor | scale_tmr | 0 | 0.000 | 0.00 | 55,308 |
| features.14.conv.1.0 | INT4 per-tensor | secded | 0 | 0.000 | 0.00 | 3,164 |
| features.14.conv.2 | FP8 E4M3FN | secded | 0 | 0.000 | 0.00 | 112,320 |
| features.15.conv.0.0 | INT4 per-tensor | scale_tmr | 0 | 0.000 | 0.00 | 76,812 |
| features.15.conv.1.0 | INT4 per-tensor | secded | 0 | 0.000 | 0.00 | 5,270 |
| features.15.conv.2 | INT8 per-tensor | scale_tmr | 0 | 0.000 | 0.00 | 153,612 |
| features.16.conv.0.0 | INT4 per-tensor | scale_tmr | 0 | 0.000 | 0.00 | 76,812 |
| features.16.conv.1.0 | INT4 per-tensor | scale_tmr | 0 | 0.000 | 0.00 | 4,332 |
| features.16.conv.2 | INT8 per-tensor | scale_tmr | 0 | 0.000 | 0.00 | 153,612 |
| features.17.conv.0.0 | INT4 per-tensor | scale_tmr | 0 | 0.000 | 0.00 | 76,812 |
| features.17.conv.1.0 | INT4 per-tensor | scale_tmr | 0 | 0.000 | 0.00 | 4,332 |
| features.17.conv.2 | INT4 per-tensor | scale_tmr | 0 | 0.000 | 0.00 | 153,612 |
| features.18.0 | INT4 per-tensor | scale_tmr | 0 | 0.000 | 0.00 | 204,812 |
| classifier.1 | INT4 per-tensor | scale_tmr | 0 | 0.000 | 0.00 | 6,412 |

Totals (layer-additive prediction): 1,611,808 bytes (5.854 bits/weight), predicted CE increase 0.0000, predicted weight-fault drop 0.000 pp, largest scale-word risk 0.00 pp.
