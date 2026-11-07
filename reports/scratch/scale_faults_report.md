# Scale-metadata faults: all-INT8 per-channel vs per-tensor

* images: `val` split (2952 images); scales are float32 words; see `fr/faults_mixed.py` for the fault model

## Storage of the scale metadata

| layout | scale words | scale bits | payload bits (weights) | scale bits / payload | extra bits per weight |
|---|---|---|---|---|---|
| all_int8_pc | 4810 | 153920 | 89314816 | 0.1723 % | 0.01379 |
| all_int8_pt | 21 | 672 | 89314816 | 0.0008 % | 0.00006 |

## Realistic load: scales fault-free (off) vs sharing the memory (on)

Bit-uniform sampling (`bits`): each strike lands on a uniformly chosen bit of the whole buffer. Columns: mean / median / p10 / min test accuracy %, number of non-finite trials; `scale hits` = mean number of strikes per trial that landed in scale words.

| layout | group | k | mode | mean | median | p10 | min | non-finite | scale hits/trial |
|---|---|---|---|---|---|---|---|---|---|
| all_int8_pc | any | 256 | off | 95.17 | 95.19 | 95.12 | 95.09 | 0 | 0.00 |
| all_int8_pc | any | 256 | on | 94.26 | 95.19 | 95.12 | 12.03 | 0 | 0.46 |
| all_int8_pc | any | 1024 | off | 95.18 | 95.19 | 95.12 | 94.75 | 0 | 0.00 |
| all_int8_pc | any | 1024 | on | 90.21 | 95.16 | 94.60 | 9.45 | 3 | 1.72 |
| all_int8_pc | any | 4096 | off | 95.13 | 95.12 | 95.05 | 94.92 | 0 | 0.00 |
| all_int8_pc | any | 4096 | on | 71.57 | 94.99 | 10.23 | 6.67 | 12 | 6.86 |
| all_int8_pt | any | 256 | off | 95.12 | 95.12 | 95.05 | 94.99 | 0 | 0.00 |
| all_int8_pt | any | 256 | on | 95.12 | 95.12 | 95.05 | 94.99 | 0 | 0.00 |
| all_int8_pt | any | 1024 | off | 95.11 | 95.11 | 95.02 | 94.78 | 0 | 0.00 |
| all_int8_pt | any | 1024 | on | 95.11 | 95.11 | 95.02 | 94.78 | 0 | 0.03 |
| all_int8_pt | any | 4096 | off | 95.06 | 95.09 | 94.88 | 94.75 | 0 | 0.00 |
| all_int8_pt | any | 4096 | on | 93.83 | 95.07 | 94.85 | 8.67 | 1 | 0.06 |
| e4m11 | any | 256 | off | 92.79 | 94.14 | 87.62 | 79.27 | 0 | 0.00 |
| e4m11 | any | 1024 | off | 82.33 | 86.64 | 60.68 | 35.74 | 0 | 0.00 |
| e4m11 | any | 4096 | off | 27.97 | 24.51 | 11.64 | 7.96 | 0 | 0.00 |

Paired difference (on minus off, same trial index and same weight strikes where kept); `hit trials` = trials with at least one scale strike:

| layout | group | k | trials | mean diff (pp) | s.e. | hit trials | mean diff on hit trials (pp) | worst diff (pp) |
|---|---|---|---|---|---|---|---|---|
| all_int8_pc | any | 256 | 100 | -0.914 | 0.832 | 41 | -2.230 | -83.13 |
| all_int8_pc | any | 1024 | 100 | -4.968 | 1.803 | 87 | -5.711 | -85.70 |
| all_int8_pc | any | 4096 | 100 | -23.559 | 3.598 | 100 | -23.559 | -88.45 |
| all_int8_pt | any | 256 | 100 | +0.000 | 0.000 | 0 | +nan | +0.00 |
| all_int8_pt | any | 1024 | 100 | +0.000 | 0.000 | 3 | +0.000 | +0.00 |
| all_int8_pt | any | 4096 | 100 | -1.230 | 0.900 | 5 | -24.607 | -86.42 |

## Damage per scale hit (every strike hits a scale word)

| layout | group | k | mean | median | p10 | min | non-finite | trials with drop > 1 pp | drop > 10 pp |
|---|---|---|---|---|---|---|---|---|---|
| all_int8_pc | any | 1 | 94.20 | 95.16 | 95.08 | 60.50 | 1 | 5 % | 3 % |
| all_int8_pc | any | 8 | 71.71 | 95.00 | 11.62 | 9.55 | 14 | 38 % | 33 % |
| all_int8_pc | exponent | 1 | 81.32 | 95.16 | 18.15 | 8.74 | 8 | 26 % | 21 % |
| all_int8_pc | mantissa_lsb | 1 | 95.16 | 95.16 | 95.16 | 95.16 | 0 | 0 % | 0 % |
| all_int8_pc | mantissa_msb | 1 | 95.16 | 95.16 | 95.16 | 95.16 | 0 | 0 % | 0 % |
| all_int8_pc | sign | 1 | 95.00 | 95.16 | 94.81 | 92.58 | 0 | 4 % | 0 % |
| all_int8_pt | any | 1 | 85.40 | 95.12 | 37.40 | 2.95 | 2 | 18 % | 16 % |
| all_int8_pt | any | 8 | 34.42 | 10.76 | 9.58 | 0.10 | 24 | 82 % | 76 % |
| all_int8_pt | exponent | 1 | 50.98 | 40.04 | 9.69 | 7.01 | 11 | 86 % | 73 % |
| all_int8_pt | mantissa_lsb | 1 | 95.12 | 95.12 | 95.12 | 95.09 | 0 | 0 % | 0 % |
| all_int8_pt | mantissa_msb | 1 | 95.10 | 95.12 | 95.02 | 94.00 | 0 | 1 % | 0 % |
| all_int8_pt | sign | 1 | 17.13 | 9.86 | 0.06 | 0.00 | 0 | 96 % | 95 % |

## Weight bit groups (scales fault-free)

| layout | group | k | mean | median | p10 | min | non-finite |
|---|---|---|---|---|---|---|---|
| all_int8_pc | mantissa_lsb | 1024 | 95.16 | 95.16 | 95.16 | 95.16 | 0 |
| all_int8_pc | mantissa_msb | 1024 | 95.16 | 95.16 | 95.12 | 95.09 | 0 |
| all_int8_pc | sign | 1024 | 95.11 | 95.11 | 94.95 | 94.88 | 0 |
| all_int8_pt | mantissa_lsb | 1024 | 95.12 | 95.12 | 95.09 | 95.09 | 0 |
| all_int8_pt | mantissa_msb | 1024 | 95.11 | 95.09 | 95.05 | 95.02 | 0 |
| all_int8_pt | sign | 1024 | 95.02 | 95.05 | 94.82 | 94.68 | 0 |
| e4m11 | exponent | 1024 | 27.16 | 25.49 | 11.99 | 9.86 | 0 |
| e4m11 | mantissa_lsb | 1024 | 95.29 | 95.29 | 95.29 | 95.29 | 0 |
| e4m11 | mantissa_msb | 1024 | 95.28 | 95.29 | 95.26 | 95.26 | 0 |
| e4m11 | sign | 1024 | 95.16 | 95.19 | 95.05 | 94.99 | 0 |
