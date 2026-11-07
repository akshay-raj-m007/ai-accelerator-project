# Fisher stability

Layer metric = Fisher trace per parameter (the quantity the lambda-sum and the Fisher-only rule rank).
Spearman = rank correlation of the 21 layer values. Jaccard is counted in weights.

## Source: train (4096 images, mean NLL 0.0010, 0 misclassified)

### Ranking stability vs sample size

| n | chunks | Spearman between chunks (mean / min) | Spearman vs full pool | mean CV per layer |
|---|---|---|---|---|
| 64 | 64 | 0.944 / 0.179 | 0.909 | 5.081 |
| 128 | 32 | 0.925 / 0.171 | 0.891 | 3.556 |
| 256 | 16 | 0.881 / 0.179 | 0.876 | 2.559 |
| 512 | 8 | 0.804 / 0.270 | 0.869 | 1.734 |
| 1024 | 4 | 0.706 / 0.340 | 0.864 | 1.052 |
| 2048 | 2 | 0.927 / 0.927 | 0.977 | 0.386 |
| 4096 | 1 | - / - | 1.000 | - |

### Concentration

* effective sample size per layer (of 4096): min 2.0, median 2.4, max 3.1
* share of a layer's trace from the top 1% of images: min 0.99, median 1.00, max 1.00
* all layers: top 1% of images carry 1.00, top 10% carry 1.00 of the total trace
* misclassified images (0) carry 0.00 of the total trace; Spearman(per-image trace, per-image NLL) = 0.958
* per-parameter vs total trace ranking: Spearman -0.766

### Fisher-only selection stability

| target bits | chunk n | chunks | Jaccard vs full (mean / min) | Jaccard between chunks | chunks identical to full |
|---|---|---|---|---|---|
| 14 | 1024 | 4 | 1.000 / 1.000 | 1.000 | 4 |
| 12 | 1024 | 4 | 0.611 / 0.310 | 0.480 | 1 |
| 10 | 1024 | 4 | 0.922 / 0.754 | 0.858 | 1 |

## Source: val (4096 images, mean NLL 0.1747, 191 misclassified)

### Ranking stability vs sample size

| n | chunks | Spearman between chunks (mean / min) | Spearman vs full pool | mean CV per layer |
|---|---|---|---|---|
| 64 | 64 | 0.986 / 0.918 | 0.991 | 0.767 |
| 128 | 32 | 0.990 / 0.971 | 0.994 | 0.481 |
| 256 | 16 | 0.993 / 0.984 | 0.995 | 0.348 |
| 512 | 8 | 0.995 / 0.990 | 0.997 | 0.203 |
| 1024 | 4 | 0.995 / 0.991 | 0.997 | 0.155 |
| 2048 | 2 | 0.997 / 0.997 | 0.999 | 0.090 |
| 4096 | 1 | - / - | 1.000 | - |

### Concentration

* effective sample size per layer (of 4096): min 59.4, median 101.0, max 114.7
* share of a layer's trace from the top 1% of images: min 0.50, median 0.54, max 0.63
* all layers: top 1% of images carry 0.54, top 10% carry 1.00 of the total trace
* misclassified images (191) carry 0.40 of the total trace; Spearman(per-image trace, per-image NLL) = 0.977
* per-parameter vs total trace ranking: Spearman 0.200

### Fisher-only selection stability

| target bits | chunk n | chunks | Jaccard vs full (mean / min) | Jaccard between chunks | chunks identical to full |
|---|---|---|---|---|---|
| 14 | 1024 | 4 | 1.000 / 1.000 | 1.000 | 4 |
| 12 | 1024 | 4 | 0.976 / 0.903 | 0.951 | 3 |
| 10 | 1024 | 4 | 1.000 / 1.000 | 1.000 | 4 |

## Training images vs held-out images

* Spearman of layer rankings: per parameter 0.862, total 0.518
* held-out / training trace ratio per layer: median 480.1, min 15.1, max 609.3

| target bits | Jaccard of Fisher-only selections |
|---|---|
| 14 | 1.000 |
| 12 | 0.407 |
| 10 | 0.967 |
