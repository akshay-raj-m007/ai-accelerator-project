# Step 1 report: scratch

* checkpoint: `/home/ssl25/fr_runs/checkpoints/scratch/best.pth`
* normalization: `{'mean': [0.4914, 0.4822, 0.4465], 'std': [0.247, 0.2435, 0.2616]}`
* test images used: 10000 (full test set)
* device: cuda, torch 2.5.1+cu121, git 239c285016

## FP32

| metric | value |
|---|---|
| accuracy (%) | 95.11 |
| correct/total | 9511/10000 |
| mean cross-entropy | 0.1855 |

Weight statistics (paper Table 3 style):

| tensors | count | min | max | abs max | mean | std |
|---|---|---|---|---|---|---|
| convlinear | 11164352 | -0.411 | 0.556 | 0.556 | -0.00046 | 0.00644 |
| all_parameters | 11173962 | -0.411 | 0.556 | 0.556 | -0.00043 | 0.00705 |

## E4M11, mode=`paper`, scope=`convlinear`, all-ones decode=`clamp`

| metric | FP32 | E4M11 |
|---|---|---|
| accuracy (%) | 95.11 | 28.44 |
| mean cross-entropy | 0.1855 | 1.8489 |

* accuracy drop: **+66.67 pp**
* SQNR (Eq. 6, all 21 Conv+Linear weight tensors, 11,164,352 weights): **6.6410 dB**

SQNR by |w| bin (original weight magnitude):

| bin | count | % weights | % signal power | % error power | SQNR (dB) |
|---|---|---|---|---|---|
| abs_lt_2^-7 (flushed) | 9,736,012 | 87.21 | 15.685 | 72.373 | 0.00 |
| 2^-7<=abs<2^-6 | 1,022,240 | 9.16 | 27.017 | 27.627 | 6.54 |
| abs>=2^-6 | 406,100 | 3.64 | 57.298 | 0.000 | 79.42 |

Per-layer SQNR:

| layer | weights | % abs(w)<2^-6 | abs max | std | SQNR (dB) |
|---|---|---|---|---|---|
| conv1.weight | 1,728 | 67.7 | 0.556 | 0.06743 | 33.68 |
| layer1.0.conv1.weight | 36,864 | 88.0 | 0.306 | 0.01439 | 16.05 |
| layer1.0.conv2.weight | 36,864 | 80.4 | 0.207 | 0.01580 | 13.04 |
| layer1.1.conv1.weight | 36,864 | 81.7 | 0.155 | 0.01534 | 12.71 |
| layer1.1.conv2.weight | 36,864 | 77.5 | 0.102 | 0.01572 | 12.38 |
| layer2.0.conv1.weight | 73,728 | 72.9 | 0.109 | 0.01495 | 11.79 |
| layer2.0.conv2.weight | 147,456 | 76.8 | 0.115 | 0.01353 | 10.67 |
| layer2.0.downsample.0.weight | 8,192 | 53.1 | 0.289 | 0.02672 | 18.05 |
| layer2.1.conv1.weight | 147,456 | 84.0 | 0.085 | 0.01155 | 8.89 |
| layer2.1.conv2.weight | 147,456 | 88.2 | 0.103 | 0.01026 | 7.83 |
| layer3.0.conv1.weight | 294,912 | 81.8 | 0.131 | 0.01216 | 9.39 |
| layer3.0.conv2.weight | 589,824 | 84.8 | 0.088 | 0.01115 | 8.48 |
| layer3.0.downsample.0.weight | 32,768 | 71.6 | 0.142 | 0.01540 | 12.04 |
| layer3.1.conv1.weight | 589,824 | 87.5 | 0.084 | 0.01040 | 7.77 |
| layer3.1.conv2.weight | 589,824 | 94.9 | 0.083 | 0.00774 | 5.34 |
| layer4.0.conv1.weight | 1,179,648 | 98.9 | 0.078 | 0.00556 | 2.67 |
| layer4.0.conv2.weight | 2,359,296 | 99.9 | 0.059 | 0.00291 | 0.72 |
| layer4.0.downsample.0.weight | 131,072 | 97.7 | 0.062 | 0.00638 | 3.74 |
| layer4.1.conv1.weight | 2,359,296 | 100.0 | 0.040 | 0.00211 | 0.52 |
| layer4.1.conv2.weight | 2,359,296 | 100.0 | 0.020 | 0.00136 | 0.05 |
| fc.weight | 5,120 | 3.4 | 0.307 | 0.08120 | 40.53 |

Comparison with paper targets (`configs/paper_targets.yaml`; reported, not tuned):

| item | ours | paper | status | source |
|---|---|---|---|---|
| FP32 accuracy (%) | 95.1100 | 96.5500 | delta -1.44 pp | Sec 5.1, p.4: '... FP32 baselines on CIFAR-10 (96.55%, 94.0%, 92.58% respectively)' (ResNet-18 first) |
| E4M11 accuracy (%) | 28.4400 | n/a | no paper value (figure only) | Only shown as a bar in Fig. 2 (p.4); no ResNet-18/E4M11 number in the text |
| E4M11 accuracy drop (pp) | 66.6700 | 0.1-0.2 (all E>=4 formats) | above range | Sec 5.1, p.4: 'all formats with E>=4 remain within 0.1-0.2% of the ... FP32 baselines' |
| E4M11 SQNR, Conv+Linear weights (dB) | 6.6410 | 78.7-79.8 (all 9 model-dataset pairs); R18-C10 = 78.7 | below range by 72.06 dB | Sec 5.2, p.5: 'E4M11 ... across all nine model-dataset combinations (78.7-79.8 dB)'; Fig. 5 row R18-C10 |
| w_min (Table 3), Conv+Linear weights | -0.4105 | -0.8460 | delta +0.435 | Appendix A.2 Table 3 (p.8), ResNet-18 / CIFAR-10; Table 3 does not state which tensors (Conv/Linear only or all parameters) are included |
| w_min (Table 3), all parameters | -0.4105 | -0.8460 | delta +0.435 | Appendix A.2 Table 3 (p.8), ResNet-18 / CIFAR-10; Table 3 does not state which tensors (Conv/Linear only or all parameters) are included |
| w_max (Table 3), Conv+Linear weights | 0.5556 | 2.2610 | delta -1.705 | Appendix A.2 Table 3 (p.8), ResNet-18 / CIFAR-10; Table 3 does not state which tensors (Conv/Linear only or all parameters) are included |
| w_max (Table 3), all parameters | 0.5556 | 2.2610 | delta -1.705 | Appendix A.2 Table 3 (p.8), ResNet-18 / CIFAR-10; Table 3 does not state which tensors (Conv/Linear only or all parameters) are included |

## E4M11, mode=`legacy`, scope=`convlinear`, all-ones decode=`clamp`

| metric | FP32 | E4M11 |
|---|---|---|
| accuracy (%) | 95.11 | 28.44 |
| mean cross-entropy | 0.1855 | 1.8489 |

* accuracy drop: **+66.67 pp**
* SQNR (Eq. 6, all 21 Conv+Linear weight tensors, 11,164,352 weights): **6.6410 dB**

SQNR by |w| bin (original weight magnitude):

| bin | count | % weights | % signal power | % error power | SQNR (dB) |
|---|---|---|---|---|---|
| abs_lt_2^-7 (flushed) | 9,736,012 | 87.21 | 15.685 | 72.373 | 0.00 |
| 2^-7<=abs<2^-6 | 1,022,240 | 9.16 | 27.017 | 27.627 | 6.54 |
| abs>=2^-6 | 406,100 | 3.64 | 57.298 | 0.000 | 79.42 |

Per-layer SQNR:

| layer | weights | % abs(w)<2^-6 | abs max | std | SQNR (dB) |
|---|---|---|---|---|---|
| conv1.weight | 1,728 | 67.7 | 0.556 | 0.06743 | 33.68 |
| layer1.0.conv1.weight | 36,864 | 88.0 | 0.306 | 0.01439 | 16.05 |
| layer1.0.conv2.weight | 36,864 | 80.4 | 0.207 | 0.01580 | 13.04 |
| layer1.1.conv1.weight | 36,864 | 81.7 | 0.155 | 0.01534 | 12.71 |
| layer1.1.conv2.weight | 36,864 | 77.5 | 0.102 | 0.01572 | 12.38 |
| layer2.0.conv1.weight | 73,728 | 72.9 | 0.109 | 0.01495 | 11.79 |
| layer2.0.conv2.weight | 147,456 | 76.8 | 0.115 | 0.01353 | 10.67 |
| layer2.0.downsample.0.weight | 8,192 | 53.1 | 0.289 | 0.02672 | 18.05 |
| layer2.1.conv1.weight | 147,456 | 84.0 | 0.085 | 0.01155 | 8.89 |
| layer2.1.conv2.weight | 147,456 | 88.2 | 0.103 | 0.01026 | 7.83 |
| layer3.0.conv1.weight | 294,912 | 81.8 | 0.131 | 0.01216 | 9.39 |
| layer3.0.conv2.weight | 589,824 | 84.8 | 0.088 | 0.01115 | 8.48 |
| layer3.0.downsample.0.weight | 32,768 | 71.6 | 0.142 | 0.01540 | 12.04 |
| layer3.1.conv1.weight | 589,824 | 87.5 | 0.084 | 0.01040 | 7.77 |
| layer3.1.conv2.weight | 589,824 | 94.9 | 0.083 | 0.00774 | 5.34 |
| layer4.0.conv1.weight | 1,179,648 | 98.9 | 0.078 | 0.00556 | 2.67 |
| layer4.0.conv2.weight | 2,359,296 | 99.9 | 0.059 | 0.00291 | 0.72 |
| layer4.0.downsample.0.weight | 131,072 | 97.7 | 0.062 | 0.00638 | 3.74 |
| layer4.1.conv1.weight | 2,359,296 | 100.0 | 0.040 | 0.00211 | 0.52 |
| layer4.1.conv2.weight | 2,359,296 | 100.0 | 0.020 | 0.00136 | 0.05 |
| fc.weight | 5,120 | 3.4 | 0.307 | 0.08120 | 40.53 |

Comparison with paper targets (`configs/paper_targets.yaml`; reported, not tuned):

| item | ours | paper | status | source |
|---|---|---|---|---|
| FP32 accuracy (%) | 95.1100 | 96.5500 | delta -1.44 pp | Sec 5.1, p.4: '... FP32 baselines on CIFAR-10 (96.55%, 94.0%, 92.58% respectively)' (ResNet-18 first) |
| E4M11 accuracy (%) | 28.4400 | n/a | no paper value (figure only) | Only shown as a bar in Fig. 2 (p.4); no ResNet-18/E4M11 number in the text |
| E4M11 accuracy drop (pp) | 66.6700 | 0.1-0.2 (all E>=4 formats) | above range | Sec 5.1, p.4: 'all formats with E>=4 remain within 0.1-0.2% of the ... FP32 baselines' |
| E4M11 SQNR, Conv+Linear weights (dB) | 6.6410 | 78.7-79.8 (all 9 model-dataset pairs); R18-C10 = 78.7 | below range by 72.06 dB | Sec 5.2, p.5: 'E4M11 ... across all nine model-dataset combinations (78.7-79.8 dB)'; Fig. 5 row R18-C10 |
| w_min (Table 3), Conv+Linear weights | -0.4105 | -0.8460 | delta +0.435 | Appendix A.2 Table 3 (p.8), ResNet-18 / CIFAR-10; Table 3 does not state which tensors (Conv/Linear only or all parameters) are included |
| w_min (Table 3), all parameters | -0.4105 | -0.8460 | delta +0.435 | Appendix A.2 Table 3 (p.8), ResNet-18 / CIFAR-10; Table 3 does not state which tensors (Conv/Linear only or all parameters) are included |
| w_max (Table 3), Conv+Linear weights | 0.5556 | 2.2610 | delta -1.705 | Appendix A.2 Table 3 (p.8), ResNet-18 / CIFAR-10; Table 3 does not state which tensors (Conv/Linear only or all parameters) are included |
| w_max (Table 3), all parameters | 0.5556 | 2.2610 | delta -1.705 | Appendix A.2 Table 3 (p.8), ResNet-18 / CIFAR-10; Table 3 does not state which tensors (Conv/Linear only or all parameters) are included |
