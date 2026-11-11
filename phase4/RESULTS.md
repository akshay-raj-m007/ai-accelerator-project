# Phase 4 results - global and layer-wise fault screening (runbook Stage 4)

Run `20261111T083515Z-fff093af` on the GPU machine. 20 trials per configuration, 1,000 validation images, k = 1 and 8, seed 0.
Files: `reports/scratch/jfamp/fault_screening/`. **Screening evidence only (20 trials, 1,000 images), not paper-grade.**

## Verification (done after pulling)
| check | result |
|---|---|
| trials recorded | 19,520 of 19,520 expected, 19,520 unique ids, 0 duplicates, 0 missing |
| per format | E4M11-SN 3,520; FP16 3,520; FP8 3,520; INT8 4,480; INT4 4,480 |
| every trial restored byte-exactly / flip applied to encoded bytes | 19,520 / 19,520 both |
| failures (non-finite) | 187, all with output flag activations+logits+loss non-finite; none are weight-level |
| test set | not used |

Clean accuracy on the 1,000 images differs per format (E4M11-SN/FP16/INT8 96.3 %, FP8 96.0 %, INT4 95.4 %), so formats are compared by
fault accuracy and drop together, not by drop alone.

## Global scope (weights, all 11.16 M, mean drop in pp over 20 trials)
| format | sign k1/k8 | exponent k1/k8 | fraction k1/k8 | all k1/k8 |
|---|---|---|---|---|
| E4M11-SN | 0.00 / 0.00 | 0.00 / 0.11 | 0.00 / 0.00 | 0.00 / 0.02 |
| FP16 | 0.00 / 0.00 | **8.12 / 22.06** | 0.00 / 0.00 | 0.00 / **6.91** |
| FP8 E4M3FN | 0.00 / 0.00 | 0.00 / 0.10 | 0.00 / 0.00 | 0.00 / -0.02 |
| INT8 | 0.00 / 0.00 | n/a | n/a (value 0.00 / 0.00) | 0.00 / 0.00 |
| INT4 | 0.00 / 0.00 | n/a | n/a (value 0.00 / 0.00) | 0.00 / 0.00 |

Reading: a random flip among 11 M weights almost never hurts. FP16 is the exception because its exponent MSB (value 2^15 scale) can turn a
0.02-sized weight into about 1e3-1e4; worst single FP16 exponent k=1 trial lost 71.8 pp. E4M11 and FP8 mostly stay flat because E4M11 clamps its
reserved exponent and E4M3's top exponent only reaches 448. Small negative drops (-0.02) are noise on 1,000 images, not gains.

## Layer-wise: where the damage is (mean drop, pp)
Largest layer values per format/group (weights): 
| format, group, k | worst layer (mean drop) | layers with mean drop > 1 pp |
|---|---|---|
| FP16 exponent k8 | layer3.0.conv1 (59.9); layer2.0.conv1 54.7; layer1.0.conv2 50.5 | 20 of 21 |
| FP16 exponent k1 | layer3.0.conv1 (17.1) | 11 |
| FP16 all k8 | layer2.0.conv2 (26.0) | 20 |
| FP8 exponent k8 | conv1 (42.1) | 13 |
| FP8 exponent k1 | conv1 (12.6) | 3 |
| E4M11-SN exponent k8 | conv1 (31.2); layer2.0.downsample.0 20.1 | 12 |
| E4M11-SN exponent k1 | layer2.0.downsample.0 (8.6) | 1 |
| INT4 sign k8 | conv1 (1.64) | 1 |
| INT8 sign k8 | conv1 (0.85) | 0 |

Observations (what the numbers show; causes beyond that are not tested here):
- The most damaging layers are the early and middle ones (conv1, layer1-3) and the FP16 exponent field; layer4 (75 % of the weights) is
  not among the top rows, consistent with the Stage 1/Plan B finding that damage tracks per-weight sensitivity, not weight count.
- Integer weight faults are harmless at k <= 8 in every layer (max mean drop 1.64 pp, INT4 sign in conv1). Integer weights are bounded by the scale.
- FP8's exposure is concentrated in conv1 and a few layers; E4M11's in conv1 and the downsample convolutions.

## INT scale words (extension, float32 scale per tensor)
| format | group | k=1 mean drop | k=8 mean drop | failures k1 / k8 (of 20) |
|---|---|---|---|---|
| INT8 | sign | 86.81 | 86.00 | 0 / 0 |
| INT8 | exponent | 31.49 | **94.00** | 1 / 16 |
| INT8 | fraction | 0.01 | 0.33 | 0 / 0 |
| INT8 | all | 17.53 | 58.74 | 2 / 6 |
| INT4 | sign | 78.99 | 85.42 | 0 / 0 |
| INT4 | exponent | 44.16 | **91.77** | 4 / 13 |
| INT4 | fraction | 0.12 | 0.76 | 0 / 0 |
| INT4 | all | 23.29 | 61.54 | 2 / 6 |

The scale word is by far the most fragile part of the INT formats. A scale sign flip negates a whole layer: 79-87 pp mean drop in the global scope, and in the
layer-wise scope (k=1, sign) the late layers are worst: mean drop 96.3 pp for INT8 layer4.1.conv1 and fc, 96.2 for layer4.1.conv2 (accuracy falls to about chance),
and 95.4 / 95.3 for INT4 layer4.1.conv2, fc, layer4.1.conv1. A scale exponent flip rescales a layer by a power of two and can give non-finite activations.
The integer payload that looked immune above is not the weak point; the float32 scale is. Full per-layer values are in `fault_summary.csv`.
Whether per-channel scales or protecting the 21 scale words (84 bytes) is worthwhile is a Stage 5 question.

## Figures
The first set of plots was removed (not readable enough); redesigned plots are being prepared and will be listed here once verified.

### Weight and activation distributions, FP16 vs E4M11-SN, clean vs faulted
Formats chosen: FP16 (largest
global damage) and E4M11-SN (the proposed format). Each format shows the worst RECORDED trial of the same configuration, exponent group, k = 8, layer conv1
(the layer with the largest summed mean drop of the two formats), replayed with the same encoded elements and bits. Both replays reproduce the recorded fault
accuracy exactly. 1,000 validation images; activations are conv1 outputs.

| | FP16 (trial t11) | E4M11-SN (trial t19) |
|---|---|---|
| accuracy | 96.3 -> 9.6 % (-86.7 pp) | 96.3 -> 12.0 % (-84.3 pp) |
| largest flipped weight | 0.124 -> 8120 | 0.321 -> 82.2 |
| largest conv1 output (clean 4.64) | 17,300 | 170 |
| largest logit magnitude (clean 10.9) | 811 | 9.1 |
| non-finite values | none | none |

Reading: only 8 of 1,728 conv1 weights change, but the exponent flips create outliers (bit 14 changes the exponent by 16 steps in FP16, 8 in E4M11), which
propagate: FP16's largest activation stays in the hundreds-thousands in nearly every layer, E4M11-SN's falls from about 20-50 early to about 1 by layer4. FP16 logits are
displaced to about -1000..+900 (clean about +-10); E4M11-SN logits stay in the clean range but are scrambled. E4M11-SN's advantage is in outlier magnitude, not in avoiding
failure: eight exponent flips in conv1 destroy accuracy in both formats (layer-wise means 31.2 pp E4M11-SN, 38.7 pp FP16). These are the worst of 20 trials chosen for illustration;
use the layer-wise means for decisions. The conv1 weakness of E4M11-SN suggests protecting or constraining the first layer in Stage 5 (not tested here).

## Issues found (honest)
1. **One non-finite loss is not flagged.** In INT4 global scales, exponent, k=8, one trial has finite logits but `fault_loss = inf` (float32 overflow in the loss).
   The code counts it as a finite loss (n_loss_finite = 7 not 6), so `loss_increase_mean/p90/worst` of that row are `inf`, and it triggered the
   NumPy `invalid value in subtract` warning (inf - mean in the std). The trial is kept; the accuracy and drop columns of that row are unaffected, and no other
   row has NaN/inf. The failure count (13) misses this one trial. Not fixed yet because it needs a decision (see below); it affects loss columns of one row only.
2. Clean accuracy differs per format on the same images (see above). The manifest's `git dirty: true` came from untracked files only (`.venv312/`, `stage4.log`); `git diff` on the GPU machine was empty, so the code run equals commit ba3a3d8.
3. 20 trials give coarse p90/worst values; rows with k=1 and a rare catastrophic outcome (e.g. FP16 exponent) have large std.

## Status
Stage 4 complete for the planned 976 configurations. Not yet done: extra trials for the worst/most promising configurations (runbook 6.1).
