# Experiments and consolidated results

Fault robustness of custom number formats on ResNet-18 / CIFAR-10, plus a project extension: Fisher-information
sensitivity to choose which layers can be stored in 8 bits.

* Reference paper: Haripriya & Tudu, *Fault Robustness of Custom Floating-Point and Integer Formats* (E4M11 etc.).
* Extension idea: Kim et al., *LAMPQ* (AAAI 2026), Fisher information as layer sensitivity. Only the Fisher sensitivity is
  used; its ILP, the iterative bit update and the alpha_t weighting are **not** ported.
* All numbers below are copied from the files in `reports/scratch/` (exported from the GPU run). Where a number is not in
  those files, the text says where it comes from.
* Hardware / software of the runs: NVIDIA RTX 3050, torch 2.5.1+cu121, Python 3.12.3 (`reports/scratch/run_meta.json`).
  The recorded git states were marked *dirty* (uncommitted changes at run time): train/step1 at 239c285, Fisher at 7ad4085.

## 0. Summary

| # | Experiment | Question | Headline result | Why the next step |
|---|---|---|---|---|
| 1 | FP32 baseline | Do we have a trained ResNet-18 to quantize? | 95.11 % test (paper: 96.55 %) | Need the E4M11 format applied to it |
| 2 | E4M11 PTQ, paper rules | Does paper-exact E4M11 keep accuracy? | **28.44 %**, 6.64 dB: it does not | 87 % of weights are flushed; test the flush rule |
| 3 | E4M11 with subnormals, bias sweep | Is the flush rule the cause? | Accuracy 95.11 %, SQNR 69.27 dB at bias 7 | Fixed the baseline; now run the fault protocol |
| 4 | Paper fault campaign on E4M11 | Do bit flips behave as the paper reports? | Exponent k=8 drop 0.73 pp (paper 1.1); sign / mantissa about 0 | Faults hurt only via the exponent; baseline is faithful |
| 5 | Fisher sensitivity | Which layers are sensitive? | Fisher from training images is dominated by 2-3 images | Check how trustworthy it is (exp. 7) |
| 6 | Mixed E4M11 + 8-bit, clean metrics | Can less-sensitive layers use 8 bits at no loss? | Yes: 25-37 % less storage, accuracy 95.09-95.19 % | Accuracy is saturated; need finer metrics (exp. 7) |
| 7 | Prediction churn + Fisher stability | Does Fisher selection beat random? | KL below random in 12/12 cells; Fisher caveat found | Test under bit flips (exp. 8) |
| 8 | Bit flips on mixed storage | Does mixed storage hold up under faults? | No measurable loss at k=8; layouts not separable | Heavier fault load needed |
| 9 | 4-bit formats (INT4 pt/pc, FP4 scaled), held-out Fisher | Is there an accuracy-vs-storage tradeoff, and does layer selection matter? | Uniform 4-bit: int4_pc -0.12 pp, int4_pt -1.2 pp, fp4 -1.3 pp. Any sensitivity-based selection recovers it; random does not | Fisher is not shown to beat plain error; need fault load |
| 10 | Held-out vs training-image Fisher selections | Does the memorized-training-image problem change the selected layers? | Mean weighted Jaccard 0.985 (8-bit arms) and 0.967 (4-bit arms); most selections identical | Conflicts with the 0.407 of exp. 7b: unexplained |
| 11 | Fault-load calibration on references | Which k separates E4M11 from low-bit storage? | E4M11 median 94.55 / 88.99 / 22.35 % at k = 256 / 1024 / 4096; all-INT8 and all-INT4 unchanged | k = 1024 chosen for the main comparison (rule L7) |

What the evidence supports today:

* **Supported:** 12-bit mixed storage (25 % smaller than 16-bit E4M11) keeps clean accuracy, and under light random bit
  flips (k=8) shows no measurable accuracy loss.
* **Supported:** Fisher / product-selected layers sit closer to the E4M11 model than random layers of the same size
  (prediction KL, 12 of 12 cells on both validation and test) and give higher SQNR.
* **Not supported (not yet shown):** that Fisher-chosen layouts are more *fault*-robust than random ones. Exp. 8 could not
  tell them apart. The Fisher used was computed on training images the network has memorized (see exp. 5 and 7).

---

## 1. FP32 baseline

**Why.** The paper's models are not available, so a ResNet-18 was trained from scratch to quantize.

**Setup.** CIFAR-style ResNet-18, 200 epochs, SGD lr 0.1 (Nesterov, momentum 0.9, weight decay 5e-4, 5 warm-up epochs,
batch 128, AMP), seed 0, 5,000 training images held out as validation (`run_meta.json`). Checkpoint selected on validation.

**Result** (`final_metrics.json`, `fp32_report.json`):

| | value |
|---|---|
| Test accuracy | 95.11 % (9511 / 10000) |
| Validation accuracy (best.pth, epoch 200) | 95.48 % |
| Test cross-entropy | 0.1855 |
| Conv+Linear weights | 11,164,352; min -0.411, max 0.556, std 0.00644 |

Versus the paper: FP32 accuracy 96.55 % (delta -1.44 pp); weight range min -0.846 / max 2.261 in the paper's Table 3
(delta +0.435 / -1.705). Our network is a different training run, so these are reported, not tuned.

**Why next.** With a baseline fixed, apply the paper's E4M11 quantizer.

## 2. E4M11 post-training quantization, paper rules

**Why.** Reproduce the paper's claim that E4M11 stays within 0.1-0.2 pp of FP32 with SQNR 78.7 dB.

**Setup.** 1 sign + 4 exponent (bias 7) + 11 mantissa bits, Algorithms 1/2, Conv+Linear weights, full test set
(`step1_report.md`).

**Result.**

| mode | accuracy | drop | SQNR |
|---|---|---|---|
| `paper` (flush subnormals to zero) | **28.44 %** | +66.67 pp | **6.64 dB** |
| `legacy` | 28.44 % | +66.67 pp | 6.64 dB (identical) |

87.21 % of the weights have |w| < 2^-7 and are flushed to zero; they carry 15.7 % of the signal power but **72.4 % of the
error power**. The deep layers are the worst: layer4.1.conv2 has SQNR 0.05 dB with 100 % of weights below 2^-6.
Versus the paper: accuracy drop 66.67 pp (paper 0.1-0.2), SQNR 6.64 dB (paper 78.7-79.8): far off.

**Cross-checks** (`docs/CODE_SUMMARY.md` section 6): a line-by-line transcription of Algorithms 1/2 matches our
encoder/decoder, and even an untrained Kaiming-init ResNet-18 only reaches 18.3 dB under the paper's rules, so the paper's
78.7 dB cannot come from the format as stated on typical ResNet-18 weights.

**Why next.** The suspected cause is the flush-to-zero rule. Test it with the smallest possible change.

## 3. E4M11 with subnormals; bias sweep (baseline decision D10)

**Why.** Check whether storing stored-exponent 0 as a subnormal (value m * 2^-17, smallest nonzero 2^-17) instead of zero
restores the paper's behaviour, and whether the exponent bias matters.

**Result** (full test set; table from `docs/CODE_SUMMARY.md` section 8 - `bias_sweep.json` was not exported, so these
numbers are not in `reports/`):

| bias | paper rule (flush) SQNR / acc | subnormal SQNR / acc |
|---|---|---|
| 7 (paper) | 6.64 dB / 28.44 % | **69.27 dB / 95.11 %** |
| 8 | 11.53 / 94.76 | 74.62 / 95.11 |
| 9 | 17.20 / 95.18 | 78.09 / 95.11 |
| 10 | 23.97 / 95.15 | 79.37 / 95.11 |
| 11 | 31.79 / 95.11 | 79.65 / 95.11 |
| 12-15 | 40.33-66.90 / 95.11 | 79.70-79.71 / 95.11 |
| 16 | 49.41 / 95.16 | 49.42 / 95.16 |

**Decision (documented deviation D10).** Baseline = E4M11 with subnormals at the paper's bias 7. Accuracy drop 0.00 pp,
SQNR 69.27 dB (9.4 dB below the paper's 78.7 dB, accepted and reported). The paper-exact numbers are always reported
alongside. This deviates from the paper's "subnormals are flushed" (Sec. 3 / App. B.1).

Findings carried forward: SQNR plateaus at about 79.7 dB (11-bit rounding noise), which is where the paper's range sits;
SQNR is a poor accuracy predictor here; clipping a handful of the largest weights (bias 16) costs about 30 dB.

**Why next.** With a working baseline, run the paper's fault protocol.

## 4. Paper-protocol bit-flip campaign on E4M11

**Why.** Reproduce the paper's fault results (Sec. 5.3-5.4) on our baseline, and have a pure-E4M11 reference for the
extension.

**Setup** (`faults_subnormal_b7/summary.md`): E4M11 subnormal bias 7; groups exponent, sign, mantissa MSB, mantissa LSB;
k = 1..8 flips; 100 trials per cell; seed 0; whole-buffer fault index; full test set; 3,200 trials in total.

**Result:**

| group | k=8 mean acc % | std | min | drop (pp) | paper |
|---|---|---|---|---|---|
| exponent | 94.38 | 3.71 | 65.52 | **0.73** | 1.1 % |
| sign | 95.11 | 0.00 | 95.10 | -0.00 | < 0.3 % |
| mantissa MSB | 95.11 | 0.00 | 95.11 | -0.00 | < 0.05 % |
| mantissa LSB | 95.11 | 0.00 | 95.11 | -0.00 | < 0.01 % |

* Exponent damage is **heavy-tailed**: the worst single trial at k=4 falls to 13.44 %, while many trials do not move.
  The mean over 100 trials is therefore noisy (standard deviation 3.7 pp at k=8, so a standard error of about 0.4 pp):
  0.73 vs 1.1 is within about one standard error, but the mean does not rise steadily with k (k=4: 1.14 pp, k=7: 0.16 pp).
* Sign and mantissa flips are harmless at these k, consistent with the paper's ordering.

**Why next.** The baseline reproduces the paper's fault ordering. The project extension asks whether the storage can be
reduced further by using 8 bits where the network is insensitive.

## 5. Fisher-information sensitivity

**Why.** Rank layers by how much the loss reacts to weight perturbations, following LAMPQ (Lemma 1/2: the Hessian trace is
a layer's sensitivity, and the expected Hessian is approximated by the Fisher information).

**Setup** (`sensitivity/sensitivity_meta.json`): diagonal Fisher of the FP32 model from per-sample gradients, 1,024
**training-split** images (un-augmented), labels sampled from the model (`true` Fisher; `empirical` with dataset labels is
also saved). Quantities used for selection: trace per parameter and the damage proxy sum F_jj * (w - w_hat)^2.

**Result.** On these images the mean NLL is 0.00093: the network has memorized them. The stability analysis (exp. 7) shows
this makes the Fisher dominated by a handful of images.

## 6. Mixed E4M11 + 8-bit storage, clean metrics

**Why.** Check whether the least sensitive layers can be stored in 8 bits (INT8 per-tensor, INT8 per-channel, FP8 E4M3,
FP8 E4M3 with a per-tensor power-of-two scale) while the others stay E4M11, at 14, 12 and 10 bits per weight on average.

**Setup** (`mixed/mixed_report.md`, `mixed/sweep_meta.json`):
* Four arms: `int8_pt`, `int8_pc`, `fp8`, `fp8_scaled`.
* Selector: score S_i = lambda * norm(error_i) + (1 - lambda) * norm(Fisher_i), lambda in {0, 0.25, 0.5, 0.75, 1}, rank
  normalization; also `product` (sum F * dw^2), 5 `random` draws and `worst` (largest damage) as an envelope.
* Exact enumeration over all 2^21 layer subsets within the matched-budget window (need to need * 1.05).
* Lambda chosen on the validation split (5,000 images, standard error about 0.31 pp); test reported for the saved
  selections only.
* 8-bit scale metadata counted separately (`+meta` column); never more than 0.02 bits/weight.

**Result.**

| reference | val acc % | test acc % | SQNR dB | bits/weight |
|---|---|---|---|---|
| FP32 | 95.48 | 95.11 | - | 32 |
| E4M11 (subnormal) | 95.48 | 95.11 | 69.27 | 16 |
| INT8 per-tensor, all layers | 95.36 | 95.15 | 32.58 | 8 |
| INT8 per-channel, all layers | 95.38 | 95.14 | 39.55 | 8 |
| FP8 E4M3 (no scale), all layers | 95.30 | 95.12 | 21.20 | 8 |
| FP8 E4M3 + power-of-two scale, all layers | 95.40 | 95.18 | 31.55 | 8 |

At 12 bits per weight (25 % less than E4M11; 5 or fewer 8-bit layers for Fisher-based selections), test accuracy:

| arm | Fisher-only (lambda=0) | product | random (mean +- std) | worst | SQNR: product / worst |
|---|---|---|---|---|---|
| int8_pt | 95.11 | 95.11 | 95.13 +- 0.01 | 95.15 | 39.21 / 33.42 dB |
| int8_pc | 95.12 | 95.12 | 95.13 +- 0.01 | 95.17 | 46.88 / 40.23 dB |
| fp8 | 95.11 | 95.11 | 95.13 +- 0.02 | 95.15 | 24.37 / 23.85 dB |
| fp8_scaled | 95.13 | 95.11 | 95.16 +- 0.02 | 95.16 | 39.61 / 32.06 dB |

Findings:
* Every row of the report lies between 95.09 % and 95.19 % test accuracy, against 95.11 % for E4M11. Differences are far
  below the 0.22 pp test standard error: **accuracy is saturated and cannot rank the selectors.** Even the deliberately
  worst sets show no accuracy loss.
* SQNR does separate them: product / Fisher sets have higher SQNR than `worst` in every arm, and INT8 per-channel is the
  best 8-bit arm (SQNR 39.55 dB all-INT8 against 32.58 for per-tensor). Unscaled FP8 has the lowest SQNR (21.20 dB).
* Lambda is not a meaningful knob here: the val-chosen lambda is 0 (pure Fisher) in 7 of 12 (arm, target) cells and
  differences between lambdas are inside the validation noise.

**Why next.** Accuracy cannot discriminate, so use a finer metric (prediction churn vs the E4M11 model) and check whether
the Fisher itself is reliable.

## 7. Prediction churn and Fisher stability

**Why.** Compare each configuration to the all-E4M11 model by how many predictions change and by KL divergence, which
reacts to small output changes that accuracy hides; and check how reliable the Fisher ranking is.

### 7a. Prediction churn (`mixed/mixed_churn_report.md`)

KL = mean KL(softmax(E4M11) || softmax(config)) in nats. Test set, 12 bits per weight:

| arm | Fisher only: flips / KL | product: flips / KL | random (5 draws): flips / KL |
|---|---|---|---|
| int8_pt | 5 / 1.26e-05 | 0 / 5.97e-06 | 6 +- 4 / 5.19e-05 |
| int8_pc | 4 / 4.17e-06 | 2 / 1.80e-06 | 4 +- 3 / 2.27e-05 |
| fp8 | 7 / 1.36e-04 | 4 / 1.06e-04 | 17 +- 6 / 4.81e-04 |
| fp8_scaled | 4 / 2.60e-05 | 2 / 1.43e-05 | 12 +- 6 / 3.35e-04 |

All-8-bit references (test): int8_pt 12 flips, KL 1.59e-04; int8_pc 8 flips, KL 5.23e-05; fp8 32 flips, KL 1.32e-03;
fp8_scaled 25 flips, KL 1.04e-03.

Findings:
* Fisher-only and product have KL **below the random-draw mean in 12 of 12 (arm, target) cells**, on both validation and
  test. Product KL is about 4x to 23x below the random mean at 12 bits.
* Across 144 validation configurations KL correlates with -SQNR (Spearman 0.81; test 0.82), but only weakly with the number
  of weights switched to 8 bits (0.15; test 0.21). So *which* layers are switched matters more than how many.
* Unscaled FP8 is a distinctly poorer 8-bit format than the other three (all-FP8 KL 1.3e-3 vs 5.8e-5 for INT8
  per-channel, validation). Mean |dlogit| for fp8 is 0.06 even for the best sets.
* Flip counts are small (single digits), so an individual flips figure is noisy; KL is the more reliable signal.
* Note: validation churn was used to pick sets in the report, so test churn is the unbiased comparison.

### 7b. Fisher stability (`sensitivity/fisher_stability.md`)

| property | training images (4,096) | held-out validation images (4,096) |
|---|---|---|
| mean NLL | 0.0010 (0 misclassified) | 0.1747 (191 misclassified) |
| effective number of images (min / median / max) | 2.0 / 2.4 / 3.1 | 59 / 101 / 115 |
| trace from the top 1 % of images | 0.99-1.00 | 0.50-0.63 |
| Spearman between 1,024-image chunks (mean / min) | 0.706 / 0.340 | 0.995 / 0.991 |
| 12-bit Fisher-only selection: Jaccard vs full | 0.611 / 0.310 | 0.976 / 0.903 |

Training-vs-held-out comparison: layer ranking Spearman 0.862 (per parameter); held-out traces are about 480x larger
(median ratio; min 15, max 609); Jaccard of the Fisher-only selections 1.000 / **0.407** / 0.967 at 14 / 12 / 10 bits.

**Interpretation.** The network assigns near-certain predictions to its training images, so the Fisher computed from them
(what all saved selections used) is determined by 2-3 images and is unstable at 12 bits. Held-out Fisher is stable.
The rankings from both sources are still similar overall (0.862), and the selections agree at 14 and 10 bits, which is why
the churn result above is still positive; at 12 bits the selection depends on the data source.

**Why next.** Test the layouts under the paper's bit-flip fault model (the question the project actually asks), using the
saved selections as they are.

## 8. Bit flips on mixed E4M11 / 8-bit storage (part D)

**Why.** The project question: does a system with less storage (Fisher-selected mixed layout) lose only a little under the
paper's random bit-flip fault model, compared with all-E4M11 and with random layouts of the same size?

**Setup** (`mixed_faults/weights/comparison.md`): 19 layouts (all-E4M11; all-8-bit for three arms; at 12 bits: product,
lambda=0, random 0, random 1, worst for INT8 per-channel, INT8 per-tensor and scaled FP8); fault group `any` (every bit of
every weight eligible); k = 8 flips; 100 trials per layout; 1,900 full-test evaluations (about 71 minutes); the same
struck weights for every layout in a given trial. Extension assumptions: each format's bit groups, a weight chosen
uniformly then a bit within it, scale metadata fault-free (details in `docs/CODE_SUMMARY.md` section 11).

**Result** (group `any`, k = 8, mean over 100 trials):

| layout | bits/weight | clean acc | mean acc under faults | median KL | mean KL | non-finite |
|---|---|---|---|---|---|---|
| e4m11 | 16.00 | 95.11 | 95.11 | 5.0e-07 | 3.6e-04 | 0 |
| product (3 arms) | 12.00 | 95.11-95.12 | 95.11 | 1.1-2.0e-07 | 3.55e-04 | 0 |
| lambda=0 (3 arms) | 11.98 | 95.11-95.13 | 95.12-95.14 | 1.6-2.4e-07 | 4.8e-05 | 0 |
| random 1 (3 arms) | 11.81 | 95.13-95.14 | 95.13-95.14 | 1.9-3.2e-07 | 3.5e-04 | 0 |
| random 0 (3 arms) | 11.81 | 95.12-95.18 | 95.12-95.18 | 1.1-1.9e-07 | 3.3e-05 | 0 |
| worst (3 arms) | 11.80 | 95.15-95.17 | 95.15-95.17 | 1.7-3.9e-07 | 7e-06 | 0 |
| all 8-bit (3 arms) | 8.00 | 95.14-95.18 | 95.14-95.18 | 0.4-2.0e-07 | 0.4-1.3e-06 | 0 |

Findings (read these together with the caveats):
* For every layout the mean accuracy under faults is within about 0.01 pp of that layout's own clean accuracy, and the
  median equals the clean value. **No fault-induced accuracy loss is measurable at k = 8, in any layout, including the
  50 %-smaller all-8-bit ones.**
* The "drop vs E4M11" column in `comparison.md` mostly reflects clean-accuracy differences between layouts (for example
  95.18 % vs 95.11 %), not faults. Do not read it as robustness.
* The mean-KL differences (3.5e-4 for product and random 1 vs 7e-6 for `worst`) are probably dominated by one or two
  catastrophic exponent hits that fell in an E4M11 layer for some layouts and an 8-bit layer for others (identical
  value 3.55e-4 across all three product arms). This explanation has **not** been verified against the per-trial KL
  values. The median KL is about 1e-7 for every layout, i.e. noise level. The `worst` layouts, which have the most weights in
  8 bits, have the lowest mean KL, which does not support a Fisher advantage.
* About 4.1 of the 8 hits per trial land in 8-bit layers in all 12-bit layouts, so they are exposed to similar numbers of
  8-bit faults.

**Why this is the end of the current run.** The fault load is too light to separate any of the layouts. k = 8 gives about
two exponent flips per trial for 16-bit words, and INT8 words have no exponent bits.

## 8b. Lower-bit formats and held-out Fisher (experiments 9, 10)

**Why.** At 8 bits every layout kept 95.1-95.2 %, so there was no accuracy cost to trade against storage. Experiment 9 adds
4-bit formats; the Fisher is also recomputed from the first 2,048 validation images (the network has memorized the training
images) and the sweep validates on the remaining 2,952 (standard error about 0.4 pp). Test set untouched until the final
phase. Design decisions L1-L7 in `docs/CODE_SUMMARY.md` section 12. FP4 (E2M1, all-ones exponent not reserved) is an
assumption taken from the OCP/MX convention, not from the paper.

**Uniform all-layer results** (`reports/scratch/mixed_heldout/mixed_report.md`, test accuracy; E4M11 = 95.11 %):

| format | bits | test acc % | SQNR dB | prediction flips vs E4M11 (val) |
|---|---|---|---|---|
| int8_pc / int8_pt / fp8_scaled | 8 | 95.14 / 95.15 / 95.18 | 39.6 / 32.6 / 31.6 | 5 / 7 / 13 |
| int4_pc | 4 | 94.99 | 14.4 | 37 |
| int4_pt | 4 | 93.88 | 7.9 | 94 |
| fp4_scaled | 4 | 93.78 | 9.8 | 114 |

**Frontier** (test accuracy %, mixed storage at the stated average bits per weight; `random` is mean +/- std of 5 draws):

| arm | budget (avg bits) | Fisher only | product | error only | random | worst | uniform 4-bit |
|---|---|---|---|---|---|---|---|
| int4_pt | 10 (9.98) | 95.12 | 95.10 | 94.95 | 94.60 +/- 0.22 | 93.87 | 93.88 |
| int4_pt | 8 (7.76) | 95.08 | 95.08 | 94.92 | 94.45 +/- 0.17 | 93.96 | 93.88 |
| int4_pt | 5 (4.91) | 94.81 | 94.86 | 94.78 | 94.41 +/- 0.24 | 93.84 | 93.88 |
| fp4_scaled | 8 (7.76) | 95.18 | 95.18 | 95.15 | 94.54 +/- 0.28 | 93.60 | 93.78 |
| fp4_scaled | 5 (4.91) | 95.12 | 95.08 | 95.17 | 94.42 +/- 0.36 | 93.83 | 93.78 |

(The full table covers all arms and budgets 14-5. For int8 arms and int4_pc every row is inside the noise of 95.1 %.)

**Reading.**

* A real accuracy-vs-storage tradeoff now exists for int4_pt and fp4_scaled (about -1.2 pp at 4 bits). int4_pc is almost
  free at 4 bits, so it is not a useful arm for this question.
* Choosing layers by a sensitivity score (Fisher, product, or plain quantization error) recovers most of the loss;
  random choice at the same storage recovers about half, and the worst feasible set recovers none.
* **Fisher is not shown to be better than plain quantization error ("error only").** Differences are 0-0.2 pp, below the
  test standard error (0.22 pp). The only consistent but small edge is int4_pt at budgets 10, 8 and 6.

**Experiment 10: selections with training-image Fisher vs held-out Fisher**
(`selection_compare_8bit.md`, `selection_compare_4bit.md`): mean weighted Jaccard 0.985 over 63 configurations
(8-bit arms) and 0.967 over 126 (4-bit arms); 58 and 110 selections identical. This conflicts with exp. 7b (Jaccard 0.407 for
a 12-bit selection). The cause is **not established**; a possible reason is that the budget window leaves few feasible
layer subsets, but this has not been checked.

## 8c. Fault-load calibration (experiment 11)

**Why.** At k = 8 nothing separated. Rule L7: choose k from the all-E4M11 curve only. Group `any`, sampling `weights`,
40 trials per cell (`reports/scratch/mixed_faults_load/weights/comparison.md`):

| k | E4M11 mean | E4M11 median | E4M11 p10 | all-INT8 (pc) median | all-INT4 (pc) median |
|---|---|---|---|---|---|
| 16 | 95.09 | 95.11 | 95.11 | 95.14 | 94.99 |
| 64 | 94.99 | 95.12 | 94.74 | 95.15 | 94.98 |
| 256 | 91.77 | 94.55 | 86.55 | 95.15 | 94.98 |
| 1024 | 80.27 | 88.99 | 66.69 | 95.14 | 94.97 |
| 4096 | 27.17 | 22.35 | 11.05 | 95.16 | 94.95 |

**Reading and caveat.** k = 1024 is where the E4M11 median has clearly fallen but not collapsed; it is the load for the main
comparison. All-INT8 and all-INT4 do not move at all: a flip in a bounded integer code cannot exceed the tensor scale, and
**scale metadata is assumed fault-free** (not in the paper), so this robustness is partly a consequence of that
assumption. In a mixed layout the fault exposure is dominated by the share of weights left in E4M11 (about 31 % at 8 bits/weight
for the 4-bit arms, 4-8 % at the 5-bit budget), so the question for the next experiment is whether *which* layers stay in
E4M11 matters at equal exposure.

## 8d. Pareto under bit flips, pilot (experiment 12)

100 trials per layout, k = 1024 weights struck per trial, group `any`, same struck weights for all layouts, held-out-Fisher
selections (`reports/scratch/mixed_faults_pareto/weights/comparison.md`). Test accuracy %, `int4_pt` shown (fp4_scaled behaves
alike). Pilot only: 500-trial top-up not run.

| layout | avg bits | clean | faulty median | faulty p10 | faulty mean |
|---|---|---|---|---|---|
| all-E4M11 | 16.00 | 95.11 | 83.68 | 55.95 | 78.43 |
| uniform int4_pt | 4.00 | 93.88 | 93.84 | 93.77 | 93.84 |
| T8 Fisher-only = product | 7.76 | 95.08 | 86.92 | 60.16 | 80.06 |
| T8 error only | 7.76 | 94.92 | 88.97 | 64.28 | 82.32 |
| T8 random (3 draws) | 7.69-7.91 | 94.47-94.54 | 90.75-91.00 | 74.6-79.8 | 86.5-87.7 |
| T5 Fisher-only | 4.91 | 94.81 | 94.18 | 68.73 | 87.32 |
| T5 product | 4.92 | 94.86 | 93.50 | 64.56 | 86.47 |
| T5 error only | 4.98 | 94.78 | 94.11 | 79.13 | 89.58 |
| T5 random (3 draws) | 4.45-4.79 | 94.09-94.70 | 92.66-94.11 | 82.1-89.5 | 89.9-90.9 |

**Reading (pilot; wide intervals, mean CIs overlap in most pairs).**

* Every mixed layout is far more fault-robust than all-E4M11 at much lower storage (median +3 to +11 pp, bits down 50-70 %),
  because most struck weights now sit in bounded integer codes. This relies on the assumption that scale metadata is fault-free.
* **Uniform 4-bit is the most fault-robust layout, with the smallest tail, and the cheapest.** Selective E4M11 layers buy
  clean accuracy (+0.9 to +1.3 pp) at the price of a heavy tail (p10 60-80 %).
* **At equal exposure (T8, about 31 % of weights in E4M11) Fisher-based selection is the least fault-robust mixed choice**
  (median 86.9 vs 89.0 error-only vs 90.8-91.0 random; p10 60 vs 64 vs 75-80), while being the best on clean accuracy.
  Hypothesis, not tested: Fisher keeps the most sensitive layers in E4M11, where exponent flips do the most damage.
* At T5 the medians are within about 1.5 pp of each other and the p10 favours random and error-only over Fisher. T5 random draws
  differ in E4M11 share (3.8-8.2 % of weights), so part of that gap is exposure, not selection.
* Not supported: that Fisher selection improves fault robustness. Supported: it gives the best clean accuracy at a given
  storage among mixed layouts.

## 9. Limitations and open points

1. **Fault comparison is underpowered.** Exponent-fault accuracy is heavy-tailed (std 3.7 pp at k = 8 in exp. 4); 100
   trials cannot resolve small differences. A heavier load (for example k = 64, 512) or 500+ trials per key layout is needed.
2. **Fisher source.** All saved selections used Fisher from training images the network has memorized (effective sample
   size about 2). Held-out Fisher (for example the first 2,048 validation images) with evaluation on the rest has not been run.
3. **Fisher vs fault damage is untested.** Whether per-layer Fisher / sum F * dw^2 predicts the damage of single-layer
   exponent-bit faults (part E) has not been measured.
4. **Single trained model, single seed, single dataset.** Test standard error about 0.22 pp; validation about 0.31 pp.
5. **Baseline deviation D10** (subnormals) and its SQNR gap to the paper (69.27 vs 78.7 dB) remain; FP32 accuracy is 1.44 pp
   below the paper's.
6. **Unverified explanation** in exp. 8 about catastrophic hits; checking the per-trial maximum KL would settle it.
7. **Scale metadata** (INT8 scales, scaled-FP8 exponent) is assumed fault-free in exp. 8; not in the paper.
8. **Recorded runs used a dirty git tree** (uncommitted changes), so the exact code state is only approximated by the
   commit hashes in the metadata.

## 10. Files and reproduction

| path | content |
|---|---|
| `models/scratch/best.pth` | trained FP32 ResNet-18 (85.4 MB; sha256 prefix c612da6a4b20) |
| `reports/scratch/step1_report.md`, `fp32_report.json`, `final_metrics.json` | baseline and E4M11 PTQ (paper / legacy modes) |
| `reports/scratch/train.log`, `train_log.csv` | training curve |
| `reports/scratch/faults_subnormal_b7/` | exp. 4: summary and per-trial data |
| `reports/scratch/sensitivity/` | exp. 5, 7b: per-layer Fisher, per-image traces, stability report |
| `reports/scratch/mixed/` | exp. 6, 7a: sweep, selections, churn |
| `reports/scratch/mixed_faults/weights/` | exp. 8: per-layout trials and `comparison.md` |
| `reports/scratch/sensitivity_heldout/`, `mixed_heldout/`, `mixed_lowbit/` | exp. 9: held-out Fisher, 6-arm sweep, churn |
| `reports/scratch/selection_compare_{8,4}bit.md` | exp. 10 |
| `reports/scratch/mixed_faults_load/weights/` | exp. 11: fault-load calibration |
| `reports/scratch/EXPORT_MANIFEST.md` | what was exported and what was skipped |

Not exported (42.6 MB each): `sensitivity/fisher_true.pt`, `fisher_empirical.pt` (full diagonal Fisher tensors; recomputable
with `scripts/fisher_sensitivity.py`). The bias-sweep output (`bias_sweep.json`) and the paper-mode fault campaign were
also not exported.

Commands to reproduce each step are in `README.md` (Steps 1-4); design decisions and deviations from the paper are in
`docs/CODE_SUMMARY.md`.
