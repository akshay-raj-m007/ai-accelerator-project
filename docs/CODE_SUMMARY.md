# Code summary: existing notebook vs. the paper

Sources: `Day_9_fault_robustness (1).ipynb` (the "notebook", cells 5-6 hold all model/E4M11/eval code) and
Haripriya & Tudu, *Fault Robustness of Custom Floating-Point and Integer Formats* (the "paper").
Nothing in the notebook was changed. The original (non-"(1)") notebook was not present, so it was not consulted.
`lampq_paper.pdf` in the same folder is a different paper and was not used.

## 1. Model and checkpoint

* `torchvision.models.resnet18(weights=None)` with three edits: `conv1 = Conv2d(3, 64, 3, stride=1, padding=1, bias=False)`,
  `maxpool = Identity()`, `fc = Linear(512, 10)`. This is the standard CIFAR variant. It has 21 Conv2d/Linear layers
  (20 conv incl. 3 downsample 1x1, 1 fc) and **11,164,352** weights (asserted in notebook cell 6).
* Checkpoint `resnet18_cifar10_checkpoint.pth` (89,494,055 B) has keys
  `epoch, model_state_dict, optimizer_state_dict, train_accuracy, average_loss, config`;
  it is loaded with `strict=True` from `["model_state_dict"]`.
* **`checkpoint["config"]` is never printed or used in the notebook**, and the checkpoint is not on this machine, so the
  training recipe, transforms and normalization are *unknown to me*. The only evidence is that the notebook evaluates with
  `transforms.ToTensor()` only (no mean/std), batch 128, and obtains FP32 = 80.31 % (paper: 96.55 %). Whether the
  80.31 % reflects under-training or a normalization mismatch cannot be decided from the notebook.
  The new code therefore refuses to guess: for `--init checkpoint` it reads `config`, and stops with an explicit message
  if normalization cannot be determined (`scripts/inspect_checkpoint.py` prints the config).

## 2. E4M11 encode/decode, notebook vs. paper

| Step | Notebook | Paper | Match |
|---|---|---|---|
| Format | bias 7, 4 exp bits, 11 mantissa bits, `E4M11_MAX=(2-2^-11)*2^7` | Sec 3.3: "(2 - 2^-11) x 2^7 ~ 256"; App A.2 Eq 8: 255.9375 | yes |
| Non-finite input | raises `ValueError` | not specified | n/a |
| Clip | `min(\|x\|, MAX)` | Alg 1 step 2 "a <- min(\|w\|, vmax)" | yes |
| Flush | `\|x\| >= 2^-7` is "normal", else word = sign only (signed zero) | Alg 1 step 3 "a <- 0 where a < 2^-7"; B.1 "values smaller than 2^-7 are flushed to signed zero" | yes. Note it flushes below 2^-7, not below the minimum normal 2^-6 (Eq 9); values in [2^-7, 2^-6) become 2^-6 via the fraction clip |
| Exponent | `clip(floor(log2 a), -6, 7)` | Eq 4 "e = clamp(floor(log2 \|w\|), -6, 7)"; Alg 1 step 5 clamps to [1-bias, 14-bias] = [-6, 7] | yes |
| Fraction | `clip(a*2^-e - 1, 0, 1-2^-11)` | Alg 1 step 7 identical | yes (with Alg 1; see carry below) |
| Rounding | `np.rint` (half-to-even) | "round" (unspecified) | assumed same |
| Mantissa | `clip(rint(f*2^11), 0, 2047)` | Alg 1 step 8 "m <- min(round(f*2^M), 2^M - 1)" | yes with Alg 1 |
| **Carry** | **none** (fraction pre-clipped, mantissa clipped) | **Eq 4: "(if m >= 2^11: set e <- e+1, m <- 0)"** | **NO**: matches Alg 1 but not Eq 4. The paper is internally inconsistent here |
| Packing | `sign<<15 \| (e+7)<<11 \| m` | Eq 4 / Alg 1 step 9 | yes; check `encode(3.14)=0x448F` in tests |
| Decode formula | `(1+m/2^11) * 2^(es-7)`, zero if `es == 0`, sign applied (gives -0.0) | Eq 5; Alg 2 step 6 "w <- 0 where es = 0" | yes |
| **All-ones exponent (es=15)** | decoder `clip(es, 1, 14)`: es=15 decodes as es=14 (finite, `(1+m/2^11)*2^7`) | Sec 3: "the all-ones exponent pattern is reserved for +-inf/NaN"; App A.1: es in {1..2^E-2}; but Alg 2 step 4 "e_u <- clamp(es, 1, 14) - bias" | matches Alg 2, contradicts the Sec 3 sentence. The encoder never emits es=15 (clip to vmax), so clean weights are unaffected. It only matters for fault injection (later step), e.g. flipping bit 14 of a weight in [1,2) |

Consequences: the notebook equals the paper's **Algorithm 1/2** but not **Eq. 4** (carry). `mode="legacy"` reproduces the notebook,
`mode="paper"` adds the Eq. 4 carry (keeping Alg 1's clip/flush and Alg 2's decode). The all-ones decode policy is an open
question for the fault step (default `clamp` = Alg 2; `ieee` and `literal` options are provided but not default).

## 3. Which parameters are quantized

* Notebook (`make_e4m11_model`): **every `named_parameters()` entry**: Conv/Linear weights, the fc bias, and BatchNorm
  gamma/beta (running mean/var are buffers, untouched). The packed fault buffers (cell 6) cover Conv/Linear weights only.
* Paper evidence is mixed and does not settle the accuracy path:
  * SQNR: Sec 5.2 "SQNR measured over convolutional and fully connected weight tensors" (also Fig 5 caption).
  * Method: App C "Post-Training Quantization - Weight-Only Quantization (PTQ-WOQ)"; App B.3 / Sec 4.3 "k weights are selected uniformly at random from the full weight tensor", "full weight buffer".
  * Table 3 (App A.2) lists "FP32 weight statistics" and says EfficientNet's 15.566 maximum "originate[s] exclusively in BatchNorm weight and bias parameters ... not in convolutional or linear weight tensors", yet claims E4M11 "represents all FP32 weights ... without overflow". So Table 3 statistics include BatchNorm parameters; whether BN is *stored* in E4M11 is not stated.
* Decision: default `--quantize-scope convlinear` (the only scope the paper states explicitly); `all` reproduces the notebook.

## 4. SQNR

* **There is no SQNR code in the notebook.** The 13.6279 dB figure (68 % of weights below 2^-6) comes from earlier work that is
  not in the files provided, so I cannot say which tensors it covered. This is an open item, not an assumption.
* Paper Eq 6: `SQNR = 10 log10( sum w_i^2 / sum (w_i - w_hat_i)^2 )`, over "convolutional and fully connected weight tensors".
  Whether the sums are global (one ratio over all tensors) or averaged per tensor is not stated; Eq 6 is written as one ratio, so I use
  the global form and also report per-layer values.
* `src/fr/sqnr.py` computes the global ratio over all Conv2d+Linear weights in float64, plus a per-layer table and a three-way
  breakdown by |w| (< 2^-7 flushed, [2^-7, 2^-6), >= 2^-6).

## 5. Deviation table (current code vs. paper)

| # | Item | Current notebook | Paper | Handling in new code |
|---|---|---|---|---|
| D1 | Mantissa carry | absent | Eq 4 has it; Alg 1 does not | `paper` adds carry, `legacy` keeps old behaviour |
| D2 | Quantized tensors | all parameters incl. BN gamma/beta, fc bias | SQNR over Conv+FC weights; accuracy scope unstated | default `convlinear`; `--quantize-scope all` = old |
| D3 | All-ones exponent decode | clamped to 14 | Sec 3: reserved for inf/NaN; Alg 2: clamp | `clamp` default (Alg 2); flagged for fault step |
| D4 | SQNR | not implemented in the notebook; 13.63 dB from unseen code | Eq 6, 78.7-79.8 dB | implemented per Eq 6 + breakdown |
| D5 | FP32 accuracy | 80.31 % | 96.55 % (Sec 5.1) | train/fine-tune on GPU machine; report, do not tune |
| D6 | Normalization / transforms | `ToTensor()` only; checkpoint config unseen | not stated | read from checkpoint config, else stop and ask |
| D7 | Rounding mode | half-to-even (`np.rint`) | "round" unspecified | half-to-even kept in both modes |
| D8 | Weight stats | never computed | Table 3 (min/max) | `eval_fp32.py` reports conv+linear and all-parameter stats |
| D9 | Hard-coded Colab paths, notebook state | Drive paths | n/a | all paths via CLI / `FR_ROOT` |

## 6. Addendum: findings after the first full run (ResNet-18, scratch-trained, 95.11 % FP32)

* `paper`/`legacy` E4M11: 28.44 % accuracy, 6.64 dB SQNR. 87.2 % of Conv+Linear weights are < 2^-7 and are flushed.
* A line-by-line transcription of Alg. 1 / Alg. 2 matches our encoder/decoder. The only difference is that Alg. 1 read literally
  maps flushed values to `a_safe = 1` (stored as 0x3800 = 1.0); we follow App. B.1 ("flushed to signed zero").
* Even an untrained Kaiming-init ResNet-18 gives only 18.3 dB under the paper's rules, so the paper's 78.7-79.8 dB cannot be
  produced by the stated format on typical ResNet-18 weights. The `subnormal` mode (not in the paper) tests the flush rule as the cause.

## 7. Baseline decision (D10)

Measured on the scratch-trained ResNet-18 (FP32 95.11 %, 9511/10000), Conv+Linear weights, full test set:

| E4M11 mode | accuracy drop | SQNR (Eq. 6) | note |
|---|---|---|---|
| `paper` | +66.67 pp (28.44 %) | 6.64 dB | paper-exact; 87.2 % of weights flushed |
| `legacy` | +66.67 pp | 6.64 dB | identical to `paper` (carry never matters here) |
| `subnormal` | **0.00 pp** | **69.27 dB** | not in the paper; es=0 stores m*2^-17 |

Decision: `subnormal` is used as the working E4M11 baseline for the fault-injection step, **as a documented deviation (D10)**
from Sec. 3 / App. B.1 ("subnormal numbers are flushed to zero"). Accuracy agrees with the paper's claim (<= 0.2 pp drop);
SQNR does **not** reproduce the paper's 78.7 dB (9.4 dB lower). `paper`-exact results are always reported alongside.
Fault injection must decode es = 0 words as subnormal values, not zero.

## 8. Baseline fixed (final): `subnormal`, bias 7

Bias sweep on the scratch ResNet-18 (FP32 95.11 %, max |w| 0.5556), Conv+Linear weights, full test set:

| bias | paper-rule (flush) SQNR / acc | subnormal SQNR / acc |
|---|---|---|
| 7 (paper) | 6.64 dB / 28.44 % | **69.27 dB / 95.11 %** |
| 8 | 11.53 / 94.76 | 74.62 / 95.11 |
| 9 | 17.20 / 95.18 | 78.09 / 95.11 |
| 10 | 23.97 / 95.15 | 79.37 / 95.11 |
| 11 | 31.79 / 95.11 | 79.65 / 95.11 |
| 12-15 | 40.33-66.90 / 95.11 | 79.70-79.71 / 95.11 |
| 16 | 49.41 / 95.16 | 49.42 / 95.16 (max value 0.4999 < max |w|: a few weights clip) |

Decision: **baseline = E4M11 with subnormals at the paper's bias 7** (`--e4m11-mode subnormal`, default `--e4m11-bias 7`).
This is the smallest deviation that restores accuracy: the single change is that stored exponent 0 holds `m*2^-17` instead of
zero (all es = 1..14 words are identical to the paper). SQNR is 69.27 dB, 9.4 dB below the paper's 78.7 dB, and is accepted.
Findings to carry forward: (i) SQNR plateaus at 79.71 dB (11-bit rounding noise) once the range fits, which is where the paper's
78.7-79.8 dB sits; (ii) SQNR is a poor accuracy predictor here; (iii) clipping a handful of the largest weights costs ~30 dB SQNR.
Bias != 7 remains available (`--e4m11-bias`, `scripts/bias_sweep.py`) as a sensitivity study, not the baseline.
Fault injection must decode es = 0 words as subnormal values.

## 9. Mixed-precision experiment: Fisher sensitivity + 8-bit formats (new, not in the fault-robustness paper)

Goal: keep sensitive Conv/Linear layers in E4M11 (fixed baseline above) and store the least sensitive ones in an
8-bit format, chosen with a Fisher-based score, then compare clean metrics (faults come in a later step).
Sources: LAMPQ (Kim et al., AAAI 2026, `lampq_paper.pdf`): Lemma 1 (Hessian trace as layer sensitivity), Lemma 2
(expected Hessian ~ Fisher), `Omega_i = alpha_t tr(F_i)`, ILP bit allocation under an average-bit budget; fault paper
Eq. 3 (per-tensor symmetric INT) and Table 2 (F8E4M3, max ~240). None of this is part of the fault paper's protocol.

Code: `formats8.py` (codecs), `sensitivity.py` (Fisher), `allocation.py` (scores, exact selection), `mixed.py`
(mixed quantizer, bit accounting), `mixed_run.py` (pipelines), scripts `fisher_sensitivity.py`, `mixed_precision.py`.

| # | Decision | Reason / status |
|---|---|---|
| M1 | Fisher is the **diagonal**, computed on the **FP32** model, one backward pass per calibration image (BN in eval mode), 1,024 images from the *training* split (LAMPQ uses 1,024). `true` Fisher samples the label from the model (Lemma 2); `empirical` uses dataset labels. Both are produced and labelled. | The true Fisher is what Lemma 2 relates to the Hessian. A model that fits its training data has tiny empirical-Fisher gradients, so the two can differ a lot (visible even on the smoke model). Which one to use is a choice; default `true`. |
| M2 | `Error_i` = relative weight error `sum (w - w^)^2 / sum w^2` of the 8-bit format on layer i. `Fisher_i` = trace per parameter (`--fisher-mode total` for the raw trace). `S_i = lam * norm(Error_i) + (1 - lam) * norm(Fisher_i)`, rank normalization by default (`zscore`, `minmax` available). | Requested weighted sum; rank normalization is robust to one outlier layer. lam = 0 is Fisher only and does not depend on the 8-bit format, so it is the shared selection across arms. |
| M3 | An extra scorer `product`: `damage_i = sum F_jj (w_j - w^_j)^2` (second-order loss estimate, additive over layers), plus `random` and `worst` controls. | The weighted sum is a heuristic; the product is the form the Taylor argument actually gives, so it is the natural reference. |
| M4 | Selection = exact minimisation of the summed score over **all 2^21 layer subsets** whose switched-weight count lies in `[need, need (1 + tol)]`, `need = N (16 - T) / 8`, `tol = 5%` (doubled if the window is empty). `random` and `worst` use the same window. | This is the LAMPQ ILP solved exactly; with only two options per layer the `gamma^-B` penalty only rescales and is dropped. A common window keeps the budgets of all methods matched. |
| M5 | Arms: `int8_pt` (per-tensor, Eq. 3; primary), `int8_pc` (per-channel; ablation), `fp8` (E4M3, bias 7, **subnormals on**, no scale), `fp8_scaled` (+ per-tensor power-of-two scale). Round to nearest, ties to even. | Requested. FP8 with subnormals deviates from the paper's flush-to-zero rule for the same reason as D10; `fp8` shows what happens without a scale (weights are far below 2^-6). |
| M6 | Size = payload bits per weight (16 / 8). Scale metadata (32 bit per INT8 tensor or channel, 8 bit per scaled-FP8 tensor) is reported separately and is assumed fault-free. | Keeps the target comparable across arms; metadata is tiny but not zero. |
| M7 | Layer choice (`lambda`) is made on the **validation** split (5,000 images, SE about 0.3 pp); the test set is evaluated only for the saved selections in `--phase final`. Selections are stored in `selections.json`. | Prevents tuning on the test set. |
| M8 | LAMPQ's type-aware scaling `alpha_t` is not used. | ResNet-18 has only conv and fc layers; `alpha_t` exists to compare the four ViT layer types. |

Limitation to keep in mind: Fisher/Hessian sensitivity assumes a small perturbation around a minimum. That is
reasonable for 8-bit rounding noise, but it is not expected to predict the effect of exponent bit flips (weights
scaled by up to 2^8), where the loss saturates; that comparison belongs to the fault step.

## 10. Findings from the first mixed-precision run and the two follow-up diagnostics

First run (scratch ResNet-18, FP32 test 95.11 %): every configuration, including all-8-bit, stays within noise of
FP32 on test accuracy (95.09-95.23 %, standard error about 0.22 pp), so accuracy cannot rank formats or selection
methods. Only SQNR separates them (Fisher-only / product selections are 2-5 dB above random for INT8 and scaled
FP8). The lambda sweep is nearly inert (many lambdas give identical layer sets). `true` and `empirical` Fisher are
identical to four digits because the model is extremely confident (mean NLL 0.0009 on the calibration images).

Two diagnostics were added (no change to the selections or to the accuracy tables):

| # | Item | Definition / reason |
|---|---|---|
| S1 | Prediction churn (`churn.py`, `--phase churn`) | Against the all-E4M11 model: images whose predicted class changes (flips), right->wrong (c2w), wrong->right (w2c), mean KL(softmax(E4M11) \|\| softmax(config)) and mean absolute logit difference. Accuracy nets out c2w and w2c, so it hides small perturbations; flips and KL do not. Validation: every configuration; test: only the configurations of the accuracy report. |
| S2 | Fisher stability (`stability.py`, `scripts/fisher_stability.py`) | Per-sample layer traces on 4,096 training images and 4,096 held-out validation images. Reports (a) rank stability (Spearman of the 21 layer values) between disjoint chunks of n images, (b) concentration (effective sample size, share of the trace from the top 1 % / 10 % of images and from misclassified images), (c) stability of the Fisher-only layer selection (weighted Jaccard), (d) training vs held-out rankings. |

Caveat for S1: churn is measured against the E4M11 model, not against the dataset labels; a flip is not an error.
Caveat for S2: the stability analysis ranks layers by trace per parameter (the quantity of the lambda sum); it does
not test whether the per-weight diagonal gives the same damage ranking.

## 11. Part D: bit-flip faults on mixed E4M11 / 8-bit storage (project extension)

Goal: test whether the Fisher-chosen mixed layouts (less storage) lose only a little under the paper's random
bit-flip fault model, compared with all-E4M11 and with random mixed layouts of the same size.
Code: `faults_mixed.py` (injector, sampling, campaign, comparison), `scripts/mixed_fault_campaign.py`. The pure-E4M11
campaign of `faults.py` is unchanged and remains the paper-protocol reference.

| # | Decision | Status |
|---|---|---|
| F1 | Bit groups per format: E4M11 as before (App. B.3: exp 14-11, sign 15, MSB 10-6, LSB 5-0); FP8 E4M3: exp 6-3, sign 7, MSB 2-1, LSB 0; INT8: sign 7, MSB 6-3, LSB 2-0, exponent empty. | FP8/INT8 follow Sec. 4.2 (upper ceil(M/2)); E4M11 follows App. B.3. The paper itself is inconsistent between the two (Sec. 4.2 vs App. B.3). INT8 bit 7 is the two's-complement sign position (a flip is +-128 steps). The group `any` does not depend on this split. |
| F2 | Two sampling models. `weights` (default): k weights uniformly with replacement among weights that have bits in the group, then a bit uniformly in that weight. `bits`: k bits uniformly over all group bits (a 16-bit weight is struck twice as often as an 8-bit one). | The paper says "k weights are selected uniformly" (Sec. 4.3). For a pure-E4M11 buffer both reduce to the validated protocol. With group `any` and `weights` the struck weights are identical for every layout, so layouts are compared pairwise. |
| F3 | Groups are conditional on the bits that exist. An INT8 layer has no exponent bits, so an exponent-group experiment cannot hit it; a layout with no such bits skips the group (recorded as `skipped_groups`). | Paper: the exponent group "is empty by construction" for integer formats. Consequence: exponent-group results of different layouts are not on the same footing; the system-level comparison is the group `any`. |
| F4 | Scale metadata (INT8 scales, scaled-FP8 exponent) is assumed fault-free. | Not in the paper. A fault in a per-tensor scale would corrupt a whole layer; the result files state the assumption. |
| F5 | Per trial, besides accuracy and cross-entropy: prediction flips and KL(softmax(clean) \|\| softmax(faulty)) against the same layout's fault-free output, non-finite output flag, number of flips landing in 8-bit layers. Summaries report the mean with a 95 % bootstrap interval, the median and the 10th percentile. | Exponent damage is heavy-tailed (a few catastrophic trials dominate the mean), so the mean alone is a poor summary. |
| F6 | One random stream per (seed, group, sampling, k, trial), independent of the layout. | Reproducible, resumable, and the basis of the paired comparison. |

Statistical power: in the pure-E4M11 campaign the std of the accuracy at k = 8 exponent flips was 3.7 pp, so 100
trials give a standard error of about 0.4 pp for the mean. Differences between layouts below that need more trials
(or the paired differences, medians and KL), and should not be reported as findings from 100 trials alone.

## 12. Lower-bit formats, held-out Fisher, heavier fault loads (experiments 1-3 of the LAMPQ + fault study)

Motivation (docs/EXPERIMENTS.md): at 8 bits per weight every layout keeps 95.1-95.2 % accuracy and k = 8 bit flips
change nothing measurable, so neither the storage-accuracy tradeoff nor the fault tradeoff existed. These three steps
create them. Everything here is a project extension; the paper has no 4-bit format, no Fisher and no mixed storage.

| # | Decision | Status |
|---|---|---|
| L1 | 4-bit formats in `formats8.py`: `int4_pt`, `int4_pc` (paper Eq. 3 with b = 4: s = max/7, codes -8..7, two's complement in the low nibble of a uint8) and `fp4_scaled` (E2M1, bias 1, subnormal 0.5, values {0, .5, 1, 1.5, 2, 3, 4, 6}, per-tensor power-of-two scale, 8 metadata bits). | INT4: follows Eq. 3. **FP4: assumption, not in the paper.** The all-ones exponent is NOT reserved (OCP/MX E2M1 convention) because reserving it would leave six magnitudes; an unscaled FP4 cannot hold weights of size 0.01, so only the scaled variant exists. Easy to drop: it is an independent arm. |
| L2 | Budget for a low-bit format of width b: `need = N (16 - T) / (16 - b)` weights must be low-bit (was `/8`). A target below an arm's width (or >= 16) is skipped for that arm, so one `--targets` list serves all arms. | Same exact enumeration over layer subsets; random and worst use the identical feasible set, as before. |
| L3 | Fault groups for 4-bit words (Sec. 4.2 split of the value field): INT4 sign 3, mantissa MSB 2-1, LSB 0, exponent empty; FP4 exponent 2-1, sign 3, mantissa MSB 0, LSB empty. Only the low nibble is a fault target. Scale metadata is still assumed fault-free. | Extension. Group `any` does not depend on the split. |
| L4 | Held-out Fisher: `fisher_sensitivity.py --source val --calib-size 2048` uses the first 2,048 validation images; the sweep then skips them (`--val-skip 2048`, 2,952 images left, standard error about 0.4 pp). `run_sweep` raises an error if a validation-source Fisher is combined with a smaller `val_skip`. Test set untouched. | Reason: the network has memorised its training images, so the training-image Fisher is determined by 2-3 images (docs/EXPERIMENTS.md section 7b). LAMPQ itself calibrates on training data (1,024 images). |
| L5 | `flips_in_8bit` in the fault CSV now counts hits in any layer narrower than 16 bits (column name kept so existing result files and summaries stay compatible). | Identical values for 8-bit-only runs. |
| L6 | `mixed_report.md` gains a **Frontier** table: per (arm, budget) the accuracy of Fisher only, product, error only, random (mean +- std), worst, and the uniform all-layer model, so "with Fisher" and "without Fisher" are side by side. | Test accuracy where evaluated, otherwise validation (marked `v`). |
| L7 | Fault-load calibration (exp 3) uses a fixed k grid {16, 64, 256, 1024, 4096} on the *references* (all-E4M11, all-INT8, all-INT4). The k used for the main comparison is chosen from the all-E4M11 curve only (where its median accuracy has clearly fallen but not collapsed), **not** from the layouts being compared, so k cannot be picked to favour a selector. | Rule fixed before the run. |

Files: `scripts/run_lowbit_heldout.sh` (all stages, resumable), `scripts/compare_selections.py` (Jaccard of two
`selections.json`), tests `tests/test_lowbit_heldout.py`.

## 13. Faults in the INT scale metadata (Plan A)

The earlier fault experiments assumed that the INT scales cannot be struck. This step removes that assumption for INT8
(and INT4). Project extension, nothing here is in the paper.

| # | Decision | Status |
|---|---|---|
| S1 | A scale is one IEEE-754 float32 word (`formats8.int_encode` already returns float32): one per tensor for `*_pt` (21 for ResNet-18), one per output channel for `*_pc` (4,810). Storage: 21 x 32 = 672 bit (0.0008 % of the 89.3 Mbit INT8 weight payload, 0.00006 bit/weight) vs 4,810 x 32 = 153,920 bit (0.17 %, 0.0138 bit/weight). | Per the user: scale kept as FP32; count and storage are printed by `mixed_fault_campaign.py` and tabulated in the report. |
| S2 | `--scale-faults off | on | only`. `on` (physical): scale words share the memory, so each of the k bit strikes lands in the scale region with probability S/(B+S) (S, B = scale / weight-word bits of the struck group), else as in `off`; the weight strikes kept are those `off` would draw with the same seed (paired). Needs sampling `bits`. `only`: every strike hits a uniformly chosen scale bit (damage per hit, independent of how rare hits are; not a physical load). | `off` reproduces all earlier results exactly. |
| S3 | Bit groups of the float32 scale: exponent 30-23, sign 31, mantissa MSB 22-11 (upper ceil(23/2) = 12 bits), mantissa LSB 10-0 (Sec. 4.2 split). | Extension. |
| S4 | A struck scale re-decodes the whole tensor (per-tensor) or one output channel (per-channel) as q * s'. No clamping: inf / NaN scales give non-finite weights and are counted in the `non-finite` column; accuracy of such trials is whatever `argmax` of the non-finite logits returns, so read `non-finite` beside it. A weight strike and a scale strike in the same channel combine. | Assumption: no saturation hardware on the scale register. |
| S5 | `--split val --val-skip 2048` evaluates the faulty model on held-out validation images (2,952) so that choices such as per-channel vs per-tensor are not made on the test set. | Test-set numbers only for final reporting. |

FP8-scaled / FP4-scaled scales (8-bit exponents) are not supported by scale faults (the injector raises). Files:
`scripts/run_scale_faults.sh`, `scripts/compare_scale_faults.py`, `tests/test_scale_faults.py` (14 tests).

