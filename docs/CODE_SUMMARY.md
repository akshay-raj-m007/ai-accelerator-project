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
