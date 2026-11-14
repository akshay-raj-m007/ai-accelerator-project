# MobileNetV2 / CIFAR-10 run (protocol v1.0)

Reference: `JFAMP_MobileNetV2_CIFAR10_CIFAR100_Experimental_Protocol.pdf` (5 pages). This is the CIFAR-10 owner's run; the other owner runs CIFAR-100 with the same protocol and the same codec commit.
The ResNet-18 pipeline ([phase1](../phase1/README.md) ... [phase6](../phase6/README.md)) is reused unchanged; every change below is backward compatible (the ResNet-18 results are not affected).

## Status by protocol section
| protocol section | status |
|---|---|
| 1 Experiment identity (architecture, scope, formats, reproducibility record) | **done** (code + tests); needs the GPU runs below |
| 2 Dataset-specific records (checkpoint SHA-256, split hash, normalization, FP32 val/test) | **done** (`baseline/baseline_manifest.json`, `baseline_metrics.json`) |
| 3 Quantization configuration and clean measurements (7 formats) | **done** (`quantization/`), needs the GPU runs below |
| 4 onwards (fault injection budgets, ECC protection, optimiser, handoff) | **not started**; to be built on the frozen Stage 2 output |

## Decisions and assumptions (each is flagged because the protocol leaves it open or leaves the choice to the owner)
| # | decision | why | to confirm |
|---|---|---|---|
| D1 | **Architecture** = torchvision `mobilenet_v2(weights=None)` with only the stride of `features[0][0]` changed 2 -> 1, `num_classes = 10`, dropout 0.2 as in torchvision (`fr.model.build_mobilenetv2_cifar`). The final feature map is 2x2. | protocol Section 1 | identical for both owners (CIFAR-100: `num_classes = 100`) |
| D2 | **Baseline = fine-tuned from torchvision's ImageNet weights (primary); from scratch is the fallback.** `init: imagenet` copies all 312 tensors except the 1000-class classifier (re-initialised for 10 classes) into the protocol model and fine-tunes EVERY layer on 32x32 CIFAR-10 images (`configs/finetune_mobilenetv2_imagenet.yaml`: SGD Nesterov, lr 0.05, wd 5e-4, 3 warm-up epochs + cosine, 100 epochs, batch 128, AMP, same 45,000/5,000 split, CIFAR-10 mean/std, random crop + flip, no label smoothing). Fallback `configs/scratch_mobilenetv2.yaml`: the ResNet-18 recipe unchanged (200 epochs). Settings were fixed before any run. If both are run, the baseline is the one with the higher **best validation accuracy** (never the test accuracy). The weight source is stored in the checkpoint config (`pretrained_source`) and therefore in `baseline_manifest.json`. | The weights fit exactly because only the stem stride differs and a stride is not a parameter. No CIFAR-10 checkpoint of the protocol architecture exists: the only public CIFAR-10 MobileNetV2 weights (chenyaofo/pytorch-cifar-models) are a different implementation that also changes the stride of the second stage. | ask the other owner how theirs was trained; accuracies are not comparable across datasets anyway |
| D3 | **INT scale policy: per-tensor is the PRIMARY INT8/INT4 pair** (the policy of the ResNet-18 implementation; protocol: "if the previous JFAMP implementation used another policy, retain it for both datasets and document it"). Per-output-channel INT8/INT4 are generated in the same run as labelled `comparison` rows (`int8_pc`, `int4_pc`; `--int-policy per_channel` swaps the roles without regenerating anything else). Per-tensor and per-channel rows are never mixed: every row carries the granularity in its label and a `role`. | The protocol recommends per-channel as the *default*; our earlier implementation used per-tensor. The other owner's ResNet-18 INT8 SQNR (38.4 dB vs our 32.6 dB) suggests their implementation may differ, so **the two owners must agree before the fault stages**. With per-channel the model has 17,066 float32 scale words (68,264 B = 3 % of the INT8 stored bits, about 31 % inside the depthwise layers), which would make scale-word faults first-order. | **needs a joint decision** |
| D4 | **Format set** = the protocol's seven formats in protocol order: FP16, BF16, E4M11-SN, FP8 E4M3FN, FP8 E5M2, INT8, INT4 (INT3 deferred); E4M11-FTZ kept as a labelled comparison (protocol: "distinguish from FTZ"). | protocol Section 3 | - |
| D5 | **Clean-accuracy shortlist**: column `meets_delta_clean` in `clean_accuracy.csv` flags primary formats whose validation drop is <= `--delta-clean` (default 0.5 pp, the value declared in the ResNet-18 run). It only flags; nothing is removed. | protocol: "shortlist candidates meeting the clean-accuracy constraint, freeze before the test set" | the value must be identical for both owners |
| D6 | **No reference values**: the ResNet-18 counts/accuracy are not used (protocol). `stage01_baseline.py` gates are structural only (strict load, 10 outputs, 3x3 stride-1 first conv, finite parameters/logits). | protocol Section 1 | - |

## What was added (all CPU-tested)
* `src/fr/model.py`: `build_mobilenetv2_cifar`, `ARCHS`, `build_model`, `arch_of_checkpoint`, `stem_conv`, `classifier_linear`; `load_model` reads the architecture from `checkpoint['config']['arch']`.
* `src/fr/config.py` / `train.py`: `arch` option (default `resnet18_cifar`, so ResNet-18 behaviour is unchanged); `init: imagenet` (+ `pretrained_weights`, or a local file via `init_checkpoint`) starts from torchvision's ImageNet weights (`fr.model.load_imagenet_weights`: strict check that exactly the classifier is missing; source recorded in the checkpoint config).
* `src/fr/stage2.py`: codecs **BF16** (`bf16`, 1+8+7, round-to-nearest-even, saturating), **FP8 E5M2** (`fp8_e5m2`, 1+5+2, bias 15, subnormals, inf/NaN codes never produced, saturating), **per-output-channel INT8/INT4** (`int8_pc`, `int4_pc`; axis 0, float32 scale per channel, INT4 nibbles packed over the flattened tensor); `make_codecs_protocol`, `codec_registry`, `reference_decode` (independent torch/NumPy reference); zero-rate fields; `CODEC_VERSION = "jfamp-codecs-2.0"`. 46 self-tests (every codec checked against `torch.bfloat16`, `float8_e4m3fn`, `float8_e5m2`, `float16`) are stored in the manifest.
* `src/fr/baseline_profile.py`: architecture-aware `layer_group` (MobileNetV2: stem, stage1..7, head, classifier) and `layer_role` (stem / depthwise / pointwise / conv3x3 / classifier), output-channel counts, zero counts.
* `scripts/stage01_baseline.py`, `scripts/stage2_quantize.py`: architecture-aware; new outputs listed below.

## Train the baseline (GPU machine)
```bash
cd ~/Music/aia/ai-accelerator-project && git pull
source .venv312/bin/activate
export FR_ROOT=$HOME/fr_runs
# 0. (only if the machine has no internet) download the weights elsewhere and pass them with --init-checkpoint <file>:
#    https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
python -c "from torchvision.models import MobileNet_V2_Weights as W; W.IMAGENET1K_V1.get_state_dict(progress=True)"   # caches the 14 MB file
# 1. pre-flight (about 1-2 minutes): 2 epochs on 5,000 images; proves the data, the weights and the GPU work
python scripts/train_baseline.py --config configs/finetune_mobilenetv2_imagenet.yaml --run-name mnv2_smoke \
    --epochs 2 --train-subset 5000 --test-subset 1000
# 2. the baseline: full fine-tune (100 epochs; --resume continues an interrupted run)
python scripts/train_baseline.py --config configs/finetune_mobilenetv2_imagenet.yaml --run-name mobilenetv2_cifar10
# 3. (optional fallback / comparison, from scratch, 200 epochs)
python scripts/train_baseline.py --config configs/scratch_mobilenetv2.yaml --run-name mobilenetv2_cifar10_scratch
```
Progress: `tail -f $FR_ROOT/results/mobilenetv2_cifar10/train.log`; per-epoch table `train_log.csv` (columns `val_acc`, `val_loss`). `best.pth` is the best-on-validation epoch; `final_metrics.json` reports the test accuracy of the last and the best-on-validation checkpoint (reporting only).
Choosing between runs 2 and 3: higher best `val_acc` (in `train_log.csv`), not the test accuracy.

## Run the protocol stages (after training)
```bash
RUN=mobilenetv2_cifar10                       # or mobilenetv2_cifar10_scratch if that run was chosen
CK=$FR_ROOT/checkpoints/$RUN/best.pth         # best-on-validation checkpoint (never selected on test)
python scripts/stage01_baseline.py --checkpoint $CK --run-name $RUN --inventory-dirname inventory \
    --supplied-by "trained in this project" --source-location "fine-tuned from torchvision ImageNet weights, configs/finetune_mobilenetv2_imagenet.yaml"
python scripts/stage2_quantize.py --checkpoint $CK --run-name $RUN --stage quantize                  # validation only; freezes the configuration
python scripts/stage2_quantize.py --checkpoint $CK --run-name $RUN --stage test                      # official test set, after the freeze
python scripts/export_results.py --fr-root $FR_ROOT --run-name $RUN                                  # then git add -f reports/$RUN ... and push
```
Both stage scripts print the gate / shortlist lines; `stage2_quantize.py` exits non-zero if any decoded tensor differs from its independent reference.

## Outputs and the protocol's required records (`reports/<run>/jfamp/`)
| protocol requirement | file |
|---|---|
| checkpoint SHA-256, byte size, epoch, architecture, source; split indices hash; normalization; seeds; Python/PyTorch/torchvision/CUDA/GPU versions; source commit | `baseline/baseline_manifest.json`; exact train/validation index lists: `baseline/split_indices.npz` |
| FP32 correct / N / accuracy / loss on validation and test; confusion matrix; per class | `baseline/baseline_metrics.json`, `confusion_matrix.csv`, `per_class_metrics.csv`, `checkpoint_validation.txt` |
| layer inventory (53 layers, shapes, roles, groups, output channels, statistics), parameter summary, FP32 storage | `inventory/layer_inventory.csv/.json`, `layer_storage_summary.csv`, `parameter_summary.json` |
| one frozen quantizer manifest per format (rounding, overflow/underflow, subnormals, NaN/Inf, bias, saturation, scale granularity and axis, nibble order, packing, padding, self-tests) | `quantization/quantizer_manifests/<format>.json`; combined `quantizer_manifest.json` (frozen-configuration SHA-256) |
| clean top-1 (% and count), drop (pp), cross-entropy, shortlist flag, test columns after the freeze | `quantization/clean_accuracy.csv` |
| SQNR, MSE, RMSE, max abs error, relative L2, zero rate (count and fraction), clipped count, per layer and GLOBAL | `quantization/quantization_metrics.csv` |
| theoretical bits vs actual encoded bytes (payload, scales, padding, packing), compression | `quantization/storage_summary.csv` |
| FP32 -> bytes -> decoded round trip; no-fault output equals the clean quantized reference | `quantization/roundtrip_check.csv`, `quantized_weights/<format>.npz` (immutable, not exported to git) |

## Stage 3 (layer-wise sensitivity) and Section 4 (encoded-weight fault injection)
```bash
OUT=reports/$RUN/stage_04_encoded_fault_injection
python scripts/stage3_layerwise_sensitivity.py --checkpoint $CK --run-name $RUN --out-dir reports/$RUN/stage_03_layerwise_sensitivity
# Section 4 - one script, one campaign per call; each is resumable (re-run the same command after an interruption)
S4="python scripts/stage4_encoded_fault_injection.py --checkpoint $CK --run-name $RUN --out-dir $OUT"
$S4 --campaign validate  --inventory reports/$RUN/stage_01_layer_inventory/layer_inventory.csv   # 4.3 gate; the campaigns refuse to start without PASS
$S4 --campaign global                                                                           # 30 trials x format x group x k in {1,8}   (1,140 trials)
$S4 --campaign layerwise                                                                        # 10 trials x layer x format x group x k   (20,140 trials)
$S4 --campaign shortlist --top-n 5 --min-mean-drop-pp 0.5                                       # writes targeted_shortlist.csv from the layer-wise summary
$S4 --campaign targeted  --targets-csv $OUT/targeted_shortlist.csv                              # 100 trials per shortlisted configuration, FULL validation
```
Design points (Section 4): faults are XOR flips of the stored encoded bytes (`fr.stage3.EncodedFaultInjector`: words/bytes/packed INT4 nibbles) which are decoded and evaluated; `E4M11-SN`, BF16 and FP8 E5M2 bit maps are validated by the Section 4.3 gate (every bit of a synthetic tensor against an independent decoder, INT4 nibble order/padding, no-fault control within 1e-5 logit difference, byte-for-byte restoration hash after every trial, non-finite tracking in weights/activations/logits/loss, trial scope regenerated from the recorded seed). Groups: sign/exponent/fraction (fp), sign/value (INT, two's-complement, no exponent). k = 8 means eight DISTINCT weights, one bit each. Screening (global, layer-wise) uses a fixed class-stratified 1,000-image validation subset declared in `eval_subset` of every manifest (`--subset-size 0` = full 5,000); targeted follow-up always uses all 5,000 images. The INT per-tensor scale words are not faulted in this campaign (Section 4 covers weight codes).

## Section 5 (protection analysis: None / Parity / SECDED, plus DMR)
```bash
python scripts/stage5_protection_analysis.py --checkpoint $CK --run-name $RUN --out-dir reports/$RUN/stage_05_protection \
    --stage4-dir reports/$RUN/stage_04_encoded_fault_injection          # CPU only, no dataset; about a minute
```
Codeword = one 32-bit word of the little-endian serialized encoded payload of a layer (last word zero-padded, padding never a target); the float32 INT scale is one more protected word. Parity = 1 even-parity bit per word (33 stored bits), SECDED = extended Hamming (39,32) (7 redundancy bits), DMR = two copies (detect-only, an extension beyond the protocol table). 100 codewords per layer and format (all of them if the layer has fewer), the same words for every scenario and strategy; primary scenarios flip 1 or 2 distinct DATA bits at identical positions for every strategy; supplementary scenarios (parity and SECDED only) may also hit the stored redundancy bits. Detected-but-uncorrected words are consumed as read (residual error). Before any campaign the script validates the codes (known answer, independent reference encoder, every single/double/triple error pattern of the 39 stored bits) and cross-checks that the word view addresses the same physical bit as the Stage 4 injector. Byte overhead is reported as actual (alignment padding + densely packed redundancy bits rounded up per layer) and nominal. Output: `protection_raw_records.csv.gz` (ECC raw records), `protection_summary.csv`, `protection_layer_summary.csv`, `protection_storage.csv` (per layer x format x strategy: the S(l,d,p) needed by an allocation step), `protection_storage_summary.csv`, `fault_protocol.json` (codeword definition and the map from the protocol's required artifact names to the Stage 4/5 files), `ecc_validation.json`, `protection_report.json`.

## Limits to state in any report
Weights-only; encoded weights are decoded to FP32 for evaluation (no native low-precision kernels); one checkpoint, one seed; INT3 deferred; the baseline is trained here with the ResNet-18 recipe, not tuned for MobileNetV2.
