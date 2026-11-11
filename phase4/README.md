# Phase 4 - Global and layer-wise fault screening (J-FAMP runbook Stage 4, Sections 6.1-6.3)

Reference: `JFAMP_ResNet18_CIFAR10_Experimental_Flow.pdf` Section 6. Previous phases: [phase1](../phase1/README.md),
[phase2](../phase2/README.md), [phase3](../phase3/README.md).
Status: **done** - GPU run complete (19,520 / 19,520 trials, 2026-11-11). Results and analysis: [RESULTS.md](RESULTS.md).

## Aim
Identify, for each candidate format, which layers and which encoded bit fields cause the greatest degradation under random bit
flips (k = 1 and k = 8), on the validation images only, using the Stage 3-validated injector (faults in the stored encoded bytes).

## Settings (decided for this run)
| item | value |
|---|---|
| Trials per configuration | **20** (the runbook's lower bound, "if runtime is high"); more trials are reserved for the worst / most promising configurations afterwards |
| Images per trial | **1,000** validation images (first 1,000 of the held-out validation split; the same set as Stage 3) |
| k | 1 and 8 |
| Formats | the five primary formats: E4M11-SN, FP16, FP8 E4M3FN, INT8 per-tensor, INT4 per-tensor. E4M11-FTZ is a comparison format (clean accuracy 28.44 %) and is not screened unless `--include-comparison` is given |
| INT scale-word faults | **included** as an extension (INT8 and INT4), see below |
| Base seed / sampling | seed 0; k distinct elements without replacement, bit uniform within the group; one derived seed per trial (Stage 3 rule) |

## Configurations
A configuration is format x scope x fault group x k, run for 20 trials.

| scope | groups | k | configurations per format |
|---|---|---|---|
| global (all 11,164,352 weights) | float: sign, exponent, fraction, all. INT: sign, value, all (**no exponent group**) | 1, 8 | 8 (float) / 6 (INT) |
| layer-wise (one of 21 layers, all other layers fixed) | same | 1, 8 | 168 (float) / 126 (INT) |
| INT scale words, global (21 float32 scales) | sign, exponent, fraction, all | 1, 8 | 8 (INT only) |
| INT scale words, layer-wise (1 scale word per layer) | sign, exponent, fraction, all | **1 only** | 84 (INT only) |

Total: E4M11-SN, FP16, FP8 176 configurations each; INT8, INT4 224 each (132 weight + 92 scale); **976 configurations,
19,520 trials**. Skipped on purpose (recorded in the manifest): the `exponent` group for INT weights (no such field) and k = 8
for the layer-wise scale scope (a layer has one scale word, so 8 distinct words cannot be drawn).

The group `all` (every bit of the element) is added to the runbook's groups because it is the realistic random-flip case.

## Requirement coverage (runbook text -> where it is satisfied)
| runbook requirement | covered by |
|---|---|
| 6.1 First: model-level smoke tests on a small deterministic validation batch for each format and fault group | pre-flight in the script: one k = 8 global trial per format and fault group before the campaign; a restoration or flip-applied failure aborts (not written to the trial file). The Stage 3 smoke campaign also covered this |
| 6.1 Global screening, random encoded weights across all layers, k = 1 and 8 | `global` configurations, `global_fault_summary.csv` |
| 6.1 Layer-wise screening, one layer at a time, same budget and sampling policy | `layerwise` configurations: identical trials, k and sampling for every layer |
| 6.1 Float groups sign / exponent / fraction; integer sign and value, no exponent | `S3.SPECS` bit maps; the configuration list never creates an INT exponent group |
| 6.1 20 trials if runtime is high; more for worst / most promising | 20 trials; the campaign is resumable and can be extended with a larger `--trials` (new trial ids only) |
| 6.1 Validation results shortlist candidates; test set only for the final configurations | validation images only; the test set is not read in this stage |
| 6.2 Accuracy drop (pp), relative drop with pp | `drop_pp_*`, `rel_drop_percent_mean` |
| 6.2 Loss increase | `loss_increase_*` (trials with a finite fault loss; `n_loss_finite` reported) |
| 6.2 Failure rate, specifying which (weights, outputs, loss) | `failures`, `failure_rate`, and `fail_weights_nonfinite`, `fail_activations_nonfinite`, `fail_logits_nonfinite`, `fail_loss_nonfinite` |
| 6.2 Mean / median / std; p90 / worst per layer x format x group x k | `drop_pp_mean/median/std/p90/worst` (std with ddof = 1, p90 by linear interpolation) |
| 6.2 Robustness score only if explicitly defined | **not defined**; no composite score is computed |
| 6.3 Exact sample count and trial count for every aggregate | `n_trials` and `sample_count` in every summary row |
| 6.3 Do not compare 1,000-image validation with 10,000-image test | stated in the manifest; no test evaluation here |
| 6.3 Do not compare absolute drops across formats unless clean accuracy and subset match | `clean_accuracy` and `fault_accuracy_mean` in every row; the formats' clean accuracies on the 1,000 images differ (about 95.4 - 96.3 %), so formats are compared by fault accuracy and drop together |
| 6.3 Keep raw trial records; summaries regenerable | `fault_trials.jsonl`; `--summarize-only` regenerates all three CSVs byte-identically (checked) |
| 6.3 NaN/Inf counted as failure by the declared policy, never omitted or zeroed silently | Stage 3 convention: images with non-finite logits count as incorrect; loss is null; the trial is kept and flagged |
| 6.3 Resumable unique trial IDs; no trial duplicated or missing | id = `format|scope_kind|scope_name|group|k<k>|t<trial>`; completed ids are skipped on restart; `verify_completeness` checks duplicates, missing and unexpected ids; the exit code is non-zero if incomplete |

## Outputs (`results/<run>/jfamp/fault_screening/`)
| file | content |
|---|---|
| `fault_trials.jsonl` | one record per trial with every Section 5.3 field, the seed, the three hashes, and the flips as `[target, layer_id, element, bit_index]` (about 1.4 KB per record, about 27 MB for the full run) |
| `fault_summary.csv` | layer-wise aggregates: n, mean, median, std, p90, worst, sem, relative drop, loss increase, failure counts |
| `global_fault_summary.csv` | the same for the global scope (weights and scale words) |
| `layer_fault_heatmap.csv` | 21 layers x (format, scope kind, group, k) matrix of mean drop and p90 drop |
| `fault_campaign_manifest.json` | settings, per-format configuration and trial counts, clean accuracy per format, skipped configurations, completion check, failure counts by cause, sessions (resume history), software versions |
| `stage4_report.md` | short automatic overview |

## Gates checked at start
The script refuses to run unless (a) the checkpoint hash and the Stage 2 frozen-configuration hash match, (b) the Stage 3
validation was run on the same checkpoint and frozen configuration, and (c) Gate 3 is PASS for every chosen format. The encoded
weights are verified against their frozen content hashes before use.

## Expected cost
About 19,520 trials at roughly 0.4 s each (Stage 3 measured 0.39 s per trial including hashing) = **about 2 hours** on the RTX 3050.
Measured time per trial is written to the manifest.

## Run (GPU machine)
```bash
cd ~/Music/aia/ai-accelerator-project && git pull
export FR_ROOT=$HOME/fr_runs
nohup python -u scripts/stage4_fault_screening.py \
    --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch > stage4.log 2>&1 &
tail -f stage4.log          # Ctrl-C leaves the job running; re-running the same command resumes
```
Code: `src/fr/stage4.py`, `scripts/stage4_fault_screening.py`; tests `tests/test_stage4.py` (4 tests).
