# Phase 6 - final analysis and presentation package (runbook Stage 6, Section 8)

Reference: `JFAMP_ResNet18_CIFAR10_Experimental_Flow.pdf` Section 8. Previous phases: [phase1](../phase1/README.md) ... [phase5](../phase5/README.md), results [phase5/RESULTS_ROUND2.md](../phase5/RESULTS_ROUND2.md).

Stage 6 runs no experiment. `scripts/stage6_final_package.py` rebuilds every final table, figure and the summary from the saved raw records and manifests of Stages 1-5 (CPU, seconds, no model, no test data):

```
python scripts/stage6_final_package.py --base reports/scratch/jfamp --model-tag resnet18_cifar10
```

## What it produces (under `--base`)
| runbook item | output |
|---|---|
| 8.1 FP32 vs each format: accuracy, loss, storage, compression | `final_tables/<tag>_format_comparison.csv` |
| 8.1 per-format fault robustness: mean, tail, loss increase, non-finite rate | `final_tables/<tag>_fault_robustness.csv` |
| 8.1 layer sensitivity and whether risk aligns with size, weight distribution, quantization error | `final_tables/<tag>_layer_sensitivity.csv`, `<tag>_risk_alignment.csv` (Spearman, p-values, n = 21 layers) |
| 8.1 datatype-only vs datatype + protection (only the implemented `scale_tmr`) | `final_tables/<tag>_protection_pairs.csv` |
| 8.1 validation-derived selection vs final test | `final_tables/<tag>_validation_vs_test.csv` |
| trade-off data incl. Pareto flags | `final_tables/<tag>_tradeoff_assignments.csv` |
| 8.2 Fig. 1-6 (baseline, storage, SQNR, fault robustness, layer heatmap, trade-off) | `figures/<tag>_fig1_...` to `fig6_...` |
| extra figures: risk alignment, risk vs layer size, protection pairs, validation vs test | `figures/<tag>_fig7_...` to `fig10_...` |
| 8.3 final summary with all required statements | `final_summary.md` |
| reproducibility | `final_manifest.json` (SHA-256 of the 48 input files and of the tables and summary) |

The data folders themselves (`baseline/`, `layer_profile/`, `quantization/`, `fault_injection/`, `fault_screening/`, `optimization/`) already follow the runbook's handoff layout under
`reports/<run>/jfamp/`; the Stage 6 outputs are added next to them. Rebuilding twice gives byte-identical tables and summary (checked); PNG bytes can differ between matplotlib versions.

## Naming for more than one model
Every figure and table carries the model tag as prefix (`resnet18_cifar10_fig6_tradeoff_pareto.png`). For the next model run the pipeline with another `--run-name`, then
`python scripts/stage6_final_package.py --base reports/<new-run>/jfamp --model-tag <new-tag>`. The earlier-stage plot scripts take the same `--model-tag` and write to
`plots_preview/<model-tag>/stage4|stage5/<model-tag>_<name>.png` (`plot_stage4_summary.py`, `plot_stage5_summary.py`, `plot_feature_maps.py`); the previous figures were moved there.
`plots_preview/` is excluded from git on this machine (`.git/info/exclude`); the Stage 6 figures live in the tracked `reports/` tree.

## Findings recorded in the package (ResNet-18 / CIFAR-10)
- Frozen assignment A128 (19 INT8 + `scale_tmr`, 1 INT4 + `scale_tmr`, 1 E4M11-SN; 7.58 bit/weight): test accuracy 95.20 % vs FP32 95.11 % (read once), no measurable loss; scale-word fault 0.0 pp.
- Layer risk (k = 8 flips in one layer) falls with layer size and depth and rises with weight std; SQNR does not explain it (positive association, opposite to "more error, more risk"). These are associations: size, std and depth are mutually correlated.
- Limits as in `final_summary.md` section 7 (screening depth, only `scale_tmr` implemented, one checkpoint, weights only).

Code: `scripts/stage6_final_package.py`; tests `tests/test_stage6.py`.
