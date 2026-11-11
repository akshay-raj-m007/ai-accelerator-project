# Stage 4 - fault screening (validation images, screening protocol)

run `20261111T083515Z-fff093af`; 19520 of 19520 expected trials recorded; completeness: **COMPLETE** (duplicates 0, missing 0)

## Global scope (all weights), mean accuracy drop in pp (p90) over 20 trials

| format | group | k | clean % | mean drop | p90 | worst | failures |
|---|---|---|---|---|---|---|---|
| e4m11_sn | all | 1 | 96.30 | 0.00 | 0.00 | 0.00 | 0/20 |
| e4m11_sn | all | 8 | 96.30 | 0.02 | 0.00 | 0.60 | 0/20 |
| e4m11_sn | exponent | 1 | 96.30 | 0.00 | 0.00 | 0.10 | 0/20 |
| e4m11_sn | exponent | 8 | 96.30 | 0.11 | 0.52 | 1.20 | 0/20 |
| e4m11_sn | fraction | 1 | 96.30 | 0.00 | 0.00 | 0.00 | 0/20 |
| e4m11_sn | fraction | 8 | 96.30 | 0.00 | 0.00 | 0.00 | 0/20 |
| e4m11_sn | sign | 1 | 96.30 | 0.00 | 0.00 | 0.00 | 0/20 |
| e4m11_sn | sign | 8 | 96.30 | 0.00 | 0.00 | 0.00 | 0/20 |
| fp16 | all | 1 | 96.30 | 0.00 | 0.00 | 0.00 | 0/20 |
| fp16 | all | 8 | 96.30 | 6.91 | 21.58 | 81.90 | 0/20 |
| fp16 | exponent | 1 | 96.30 | 8.12 | 32.10 | 71.80 | 0/20 |
| fp16 | exponent | 8 | 96.30 | 22.06 | 81.73 | 86.40 | 0/20 |
| fp16 | fraction | 1 | 96.30 | 0.00 | 0.00 | 0.00 | 0/20 |
| fp16 | fraction | 8 | 96.30 | 0.00 | 0.00 | 0.00 | 0/20 |
| fp16 | sign | 1 | 96.30 | 0.00 | 0.00 | 0.00 | 0/20 |
| fp16 | sign | 8 | 96.30 | 0.00 | 0.00 | 0.00 | 0/20 |
| fp8_e4m3fn | all | 1 | 96.00 | 0.00 | 0.00 | 0.00 | 0/20 |
| fp8_e4m3fn | all | 8 | 96.00 | -0.02 | 0.20 | 0.30 | 0/20 |
| fp8_e4m3fn | exponent | 1 | 96.00 | -0.00 | 0.00 | 0.00 | 0/20 |
| fp8_e4m3fn | exponent | 8 | 96.00 | 0.10 | 0.03 | 3.00 | 0/20 |
| fp8_e4m3fn | fraction | 1 | 96.00 | 0.00 | 0.00 | 0.00 | 0/20 |
| fp8_e4m3fn | fraction | 8 | 96.00 | 0.00 | 0.00 | 0.00 | 0/20 |
| fp8_e4m3fn | sign | 1 | 96.00 | 0.00 | 0.00 | 0.00 | 0/20 |
| fp8_e4m3fn | sign | 8 | 96.00 | -0.00 | 0.00 | 0.00 | 0/20 |
| int4_pt | all | 1 | 95.40 | 0.00 | 0.00 | 0.00 | 0/20 |
| int4_pt | all | 8 | 95.40 | 0.00 | 0.00 | 0.00 | 0/20 |
| int4_pt | sign | 1 | 95.40 | 0.00 | 0.00 | 0.00 | 0/20 |
| int4_pt | sign | 8 | 95.40 | 0.00 | 0.00 | 0.10 | 0/20 |
| int4_pt | value | 1 | 95.40 | 0.00 | 0.00 | 0.00 | 0/20 |
| int4_pt | value | 8 | 95.40 | 0.00 | 0.00 | 0.00 | 0/20 |
| int8_pt | all | 1 | 96.30 | 0.00 | 0.00 | 0.00 | 0/20 |
| int8_pt | all | 8 | 96.30 | 0.00 | 0.00 | 0.10 | 0/20 |
| int8_pt | sign | 1 | 96.30 | 0.00 | 0.00 | 0.00 | 0/20 |
| int8_pt | sign | 8 | 96.30 | 0.00 | 0.00 | 0.10 | 0/20 |
| int8_pt | value | 1 | 96.30 | 0.00 | 0.00 | 0.00 | 0/20 |
| int8_pt | value | 8 | 96.30 | 0.00 | 0.00 | 0.00 | 0/20 |

Full tables: `global_fault_summary.csv`, `fault_summary.csv`, `layer_fault_heatmap.csv`; raw records: `fault_trials.jsonl`.
