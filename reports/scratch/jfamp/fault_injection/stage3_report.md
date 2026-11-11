# Stage 3 - fault-injection protocol validation

run `20261111T073720Z-35b172e9`; checkpoint `c612da6a4b20c34f...`; Stage 2 frozen config `387ff9741de4a14d...`
validation images for the model-level tests: 1000; synthetic bit cases: 497

| format | role | tests passed | Gate 3 |
|---|---|---|---|
| e4m11_sn | primary | 21/21 | **PASS** |
| fp16 | primary | 21/21 | **PASS** |
| fp8_e4m3fn | primary | 21/21 | **PASS** |
| int8_pt | primary | 22/22 | **PASS** |
| int4_pt | primary | 23/23 | **PASS** |
| e4m11_ftz | comparison | 21/21 | **PASS** |

Overall (primary formats): **PASS**
Measured mean time per trial (incl. hashing): 0.39 s

