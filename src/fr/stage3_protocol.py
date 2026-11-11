"""Content of fault_protocol.json (J-FAMP runbook Stage 3, Section 5.1)."""
from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Optional, Sequence

from . import stage3 as S3


def build_protocol(*, run_id: str, checkpoint_sha: str, stage2_frozen: str, layers: Sequence[dict], base_seed: int,
                   n_images: int, trials_screening: int, smoke_trials: int, seconds_per_eval: Optional[float],
                   codec_versions: Mapping[str, str]) -> dict:
    formats = OrderedDict()
    for fmt, sp in S3.SPECS.items():
        groups = {g: list(S3.group_bits(fmt, g)) for g in S3.format_groups(fmt)}
        d = {"role": "comparison" if fmt == "e4m11_ftz" else "primary", "kind": sp["kind"], "bits_per_element": sp["width"],
             "stored_unit": "packed byte (two INT4 elements)" if fmt == "int4_pt" else f"{sp['width']}-bit word",
             "bit_numbering": "0 = least significant bit of the element's stored word",
             "groups": groups, "fields": {k: list(v) for k, v in sp["fields"].items()},
             "codec_version": codec_versions.get(fmt)}
        if sp["kind"] == "float":
            d["layout"] = f"S(1) E({sp['e_bits']}) M({sp['m_bits']}), exponent bias {sp['bias']}"
            d["groups_missing"] = []
        else:
            b = sp["width"]
            d["layout"] = f"{b}-bit two's complement code, value = code * float32 scale"
            d["groups_missing"] = ["exponent (integer formats have no exponent field; no 'exponent' group is created)"]
            d["sign_bit_interpretation"] = (
                f"bit {b - 1} is the two's complement MSB with weight -{1 << (b - 1)}: flipping it changes the code by "
                f"+/-{1 << (b - 1)} (not a plain sign change); value bits 0..{b - 2} have weights +1..+{1 << (b - 2)}")
            d["scale_metadata"] = ("one float32 per tensor (per-tensor scale); optional scope 'scales' with bit groups "
                                   + str({k: list(v) for k, v in S3.SCALE_FIELDS.items()}))
        if fmt == "int4_pt":
            d["packing"] = {
                "element_to_stored_bit": "element e, bit b -> byte e // 2, bit (e % 2) * 4 + b",
                "nibble_order": "element 2j in the LOW nibble (bits 0-3) of byte j, element 2j+1 in the HIGH nibble (bits 4-7)",
                "byte_alignment": "one byte holds two elements; tensor payloads are byte aligned",
                "padding": "an odd element count leaves one zero padding nibble (high nibble of the last byte); it is never a "
                           "fault target (all 21 ResNet-18 tensors have even counts, so none occurs)",
                "verified_by": "fault_injection_smoke_tests.csv (every element x bit of a 7-element tensor, plus the padding test)"}
        formats[fmt] = d
    n_total = sum(int(r["num_weights"]) for r in layers)
    est = None
    if seconds_per_eval:
        prim = [f for f in S3.SPECS if f != "e4m11_ftz"]
        n_cfg = sum(len([g for g in S3.format_groups(f) if S3.group_bits(f, g)]) for f in prim) * len(layers) * 2
        est = {"configurations_layerwise_primary_formats": n_cfg, "trials_per_configuration": trials_screening,
               "evaluations": n_cfg * trials_screening, "seconds_per_evaluation_measured": seconds_per_eval,
               "estimated_hours": n_cfg * trials_screening * seconds_per_eval / 3600.0,
               "note": "layer-wise screening of the five primary formats over all groups (incl. 'all'), k in {1, 8}; "
                       "excludes global screening"}
    return {
        "stage": "J-FAMP runbook Stage 3 - fault-injection protocol (Section 5.1)",
        "label": "SCREENING protocol (not paper-grade statistical coverage)",
        "run_id": run_id, "checkpoint_sha256": checkpoint_sha, "stage2_frozen_config_sha256": stage2_frozen,
        "injector_version": S3.INJECTOR_VERSION,
        "fault_model": "random bit flips (DRAM disturbance) applied to the STORED ENCODED representation of the weights; weights "
                       "are decoded from the faulted stored bytes only; FP32 source values are never modified",
        "formats": formats,
        "target_scope": {
            "primary": "Conv2d/Linear weight words, either one layer at a time (layer-wise) or all %d weights (global)" % n_total,
            "layers": [{"layer_id": int(r["layer_id"]), "layer_name": r["module_name"], "n_elements": int(r["num_weights"])}
                       for r in layers],
            "total_elements": n_total,
            "not_in_scope": "biases, BatchNorm parameters and buffers (kept FP32, weight-only study)",
            "optional_extension_scope": "'scales': the float32 per-tensor scale of INT formats (not part of the runbook screening; "
                                        "implemented and validated, off by default)"},
        "fault_counts_k": [1, 8],
        "sampling_policy": {
            "locations": "k DISTINCT elements per trial, drawn uniformly WITHOUT replacement from the scope (recorded policy; the "
                         "paper's wording draws with replacement, immaterial for k<=8)",
            "bit": "uniformly at random within the target group, independently per struck element",
            "independence": "one seed per trial derived from (base seed, format, scope, scope kind, group, k, trial) with numpy "
                            "SeedSequence; the seed is stored in every trial record",
            "application": "all k flips applied simultaneously before decoding; distinct elements, so no element is hit twice"},
        "seeds": {"base_seed": base_seed,
                  "derivation": "seed = SeedSequence([base_seed, crc32(format), crc32(scope name), crc32(scope kind), "
                                "crc32(group), k, trial]).generate_state(1)[0] % 2**63"},
        "trial_budget": {"screening_trials_per_layer_format_group": f"{trials_screening} (runbook: 20-30; 20 if runtime is high)",
                         "smoke_trials_per_configuration_in_this_stage": smoke_trials, "estimate": est},
        "evaluation": {"images": f"first {n_images} images of the validation holdout (deterministic, held out from training)",
                       "sample_count": n_images, "precision": "FP32 compute on weights decoded from the stored bytes",
                       "note": "never compare this screen with a 10,000-image test result as the same protocol"},
        "non_finite_policy": {
            "detection": "decoded weights (all Conv/Linear weights), activations (outputs of every Conv/Linear/BatchNorm module), "
                         "logits and loss",
            "accuracy_convention": "an image whose logits contain NaN/Inf counts as incorrect",
            "loss_convention": "fault_loss is null when any logit is non-finite",
            "never_dropped": "the trial is kept with output_flag set (weights_nonfinite, activations_nonfinite, logits_nonfinite, "
                             "loss_nonfinite) and failure=true"},
        "restoration_policy": "the stored bytes (and scales) are restored after every trial, also on errors; SHA-256 of the whole "
                              "encoded state is stored before the fault, during the fault and after restoration",
        "trial_record_fields": list(S3.REQUIRED_TRIAL_FIELDS) + ["scope_kind", "requested_k", "failure", "encoded_hash_faulted",
                                                                 "restored_exactly", "flip_applied_to_encoded_bytes", "flips", "sampling"],
        "validation_suite": ["round-trip exactness and stable bytes", "bit mapping of every bit of a synthetic tensor",
                             "special values (inf/NaN, reserved exponent, subnormal/FTZ, two's complement sign bit)",
                             "INT4 packing and padding", "INT scale bit mapping", "scope", "no-fault control", "restoration",
                             "non-finite handling"],
        "codec_versions": dict(codec_versions)}
