#!/usr/bin/env python
"""JFAMP Stage 1 - MobileNetV2 layer inventory, written to the shared spec ("JFAMP - Stage 1: MobileNetV2 Layer Inventory").

Both owners run THIS script (CIFAR-10: --num-classes 10, CIFAR-100: --num-classes 100) on their own checkpoint:

  python scripts/stage1_layer_inventory.py --checkpoint $CK --num-classes 10 --out-dir reports/mobilenetv2_cifar10/stage_01_layer_inventory

It is deliberately independent of the rest of the project (torchvision + torch only): the model is built exactly as in the spec,
    model = mobilenet_v2(weights=None, num_classes=NUM_CLASSES); model.features[0][0].stride = (1, 1)
and the checkpoint is loaded with strict=True from checkpoint["model_state_dict"]. Inference only; the checkpoint is read, never written.

Writes <out-dir>/layer_inventory.csv, layer_inventory.json, inventory_summary.json (no timestamps or paths inside, so the files are
byte-reproducible) and sha256sums.txt, prints the eight required validation checks and the SHA-256 of the three artifacts.
Exit code 0 only if all eight checks pass. The numbers of the other owner's inventory are NOT inputs: expected values from the spec
are only compared against and any difference is reported (they never replace what is measured).
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

# Block index (features.N) -> torchvision stage, for the "Stage" column. Documented in inventory_summary.json.
STAGE_OF_BLOCK = {1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3, 7: 4, 8: 4, 9: 4, 10: 4, 11: 5, 12: 5, 13: 5, 14: 6, 15: 6, 16: 6, 17: 7}

# Values stated in the shared spec (section 5 / 6). VALIDATION REFERENCES ONLY.
SPEC_REFERENCE = {
    100: {"total_parameters": 2351972, "trainable_parameters": 2351972, "quantizable_layers": 53, "quantizable_weights": 2317760,
          "quantizable_biases": 100, "depthwise": 17, "pointwise": 34, "standard": 1, "linear": 1},
    10: {"total_parameters": 2236772, "quantizable_layers": 53, "quantizable_weights": 2202560, "quantizable_biases": 10,
         "depthwise": 17, "pointwise": 34, "standard": 1, "linear": 1},
}
CSV_FIELDS = ["layer_index", "module_name", "stage", "layer_type", "weight_key", "weight_shape", "weight_dtype", "num_weights",
              "bias_present", "bias_count", "in_channels", "out_channels", "kernel_size", "stride", "padding", "groups",
              "quantization_axis"]


def build_model(num_classes: int) -> nn.Module:
    model = mobilenet_v2(weights=None, num_classes=num_classes)
    model.features[0][0].stride = (1, 1)                       # CIFAR-adapted MobileNetV2 (spec section 2)
    return model


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def stage_of(name: str) -> str:
    if name.startswith("classifier"):
        return "classifier"
    idx = int(name.split(".")[1])
    return "stem" if idx == 0 else (f"stage{STAGE_OF_BLOCK[idx]}" if idx in STAGE_OF_BLOCK else "head")


def classify(m: nn.Module) -> str:
    """Spec section 4: depthwise = Conv2d with groups == in_channels; pointwise = Conv2d with kernel (1, 1); standard = other Conv2d; linear."""
    if isinstance(m, nn.Linear):
        return "linear"
    if m.groups == m.in_channels:
        return "depthwise"
    if tuple(m.kernel_size) == (1, 1):
        return "pointwise"
    return "standard"


def make_inventory(model: nn.Module) -> list:
    rows = []
    for name, m in model.named_modules():
        if not isinstance(m, (nn.Conv2d, nn.Linear)):
            continue
        conv = isinstance(m, nn.Conv2d)
        rows.append({
            "layer_index": len(rows), "module_name": name, "stage": stage_of(name), "layer_type": classify(m),
            "weight_key": f"{name}.weight", "weight_shape": list(m.weight.shape), "weight_dtype": str(m.weight.dtype).replace("torch.", ""),
            "num_weights": int(m.weight.numel()), "bias_present": m.bias is not None,
            "bias_count": int(m.bias.numel()) if m.bias is not None else 0,
            "in_channels": int(m.in_channels if conv else m.in_features), "out_channels": int(m.out_channels if conv else m.out_features),
            "kernel_size": list(m.kernel_size) if conv else None, "stride": list(m.stride) if conv else None,
            "padding": list(m.padding) if conv else None, "groups": int(m.groups) if conv else 1,
            "quantization_axis": 0})
    return rows


def validate(model: nn.Module, rows: list, num_classes: int, load_ok: bool, state_dict: dict) -> tuple:
    """The eight checks of spec section 7, each computed from the loaded model / checkpoint (nothing compared with hard-coded counts).
    Returns (checks, measured)."""
    with torch.inference_mode():
        out_shape = list(model.eval()(torch.zeros(1, 3, 32, 32)).shape)
    modules = [m for _, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    types = [r["layer_type"] for r in rows]
    counts = {t: types.count(t) for t in ("depthwise", "pointwise", "standard", "linear")}
    total_params = sum(p.numel() for p in model.parameters())
    inv_weights = sum(r["num_weights"] for r in rows)
    real_weights = sum(m.weight.numel() for m in modules)
    inv_biases = sum(r["bias_count"] for r in rows)
    param_names = {n for n, _ in model.named_parameters()}
    ckpt_param_elems = sum(int(v.numel()) for k, v in state_dict.items() if k in param_names)      # parameter tensors stored in the checkpoint
    bn_params = sum(p.numel() for m in model.modules() if isinstance(m, nn.BatchNorm2d) for p in m.parameters())
    return {
        "1_strict_checkpoint_load": bool(load_ok),
        "2_dummy_input_1x3x32x32_gives_1xNUM_CLASSES": out_shape == [1, num_classes],
        "3_exactly_as_many_entries_as_conv2d_linear_modules": len(rows) == len(modules),
        "4_inventory_weight_count_equals_model_weight_count": inv_weights == real_weights,
        "5_layer_names_unique": len({r["module_name"] for r in rows}) == len(rows),
        "6_every_weight_count_positive": all(r["num_weights"] > 0 for r in rows),
        "7_layer_type_counts_sum_to_number_of_layers": sum(counts.values()) == len(rows),
        "8_total_parameters_match_checkpoint_and_decompose_into_weights_biases_batchnorm":
            total_params == ckpt_param_elems and total_params == inv_weights + inv_biases + bn_params,
    }, {"output_shape_for_dummy_input": out_shape, "layer_type_counts": counts, "total_parameters": int(total_params),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "quantizable_layers": len(rows), "quantizable_weights": int(inv_weights), "quantizable_biases": int(inv_biases),
        "batchnorm_affine_parameters": int(bn_params), "parameter_elements_in_checkpoint": int(ckpt_param_elems)}


def write_outputs(out: Path, rows: list, summary: dict, ck_sha: str, num_classes: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "layer_inventory.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(CSV_FIELDS)
        for r in rows:
            w.writerow([json.dumps(r[k]) if isinstance(r[k], list) else ("" if r[k] is None else r[k]) for k in CSV_FIELDS])
    with open(out / "layer_inventory.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"model": "torchvision mobilenet_v2(weights=None), features[0][0].stride = (1, 1)", "num_classes": num_classes,
                   "checkpoint_sha256": ck_sha, "ordering": "model definition order (named_modules)", "layers": rows}, fh, indent=2)
        fh.write("\n")
    with open(out / "inventory_summary.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    hashes = {f: sha256_file(out / f) for f in ("layer_inventory.csv", "layer_inventory.json", "inventory_summary.json")}
    with open(out / "sha256sums.txt", "w", encoding="utf-8", newline="\n") as fh:
        for f, h in hashes.items():
            fh.write(f"{h}  {f}\n")
    return hashes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-classes", type=int, required=True, help="10 for CIFAR-10, 100 for CIFAR-100")
    ap.add_argument("--out-dir", default="experiments/stage_01_layer_inventory")
    args = ap.parse_args(argv)
    ck_path = Path(args.checkpoint)
    ck_sha = sha256_file(ck_path)

    model = build_model(args.num_classes)
    checkpoint = torch.load(ck_path, map_location="cpu", weights_only=False)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        load_ok, load_msg = True, "strict load succeeded"
    except Exception as e:                                   # shape / key mismatch (e.g. the other dataset's checkpoint)
        load_ok, load_msg = False, f"{type(e).__name__}: {str(e)[:300]}"
    rows = make_inventory(model)
    checks, measured = validate(model, rows, args.num_classes, load_ok, checkpoint["model_state_dict"])

    ref = SPEC_REFERENCE.get(args.num_classes, {})
    comparison = {k: {"spec_value": v, "measured": measured.get(k, measured["layer_type_counts"].get(k)),
                      "matches": v == measured.get(k, measured["layer_type_counts"].get(k))} for k, v in ref.items()}
    summary = {
        "stage": "JFAMP Stage 1 - MobileNetV2 layer inventory", "num_classes": args.num_classes,
        "checkpoint": {"file_name": ck_path.name, "size_bytes": ck_path.stat().st_size, "sha256": ck_sha,
                       "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None, "load": load_msg},
        "architecture": "torchvision.models.mobilenet_v2(weights=None, num_classes=NUM_CLASSES); features[0][0].stride = (1, 1)",
        "stage_column_definition": "stem = features.0; stage1..stage7 = the 7 torchvision inverted-residual stages (features.1 | 2-3 | 4-6 | 7-10 | "
                                   "11-13 | 14-16 | 17); head = features.18 (final 1x1 conv); classifier = classifier.1",
        "layer_type_rule": "depthwise: Conv2d with groups == in_channels; pointwise: Conv2d with kernel (1, 1); standard: other Conv2d; linear: Linear",
        "quantization_axis_note": "0 (output-channel axis) is a proposed convention, to be checked against the actual quantizer in a later stage",
        "measured": measured, "validation_checks": checks, "all_checks_passed": all(checks.values()),
        "comparison_with_spec_reference_values": comparison,
        "comparison_note": "spec values are references only; any 'matches': false is reported, not forced",
        "weights_by_layer_type": {t: sum(r["num_weights"] for r in rows if r["layer_type"] == t) for t in ("depthwise", "pointwise", "standard", "linear")},
    }
    hashes = write_outputs(Path(args.out_dir), rows, summary, ck_sha, args.num_classes)

    print(f"checkpoint: {ck_path}  ({summary['checkpoint']['size_bytes']} bytes)  sha256 {ck_sha}")
    print(f"strict load: {load_msg}")
    print(f"measured: {json.dumps(measured)}")
    print("\nvalidation checks (spec section 7):")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("\ncomparison with the spec's reference values (informational):")
    for k, v in comparison.items():
        print(f"  {'match   ' if v['matches'] else 'DIFFERS '} {k}: spec {v['spec_value']:,}  measured {v['measured']:,}")
    print("\nSHA256:")
    for f, h in hashes.items():
        print(f"  {h}  {f}")
    print(f"\nwritten: {Path(args.out_dir).resolve()}")
    ok = summary["all_checks_passed"]
    print("STAGE 1 " + ("COMPLETE: all checks passed" if ok else "NOT COMPLETE: a check failed"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
