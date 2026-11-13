"""Shared-spec Stage 1 inventory script (scripts/stage1_layer_inventory.py): both class counts, reproducibility, failure modes."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location("stage1inv", Path(__file__).resolve().parents[1] / "scripts" / "stage1_layer_inventory.py")
S1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S1)


def _ckpt(tmp_path, k, name=None):
    torch.manual_seed(k)
    p = tmp_path / (name or f"ck{k}.pth")
    torch.save({"epoch": 3, "model_state_dict": S1.build_model(k).state_dict()}, p)
    return p


@pytest.mark.parametrize("k,total,weights,biases", [(10, 2_236_682, 2_202_560, 10), (100, 2_351_972, 2_317_760, 100)])
def test_inventory_for_both_datasets(tmp_path, capsys, k, total, weights, biases):
    ck = _ckpt(tmp_path, k)
    out = tmp_path / "out"
    assert S1.main(["--checkpoint", str(ck), "--num-classes", str(k), "--out-dir", str(out)]) == 0
    summary = json.loads((out / "inventory_summary.json").read_text())
    m = summary["measured"]
    assert m["quantizable_layers"] == 53 and m["quantizable_weights"] == weights and m["quantizable_biases"] == biases
    assert m["total_parameters"] == m["trainable_parameters"] == total
    assert m["layer_type_counts"] == {"depthwise": 17, "pointwise": 34, "standard": 1, "linear": 1}
    assert summary["all_checks_passed"] and all(summary["validation_checks"].values()) and len(summary["validation_checks"]) == 8
    rows = list(csv.DictReader(open(out / "layer_inventory.csv", encoding="utf-8")))
    assert [r["layer_index"] for r in rows] == [str(i) for i in range(53)]
    assert rows[0]["module_name"] == "features.0.0" and rows[0]["layer_type"] == "standard" and rows[0]["stride"] == "[1, 1]"
    assert rows[-1]["module_name"] == "classifier.1" and rows[-1]["layer_type"] == "linear" and rows[-1]["weight_shape"] == f"[{k}, 1280]"
    assert rows[-1]["kernel_size"] == "" and rows[-1]["bias_count"] == str(k) and all(r["quantization_axis"] == "0" for r in rows)
    assert rows[1]["layer_type"] == "depthwise" and rows[1]["groups"] == rows[1]["in_channels"] == "32"
    assert list(rows[0]) == S1.CSV_FIELDS
    out_txt = capsys.readouterr().out
    assert "STAGE 1 COMPLETE" in out_txt and "[FAIL]" not in out_txt
    sums = dict(reversed(line.split("  ")) for line in (out / "sha256sums.txt").read_text().splitlines())
    assert set(sums) == {"layer_inventory.csv", "layer_inventory.json", "inventory_summary.json"}
    assert all(S1.sha256_file(out / f) == h for f, h in sums.items())


def test_outputs_are_byte_reproducible(tmp_path):
    ck = _ckpt(tmp_path, 10)
    for d in ("a", "b"):
        assert S1.main(["--checkpoint", str(ck), "--num-classes", "10", "--out-dir", str(tmp_path / d)]) == 0
    assert (tmp_path / "a" / "sha256sums.txt").read_bytes() == (tmp_path / "b" / "sha256sums.txt").read_bytes()


def test_spec_reference_comparison_reports_differences_instead_of_hiding_them(tmp_path):
    out100, out10 = tmp_path / "o100", tmp_path / "o10"
    S1.main(["--checkpoint", str(_ckpt(tmp_path, 100)), "--num-classes", "100", "--out-dir", str(out100)])
    S1.main(["--checkpoint", str(_ckpt(tmp_path, 10)), "--num-classes", "10", "--out-dir", str(out10)])
    c100 = json.loads((out100 / "inventory_summary.json").read_text())["comparison_with_spec_reference_values"]
    assert all(v["matches"] for v in c100.values())                                  # the spec's CIFAR-100 numbers are reproduced
    c10 = json.loads((out10 / "inventory_summary.json").read_text())["comparison_with_spec_reference_values"]
    assert {k for k, v in c10.items() if not v["matches"]} == {"total_parameters"}   # the spec's CIFAR-10 total is 90 too high
    assert c10["total_parameters"]["spec_value"] - c10["total_parameters"]["measured"] == 90


def test_checkpoint_of_the_other_dataset_fails_the_strict_load(tmp_path, capsys):
    ck = _ckpt(tmp_path, 100)
    out = tmp_path / "bad"
    assert S1.main(["--checkpoint", str(ck), "--num-classes", "10", "--out-dir", str(out)]) == 1
    text = capsys.readouterr().out
    assert "[FAIL] 1_strict_checkpoint_load" in text and "NOT COMPLETE" in text


def test_type_rule_and_stage_labels():
    assert [S1.stage_of(n) for n in ("features.0.0", "features.1.conv.0.0", "features.3.conv.2", "features.17.conv.1.0", "features.18.0", "classifier.1")] == \
        ["stem", "stage1", "stage2", "stage7", "head", "classifier"]
    net = S1.build_model(10)
    rows = S1.make_inventory(net)
    assert sum(r["num_weights"] for r in rows) == 2_202_560
    assert len({r["module_name"] for r in rows}) == 53
