import importlib.util
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
SPEC = importlib.util.spec_from_file_location("s3lw", Path(__file__).resolve().parents[1] / "scripts" / "stage3_layerwise_sensitivity.py")
S3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S3)
from fr.model import build_mobilenetv2_cifar, quantizable_modules  # noqa: E402


def test_layer_types_follow_the_shared_rule():
    c = {}
    for _, m in quantizable_modules(build_mobilenetv2_cifar(10)):
        c[S3.layer_type(m)] = c.get(S3.layer_type(m), 0) + 1
    assert c == {"depthwise": 17, "pointwise": 34, "standard": 1, "linear": 1}


def test_metrics_and_summaries():
    lg0 = torch.tensor([[3., 0, 0], [0, 3, 0], [0, 0, 3], [3, 0, 0]])
    lg = torch.tensor([[3., 0, 0], [0, 0, 3], [0, 0, 3], [0, 3, 0]])
    y = torch.tensor([0, 1, 2, 1])
    m = S3.metrics(lg, lg0, y)
    assert m["changed_predictions"] == 2 and m["prediction_agreement_percent"] == 50.0
    assert m["correct_to_wrong"] == 1 and m["wrong_to_correct"] == 1 and m["correct"] == 3
    rows = [{"layer_name": "a", "layer_type": "depthwise", "format_key": "f", "accuracy_change_pp": -2.0, "changed_predictions": 10,
             "prediction_agreement_percent": 99.8, "weight_sqnr_db": 20.0, "payload_bytes": 5},
            {"layer_name": "b", "layer_type": "depthwise", "format_key": "f", "accuracy_change_pp": 0.1, "changed_predictions": 2,
             "prediction_agreement_percent": 99.96, "weight_sqnr_db": 40.0, "payload_bytes": 7}]
    s = S3.summarise(rows, lambda r: (r["layer_type"], r["format_key"]), ["layer_type", "format_key"])[0]
    assert s["n_layers"] == 2 and s["worst_layer"] == "a" and s["best_layer"] == "b" and s["layers_with_drop_gt_1pp"] == 1
    assert s["layers_with_improvement"] == 1 and s["total_payload_bytes"] == 12 and abs(s["mean_accuracy_change_pp"] + 0.95) < 1e-12
