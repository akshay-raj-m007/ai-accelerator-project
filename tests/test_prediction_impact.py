import importlib.util
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

SPEC = importlib.util.spec_from_file_location("pi", Path(__file__).resolve().parents[1] / "scripts" / "stage2_prediction_impact.py")
PI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PI)


def test_compute_impact_counts():
    fp = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3], [3, 0, 0]], dtype=np.float32)      # predictions 0,1,2,0
    q = np.array([[3, 0, 0], [0, 0, 3], [0, 0, 3], [0, 3, 0]], dtype=np.float32)       # predictions 0,2,2,1
    y = np.array([0, 1, 2, 1])                                                          # FP32 correct: 0,1,2 ; quantized correct: 0,2,3?
    r = PI.compute_impact(fp, q, y)
    assert r["changed_predictions"] == 2 and r["agreement_percent"] == 50.0
    assert r["correct_to_incorrect"] == 1 and r["incorrect_to_correct"] == 1 and r["net_change_correct"] == 0
    assert r["max_abs_logit_diff"] == 3.0 and abs(r["mean_abs_logit_diff"] - 12 / 12) < 1e-12
    same = PI.compute_impact(fp, fp, y)
    assert same["changed_predictions"] == 0 and same["agreement_percent"] == 100.0 and same["max_abs_logit_diff"] == 0.0
