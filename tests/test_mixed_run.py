"""End-to-end CPU test of fisher -> sweep (validation) -> final (test) on FakeData and a random-init ResNet-18."""
import csv
import json

import numpy as np
import pytest
import torch
from torchvision import datasets

from fr import data as fdata
from fr import mixed_run as R
from fr.config import Config
from fr.model import build_resnet18_cifar


@pytest.fixture()
def env(tmp_path, monkeypatch):
    def fake(root, train, transform, download=True):
        return datasets.FakeData(size=120 if train else 40, image_size=(3, 32, 32), num_classes=10,
                                 transform=transform, random_offset=0 if train else 7)
    monkeypatch.setattr(fdata, "_cifar10", fake)
    torch.manual_seed(0)
    m = build_resnet18_cifar().eval()
    ck = tmp_path / "ck.pth"
    torch.save({"model_state_dict": m.state_dict(),
                "config": {"normalization": None, "val_size": 30, "seed": 0}}, ck)
    cfg = Config(root=str(tmp_path), run_name="t", device="cpu", num_workers=0, eval_batch_size=20, seed=0)
    return cfg, ck


def test_pipeline_end_to_end(env):
    cfg, ck = env
    summ = R.run_fisher(cfg, ck, ("true", "empirical"), calib_size=12)
    assert set(summ["kinds"]) == {"true", "empirical"} and summ["val_size"] == 30
    fpath = cfg.results_dir() / "sensitivity" / "fisher_true.pt"
    assert fpath.exists() and (cfg.results_dir() / "sensitivity" / "fisher_empirical_layers.json").exists()

    out = R.run_sweep(cfg, ck, fpath, arms=("int8_pt", "fp8_scaled"), targets=(12.0,), lambdas=(0.0, 1.0),
                      random_draws=1, tol=0.05)
    rows = list(csv.DictReader(open(out["csv"])))
    ids = [r["config_id"] for r in rows]
    assert len(ids) == len(set(ids))
    # 3 references + (e4m11, 2 arms all-8bit) + 2 arms x (2 lambdas + product + worst + 1 random)
    assert {"baseline|fp32", "baseline|e4m11_all", "baseline|int8_pt_all", "baseline|fp8_scaled_all"} <= set(ids)
    sweep = [r for r in rows if r["method"] != "baseline"]
    assert len(sweep) == 2 * 5
    for r in sweep:
        assert float(r["avg_bits_payload"]) <= 12.0 + 1e-9 and 0 < int(r["n_layers8"]) < 21
    sel = json.load(open(out["selections"]))
    assert set(sel) == {r["config_id"] for r in sweep}

    # the validation sweep must not have created any test result
    assert not (cfg.results_dir() / "mixed" / "mixed_test.csv").exists()

    # resume: nothing is recomputed or duplicated
    R.run_sweep(cfg, ck, fpath, arms=("int8_pt", "fp8_scaled"), targets=(12.0,), lambdas=(0.0, 1.0),
                random_draws=1, tol=0.05)
    assert len(list(csv.DictReader(open(out["csv"])))) == len(rows)

    fin = R.run_final(cfg, ck)
    test = list(csv.DictReader(open(fin["csv"])))
    assert {r["config_id"] for r in test} == set(ids)          # lambda in {0,1} are both end points here
    for r in test:
        if r["method"] != "baseline":
            assert r["layers8"] == ";".join(sorted(sel[r["config_id"]]["layers8"]))   # same sets as validation
    md = open(fin["report"], encoding="utf-8").read()
    assert "## int8_pt, target <= 12 bits/weight" in md and "lambda (best on val)" in md
    assert "random (n=1)" in md

    # prediction churn: reference row is exactly zero, resumable, test limited to the final-phase configs
    ch = R.run_churn(cfg, ck)
    base = cfg.results_dir() / "mixed"
    val_ch = list(csv.DictReader(open(base / "mixed_churn_val.csv")))
    test_ch = list(csv.DictReader(open(base / "mixed_churn_test.csv")))
    assert {r["config_id"] for r in val_ch} == set(ids) and {r["config_id"] for r in test_ch} == set(ids)
    ref = [r for r in val_ch if r["config_id"] == "baseline|e4m11_all"][0]
    assert int(ref["flips"]) == 0 and float(ref["kl"]) < 1e-12
    assert all(int(r["flips"]) >= int(r["c2w"]) and int(r["flips"]) >= int(r["w2c"]) for r in val_ch)
    assert "Prediction churn" in open(ch["report"], encoding="utf-8").read()
    R.run_churn(cfg, ck)
    assert len(list(csv.DictReader(open(base / "mixed_churn_val.csv")))) == len(val_ch)


def test_run_stability_end_to_end(env):
    cfg, ck = env
    out = R.run_stability(cfg, ck, ("train", "val"), pool=24, kind="true", sizes=(4, 8, 24))
    d = cfg.results_dir() / "sensitivity"
    assert (d / "traces_train_true.npz").exists() and (d / "traces_val_true.npz").exists()
    md = open(out["report"], encoding="utf-8").read()
    assert "## Source: train (24 images" in md and "## Source: val (24 images" in md
    assert "Training images vs held-out images" in md
    z = np.load(d / "traces_train_true.npz")
    assert z["traces"].shape == (24, 21) and (z["traces"] >= 0).all()
    mtime = (d / "traces_train_true.npz").stat().st_mtime_ns
    R.run_stability(cfg, ck, ("train", "val"), pool=24, kind="true", sizes=(4, 8, 24))
    assert (d / "traces_train_true.npz").stat().st_mtime_ns == mtime             # cached traces are reused


def test_pick_lambda_best_tie_breaks():
    rows = [{"config_id": f"a|T12|lambda|{l}", "arm": "a", "target_avg_bits": "12", "method": "lambda",
             "param": str(l), "acc": acc, "ce": ce}
            for l, acc, ce in [(0.0, "90.0", "0.5"), (0.5, "91.0", "0.6"), (1.0, "91.0", "0.4")]]
    assert R.pick_lambda_best(rows)[("a", 12.0)] == "a|T12|lambda|1.0"         # equal acc -> lower CE
    rows[2]["ce"] = "0.6"
    assert R.pick_lambda_best(rows)[("a", 12.0)] == "a|T12|lambda|0.5"       # equal acc and CE -> lower lambda
