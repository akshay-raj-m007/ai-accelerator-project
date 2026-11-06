import math

import numpy as np
import pytest
import torch

from fr import e4m11 as E
from fr.model import build_resnet18_cifar, load_model, quantizable_modules
from fr.quantize import all_parameters, conv_linear_weights, quantize_model, total_words
from fr.sqnr import BIN_EDGES, sqnr_db, sqnr_report


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_resnet18_cifar(10).eval()


def test_model_structure(model):
    assert model.conv1.kernel_size == (3, 3) and model.conv1.stride == (1, 1)
    assert model.conv1.padding == (1, 1) and model.conv1.bias is None
    assert isinstance(model.maxpool, torch.nn.Identity)
    assert model.fc.out_features == 10
    assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 10)
    mods = quantizable_modules(model)
    assert len(mods) == 21
    assert sum(m.weight.numel() for _, m in mods) == 11_164_352


def test_checkpoint_strict_load(model, tmp_path):
    # same layout as the legacy checkpoint
    path = tmp_path / "legacy.pth"
    torch.save({"epoch": 3, "model_state_dict": model.state_dict(), "optimizer_state_dict": {},
                "train_accuracy": 1.0, "average_loss": 1.0, "config": {}}, path)
    m2, ck = load_model(path)
    for (k, a), (_, b) in zip(model.state_dict().items(), m2.state_dict().items()):
        assert torch.equal(a, b), k
    bad = {k: v for k, v in model.state_dict().items() if k != "fc.bias"}
    torch.save({"model_state_dict": bad}, tmp_path / "bad.pth")
    with pytest.raises(RuntimeError):
        load_model(tmp_path / "bad.pth")


def test_quantize_convlinear_scope(model):
    q, buf = quantize_model(model, "paper", "convlinear")
    assert len(buf) == 21 and total_words(buf) == 11_164_352
    starts = [b["global_start"] for b in buf.values()]
    assert starts[0] == 0 and starts == sorted(starts)
    for name, b in buf.items():
        assert b["words"].dtype == np.uint16 and b["words"].size == b["count"]
        assert b["global_start"] + b["count"] <= 11_164_352
    # BN parameters and fc bias untouched; conv/linear weights equal decoded buffers
    sd0, sd1 = model.state_dict(), q.state_dict()
    for k in sd0:
        is_cl = k.endswith(".weight") and (k in {f"{n}.weight" for n, _ in quantizable_modules(model)})
        if not is_cl:
            assert torch.equal(sd0[k], sd1[k]), k
    w = conv_linear_weights(q)
    for name, b in buf.items():
        np.testing.assert_array_equal(w[name], E.decode(b["words"]).reshape(b["shape"]))


def test_quantize_all_scope_changes_bn(model):
    m = build_resnet18_cifar(10).eval()
    with torch.no_grad():
        m.bn1.weight.fill_(0.7771)          # not representable
        m.fc.bias.fill_(0.1234)
    q, buf = quantize_model(m, "legacy", "all")
    assert len(buf) == len(list(m.named_parameters()))
    assert not torch.equal(m.bn1.weight, q.bn1.weight)
    assert not torch.equal(m.fc.bias, q.fc.bias)


def test_sqnr_math():
    assert sqnr_db(1.0, 0.0) == math.inf
    assert sqnr_db(100.0, 1.0) == pytest.approx(20.0)
    rng = np.random.default_rng(0)
    w = {"a": rng.normal(0, 0.3, (64, 32)).astype(np.float32), "b": rng.normal(0, 0.05, 1000).astype(np.float32)}
    h = {k: v + 0.01 for k, v in w.items()}
    r = sqnr_report(w, h)
    sig = sum(float((v.astype(np.float64) ** 2).sum()) for v in w.values())
    err = sum(float(((v.astype(np.float64) - h[k].astype(np.float64)) ** 2).sum()) for k, v in w.items())
    assert r["overall"]["sqnr_db"] == pytest.approx(10 * math.log10(sig / err), abs=1e-9)
    assert sum(b["count"] for b in r["breakdown"]) == r["overall"]["count"]
    assert sum(b["signal_power"] for b in r["breakdown"]) == pytest.approx(sig)
    assert sum(b["error_power"] for b in r["breakdown"]) == pytest.approx(err)
    assert set(r["per_layer"]) == {"a", "b"}
    # lossless -> inf
    assert sqnr_report(w, w)["overall"]["sqnr_db"] == math.inf


def test_sqnr_bins_follow_flush_rule():
    lo, mid = BIN_EDGES
    w = {"x": np.array([lo * 0.5, -lo * 0.5, lo * 1.2, mid * 1.5, 1.0], np.float32)}
    q = {"x": E.quantize_dequantize(w["x"], "paper")}
    r = sqnr_report(w, q)
    b0, b1, b2 = r["breakdown"]
    assert (b0["count"], b1["count"], b2["count"]) == (2, 1, 2)
    # flushed bin: error power == signal power -> 0 dB
    assert b0["sqnr_db"] == pytest.approx(0.0, abs=1e-6)


def test_sqnr_of_normal_range_weights_is_high():
    # magnitudes entirely in the normal range: only 11-bit rounding noise remains (>= ~77 dB)
    rng = np.random.default_rng(0)
    w = (rng.uniform(2.0 ** -5, 4.0, 200_000) * rng.choice([-1, 1], 200_000)).astype(np.float32)
    r = sqnr_report({"w": w}, {"w": E.quantize_dequantize(w)})
    assert r["overall"]["sqnr_db"] > 76.0


def test_sqnr_is_limited_by_flush_for_small_gaussian_weights():
    # documents WHY SQNR depends on the weight distribution: N(0, 0.02) loses mass below 2^-7
    w = np.random.default_rng(0).normal(0, 0.02, 200_000).astype(np.float32)
    r = sqnr_report({"w": w}, {"w": E.quantize_dequantize(w)})
    assert r["overall"]["sqnr_db"] < 40.0
    assert r["breakdown"][0]["fraction_of_error_power"] > 0.5


def test_subnormal_mode_removes_flush_loss_for_small_weights():
    w = np.random.default_rng(0).normal(0, 0.0064, 300_000).astype(np.float32)
    p = sqnr_report({"w": w}, {"w": E.quantize_dequantize(w, "paper")})["overall"]["sqnr_db"]
    s = sqnr_report({"w": w}, {"w": E.quantize_dequantize(w, "subnormal")})["overall"]["sqnr_db"]
    assert p < 10 and s > 60
