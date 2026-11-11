import numpy as np
import pytest
import torch

from fr import baseline_profile as BP
from fr import dist_plots as DP
from fr.model import build_resnet18_cifar


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return build_resnet18_cifar().eval()


def test_inventory_reconciles_with_the_model(net):
    rows = BP.layer_inventory(net)
    assert len(rows) == 21 and [r["layer_id"] for r in rows] == list(range(21))
    assert sum(r["num_weights"] for r in rows) == 11_164_352
    assert all(BP.reconcile(rows, net).values())
    assert {r["layer_group"] for r in rows} == {"stem", "layer1", "layer2", "layer3", "layer4", "classifier"}
    ps = BP.parameter_summary(net, rows)
    assert ps["total_parameters"] == 11_173_962 and ps["quantizable_plus_nonquantized_equals_total"]
    assert ps["fp32_quantizable_bytes"] == 44_657_408 and ps["fp32_quantizable_bytes_equals_4x_weights"]
    assert BP.storage_summary(rows)[-1]["num_weights"] == 11_164_352
    r0 = rows[0]
    assert r0["abs_p50"] <= r0["abs_p90"] <= r0["abs_p99"] <= r0["abs_p99.9"] <= r0["weight_abs_max"]


def test_evaluate_full_confusion_and_per_class(net):
    X = torch.randn(40, 3, 32, 32)
    y = torch.randint(0, 10, (40,))
    r = BP.evaluate_full(net, X, y, batch_size=16)
    assert r["total"] == 40 and r["confusion"].sum() == 40 and r["confusion"].shape == (10, 10)
    assert r["correct"] == int(np.trace(r["confusion"]))
    assert sum(c["support"] for c in r["per_class"]) == 40 and r["all_logits_finite"]
    with pytest.raises(RuntimeError):
        BP.evaluate_full(build_resnet18_cifar().train(), X, y)


def test_checkpoint_checks(net):
    sd = net.state_dict()
    c = BP.checkpoint_checks(net, {}, sd, {"n_quantizable_layers": 21, "n_quantizable_weights": 11164352})
    assert c["strict_load"] and c["output_dim_is_10"] and c["stem_is_3x3_stride1"] and c["no_maxpool"]
    assert all(v["matches"] for v in c["comparison_with_runbook_reference"].values())
    bad = dict(sd)
    bad.pop("fc.bias")
    assert not BP.checkpoint_checks(net, {}, bad, {})["strict_load"]


def test_sha256_helpers(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert BP.sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert BP.sha256_ints([1, 2, 3]) == BP.sha256_ints([1, 2, 3]) != BP.sha256_ints([1, 2, 4])


def test_activation_stats_and_plot_pages(net, tmp_path):
    pytest.importorskip("matplotlib")
    X = torch.randn(8, 3, 32, 32)
    for point in ("input", "output"):
        st = DP.activation_stats(net, X, point, batch_size=4, max_samples=20_000)
        assert len(st) == 21
        s0 = st["conv1"]
        if point == "input":
            assert s0["n"] == 8 * 3 * 32 * 32 and s0["min"] <= s0["mean"] <= s0["max"]
            assert abs(s0["mean"] - X.mean().item()) < 1e-4
        assert all(s["sample_size"] <= 20_000 for s in st.values())
    ws = DP.weight_stats(net)
    pngs = DP.plot_pages(ws, "weights", "t", tmp_path / "w.pdf", tmp_path / "w", bins=30)
    assert len(pngs) == 3 and (tmp_path / "w.pdf").stat().st_size > 1000   # 21 layers / 8 per page
