import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fr import churn as C
from fr import sensitivity as S
from fr import stability as ST
from fr.model import build_resnet18_cifar
from fr.quantize import conv_linear_weights


# ------------------------------------------------------------------------------------- statistics
def test_rankdata_spearman_jaccard():
    np.testing.assert_allclose(ST.rankdata([3.0, 1.0, 1.0, 2.0]), [4.0, 1.5, 1.5, 3.0])
    assert ST.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert ST.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(ST.spearman([1, 1, 1], [1, 2, 3]))
    sz = {"a": 10, "b": 30, "c": 60}
    assert ST.weighted_jaccard(["a", "b"], ["b", "c"], sz) == pytest.approx(30 / 100)
    assert ST.weighted_jaccard([], [], sz) == 1.0


# ------------------------------------------------------------------------------------------ churn
def test_churn_metrics_counts_and_kl():
    ref = torch.tensor([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0], [5.0, 0, 0]])
    cur = ref.clone()
    cur[0] = torch.tensor([0, 5.0, 0])          # image 0: class 0 -> 1
    cur[3] = torch.tensor([0, 0, 5.0])          # image 3: class 0 -> 2
    y = torch.tensor([0, 1, 2, 2])              # ref: right, right, right, wrong
    m = C.churn_metrics(ref, cur, y)
    assert m["flips"] == 2 and m["c2w"] == 1 and m["w2c"] == 1       # img0 right->wrong, img3 wrong->right
    assert m["acc"] == pytest.approx(75.0)                           # net accuracy change hides the 2 flips
    assert m["kl"] > 0 and m["mean_abs_dlogit"] > 0
    z = C.churn_metrics(ref, ref, y)
    assert z["flips"] == 0 and z["kl"] == pytest.approx(0.0, abs=1e-12) and z["mean_abs_dlogit"] == 0.0
    # KL(p||q) against a hand computation for one row
    p = torch.softmax(torch.tensor([[1.0, 2.0, 0.5]]).double(), 1)
    q = torch.softmax(torch.tensor([[0.0, 1.0, 1.0]]).double(), 1)
    expect = float((p * (p.log() - q.log())).sum())
    got = C.churn_metrics(torch.tensor([[1.0, 2.0, 0.5]]), torch.tensor([[0.0, 1.0, 1.0]]), torch.tensor([0]))["kl"]
    assert got == pytest.approx(expect, rel=1e-6)


# ------------------------------------------------------------------------------------ per-sample traces
def _tiny_loader(n=10):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 6, 1, 1, generator=g)
    y = torch.randint(0, 3, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=4)


def test_per_sample_traces_match_fisher_trace():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Flatten(), nn.Linear(6, 3))
    loader = _tiny_loader()
    r = S.per_sample_layer_traces(m, loader, "cpu", kind="empirical")
    f = S.compute_fisher(m, loader, "cpu", kind="empirical")
    assert r["traces"].shape == (10, 1) and r["names"] == ["1.weight"] and r["counts"][0] == 18
    assert r["traces"].mean() == pytest.approx(float(f["fisher"]["1.weight"].double().sum()), rel=1e-5)
    assert r["nll"].shape == (10,) and r["nll"].mean() == pytest.approx(f["mean_nll"], rel=1e-6)
    with torch.no_grad():
        pred = m(torch.cat([x for x, _ in loader])).argmax(1)
    labels = torch.cat([y for _, y in loader])
    assert np.array_equal(r["correct"], (pred == labels).numpy())
    t = S.per_sample_layer_traces(m, loader, "cpu", kind="true", seed=3)
    t2 = S.per_sample_layer_traces(m, loader, "cpu", kind="true", seed=3)
    assert np.array_equal(t["traces"], t2["traces"])                  # seeded


# ----------------------------------------------------------------------------------- stability analysis
@pytest.fixture(scope="module")
def layer_info():
    m = build_resnet18_cifar()
    w = conv_linear_weights(m)
    return list(w), np.array([v.size for v in w.values()], dtype=np.int64)


def _pool(names, counts, n, noise, seed=0, heavy=False):
    rng = np.random.default_rng(seed)
    per_param = np.geomspace(1e-6, 1e-9, len(names))                    # decreases with depth
    base = per_param * counts
    T = base[None, :] * np.exp(noise * rng.standard_normal((n, len(names))))
    if heavy:
        T = T * 1e-6
        T[0] = base * 1e6                                                # one image carries almost everything
    return {"names": names, "counts": counts, "traces": T, "nll": rng.random(n),
            "correct": np.ones(n, dtype=bool)}


def test_stability_low_noise_is_stable_and_homogeneous(layer_info):
    names, counts = layer_info
    r = ST.analyze_pool(_pool(names, counts, 256, noise=0.05), sizes=(16, 64, 256))
    big = [s for s in r["sizes"] if s["n"] == 64][0]
    assert big["spearman_between_chunks_mean"] > 0.99 and big["mean_cv_per_layer"] < 0.1
    c = r["concentration"]
    assert c["ess_per_layer"]["median"] > 0.9 * 256
    assert c["top1pct_share_all_layers"] < 0.05
    for s in r["selection"]:
        assert s["jaccard_vs_full_mean"] == 1.0 and s["identical_to_full"] == s["chunks"]
    assert r["per_param_vs_total_spearman"] < 1.0                       # the two metrics do not rank alike


def test_stability_detects_heavy_tail(layer_info):
    names, counts = layer_info
    r = ST.analyze_pool(_pool(names, counts, 200, noise=0.0, heavy=True), sizes=(50,))
    c = r["concentration"]
    assert c["ess_per_layer"]["max"] < 1.1                              # effectively one image
    assert c["top1pct_share_all_layers"] > 0.99


def test_compare_pools(layer_info):
    names, counts = layer_info
    a = _pool(names, counts, 64, noise=0.05)
    out = ST.compare_pools(a, a)
    assert out["spearman_per_param"] == pytest.approx(1.0) and all(s["jaccard"] == 1.0 for s in out["selection"])
    b = dict(a)
    b["traces"] = a["traces"][:, ::-1] / counts[::-1][None, :] * counts[None, :]   # reversed sensitivity order
    rev = ST.compare_pools(a, b)
    assert rev["spearman_per_param"] < 0.0
    md = ST.render_markdown({"train": ST.analyze_pool(a, sizes=(16, 64)), "val": ST.analyze_pool(b, sizes=(16, 64))}, rev)
    assert "Training images vs held-out images" in md and "Fisher-only selection stability" in md
    with pytest.raises(ValueError):
        ST.compare_pools(a, {**a, "names": names[::-1]})
