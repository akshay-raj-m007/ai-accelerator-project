import itertools

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fr import allocation as A
from fr import data as fdata
from fr import e4m11 as E
from fr import formats8 as F
from fr import mixed as M
from fr import sensitivity as S
from fr.model import build_resnet18_cifar
from fr.quantize import conv_linear_weights, quantize_model


# ------------------------------------------------------------------------------------------ Fisher
def _tiny():
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(6, 3))


def _loader(n, seed=0, same=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 6, generator=g)
    if same:
        x = x[:1].repeat(n, 1)
    y = torch.randint(0, 3, (n,), generator=g)
    return DataLoader(TensorDataset(x.view(n, 6, 1, 1), y), batch_size=4)


def test_empirical_fisher_equals_mean_squared_per_sample_gradient():
    m = _tiny()
    loader = _loader(8)
    res = S.compute_fisher(m, loader, "cpu", kind="empirical")
    manual = torch.zeros_like(m[1].weight, dtype=torch.float64)
    for x, y in loader:
        for i in range(x.shape[0]):
            m.zero_grad()
            nn.functional.cross_entropy(m(x[i:i + 1]), y[i:i + 1]).backward()
            manual += m[1].weight.grad.double() ** 2
    np.testing.assert_allclose(res["fisher"]["1.weight"].double().numpy(), (manual / 8).numpy(), rtol=1e-5)
    assert res["n_samples"] == 8 and res["kind"] == "empirical"


def test_true_fisher_converges_to_exact_expectation_and_is_seeded():
    m = _tiny()
    loader = _loader(6000, same=True)                       # one input repeated: exact expectation is closed form
    x = loader.dataset.tensors[0][:1]
    p = torch.softmax(m(x), 1).detach()[0]
    exact = torch.zeros_like(m[1].weight, dtype=torch.float64)
    for c in range(3):
        m.zero_grad()
        nn.functional.cross_entropy(m(x), torch.tensor([c])).backward()
        exact += float(p[c]) * m[1].weight.grad.double() ** 2
    a = S.compute_fisher(m, loader, "cpu", kind="true", seed=0)["fisher"]["1.weight"].double()
    assert torch.allclose(a, exact, rtol=0.1, atol=1e-6)
    b = S.compute_fisher(m, loader, "cpu", kind="true", seed=0, max_samples=300)["fisher"]["1.weight"]
    b2 = S.compute_fisher(m, loader, "cpu", kind="true", seed=0, max_samples=300)["fisher"]["1.weight"]
    c = S.compute_fisher(m, loader, "cpu", kind="true", seed=1, max_samples=300)["fisher"]["1.weight"]
    assert torch.equal(b, b2) and not torch.equal(b, c)                              # seeded sampling


def test_fisher_nonnegative_state_restored_and_summary():
    m = _tiny()
    m.train()
    flags = [p.requires_grad for p in m.parameters()]
    res = S.compute_fisher(m, _loader(10), "cpu", kind="true", max_samples=7)
    assert res["n_samples"] == 7
    assert m.training and [p.requires_grad for p in m.parameters()] == flags
    f = res["fisher"]["1.weight"]
    assert (f >= 0).all() and f.shape == m[1].weight.shape
    s = S.layer_summary(res["fisher"])["1.weight"]
    assert s["count"] == 18 and s["trace"] == pytest.approx(float(f.double().sum()))
    assert s["trace_per_param"] == pytest.approx(s["trace"] / 18)
    with pytest.raises(ValueError):
        S.compute_fisher(m, _loader(4), "cpu", kind="bogus")


def test_fisher_save_load_roundtrip(tmp_path):
    res = S.compute_fisher(_tiny(), _loader(4), "cpu", kind="empirical")
    S.save_fisher(res, tmp_path / "f.pt")
    back = S.load_fisher(tmp_path / "f.pt")
    assert torch.equal(back["fisher"]["1.weight"], res["fisher"]["1.weight"]) and back["kind"] == "empirical"


# -------------------------------------------------------------------------------------- scoring
def test_normalize_variants():
    np.testing.assert_allclose(A.normalize([5.0, 1.0, 3.0], "rank"), [1.0, 0.0, 0.5])
    np.testing.assert_allclose(A.normalize([2.0, 2.0, 4.0], "rank"), [0.25, 0.25, 1.0])     # ties share a rank
    np.testing.assert_allclose(A.normalize([1.0, 2.0, 3.0], "minmax"), [0.0, 0.5, 1.0])
    z = A.normalize([1.0, 2.0, 3.0, 10.0], "zscore")
    assert abs(z.mean()) < 1e-12 and z.std() == pytest.approx(1.0)
    assert np.all(A.normalize([3.0, 3.0], "minmax") == 0)
    with pytest.raises(ValueError):
        A.normalize([1.0], "foo")


def _rows():
    return [{"name": f"l{i}", "count": 10 * (i + 1), "error": e, "fisher_per_param": fp, "fisher_total": ft,
             "damage": d} for i, (e, fp, ft, d) in
            enumerate([(0.1, 5.0, 50.0, 1.0), (0.4, 1.0, 20.0, 2.0), (0.2, 3.0, 90.0, 3.0), (0.3, 2.0, 10.0, 4.0)])]


def test_combined_score_endpoints_and_weighting():
    r = _rows()
    np.testing.assert_allclose(A.combined_score(r, 0.0), A.normalize([x["fisher_per_param"] for x in r]))
    np.testing.assert_allclose(A.combined_score(r, 1.0), A.normalize([x["error"] for x in r]))
    half = A.combined_score(r, 0.5)
    np.testing.assert_allclose(half, 0.5 * A.combined_score(r, 0.0) + 0.5 * A.combined_score(r, 1.0))
    tot = A.combined_score(r, 0.0, fisher_mode="total")
    np.testing.assert_allclose(tot, A.normalize([x["fisher_total"] for x in r]))
    with pytest.raises(ValueError):
        A.combined_score(r, 1.5)
    with pytest.raises(ValueError):
        A.combined_score(r, 0.5, fisher_mode="x")


def test_layer_table_matches_manual_computation():
    rng = np.random.default_rng(0)
    weights = {"a.weight": rng.standard_normal((4, 3)).astype(np.float32) * 0.1,
               "b.weight": rng.standard_normal((2, 5)).astype(np.float32) * 0.5}
    fisher = {k: rng.random(v.shape) for k, v in weights.items()}
    cache = {}
    rows = A.layer_table(weights, fisher, "int8_pt", cache)
    assert len(cache) == 2
    for row, (k, w) in zip(rows, weights.items()):
        deq = F.quantize8(w, "int8_pt")["dequantized"].astype(np.float64)
        d = w.astype(np.float64) - deq
        assert row["error"] == pytest.approx(np.sum(d ** 2) / np.sum(w.astype(np.float64) ** 2))
        assert row["damage"] == pytest.approx(np.sum(fisher[k] * d ** 2))
        assert row["fisher_total"] == pytest.approx(fisher[k].sum())
        assert row["fisher_per_param"] == pytest.approx(fisher[k].mean())
        assert row["count"] == w.size
    assert A.layer_table(weights, fisher, "int8_pt", cache) == rows            # cached path identical
    with pytest.raises(ValueError):
        A.layer_table(weights, {"b.weight": fisher["b.weight"], "a.weight": fisher["a.weight"]}, "int8_pt")


# ------------------------------------------------------------------------------------ selection
SIZES = [10, 20, 30, 40, 50, 60, 70]
NAMES = [f"l{i}" for i in range(7)]


def _brute(sizes, need, tol, costs):
    best = None
    feas = []
    for r in range(len(sizes) + 1):
        for comb in itertools.combinations(range(len(sizes)), r):
            s = sum(sizes[i] for i in comb)
            if need <= s <= need * (1 + tol) + 1:
                feas.append((sum(costs[i] for i in comb), comb))
    return feas


def test_required_switched():
    assert A.required_switched(1000, 16) == 0
    assert A.required_switched(1000, 8) == 1000
    assert A.required_switched(1000, 12) == 500
    assert A.required_switched(1001, 12) == 501                                  # rounds up: target is "<="
    with pytest.raises(ValueError):
        A.required_switched(1000, 7.9)


def test_min_cost_matches_brute_force_and_meets_budget():
    costs = [0.9, 0.1, 0.5, 0.3, 0.8, 0.2, 0.6]
    total = sum(SIZES)
    for target in (14.0, 12.0, 10.0):
        r = A.select("min_cost", NAMES, SIZES, target, costs=costs, tol=0.1)
        need = A.required_switched(total, target)
        feas = _brute(SIZES, need, r["tol_used"], costs)
        assert r["total_cost"] == pytest.approx(min(c for c, _ in feas))
        assert r["avg_bits"] <= target + 1e-9 and r["n_switched"] >= need
        assert r["n_feasible"] == len(feas)
        assert sum(SIZES[NAMES.index(n)] for n in r["layers8"]) == r["n_switched"]


def test_random_and_worst_use_the_same_feasible_set():
    dam = [0.9, 0.1, 0.5, 0.3, 0.8, 0.2, 0.6]
    base = A.select("min_cost", NAMES, SIZES, 12.0, costs=dam, tol=0.1)
    worst = A.select("worst", NAMES, SIZES, 12.0, damages=dam, tol=0.1)
    assert worst["total_cost"] >= base["total_cost"] and worst["n_feasible"] == base["n_feasible"]
    seen = set()
    for s in range(30):
        r = A.select("random", NAMES, SIZES, 12.0, tol=0.1, seed=s)
        assert r["need"] <= r["n_switched"] <= r["need"] * (1 + r["tol_used"]) + 1
        assert r == A.select("random", NAMES, SIZES, 12.0, tol=0.1, seed=s)        # reproducible
        seen.add(tuple(r["layers8"]))
    assert len(seen) > 1


def test_select_errors_and_tolerance_widening():
    with pytest.raises(ValueError):
        A.select("min_cost", NAMES, SIZES, 12.0)                                    # costs missing
    with pytest.raises(ValueError):
        A.select("nope", NAMES, SIZES, 12.0)
    r = A.select("min_cost", ["a", "b"], [100, 1000], 12.0, costs=[1.0, 2.0], tol=0.0)
    assert r["tol_used"] > 0 and r["layers8"]                                         # no exact fit -> widened
    assert A.select("min_cost", ["a"], [10], 16.0, costs=[1.0])["layers8"] == []       # nothing to switch
    assert len(A.select("min_cost", NAMES, SIZES, 8.0, costs=[1.0] * 7, tol=0.0)["layers8"]) == 7   # all layers
    with pytest.raises(ValueError):
        A.feasible_subsets(SIZES, sum(SIZES) + 1)                                   # unreachable target


def test_resnet18_scale_enumeration_is_fast_and_exact():
    m = build_resnet18_cifar().eval()
    w = conv_linear_weights(m)
    sizes = [v.size for v in w.values()]
    costs = np.random.default_rng(0).random(len(sizes))
    r = A.select("min_cost", list(w), sizes, 12.0, costs=costs)
    assert r["avg_bits"] <= 12.0 + 1e-9 and r["n_feasible"] >= 1
    # the cheapest feasible set is not worse than the greedy ascending-cost fill
    order = np.argsort(costs)
    need, acc, greedy = r["need"], 0, 0.0
    for i in order:
        if acc >= need:
            break
        acc += sizes[i]
        greedy += costs[i]
    if acc <= need * (1 + r["tol_used"]) + 1:
        assert r["total_cost"] <= greedy + 1e-9


# ----------------------------------------------------------------------------- mixed quantizer
@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return build_resnet18_cifar().eval()


def test_all_e4m11_equals_existing_quantizer(net):
    q0, b0 = quantize_model(net, "subnormal", "convlinear")
    q1, b1 = M.quantize_mixed(net, {})
    assert list(b0) == list(b1)
    for k in b0:
        assert np.array_equal(b0[k]["words"], b1[k]["words"])
        assert b0[k]["global_start"] == b1[k]["global_start"] and b1[k]["bits"] == 16
    for (n, a), (_, b) in zip(q0.state_dict().items(), q1.state_dict().items()):
        assert torch.equal(a, b), n


@pytest.mark.parametrize("fmt", F.FORMATS8)
def test_switched_layer_uses_8bit_codec_others_unchanged(net, fmt):
    layer = "layer4.0.conv1.weight"
    q, buf = M.quantize_mixed(net, {layer: fmt})
    w = conv_linear_weights(net)[layer]
    ref = F.quantize8(w, fmt)
    assert buf[layer]["format"] == fmt and buf[layer]["bits"] == 8
    assert buf[layer]["words"].dtype == np.uint8 and buf[layer]["count"] == 1_179_648
    got = dict(q.named_parameters())[layer].detach().numpy()
    np.testing.assert_array_equal(got, ref["dequantized"])
    q0, _ = quantize_model(net, "subnormal", "convlinear")
    for (n, a), (_, b) in zip(q0.state_dict().items(), q.state_dict().items()):
        if n != layer:
            assert torch.equal(a, b), n
    acc = M.bit_accounting(buf)
    N = 11_164_352
    assert acc["n_weights"] == N and acc["weights_by_format"] == {"e4m11": N - 1_179_648, fmt: 1_179_648}
    assert acc["avg_bits_payload"] == pytest.approx(16 - 8 * 1_179_648 / N)
    assert acc["metadata_bits"] == ref["metadata_bits"]
    starts = [b["global_start"] for b in buf.values()]
    assert starts[0] == 0 and all(b2 - b1 == c for b1, b2, c in
                                  zip(starts, starts[1:], [b["count"] for b in buf.values()]))


def test_dequantized_weights_roundtrip_and_cache(net):
    assign = {"layer1.0.conv1.weight": "int8_pc", "fc.weight": "fp8_scaled", "conv1.weight": "fp8"}
    cache = {}
    q, buf = M.quantize_mixed(net, assign, cache=cache)
    n_cached = len(cache)
    q2, buf2 = M.quantize_mixed(net, assign, cache=cache)
    assert len(cache) == n_cached
    deq = M.dequantized_weights(buf)
    for name, p in [(n, p) for n, p in q.named_parameters() if n in deq]:
        np.testing.assert_array_equal(p.detach().numpy(), deq[name])
    for k in buf:
        assert np.array_equal(buf[k]["words"], buf2[k]["words"])
    with pytest.raises(KeyError):
        M.quantize_mixed(net, {"nope.weight": "fp8"})
    with pytest.raises(ValueError):
        M.quantize_mixed(net, {"fc.weight": "int4"})


# ------------------------------------------------------------------------------------ data
def test_calibration_and_val_loaders_are_disjoint(monkeypatch):
    from torchvision import datasets

    def fake(root, train, transform, download=True):
        return datasets.FakeData(size=200, image_size=(3, 32, 32), num_classes=10, transform=transform,
                                 random_offset=0)
    monkeypatch.setattr(fdata, "_cifar10", fake)
    cal = fdata.get_calibration_loader("x", None, 50, 40, 0)
    val = fdata.get_val_loader("x", None, 20, 40, 0)
    tr_idx, va_idx = fdata.split_indices(200, 40, 0)
    assert list(cal.dataset.indices) == tr_idx[:50] and list(val.dataset.indices) == va_idx
    assert not set(cal.dataset.indices) & set(val.dataset.indices)
    assert len(fdata.get_val_loader("x", None, 20, 40, 0, subset=7).dataset) == 7
