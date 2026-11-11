import numpy as np
import pytest
import torch

from fr import stage5 as S5


def test_exact_fisher_matches_the_definition_on_a_tiny_model():
    import torch.nn as nn
    from fr import sensitivity as SE
    torch.manual_seed(0)
    net = nn.Sequential(nn.Flatten(), nn.Linear(6, 4, bias=False), nn.ReLU(), nn.Linear(4, 3, bias=False)).eval()
    X = torch.randn(5, 6)
    y = torch.randint(0, 3, (5,))
    ld = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=2)
    res = SE.compute_fisher_exact(net, ld, "cpu", min_prob=0.0)
    params = [p for _, p in SE.weight_params(net)]
    ref = [torch.zeros_like(p, dtype=torch.float64) for p in params]
    for i in range(5):
        lp = torch.log_softmax(net(X[i:i + 1]), 1)
        for c in range(3):
            g = torch.autograd.grad(-lp[0, c], params, retain_graph=True)
            for r, gi in zip(ref, g):
                r += float(lp[0, c].exp()) * gi.double() ** 2
    for f, r in zip(res["fisher"].values(), ref):
        assert np.allclose(f.double().numpy(), (r / 5).numpy(), rtol=1e-5, atol=1e-12)
    assert res["kind"] == "exact" and res["n_samples"] == 5 and res["n_backward"] == 15
    res2 = SE.compute_fisher_exact(net, ld, "cpu", min_prob=0.3)           # skipping classes below min_prob: fewer passes, neglected mass reported
    assert res2["n_backward"] < 15 and res2["max_neglected_probability_mass_per_image"] > 0


def _problem(seed, L=4, D=3):
    rng = np.random.default_rng(seed)
    Q = rng.random((L, D)) * np.array([1.0, 0.3, 5.0])
    R = rng.random((L, D)) * np.array([2.0, 5.0, 0.1])
    S = np.tile(np.array([16.0, 8.0, 4.0]), (L, 1)) * rng.integers(1, 5, size=(L, 1))
    Rs = rng.random((L, D)) * np.array([0.0, 0.0, 3.0])
    return S5.Problem([f"l{i}" for i in range(L)], ["a", "b", "c"], Q, R, S, Rscale=Rs)


def test_normalize_minmax_and_rank():
    v = [5.0, 1.0, 3.0, 3.0]
    assert np.allclose(S5.normalize(v, "minmax"), [1.0, 0.0, 0.5, 0.5])
    r = S5.normalize(v, "rank")
    assert r[1] == 0.0 and r[0] == 1.0 and r[2] == r[3]
    assert np.all(S5.normalize([2.0, 2.0], "minmax") == 0)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("norm", ["minmax", "rank"])
@pytest.mark.parametrize("risk", ["weights", "plus_scale"])
def test_milp_equals_brute_force(seed, norm, risk):
    p = _problem(seed)
    for lam, B, tau, eps in [(0.0, None, None, None), (1.0, 40.0, None, None), (2.0, 50.0, 2.0, None), (0.5, 60.0, 1.5, 3.0)]:
        bf = S5.brute_force(p, lam, norm, risk, B, tau, eps)
        sol = S5.solve(p, lam, norm, risk, B, tau, eps)
        if bf is None:
            assert sol["status"] == "infeasible"
        else:
            assert sol["status"] == "optimal" and sol["objective"] == pytest.approx(bf[0], abs=1e-7)


def test_infeasible_budget_is_reported():
    p = _problem(0)
    assert S5.solve(p, 1.0, max_bytes=1.0)["status"] == "infeasible"


def test_fisher_q_matches_definition_and_checks_shapes():
    w = np.array([[0.5, -0.25], [0.125, 0.0]])
    wh = np.array([[0.5, -0.3], [0.1, 0.0]])
    f = np.array([[1.0, 2.0], [4.0, 8.0]])
    assert S5.fisher_q(w, wh, f) == pytest.approx(2.0 * 0.05 ** 2 + 4.0 * 0.025 ** 2)
    with pytest.raises(ValueError):
        S5.fisher_q(w, wh, np.ones(3))


def test_risk_columns_and_missing_row():
    rows = [{"format": "int8_pt", "layer_name": "conv1", "scope_kind": "weights", "fault_group": "all", "k": "8", "drop_pp_mean": "0.4", "drop_pp_p90": "1",
             "drop_pp_worst": "2", "failure_rate": "0", "n_trials": "20", "sample_count": "1000"},
            {"format": "int8_pt", "layer_name": "conv1", "scope_kind": "scales", "fault_group": "all", "k": "1", "drop_pp_mean": "30", "drop_pp_p90": "1",
             "drop_pp_worst": "2", "failure_rate": "0", "n_trials": "20", "sample_count": "1000"}]
    rk = S5.risk_lookup(rows)
    c = S5.risk_columns(rk, "int8_pt", "conv1")
    assert c["R_any_k8_mean_drop_pp"] == 0.4 and c["R_scale_any_k1_mean_drop_pp"] == 30 and c["has_scale_word"] and c["R_n_trials"] == 20
    with pytest.raises(KeyError):
        S5.risk_columns(rk, "fp16", "conv1")
