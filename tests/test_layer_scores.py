import numpy as np
import pytest
import torch

from fr import faults_mixed as FM
from fr import layer_scores as LS
from fr import mixed
from fr.model import build_resnet18_cifar


@pytest.fixture(scope="module")
def inj():
    torch.manual_seed(0)
    net = build_resnet18_cifar().eval()
    q, buf = mixed.quantize_mixed(net, {})
    return FM.MixedFaultInjector(q, buf)


def test_rank_and_spearman():
    assert LS.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert LS.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert list(LS.rank([5, 5, 1])) == [1.5, 1.5, 0.0]
    assert np.isnan(LS.spearman([1, 1, 1], [1, 2, 3]))


def test_predicted_moments_match_brute_force(inj):
    F = {n: torch.rand(tuple(inj.shapes[i])) for i, n in enumerate(inj.names)}
    res = LS.predicted_flip_moments(inj, F, "sign")
    l = 0
    w = inj._decode(l, inj.words[l], np.arange(inj.counts[l])).astype(np.float64)
    d = inj._decode(l, inj.words[l] ^ np.uint16(1 << 15), np.arange(inj.counts[l])).astype(np.float64) - w
    assert res[l]["S2n"] == pytest.approx(0.5 * (d * d).mean())
    assert res[l]["S2"] == pytest.approx(0.5 * (F[inj.names[l]].reshape(-1).double().numpy() * d * d).mean())
    assert res[l]["n_bits"] == 1


def test_exponent_flips_cost_more_than_sign_flips(inj):
    F = {n: torch.ones(tuple(inj.shapes[i])) for i, n in enumerate(inj.names)}
    e = LS.predicted_flip_moments(inj, F, "exponent")
    s = LS.predicted_flip_moments(inj, F, "sign")
    assert all(a["S2n"] > b["S2n"] for a, b in zip(e, s))


def test_measured_damage_runs_and_restores(inj):
    X = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    rows = LS.measured_layer_damage(inj, X, y, "exponent", 4, 2, 0, log=lambda *_: None)
    assert len(rows) == len(inj.names) and all(r["trials"] == 2 for r in rows)
    assert inj.clean_weights_unchanged()
