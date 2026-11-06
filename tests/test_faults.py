import csv
import json

import numpy as np
import pytest
import torch

from fr import e4m11 as E
from fr import faults as F
from fr.model import build_resnet18_cifar
from fr.quantize import quantize_model


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    m = build_resnet18_cifar().eval()
    q, buf = quantize_model(m, "subnormal", "convlinear")
    return m, q, buf


def _injector(setup, mode="subnormal"):
    import copy
    m, _, _ = setup
    q, buf = quantize_model(m, mode, "convlinear")
    return F.FaultInjector(copy.deepcopy(q), buf, mode, 7, "clamp", "cpu")


def test_paper_group_bits():
    assert F.PAPER_GROUPS["exponent"] == (11, 12, 13, 14)
    assert F.PAPER_GROUPS["sign"] == (15,)
    assert F.PAPER_GROUPS["mantissa_msb"] == (6, 7, 8, 9, 10)
    assert F.PAPER_GROUPS["mantissa_lsb"] == (0, 1, 2, 3, 4, 5)
    all_bits = sorted(b for g in F.PAPER_GROUPS.values() for b in g)
    assert all_bits == list(range(16))                      # disjoint and complete
    assert F.parse_group("bit14")[1] == (14,)
    with pytest.raises(ValueError):
        F.parse_group("bit16")


def test_draws_are_uniform_in_group_and_reproducible():
    rng = F.trial_rng(0, "exponent", 8, 0)
    idx, bits = F.draw_flips(rng, 0, 1000, 20000, F.PAPER_GROUPS["exponent"])
    assert set(np.unique(bits)) == {11, 12, 13, 14}
    counts = np.bincount(bits, minlength=15)[11:15]
    assert counts.min() > 0.22 * 20000 and counts.max() < 0.28 * 20000
    assert idx.min() >= 0 and idx.max() < 1000 and len(np.unique(idx)) < len(idx)   # with replacement
    a = F.draw_flips(F.trial_rng(5, "sign", 3, 7), 0, 10**7, 3, (15,))
    b = F.draw_flips(F.trial_rng(5, "sign", 3, 7), 0, 10**7, 3, (15,))
    c = F.draw_flips(F.trial_rng(5, "sign", 3, 8), 0, 10**7, 3, (15,))
    assert np.array_equal(a[0], b[0]) and not np.array_equal(a[0], c[0])


def test_duplicate_flips_xor_together():
    idx = np.array([5, 5, 9, 5])
    bits = np.array([3, 3, 14, 4])
    uniq, mask = F.combine_flips(idx, bits)
    assert list(uniq) == [5, 9]
    assert int(mask[0]) == (1 << 4)                        # bit 3 flipped twice cancels, bit 4 stays
    assert int(mask[1]) == (1 << 14)


@pytest.mark.parametrize("bit,expected", [(11, 6.0), (12, 12.0), (13, 48.0), (14, 0.0), (15, -3.0), (10, 2.0)])
def test_day5_values_through_injector(setup, bit, expected):
    inj = _injector(setup, "legacy")
    with torch.no_grad():
        pass
    # put a known 3.0 into the first weight of the first layer's buffer
    inj.words[0] = E.encode(np.float32(3.0), "legacy")
    inj.flat[0][0] = 3.0
    inj.inject(np.array([0]), np.array([bit]))
    assert float(inj.flat[0][0]) == expected
    inj.restore()
    assert float(inj.flat[0][0]) == 3.0


def test_injector_matches_naive_full_buffer_decode_and_restores(setup):
    inj = _injector(setup, "subnormal")
    rng = np.random.default_rng(0)
    for group in ("exponent", "sign", "mantissa_msb", "any"):
        idx, bits = F.draw_flips(rng, 0, inj.total, 8, F.parse_group(group)[1])
        # naive: corrupt a copy of the whole buffer, decode everything
        corrupted = inj.words.copy()
        for i, b in zip(idx, bits):
            corrupted[i] ^= np.uint16(1 << int(b))
        naive = E.decode(corrupted, "subnormal")
        inj.inject(idx, bits)
        got = torch.cat([f.detach().cpu() for f in inj.flat]).numpy()
        np.testing.assert_array_equal(got, naive)
        inj.restore()
        assert inj.clean_weights_unchanged()


def test_inject_requires_restore(setup):
    inj = _injector(setup)
    inj.inject(np.array([1]), np.array([14]))
    with pytest.raises(RuntimeError):
        inj.inject(np.array([2]), np.array([14]))
    inj.restore()


def test_layer_range_and_locate(setup):
    inj = _injector(setup)
    lo, hi = inj.layer_range("layer4.0.conv1")
    assert hi - lo == 1_179_648
    assert inj.layer_range(None) == (0, 11_164_352)
    li, loc = inj.locate(np.array([lo, hi - 1]))
    assert inj.names[li[0]] == "layer4.0.conv1.weight" and list(loc) == [0, 1_179_647]


def test_sign_group_flips_only_sign(setup):
    inj = _injector(setup)
    idx = np.array([100, 2000, 500000])
    clean = np.array([float(inj.flat[inj.locate(np.array([i]))[0][0]][inj.locate(np.array([i]))[1][0]]) for i in idx])
    inj.inject(idx, np.array([15, 15, 15]))
    new = np.array([float(inj.flat[inj.locate(np.array([i]))[0][0]][inj.locate(np.array([i]))[1][0]]) for i in idx])
    inj.restore()
    np.testing.assert_array_equal(new, -clean)


def test_campaign_end_to_end_and_resume(setup, tmp_path):
    inj = _injector(setup)
    X = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, 10, (32,))
    out = tmp_path / "c"
    p = F.run_campaign(inj, X, y, out, ["exponent", "sign"], [1, 2], 3, seed=7, log=lambda *_: None)
    rows = list(csv.DictReader(open(p)))
    assert len(rows) == 2 * 2 * 3
    assert {r["group"] for r in rows} == {"exponent", "sign"}
    # resume: nothing new is written, results unchanged
    F.run_campaign(inj, X, y, out, ["exponent", "sign"], [1, 2], 3, seed=7, log=lambda *_: None)
    assert len(list(csv.DictReader(open(p)))) == 12
    # extending trials only adds the missing ones
    F.run_campaign(inj, X, y, out, ["exponent", "sign"], [1, 2], 4, seed=7, log=lambda *_: None)
    assert len(list(csv.DictReader(open(p)))) == 16
    # incompatible resume is refused
    with pytest.raises(RuntimeError):
        F.run_campaign(inj, X, y, out, ["exponent"], [1], 1, seed=8, log=lambda *_: None)
    meta = json.load(open(out / "campaign_meta.json"))
    s = F.summarize(p, meta["clean"]["accuracy_percent"])
    assert len(s) == 4 and all(r["n_trials"] >= 3 for r in s)
    assert "| exponent | 1 |" in F.render_summary_md(s, meta, None, None)
    assert inj.clean_weights_unchanged()
