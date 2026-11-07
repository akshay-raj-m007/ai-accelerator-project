import copy
import csv

import numpy as np
import pytest
import torch

from fr import faults_mixed as FM
from fr import mixed
from fr.model import build_resnet18_cifar

LAYER = "layer1.0.conv1.weight"      # 64 output channels, 36,864 weights
ALL_PC = {"layer1.0.conv1.weight": "int8_pc", "layer1.1.conv1.weight": "int8_pc", "fc.weight": "int8_pc"}


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return build_resnet18_cifar().eval()


def _inj(net, assignment, scale_mode="on"):
    q, buf = mixed.quantize_mixed(net, assignment)
    return FM.MixedFaultInjector(copy.deepcopy(q), buf, scale_mode=scale_mode), buf


def _layer(inj, name=LAYER):
    return inj.names.index(name)


def _flat(inj, name=LAYER):
    return inj.flat[_layer(inj, name)].detach().clone()


def test_scale_group_bits_partition_float32():
    bits = [b for g in FM.PAPER_GROUP_NAMES for b in FM.scale_group_bits(g)]
    assert sorted(bits) == list(range(32))
    assert FM.scale_group_bits("any") == tuple(range(32))
    assert FM.scale_group_bits("bit31") == (31,) and FM.scale_group_bits("bit32") == ()
    assert FM.scale_group_bits("exponent") == tuple(range(23, 31))
    with pytest.raises(ValueError):
        FM.scale_group_bits("nope")


def test_scale_counts_and_storage(net):
    inj, buf = _inj(net, {n: "int8_pc" for n in ALL_PC} | {"layer1.0.conv2.weight": "int8_pt"})
    assert inj.scale_u32[_layer(inj)].size == 64                       # one float32 per output channel
    assert inj.scale_u32[_layer(inj, "layer1.0.conv2.weight")].size == 1   # one per tensor
    assert inj.n_scale_words == 64 + 64 + 10 + 1
    assert sum(b["metadata_bits"] for b in buf.values()) == 32 * inj.n_scale_words


def test_off_mode_matches_the_plain_draw(net):
    inj, _ = _inj(net, ALL_PC, "off")
    a = inj.draw(np.random.default_rng(1), "any", 50, "bits")
    b = inj.draw_with_scales(np.random.default_rng(1), "any", 50, "bits")
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and all(x.size == 0 for x in b[2:])


def test_on_mode_keeps_the_weight_strikes_of_mode_off(net):
    off, _ = _inj(net, ALL_PC, "off")
    on, _ = _inj(net, ALL_PC, "on")
    k = 400
    wi, wb = off.draw(np.random.default_rng(3), "any", k, "bits")
    idx, bit, sl, sw, sb = on.draw_with_scales(np.random.default_rng(3), "any", k, "bits")
    assert idx.size + sl.size == k                                     # k strikes in total
    assert set(zip(idx.tolist(), bit.tolist())) <= set(zip(wi.tolist(), wb.tolist()))
    assert np.all(sb < 32) and np.all(sw >= 0)


def test_on_mode_hit_rate_equals_scale_share_of_memory(net):
    inj, _ = _inj(net, ALL_PC, "on")
    S = 32 * inj.n_scale_words
    B = inj.total_bits
    k = 400_000
    _, _, sl, _, _ = inj.draw_with_scales(np.random.default_rng(0), "any", k, "bits")
    p = S / (B + S)
    assert abs(sl.size - k * p) < 4 * np.sqrt(k * p * (1 - p))


def test_scale_mode_needs_bit_sampling_and_int_formats(net):
    inj, _ = _inj(net, ALL_PC, "on")
    with pytest.raises(ValueError):
        inj.draw_with_scales(np.random.default_rng(0), "any", 5, "weights")
    q, buf = mixed.quantize_mixed(net, {"layer1.0.conv1.weight": "fp8_scaled"})
    with pytest.raises(ValueError):
        FM.MixedFaultInjector(q, buf, scale_mode="on")


def test_only_mode_strikes_only_scales(net):
    inj, _ = _inj(net, ALL_PC, "only")
    idx, bit, sl, sw, sb = inj.draw_with_scales(np.random.default_rng(0), "exponent", 300)
    assert idx.size == 0 and sl.size == 300
    assert np.all((sb >= 23) & (sb <= 30))
    assert set(sl.tolist()) <= {_layer(inj, n) for n in ALL_PC}
    for l, w in zip(sl, sw):
        assert 0 <= w < inj.scale_u32[int(l)].size


def test_check_group(net):
    names = [n for n, _ in __import__("fr.quantize", fromlist=["x"]).target_parameters(net, "convlinear")]
    all_int = {n: "int8_pc" for n in names}
    off, _ = _inj(net, all_int, "off")
    with pytest.raises(ValueError):
        off.check_group("exponent", "bits")                           # INT weights have no exponent bits
    on, _ = _inj(net, all_int, "on")
    on.check_group("exponent", "bits")                                # scales do
    only, _ = _inj(net, ALL_PC, "only")
    only.check_group("exponent", "bits")
    with pytest.raises(ValueError):
        only.check_group("exponent", "bits", layer="layer2.0.conv1")   # that layer is E4M11: no scale


def test_exponent_flip_of_a_channel_scale_rescales_only_that_channel(net):
    inj, _ = _inj(net, ALL_PC, "only")
    l = _layer(inj)
    clean = _flat(inj).view(64, -1)
    s = float(inj.scales[l][5])
    inj.inject(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), ([l], [5], [23]))   # exponent LSB
    new = inj.flat[l].detach().view(64, -1)
    ratio_expected = 2.0 if (inj.scale_u32[l][5] >> 23) & 1 == 0 else 0.5    # bit 23 set -> cleared halves
    for c in range(64):
        if c == 5:
            nz = clean[c] != 0
            assert torch.allclose(new[c][nz] / clean[c][nz], torch.full_like(new[c][nz], ratio_expected), rtol=1e-6)
        else:
            assert torch.equal(new[c], clean[c])
    assert s > 0
    inj.restore()
    assert torch.equal(inj.flat[l].detach().view(64, -1), clean)
    assert inj.clean_weights_unchanged()


def test_per_tensor_scale_flip_rescales_the_whole_tensor_and_sign_bit_negates(net):
    inj, _ = _inj(net, {LAYER: "int8_pt"}, "only")
    l = _layer(inj)
    clean = _flat(inj)
    inj.inject(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), ([l], [0], [31]))
    assert torch.equal(inj.flat[l].detach(), -clean)
    inj.restore()
    assert torch.equal(inj.flat[l].detach(), clean)


def test_huge_scale_gives_huge_but_restorable_weights(net):
    inj, _ = _inj(net, ALL_PC, "only")
    l = _layer(inj)
    clean = _flat(inj)
    info = inj.inject(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), ([l], [0], [30]))   # exponent MSB
    assert info["scale_hits"] == 1 and (info["max_abs_faulted_value"] > 1e20 or not np.isfinite(info["max_abs_faulted_value"]))
    inj.restore()
    assert torch.equal(inj.flat[l].detach(), clean)


def test_weight_and_scale_strike_in_the_same_channel_combine(net):
    inj, _ = _inj(net, ALL_PC, "only")
    l = _layer(inj)
    per_channel = 36864 // 64
    widx = inj.starts[l] + 5 * per_channel + 7                        # a weight of channel 5
    inj.inject(np.array([widx]), np.array([6]), ([l], [5], [31]))      # flip weight bit 6 and negate channel 5's scale
    q_old = inj.words[l][5 * per_channel + 7]
    q_new = np.uint8(q_old ^ np.uint8(1 << 6))
    expected = float(np.int8(q_new)) * -float(inj.scales[l][5])
    got = float(inj.flat[l].detach()[5 * per_channel + 7])
    assert got == pytest.approx(expected, rel=1e-6)
    inj.restore()
    assert inj.clean_weights_unchanged()


def test_campaign_only_mode_end_to_end_and_resume(net, tmp_path):
    inj, _ = _inj(net, ALL_PC, "only")
    X = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    out = tmp_path / "c"
    FM.run_mixed_campaign(inj, X, y, out, ["exponent", "any"], [3], 2, 0, "bits", None, {}, log=lambda *_: None)
    rows = list(csv.DictReader(open(out / "trials.csv")))
    assert len(rows) == 4 and "scale_hits" in rows[0] and all(int(r["scale_hits"]) == 3 for r in rows)
    assert inj.clean_weights_unchanged()
    FM.run_mixed_campaign(inj, X, y, out, ["exponent", "any"], [3], 3, 0, "bits", None, {}, log=lambda *_: None)
    assert len(list(csv.DictReader(open(out / "trials.csv")))) == 6      # resumed: only the new trial was run
    camps = FM.collect([out])
    assert camps[0]["meta"]["scale_mode"] == "only" and camps[0]["meta"]["n_scale_words"] == inj.n_scale_words
    assert "scale metadata of INT layers is struck" in FM.render_comparison(camps, reference="none")
    off, _ = _inj(net, ALL_PC, "off")
    with pytest.raises(RuntimeError):                                   # scale modes must not be mixed in one directory
        FM.run_mixed_campaign(off, X, y, out, ["any"], [3], 1, 0, "bits", None, {}, log=lambda *_: None)


def test_compare_scale_faults_report(net, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    import compare_scale_faults as C
    X = torch.randn(12, 3, 32, 32)
    y = torch.randint(0, 10, (12,))
    root = tmp_path / "scale"
    names = [n for n, _ in __import__("fr.quantize", fromlist=["x"]).target_parameters(net, "convlinear")]
    for mode, sub in (("off", "off"), ("on", "on"), ("only", "only")):
        for arm in ("int8_pc", "int8_pt"):
            inj, buf = _inj(net, {n: arm for n in names}, mode)
            meta = {"avg_bits_payload": 8.0, "layout": f"all:{arm}"}
            FM.run_mixed_campaign(inj, X, y, root / sub / "bits" / f"all_{arm}", ["any"], [64], 3, 0, "bits",
                                  None, meta, log=lambda *_: None)
    md = C.compare(root)
    for part in ("Storage of the scale metadata", "scales fault-free (off)", "Paired difference", "Damage per scale hit"):
        assert part in md
    assert "all_int8_pc" in md and "all_int8_pt" in md
    # per-channel layout has far more scale words than per-tensor (21 tensors)
    pc = FM.collect([root / "on" / "bits" / "all_int8_pc"])[0]["meta"]["n_scale_words"]
    pt = FM.collect([root / "on" / "bits" / "all_int8_pt"])[0]["meta"]["n_scale_words"]
    assert pt == 21 and pc == 4810
