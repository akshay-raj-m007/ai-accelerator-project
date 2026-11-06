import copy
import csv
import json

import numpy as np
import pytest
import torch

from fr import e4m11 as E
from fr import faults as F
from fr import faults_mixed as FM
from fr import formats8, mixed
from fr.model import build_resnet18_cifar
from fr.quantize import quantize_model, target_parameters

ASSIGN = {"layer1.0.conv1.weight": "int8_pt", "layer1.1.conv1.weight": "int8_pc",
          "layer2.0.conv1.weight": "fp8", "layer2.1.conv1.weight": "fp8_scaled",
          "fc.weight": "int8_pc"}


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return build_resnet18_cifar().eval()


def _inj(net, assignment):
    q, buf = mixed.quantize_mixed(net, assignment)
    return FM.MixedFaultInjector(copy.deepcopy(q), buf), buf


# ----------------------------------------------------------------------------------------- groups
@pytest.mark.parametrize("fmt", list(FM.WORD_BITS))
def test_group_tables_partition_the_word(fmt):
    width = FM.WORD_BITS[fmt]
    bits = [b for g in FM.PAPER_GROUP_NAMES for b in FM.group_bits(fmt, g)]
    assert sorted(bits) == list(range(width))                          # disjoint and complete
    assert FM.group_bits(fmt, "any") == tuple(range(width))
    assert FM.group_bits(fmt, f"bit{width - 1}") == (width - 1,) and FM.group_bits(fmt, f"bit{width}") == ()
    if fmt.startswith("int8"):
        assert FM.group_bits(fmt, "exponent") == ()                     # paper: empty by construction
    with pytest.raises(ValueError):
        FM.group_bits(fmt, "nope")


# ------------------------------------------------------------------------------------- injection
def test_all_e4m11_matches_the_validated_injector(net):
    q, buf = mixed.quantize_mixed(net, {})
    new = FM.MixedFaultInjector(copy.deepcopy(q), buf)
    q0, b0 = quantize_model(net, "subnormal", "convlinear")
    old = F.FaultInjector(q0, b0, "subnormal", 7, "clamp", "cpu")
    assert new.total == old.total and new.total_bits == 16 * old.total
    rng = np.random.default_rng(0)
    idx = rng.integers(0, new.total, 60)
    bits = rng.integers(0, 16, 60)
    old.inject(idx, bits)
    new.inject(idx, bits)
    a = torch.cat([f.detach() for f in old.flat])
    b = torch.cat([f.detach() for f in new.flat])
    assert torch.equal(a, b)
    old.restore()
    new.restore()
    assert new.clean_weights_unchanged() and old.clean_weights_unchanged()


def test_matches_naive_full_decode_for_every_format(net):
    inj, buf = _inj(net, ASSIGN)
    assert inj.clean_weights_unchanged()
    rng = np.random.default_rng(1)
    for group in ("any", "sign", "mantissa_msb", "exponent"):
        for sampling in FM.SAMPLINGS:
            idx, bit = inj.draw(rng, group, 40, sampling)
            idx, bit = np.concatenate([idx, idx[:3]]), np.concatenate([bit, bit[:3]])   # repeated flips cancel
            corrupted = copy.deepcopy(buf)
            for i, b in zip(idx, bit):
                l = int(np.searchsorted(inj.starts, i, side="right") - 1)
                w = corrupted[inj.names[l]]["words"]
                w[i - inj.starts[l]] ^= w.dtype.type(1 << int(b))
            naive = mixed.dequantized_weights(corrupted)
            inj.inject(idx, bit)
            for l, n in enumerate(inj.names):
                np.testing.assert_array_equal(inj.flat[l].detach().cpu().numpy(), naive[n].reshape(-1))
            inj.restore()
            assert inj.clean_weights_unchanged()


def test_inject_requires_restore_and_duplicates_cancel(net):
    inj, _ = _inj(net, ASSIGN)
    i = int(inj.starts[3]) + 5
    inj.inject(np.array([i, i]), np.array([2, 2]))                       # same bit twice -> unchanged
    assert inj.clean_weights_unchanged()
    inj.restore()
    inj.inject(np.array([i]), np.array([2]))
    with pytest.raises(RuntimeError):
        inj.inject(np.array([i + 1]), np.array([2]))
    inj.restore()
    with pytest.raises(ValueError):
        inj.inject(np.array([int(inj.starts[1])]), np.array([9]))      # bit 9 does not exist in an 8-bit word


def test_int8_flip_effects_are_bounded_steps(net):
    inj, buf = _inj(net, ASSIGN)
    name = "layer1.0.conv1.weight"                                      # int8_pt
    l = inj.names.index(name)
    scale = float(buf[name]["scale"])
    q = int(np.array(buf[name]["words"][7], dtype=np.uint8).view(np.int8))
    gi = int(inj.starts[l]) + 7
    for bit in (0, 3, 6, 7):
        inj.inject(np.array([gi]), np.array([bit]))
        new_q = int(np.array(np.uint8(q & 0xFF) ^ np.uint8(1 << bit), dtype=np.uint8).view(np.int8))
        assert float(inj.flat[l][7]) == pytest.approx(new_q * scale, rel=1e-6)
        assert abs(new_q - q) == (1 << bit)                              # +-2^bit steps; bit 7 -> +-128 steps
        inj.restore()
    assert inj.clean_weights_unchanged()


def test_fp8_exponent_msb_flip_scales_by_256(net):
    inj, buf = _inj(net, ASSIGN)
    l = inj.names.index("layer2.1.conv1.weight")                        # fp8_scaled
    words = np.array(buf["layer2.1.conv1.weight"]["words"], dtype=np.uint8)
    es = (words >> 3) & 15
    j = int(np.nonzero((es >= 1) & (es <= 6))[0][0])                    # es + 8 stays <= 14: finite, x 2^8
    before = float(inj.flat[l][j])
    inj.inject(np.array([int(inj.starts[l]) + j]), np.array([6]))     # E4M3 exponent MSB is bit 6
    after = float(inj.flat[l][j])
    inj.restore()
    assert after / before == pytest.approx(2.0 ** 8, rel=1e-6)          # rho = 2^(2^3) = 256 (paper A.3)


# -------------------------------------------------------------------------------------- sampling
def test_exponent_group_never_hits_int8_layers(net):
    inj, _ = _inj(net, ASSIGN)
    rng = np.random.default_rng(0)
    for sampling in FM.SAMPLINGS:
        idx, bit = inj.draw(rng, "exponent", 5000, sampling)
        li = np.searchsorted(inj.starts, idx, side="right") - 1
        assert not any(inj.fmts[l].startswith("int8") for l in li)
        for l, b in zip(li, bit):
            assert b in FM.group_bits(inj.fmts[l], "exponent")
    only_int8, _ = _inj(net, {n: "int8_pc" for n, _ in target_parameters(net, "convlinear")})
    with pytest.raises(ValueError):
        only_int8.draw(rng, "exponent", 3)


def test_weights_vs_bits_sampling_probabilities(net):
    inj, _ = _inj(net, ASSIGN)
    rng = np.random.default_rng(0)
    n = 400_000
    for sampling in FM.SAMPLINGS:
        idx, bit = inj.draw(rng, "any", n, sampling)
        li = np.searchsorted(inj.starts, idx, side="right") - 1
        frac = np.bincount(li, minlength=len(inj.names)) / n
        expect = inj.counts * (inj.width if sampling == "bits" else 1)
        expect = expect / expect.sum()
        sd = np.sqrt(expect * (1 - expect) / n)
        assert np.all(np.abs(frac - expect) < 5 * sd + 1e-4), sampling
        for l in (inj.names.index("fc.weight"), 0):                       # bits uniform within the word
            sel = li == l
            if sel.sum() > 2000:
                counts = np.bincount(bit[sel], minlength=inj.width[l]) / sel.sum()
                assert np.all(np.abs(counts - 1 / inj.width[l]) < 0.04)


def test_draw_reproducible_layer_filter_and_pairing(net):
    a, _ = _inj(net, ASSIGN)
    b, _ = _inj(net, {})                                                  # a different storage layout
    for sampling in FM.SAMPLINGS:
        x = a.draw(F.trial_rng(3, "any|" + sampling, 8, 5), "any", 8, sampling)
        y = a.draw(F.trial_rng(3, "any|" + sampling, 8, 5), "any", 8, sampling)
        z = a.draw(F.trial_rng(3, "any|" + sampling, 8, 6), "any", 8, sampling)
        assert np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) and not np.array_equal(x[0], z[0])
    # group 'any' + weights: the struck weights are identical for every layout (paired comparisons)
    wa = a.draw(F.trial_rng(0, "any|weights", 8, 1), "any", 8, "weights")[0]
    wb = b.draw(F.trial_rng(0, "any|weights", 8, 1), "any", 8, "weights")[0]
    assert np.array_equal(wa, wb)
    lo = int(a.starts[a.names.index("fc.weight")])
    idx, bit = a.draw(np.random.default_rng(0), "any", 500, "bits", layer="fc")
    assert idx.min() >= lo and idx.max() < lo + 5120


# --------------------------------------------------------------------------------------- campaign
def test_trial_metrics_handles_nonfinite():
    clean = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    y = torch.tensor([0, 1])
    m = FM.trial_metrics(clean.clone(), clean, y)
    assert m["flips_vs_clean"] == 0 and m["kl_vs_clean"] == pytest.approx(0.0, abs=1e-12) and m["finite"] == 1
    bad = clean.clone()
    bad[0, 0] = float("nan")
    mb = FM.trial_metrics(bad, clean, y)
    assert mb["finite"] == 0 and np.isnan(mb["kl_vs_clean"])
    sw = FM.trial_metrics(torch.tensor([[0.0, 2.0], [0.0, 3.0]]), clean, y)
    assert sw["flips_vs_clean"] == 1 and sw["kl_vs_clean"] > 0 and sw["accuracy"] == 50.0


def test_spec_parsing():
    names = ["a.weight", "b.weight", "c.weight"]
    sel = {"int8_pc|T12|product|": {"layers8": ["b.weight"]}, "fp8|T10|random|1": {"layers8": ["a.weight"]},
           "int8_pt|T12|lambda|0.5": {"layers8": ["c.weight"]}}
    assert FM.resolve_assignment("e4m11", sel, names)[:2] == ("e4m11", {})
    assert FM.resolve_assignment("all:fp8", sel, names)[1] == {n: "fp8" for n in names}
    assert FM.resolve_assignment("int8_pc:T12:product", sel, names)[1] == {"b.weight": "int8_pc"}
    assert FM.resolve_assignment("fp8|T10|random|1", sel, names)[1] == {"a.weight": "fp8"}
    assert FM.resolve_assignment("fp8:T10:random:1", sel, names)[0] == "fp8_T10_random_1"
    assert FM.resolve_assignment("int8_pt:T12:lambda:0.5", sel, names)[1] == {"c.weight": "int8_pt"}
    with pytest.raises(KeyError):
        FM.resolve_assignment("int8_pc:T10:product", sel, names)
    for bad in ("all:int4", "foo:T12:product", "int8_pc:T12:bogus", "int8_pc"):
        with pytest.raises(ValueError):
            FM.normalize_spec(bad)


def test_campaign_end_to_end_resume_and_comparison(net, tmp_path):
    torch.manual_seed(0)
    X = torch.randn(24, 3, 32, 32)
    y = torch.randint(0, 10, (24,))
    layouts = {"e4m11": {}, "all_int8_pt": {n: "int8_pt" for n, _ in target_parameters(net, "convlinear")},
               "mix": ASSIGN}
    paths = {}
    for slug, assign in layouts.items():
        inj, buf = _inj(net, assign)
        acc = mixed.bit_accounting(buf)
        out = tmp_path / "weights" / slug
        paths[slug] = out
        FM.run_mixed_campaign(inj, X, y, out, ["any", "sign"], [2], 3, seed=7, sampling="weights",
                              meta={"avg_bits_payload": acc["avg_bits_payload"], "label": slug},
                              log=lambda *_: None)
        assert inj.clean_weights_unchanged()
        rows = list(csv.DictReader(open(out / "trials.csv")))
        assert len(rows) == 2 * 1 * 3 and set(FM.CSV_FIELDS) == set(rows[0])
        # resume does nothing, extending adds only the new trials
        FM.run_mixed_campaign(inj, X, y, out, ["any", "sign"], [2], 3, seed=7, sampling="weights", log=lambda *_: None)
        assert len(list(csv.DictReader(open(out / "trials.csv")))) == 6
        FM.run_mixed_campaign(inj, X, y, out, ["any", "sign"], [2], 4, seed=7, sampling="weights", log=lambda *_: None)
        assert len(list(csv.DictReader(open(out / "trials.csv")))) == 8
        with pytest.raises(RuntimeError):
            FM.run_mixed_campaign(inj, X, y, out, ["any"], [2], 1, seed=8, sampling="weights", log=lambda *_: None)
        with pytest.raises(RuntimeError):
            FM.run_mixed_campaign(inj, X, y, out, ["any"], [2], 1, seed=7, sampling="bits", log=lambda *_: None)
    # group 'any' + weights: identical struck weights for all layouts => identical unique-weight counts per trial
    uniq = {s: [int(r["n_unique_weights"]) for r in csv.DictReader(open(p / "trials.csv")) if r["group"] == "any"]
            for s, p in paths.items()}
    assert uniq["e4m11"] == uniq["all_int8_pt"] == uniq["mix"]
    hits = [int(r["flips_in_8bit"]) for r in csv.DictReader(open(paths["all_int8_pt"] / "trials.csv"))]
    assert all(h == 2 for h in hits)                                      # k = 2 flips, all in 8-bit layers
    camps = FM.collect(sorted(paths.values()))
    assert {c["slug"] for c in camps} == set(layouts)
    md = FM.render_comparison(camps, reference="e4m11")
    assert "group `any`, sampling `weights`, k = 2" in md and "Paired difference" in md and "| mix |" in md
    s = FM.summarize_trials(camps[0]["rows"], 50.0, 50.0)
    assert {r["group"] for r in s} == {"any", "sign"} and all(r["n_trials"] == 4 for r in s)
    # same seed, same layout => identical results in a fresh directory
    inj, _ = _inj(net, ASSIGN)
    FM.run_mixed_campaign(inj, X, y, tmp_path / "again", ["any"], [2], 3, seed=7, sampling="weights", log=lambda *_: None)
    a = [r["accuracy"] for r in csv.DictReader(open(tmp_path / "again" / "trials.csv"))]
    b = [r["accuracy"] for r in csv.DictReader(open(paths["mix"] / "trials.csv")) if r["group"] == "any"][:3]
    assert a == b


def test_bootstrap_ci_is_deterministic():
    x = np.array([95.0, 94.5, 95.1, 60.0, 95.2])
    lo, hi = FM.bootstrap_ci(x)
    assert (lo, hi) == FM.bootstrap_ci(x) and lo <= x.mean() <= hi


def test_campaign_skips_groups_a_layout_does_not_have(net, tmp_path):
    torch.manual_seed(0)
    X, y = torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))
    inj, _ = _inj(net, {n: "int8_pt" for n, _ in target_parameters(net, "convlinear")})
    msgs = []
    out = tmp_path / "i8"
    FM.run_mixed_campaign(inj, X, y, out, ["exponent", "any"], [1], 2, seed=0, log=msgs.append)
    rows = list(csv.DictReader(open(out / "trials.csv")))
    assert {r["group"] for r in rows} == {"any"} and any("skipped" in m for m in msgs)
    assert json.load(open(out / "campaign_meta.json"))["skipped_groups"] == ["exponent"]
    with pytest.raises(ValueError):                                       # a typo is still an error
        FM.run_mixed_campaign(inj, X, y, tmp_path / "typo", ["expnent"], [1], 1, seed=0, log=lambda *_: None)
