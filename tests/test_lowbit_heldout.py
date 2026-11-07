"""4-bit formats (INT4 per-tensor / per-channel, FP4 E2M1 scaled), budgets for them, faults in 4-bit words,
held-out (validation) Fisher with the leakage guard, and the selection comparison."""
import copy
import csv
import json

import numpy as np
import pytest
import torch
from torchvision import datasets

from fr import allocation, data as fdata, faults_mixed as FM, formats8 as F, mixed
from fr import mixed_run as R
from fr.config import Config
from fr.model import build_resnet18_cifar


# ------------------------------------------------------------------------------------------ formats
def test_fp4_table_and_word_roundtrip():
    assert F._T4.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] and F.FP4_MAX == 6.0
    words = np.arange(16, dtype=np.uint8)
    vals = F.fp4_decode(words)
    assert vals[:8].tolist() == F._T4.tolist() and np.array_equal(vals[8:], -vals[:8])
    assert np.array_equal(F.fp4_encode(vals), words)                       # idempotent, -0.0 keeps its sign bit
    assert np.array_equal(F.fp4_decode(words | 0xF0), vals)                # high nibble is ignored


def test_fp4_rounding_ties_to_even_and_saturation():
    x = np.array([0.25, 0.75, 5.0, 0.26, 7.0, -100.0, -0.3])
    got = F.fp4_decode(F.fp4_encode(x)).astype(np.float64)
    # ties go to the even word: 0.25 -> 0 (word 0), 0.75 -> 1.0 (word 2), 5.0 -> 4.0 (word 6)
    assert got.tolist() == [0.0, 1.0, 4.0, 0.5, 6.0, -6.0, -0.5]


def test_int_words_to_q_twos_complement():
    assert F.int_words_to_q(np.arange(16, dtype=np.uint8), 4).tolist() == list(range(8)) + list(range(-8, 0))
    assert F.int_words_to_q(np.array([127, 128, 255], dtype=np.uint8), 8).tolist() == [127, -128, -1]


@pytest.mark.parametrize("fmt", ["int4_pt", "int4_pc", "fp4_scaled"])
def test_quantize_4bit_properties(fmt):
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((6, 5, 3, 3)) * 0.01).astype(np.float32)
    r = F.quantize8(w, fmt)
    assert r["bits"] == 4 and r["words"].dtype == np.uint8 and r["words"].max() < 16
    assert r["dequantized"].shape == w.shape and r["dequantized"].dtype == np.float32
    again = F.dequantize8(r["words"], r["scale"], r["shape"], fmt)
    assert np.array_equal(again, r["dequantized"])                           # storage decodes to the same values
    if fmt == "int4_pt":
        s = np.abs(w).max() / 7
        assert np.abs(w - r["dequantized"]).max() <= s / 2 * (1 + 1e-5) and r["metadata_bits"] == 32
        q = F.int_words_to_q(r["words"], 4)
        assert q.min() >= -8 and q.max() <= 7 and np.abs(q).max() == 7       # the largest weight maps to +-7
    elif fmt == "int4_pc":
        assert np.shape(r["scale"]) == (6,) and r["metadata_bits"] == 32 * 6
        s = np.abs(w.reshape(6, -1)).max(axis=1) / 7
        err = np.abs(w - r["dequantized"]).reshape(6, -1).max(axis=1)
        assert (err <= s / 2 * (1 + 1e-5)).all()
    else:
        k = int(r["scale"])
        assert r["metadata_bits"] == 8 and np.abs(w).max() / 2.0 ** k <= 6.0 < np.abs(w).max() / 2.0 ** (k - 1)
        assert np.abs(r["dequantized"]).max() <= 6.0 * 2.0 ** k


def test_per_channel_int4_beats_per_tensor_on_uneven_channels():
    rng = np.random.default_rng(1)
    w = (rng.standard_normal((8, 64)) * np.array([1e-3, 1e-2] * 4)[:, None]).astype(np.float32)
    e = {f: np.sum((w - F.quantize8(w, f)["dequantized"]) ** 2) for f in ("int4_pt", "int4_pc")}
    assert e["int4_pc"] < e["int4_pt"]


def test_int8_unchanged_by_generalisation():
    rng = np.random.default_rng(2)
    w = (rng.standard_normal((4, 10)) * 0.05).astype(np.float32)
    words, scale = F.int8_encode(w, per_channel=False)
    q_old = np.clip(np.rint(w.reshape(-1).astype(np.float64) / float(scale)), -128, 127).astype(np.int8)
    assert np.array_equal(words, q_old.view(np.uint8))


# ----------------------------------------------------------------------------------------- allocation
def test_required_switched_and_select_for_4bit():
    assert allocation.required_switched(100, 10, lo_bits=4) == 50
    assert allocation.required_switched(100, 4, lo_bits=4) == 100
    assert allocation.required_switched(100, 12, lo_bits=8) == 50          # unchanged for 8-bit formats
    with pytest.raises(ValueError):
        allocation.required_switched(100, 3.9, lo_bits=4)
    names, sizes = list("abcd"), [10, 20, 30, 40]
    r = allocation.select("min_cost", names, sizes, 10.0, costs=[4, 3, 2, 1], lo_bits=4)
    n8 = r["n_switched"]
    assert n8 >= 50 and r["avg_bits"] == pytest.approx((16 * (100 - n8) + 4 * n8) / 100) and r["avg_bits"] <= 10 + 1e-9


# ------------------------------------------------------------------------------------------ faults
ASSIGN4 = {"layer1.0.conv1.weight": "int4_pc", "layer1.1.conv1.weight": "int4_pt",
           "layer2.0.conv1.weight": "fp4_scaled", "fc.weight": "int8_pc"}


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return build_resnet18_cifar().eval()


def _inj(net, assignment):
    q, buf = mixed.quantize_mixed(net, assignment)
    return FM.MixedFaultInjector(copy.deepcopy(q), buf), buf


def test_4bit_words_are_only_struck_in_the_low_nibble(net):
    inj, buf = _inj(net, ASSIGN4)
    assert inj.clean_weights_unchanged()
    rng = np.random.default_rng(0)
    for sampling in FM.SAMPLINGS:
        idx, bit = inj.draw(rng, "any", 20000, sampling)
        li = np.searchsorted(inj.starts, idx, side="right") - 1
        assert (bit < inj.width[li]).all()
        four = inj.width[li] == 4
        assert four.any() and set(np.unique(bit[four])) <= {0, 1, 2, 3}
    # exponent group: FP4 has bits 1-2, INT4 none, E4M11 11-14
    bits, nb, units = inj.eligible("exponent", "weights")
    by_fmt = {f: len(b) for f, b in zip(inj.fmts, bits)}
    assert by_fmt["fp4_scaled"] == 2 and by_fmt["int4_pc"] == 0 and by_fmt["e4m11"] == 4


def test_4bit_injection_matches_naive_full_decode(net):
    inj, buf = _inj(net, ASSIGN4)
    rng = np.random.default_rng(3)
    for sampling in FM.SAMPLINGS:
        for layer in ("layer1.0.conv1", "layer1.1.conv1", "layer2.0.conv1", "fc"):
            idx, bit = inj.draw(rng, "any", 60, sampling, layer=layer)
            corrupted = copy.deepcopy(buf)
            for i, b in zip(idx, bit):
                l = int(np.searchsorted(inj.starts, i, side="right") - 1)
                w = corrupted[inj.names[l]]["words"]
                w[i - inj.starts[l]] ^= w.dtype.type(1 << int(b))
            naive = mixed.dequantized_weights(corrupted)
            inj.inject(idx, bit)
            for l, n in enumerate(inj.names):
                assert np.array_equal(naive[n].reshape(-1), inj.flat[l].detach().cpu().numpy()), (sampling, layer, n)
            inj.restore()
    assert inj.clean_weights_unchanged()


def test_int4_flip_changes_the_code_by_a_power_of_two(net):
    inj, buf = _inj(net, ASSIGN4)
    name = "layer1.0.conv1.weight"
    l = inj.names.index(name)
    gi = int(inj.starts[l]) + 7                                       # weight 7 is in output channel 0
    q = float(F.int_words_to_q(buf[name]["words"][7:8], 4)[0])
    scale = float(np.asarray(buf[name]["scale"])[0])
    for bit in range(4):
        inj.inject(np.array([gi]), np.array([bit]))
        new_q = float(inj.flat[l][7]) / scale
        assert abs(new_q - q) == pytest.approx(2 ** bit, abs=1e-3)     # bit 3 (sign position) = 8 steps
        inj.restore()
    assert inj.clean_weights_unchanged()


def test_campaign_runs_on_4bit_layouts_and_skips_missing_groups(net, tmp_path):
    torch.manual_seed(0)
    X, y = torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))
    inj, _ = _inj(net, ASSIGN4)
    FM.run_mixed_campaign(inj, X, y, tmp_path / "mix", ["any", "exponent", "sign"], [2], 3, seed=0,
                          log=lambda *_: None)
    rows = list(csv.DictReader(open(tmp_path / "mix" / "trials.csv")))
    assert len(rows) == 9 and {r["group"] for r in rows} == {"any", "exponent", "sign"}
    inj4, _ = _inj(net, {n: "int4_pc" for n in
                         [n for n, _ in __import__("fr.quantize", fromlist=["x"]).target_parameters(net, "convlinear")]})
    FM.run_mixed_campaign(inj4, X, y, tmp_path / "i4", ["exponent", "any"], [1], 2, seed=0, log=lambda *_: None)
    assert json.load(open(tmp_path / "i4" / "campaign_meta.json"))["skipped_groups"] == ["exponent"]


def test_normalize_spec_accepts_4bit_arms():
    assert FM.normalize_spec("all:int4_pc") == ("all_int4_pc", None)
    slug, cid = FM.normalize_spec("fp4_scaled:T6:product")
    assert cid == "fp4_scaled|T6|product|"
    with pytest.raises(ValueError):
        FM.normalize_spec("all:int3_pc")


# ------------------------------------------------------------------------------------- held-out Fisher
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


def test_val_loader_skip_drops_the_leading_images(env):
    cfg, _ = env
    full = next(iter(fdata.get_val_loader(cfg.data_path(), None, 30, 30, 0)))[0]
    part = next(iter(fdata.get_val_loader(cfg.data_path(), None, 30, 30, 0, subset=5, skip=8)))[0]
    assert part.shape[0] == 5 and torch.equal(part, full[8:13])
    rest = next(iter(fdata.get_val_loader(cfg.data_path(), None, 30, 30, 0, skip=8)))[0]
    assert rest.shape[0] == 22


def test_heldout_fisher_pipeline_with_4bit_arms(env):
    cfg, ck = env
    out_f = cfg.results_dir() / "sensitivity_heldout"
    summ = R.run_fisher(cfg, ck, ("true",), calib_size=8, out_dir=out_f, source="val")
    assert summ["source"] == "val" and summ["kinds"]["true"]["n_samples"] == 8
    with pytest.raises(ValueError):                                        # held-out Fisher needs images left over
        R.run_fisher(cfg, ck, ("true",), calib_size=30, out_dir=out_f, source="val")
    fpath = out_f / "fisher_true.pt"
    kw = dict(arms=("int4_pc", "fp4_scaled"), targets=(12.0, 6.0, 2.0), lambdas=(0.0, 1.0), random_draws=1)
    with pytest.raises(ValueError, match="val_skip"):                      # would validate on the Fisher images
        R.run_sweep(cfg, ck, fpath, out_dir=cfg.results_dir() / "mixed_heldout", **kw)
    out = R.run_sweep(cfg, ck, fpath, out_dir=cfg.results_dir() / "mixed_heldout", val_skip=8, **kw)
    meta = json.load(open(cfg.results_dir() / "mixed_heldout" / "sweep_meta.json"))
    assert meta["val_skip"] == 8 and meta["fisher_source"] == "val" and meta["val_images"] == 22
    rows = list(csv.DictReader(open(out["csv"])))
    sweep = [r for r in rows if r["method"] != "baseline"]
    assert {float(r["target_avg_bits"]) for r in sweep} == {12.0, 6.0}     # 2 bits < 4 is skipped for 4-bit arms
    assert len(sweep) == 2 * 2 * 5
    for r in sweep:
        assert float(r["avg_bits_payload"]) <= float(r["target_avg_bits"]) + 1e-9
        assert 0 < int(r["n_layers8"]) <= 21
    assert {"baseline|int4_pc_all", "baseline|fp4_scaled_all"} <= {r["config_id"] for r in rows}
    fin = R.run_final(cfg, ck, cfg.results_dir() / "mixed_heldout")
    md = open(fin["report"], encoding="utf-8").read()
    assert "held out" in md and "## Frontier" in md and "| int4_pc | 6 |" in md
    ch = R.run_churn(cfg, ck, cfg.results_dir() / "mixed_heldout")
    assert (cfg.results_dir() / "mixed_heldout" / "mixed_churn_val.csv").exists() and ch["report"]


def test_compare_selections(tmp_path, monkeypatch):
    import pathlib
    import sys
    monkeypatch.syspath_prepend(str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    sys.modules.pop("compare_selections", None)
    import compare_selections as mod
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    meta = {"layers": ["x", "y", "z"], "layer_sizes": [10, 20, 30]}
    json.dump(meta, open(a / "sweep_meta.json", "w"))
    json.dump({"m|T12|product|": {"layers8": ["x", "y"]}, "m|T12|random|0": {"layers8": ["z"]}},
              open(a / "selections.json", "w"))
    json.dump({"m|T12|product|": {"layers8": ["y", "z"]}, "m|T12|random|0": {"layers8": ["x"]}},
              open(b / "selections.json", "w"))
    md = mod.compare(a, b)
    assert "m|T12|product|" in md and "0.333" in md and "random|0" not in md      # 20 / 60
