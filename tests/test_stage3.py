from collections import OrderedDict

import numpy as np
import pytest
import torch

from fr import stage2 as S2
from fr import stage3 as S3
from fr import stage3_validation as V
from fr.model import build_resnet18_cifar, quantizable_modules
from fr.stage3_protocol import build_protocol

CODECS = S2.make_codecs(True)


@pytest.mark.parametrize("fmt", list(S3.SPECS))
def test_bit_groups_partition_the_element(fmt):
    sp = S3.SPECS[fmt]
    allbits = sorted(b for bits in sp["fields"].values() for b in bits)
    assert allbits == list(range(sp["width"]))
    assert S3.group_bits(fmt, "all") == tuple(range(sp["width"]))
    if sp["kind"] == "int":
        assert S3.group_bits(fmt, "exponent") == ()                       # integer formats have no exponent group
        assert S3.group_bits(fmt, "sign") == (sp["width"] - 1,)
    else:
        assert S3.group_bits(fmt, "sign") == (sp["width"] - 1,)
        assert len(S3.group_bits(fmt, "exponent")) == sp["e_bits"] and len(S3.group_bits(fmt, "fraction")) == sp["m_bits"]
    assert sorted(b for v in S3.SCALE_FIELDS.values() for b in v) == list(range(32))


@pytest.mark.parametrize("fmt", list(S3.SPECS))
def test_every_bit_of_every_element_maps_as_expected(fmt):
    rows = V.bit_mapping_rows(fmt, CODECS[fmt])
    assert len(rows) == (6 if fmt in ("fp8_e4m3fn",) or S3.SPECS[fmt]["kind"] == "float" else 7) * S3.SPECS[fmt]["width"]
    bad = [r["test_id"] for r in rows if r["status"] != "PASS"]
    assert not bad, bad[:5]
    sp = V.special_value_rows(fmt, CODECS[fmt])
    assert sp and all(r["status"] == "PASS" for r in sp), [r["note"] for r in sp if r["status"] != "PASS"]
    if fmt in S3.SCALE_FORMATS:
        sc = V.scale_bit_rows(fmt, CODECS[fmt])
        assert len(sc) == 32 and all(r["status"] == "PASS" for r in sc)
    if fmt == "int4_pt":
        assert all(r["status"] == "PASS" for r in V.packing_rows(CODECS[fmt]))


def test_exponent_flip_changes_value_by_a_power_of_two():
    rows = V.bit_mapping_rows("e4m11_sn", CODECS["e4m11_sn"])
    r = next(r for r in rows if r["element_index"] == 0 and r["bit_index"] == 12)        # 1.0, exponent bit 1 (value 2)
    assert float(r["expected_change_ratio"]) == pytest.approx(2.0 ** 2) or float(r["expected_change_ratio"]) == pytest.approx(2.0 ** -2)
    r = next(r for r in rows if r["element_index"] == 0 and r["bit_index"] == 15)        # sign
    assert float(r["observed_value_after"]) == -1.0


def test_int4_location_matches_stage2_helper():
    for e in range(10):
        for b in range(4):
            assert S3.bit_location("int4_pt", e, b) == S2.int4_bit_location(e, b)


def test_sampling_is_distinct_deterministic_and_in_scope():
    net = build_resnet18_cifar().eval()
    w = {f"{n}.weight": m.weight.detach().numpy() for n, m in quantizable_modules(net)}
    enc = OrderedDict((n, CODECS["int8_pt"].encode(v)) for n, v in w.items())
    st = S3.EncodedStore("int8_pt", enc)
    s1 = S3.trial_seed(0, "int8_pt", "GLOBAL", "weights", "all", 8, 5)
    assert s1 == S3.trial_seed(0, "int8_pt", "GLOBAL", "weights", "all", 8, 5)
    assert s1 != S3.trial_seed(0, "int8_pt", "GLOBAL", "weights", "all", 8, 6)
    a = S3.draw_weight_flips(np.random.default_rng(s1), st, list(range(21)), S3.group_bits("int8_pt", "all"), 8)
    b = S3.draw_weight_flips(np.random.default_rng(s1), st, list(range(21)), S3.group_bits("int8_pt", "all"), 8)
    assert a == b and len({(l, e) for l, e, _ in a}) == 8
    for l, e, bit in a:
        assert 0 <= e < int(st.n[l]) and 0 <= bit < 8
    one = S3.draw_weight_flips(np.random.default_rng(1), st, [20], (7,), 8)           # fc only, sign bit only
    assert all(l == 20 and bit == 7 for l, _, bit in one)
    with pytest.raises(ValueError):
        S3.draw_weight_flips(np.random.default_rng(1), st, [20], (7,), 6000)         # k larger than the scope
    with pytest.raises(ValueError):
        S3.draw_weight_flips(np.random.default_rng(1), st, [20], (), 1)


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return build_resnet18_cifar().eval()


@pytest.fixture(scope="module")
def data():
    g = torch.Generator().manual_seed(0)
    return torch.randn(24, 3, 32, 32, generator=g), torch.randint(0, 10, (24,), generator=g)


def _injector(net, fmt):
    codec = CODECS[fmt]
    enc = OrderedDict((f"{n}.weight", codec.encode(m.weight.detach().numpy())) for n, m in quantizable_modules(net))
    return S3.EncodedFaultInjector(net, S3.EncodedStore(fmt, enc), codec, "cpu")


def test_evaluate_matches_plain_forward_and_hooks_do_not_replace_outputs(net, data):
    X, y = data
    ev = S3.evaluate(net, X, y, return_logits=True)
    with torch.inference_mode():
        plain = net(X).float()
    assert torch.equal(ev["logits"], plain) and not ev["activation_nonfinite"] and ev["loss"] is not None


def test_evaluate_non_finite_policy(net, data):
    import copy
    X, y = data
    bad = copy.deepcopy(net)
    with torch.no_grad():
        bad.fc.weight[0, 0] = float("nan")
    ev = S3.evaluate(bad, X, y)
    assert ev["logit_rows_nonfinite"] == 24 and ev["accuracy"] == 0.0 and ev["loss"] is None
    assert ev["activation_nonfinite"] and ev["first_nonfinite_activation_layer"] == "fc"


@pytest.mark.parametrize("fmt", ["e4m11_sn", "fp16", "fp8_e4m3fn", "int8_pt", "int4_pt"])
def test_model_level_validation_suite(net, data, fmt):
    X, y = data
    inj = _injector(net, fmt)
    ctx = {"experiment_id": "t", "run_id": "r", "model": "m", "dataset": "d", "checkpoint_hash": "c", "codec_version": "v"}
    import hashlib

    def fp32_hash():
        h = hashlib.sha256()
        for k, v in sorted(net.state_dict().items()):
            h.update(v.numpy().tobytes())
        return h.hexdigest()
    res = V.model_level_tests(fmt, inj, X, y, ctx, 0, fp32_hash, n_restore_trials=6)
    failed = [t for t in res["tests"] if not t["passed"]]
    assert not failed, failed
    for r in res["records"]:
        assert all(f in r for f in S3.REQUIRED_TRIAL_FIELDS)
        assert r["restored_exactly"] and r["encoded_hash_before"] == r["encoded_hash_after_restore"]
    assert any(r["fault_count"] == 0 for r in res["records"])                       # the no-fault control
    forced = [r for r in res["records"] if r["fault_group"] == "forced"]
    assert len(forced) == 1
    if fmt in ("fp16", "fp8_e4m3fn", "int8_pt", "int4_pt"):                       # non-finite reachable: trial kept and flagged
        assert forced[0]["failure"] and forced[0]["fault_loss"] is None and "weights_nonfinite" in forced[0]["output_flag"]
    else:                                                                          # E4M11: reserved exponent clamps, stays finite
        assert not forced[0]["failure"]


def test_flip_changes_stored_bytes_not_fp32_and_restores(net, data):
    X, y = data
    inj = _injector(net, "int4_pt")
    clean = S3.evaluate(inj.model, X, y)
    h0 = inj.store.hash()
    fc = 20
    before = inj.store.words[fc].copy()
    rec = S3.run_trial(inj, X, y, clean, {k: "x" for k in ("experiment_id", "run_id", "model", "dataset", "checkpoint_hash", "codec_version")},
                       fmt="int4_pt", scope_layers=[fc], scope_name="fc", scope_kind="weights", group="sign", k=8, trial=0, base_seed=0)
    assert rec["encoded_hash_faulted"] != h0 and rec["restored_exactly"] and np.array_equal(inj.store.words[fc], before)
    assert rec["fault_count"] == 8 and all(f["field"] == "sign" and f["layer_name"].startswith("fc") for f in rec["flips"])
    assert all(f["bit_in_stored_unit"] in (3, 7) for f in rec["flips"])               # sign nibble bit of a packed byte


def test_scale_scope_flips_one_tensor_and_restores(net, data):
    X, y = data
    inj = _injector(net, "int8_pt")
    clean = S3.evaluate(inj.model, X, y)
    ctx = {k: "x" for k in ("experiment_id", "run_id", "model", "dataset", "checkpoint_hash", "codec_version")}
    rec = S3.run_trial(inj, X, y, clean, ctx, fmt="int8_pt", scope_layers=list(range(21)), scope_name="GLOBAL-SCALES",
                       scope_kind="scales", group="exponent", k=1, trial=0, base_seed=0)
    assert rec["fault_count"] == 1 and rec["flips"][0]["target"] == "scale" and 23 <= rec["flips"][0]["bit_index"] <= 30
    assert rec["restored_exactly"] and rec["encoded_hash_faulted"] != rec["encoded_hash_before"]
    with pytest.raises(ValueError):
        S3.draw_scale_flips(np.random.default_rng(0), inj.store, list(range(21)), (30,), 22)   # only 21 scale words


def test_scope_violation_is_rejected(net, data):
    X, y = data
    inj = _injector(net, "fp16")
    clean = S3.evaluate(inj.model, X, y)
    ctx = {k: "x" for k in ("experiment_id", "run_id", "model", "dataset", "checkpoint_hash", "codec_version")}
    with pytest.raises(AssertionError):
        S3.run_trial(inj, X, y, clean, ctx, fmt="fp16", scope_layers=[0], scope_name="conv1", scope_kind="weights", group="all",
                     k=1, trial=0, base_seed=0, explicit_weight_flips=[(5, 3, 2)])
    with pytest.raises(IndexError):
        inj.store.flip(20, 5120, 0)
    assert inj.store.hash() == S3.EncodedStore("fp16", inj.store.to_encoded()).hash()


def test_protocol_document_contains_the_required_items():
    layers = [{"layer_id": i, "module_name": f"l{i}", "num_weights": 10} for i in range(3)]
    p = build_protocol(run_id="r", checkpoint_sha="c", stage2_frozen="f", layers=layers, base_seed=0, n_images=1000,
                       trials_screening=30, smoke_trials=3, seconds_per_eval=0.5, codec_versions={k: "v" for k in S3.SPECS})
    assert p["fault_counts_k"] == [1, 8] and "SCREENING" in p["label"]
    assert p["formats"]["int4_pt"]["packing"]["nibble_order"] and "sign_bit_interpretation" in p["formats"]["int8_pt"]
    assert "exponent" not in [g for g in p["formats"]["int8_pt"]["groups"] if p["formats"]["int8_pt"]["groups"][g]]
    assert p["formats"]["fp16"]["groups"]["exponent"] == [10, 11, 12, 13, 14]
    assert p["sampling_policy"]["locations"] and p["seeds"]["derivation"] and p["non_finite_policy"]["never_dropped"]
    assert p["trial_budget"]["estimate"]["estimated_hours"] > 0 and p["target_scope"]["total_elements"] == 30
