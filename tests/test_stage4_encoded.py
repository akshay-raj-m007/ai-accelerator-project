import importlib.util
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
SPEC = importlib.util.spec_from_file_location("s4enc", Path(__file__).resolve().parents[1] / "scripts" / "stage4_encoded_fault_injection.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
from fr import stage2 as S2  # noqa: E402
from fr import stage3 as S3  # noqa: E402
from fr import stage3_validation as V  # noqa: E402
from fr import stage4 as S4  # noqa: E402
from fr.faults import load_tensors  # noqa: E402,F401
from fr.model import build_mobilenetv2_cifar, quantizable_modules  # noqa: E402


@pytest.mark.parametrize("fmt", ["bf16", "fp8_e5m2", "fp16", "e4m11_sn", "fp8_e4m3fn", "int8_pt", "int4_pt"])
def test_known_bit_suites_pass_for_every_protocol_format(fmt):
    codec = S2.codec_registry()[fmt]
    rows = V.bit_mapping_rows(fmt, codec) + V.special_value_rows(fmt, codec)
    assert rows and all(r["status"] == "PASS" for r in rows), [r["test_id"] for r in rows if r["status"] != "PASS"][:5]
    assert all(t["passed"] for t in V.roundtrip_synthetic_tests(fmt, codec))
    assert len(rows) >= S3.SPECS[fmt]["width"] * 6


def test_bit_groups_follow_section_4_1():
    assert {f: M.groups_of(f) for f in M.PRIMARY} == {
        "fp16": ["sign", "exponent", "fraction"], "bf16": ["sign", "exponent", "fraction"], "e4m11_sn": ["sign", "exponent", "fraction"],
        "fp8_e4m3fn": ["sign", "exponent", "fraction"], "fp8_e5m2": ["sign", "exponent", "fraction"],
        "int8_pt": ["sign", "value"], "int4_pt": ["sign", "value"]}
    assert S3.group_bits("bf16", "exponent") == tuple(range(7, 15)) and S3.group_bits("bf16", "fraction") == tuple(range(7))
    assert S3.group_bits("fp8_e5m2", "exponent") == (2, 3, 4, 5, 6) and S3.group_bits("fp8_e5m2", "fraction") == (0, 1)
    assert S3.group_bits("int8_pt", "value") == tuple(range(7)) and S3.group_bits("int4_pt", "value") == (0, 1, 2)
    assert S3.bit_location("int4_pt", 5, 3) == (2, 7) and S3.bit_location("int4_pt", 4, 0) == (2, 0)      # nibble order: even = low, odd = high


def test_trial_budgets_and_counts():
    names = [n for n, _ in quantizable_modules(build_mobilenetv2_cifar(10))]
    g = {f: M.make_configs(f, names, "global")[0] for f in M.PRIMARY}
    lw = {f: M.make_configs(f, names, "layerwise")[0] for f in M.PRIMARY}
    assert sum(len(v) for v in g.values()) == 5 * 3 * 2 + 2 * 2 * 2 == 38
    assert sum(len(v) for v in lw.values()) == 38 * 53
    assert 30 * 38 == 1140 and 10 * 38 * 53 == 20140
    assert all(c["group"] != "all" and c["scope_kind"] == "weights" for v in lw.values() for c in v)
    assert {c["k"] for v in g.values() for c in v} == {1, 8}
    assert all(len(c["scope_layers"]) == 53 for v in g.values() for c in v) and all(len(c["scope_layers"]) == 1 for v in lw.values() for c in v)
    ids = S4.expected_ids(g["fp16"], 30)
    assert len(ids) == len(set(ids)) == 6 * 30


def test_stratified_subset_is_balanced_deterministic_and_sorted():
    y = np.repeat(np.arange(10), 500)
    a, b = M.stratified_subset(y, 1000, 42), M.stratified_subset(y, 1000, 42)
    assert np.array_equal(a, b) and np.all(np.diff(a) > 0) and a.size == 1000
    assert np.bincount(y[a]).tolist() == [100] * 10
    assert not np.array_equal(a, M.stratified_subset(y, 1000, 7))
    assert M.stratified_subset(y, 0).size == 5000 and M.stratified_subset(y, 5000).size == 5000
    assert np.bincount(y[M.stratified_subset(y, 1003, 1)]).sum() == 1003


def test_shortlist_rule():
    rows = [{"format": "int4_pt", "layer_id": i, "layer_name": f"l{i}", "fault_group": "sign", "k": 8, "n_trials": 10, "drop_pp_mean": d}
            for i, d in enumerate([5.0, 0.2, 3.0, 0.6])]
    sl = M.select_shortlist(rows, top_n=2, min_mean_drop_pp=0.5)
    assert [r["layer_name"] for r in sl] == ["l0", "l2"] and "SELECTED ON THE SCREENING RESULTS" in sl[0]["selection_rule"]


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    model = build_mobilenetv2_cifar(10).eval()
    X = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    mods = quantizable_modules(model)
    names = [f"{n}.weight" for n, _ in mods]
    ws = OrderedDict((f"{n}.weight", m.weight.detach().numpy().astype(np.float32)) for n, m in mods)
    return model, X, y, [n for n, _ in mods], names, ws


@pytest.mark.parametrize("fmt", ["bf16", "int4_pt"])
def test_layerwise_campaign_end_to_end_is_resumable_and_exact(tiny, tmp_path, fmt):
    model, X, y, mnames, lnames, ws = tiny
    codec = S2.codec_registry()[fmt]
    stored = OrderedDict((n, codec.encode(w)) for n, w in ws.items())
    h0 = S2.content_sha256(stored)
    cfgs, _ = M.make_configs(fmt, mnames, "layerwise", layers=[0, 52])
    log = S4.TrialLog(tmp_path / "t.jsonl")
    subset = {"name": "unit"}
    ctx = {"experiment_id": "t", "run_id": "r", "model": "m", "dataset": "d", "checkpoint_hash": "c", "codec_version": "v"}
    ctrl, res = M.run_format(fmt, codec, stored, model, X, y, torch.device("cpu"), cfgs, 2, log, 42, ctx, "layerwise", subset, progress=lambda *_: None)
    assert ctrl["passed"] and ctrl["max_abs_logit_diff"] == 0.0 and ctrl["encoded_hash_before"] == ctrl["encoded_hash_after"]
    assert res["new"] == len(cfgs) * 2 and res["skipped"] == 0
    log.close()
    assert S2.content_sha256(stored) == h0                      # the input encodings are never modified
    recs = S4.load_records(tmp_path / "t.jsonl")
    assert len(recs) == res["new"] and all(r["restored_exactly"] and r["scope_verified"] for r in recs)
    assert {r["layer_id"] for r in recs} == {0, 52} and {r["fault_count"] for r in recs} == {1, 8}
    assert all(0.0 <= r["agreement_percent"] <= 100.0 for r in recs)
    rows = M.summarise(recs)
    assert len(rows) == len(cfgs) and all(r["n_trials"] == 2 for r in rows)
    assert S4.verify_completeness(recs, S4.expected_ids(cfgs, 2))["complete"]
    # resume: a second pass computes nothing new
    log2 = S4.TrialLog(tmp_path / "t.jsonl")
    _, res2 = M.run_format(fmt, codec, stored, model, X, y, torch.device("cpu"), cfgs, 2, log2, 42, ctx, "layerwise", subset, progress=lambda *_: None)
    assert res2["new"] == 0 and res2["skipped"] == res["new"]


def test_scope_verification_catches_a_tampered_manifest(tiny):
    model, X, y, mnames, lnames, ws = tiny
    codec = S2.codec_registry()["fp16"]
    store = S3.EncodedStore("fp16", OrderedDict((n, codec.encode(w)) for n, w in ws.items()))
    inj = S3.EncodedFaultInjector(model, store, codec, torch.device("cpu"))
    clean = S3.evaluate(inj.model, X, y)
    ctx = {"experiment_id": "t", "run_id": "r", "model": "m", "dataset": "d", "checkpoint_hash": "c", "codec_version": "v"}
    cfg = {"scope_layers": [3], "group": "exponent", "k": 8}
    rec = S3.run_trial(inj, X, y, clean, ctx, fmt="fp16", scope_layers=[3], scope_name="x", scope_kind="weights", group="exponent", k=8, trial=0, base_seed=42)
    M.verify_trial(rec, cfg, store, "fp16")                                                     # genuine record passes
    rec["flips"][0]["element"] += 1
    with pytest.raises(AssertionError):
        M.verify_trial(rec, cfg, store, "fp16")
