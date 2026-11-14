import csv
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fr import ecc  # noqa: E402
from fr import protect as P  # noqa: E402
from fr import stage2 as S2  # noqa: E402
from fr import stage3 as S3  # noqa: E402
from fr import stage5 as S5  # noqa: E402
from fr import stage5_e2e as E2  # noqa: E402
from fr import stage6 as S6  # noqa: E402
from fr.model import build_mobilenetv2_cifar, quantizable_modules  # noqa: E402

REP = ROOT / "reports" / "mobilenetv2_cifar10"
needs_reports = pytest.mark.skipif(not (REP / "stage_05_protection" / "protection_storage.csv").exists(), reason="Stage 3-5 report files not present")


def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def fake_scale_rows(layers):
    """PLACEHOLDER scale-word results for tests only (the real ones come from stage6_scale_faults.py)."""
    rng = np.random.default_rng(0)
    return [{"layer_name": l, "format_key": f, "mean_drop_pp": str(float(rng.uniform(0, 8)))} for l in layers for f in ("int8_pt", "int4_pt")]


@pytest.fixture(scope="module")
def table():
    s5 = REP / "stage_05_protection"
    storage = rd(s5 / "protection_storage.csv")
    layers = list(OrderedDict.fromkeys(r["layer_name"] for r in sorted(storage, key=lambda r: int(r["layer_index"]))))
    rows, meta = S6.build_candidates(rd(REP / "stage_03_layerwise_sensitivity/layerwise_sensitivity_results.csv"),
                                     rd(REP / "stage_04_encoded_fault_injection/layerwise_summary.csv"), storage, rd(s5 / "protection_layer_summary.csv"),
                                     rd(s5 / "protection_summary.csv"), fake_scale_rows(layers))
    return rows, meta, layers


# ------------------------------------------------------------------------------------------------ options
def test_options_roundtrip_and_unsupported_combinations():
    opts = S6.all_options()
    assert len(opts) == 30 and len(set(opts)) == 30
    for o in opts:
        f, p = S6.parse_option(o)
        assert S6.option_token(f, p) == o and S6.long_token(S6.short_token(o)) == o
    with pytest.raises(ValueError):
        S6.parse_option("fp16+scale_tmr")
    with pytest.raises(ValueError):
        S6.parse_option("int8_pt+ecc")
    assert S6.base_token("int8_pt+secded") == "int8_pt" and S6.base_token("int4_pt+scale_tmr") == "int4_pt+scale_tmr"
    assert len({S6.short_token(o) for o in opts}) == 30


# ------------------------------------------------------------------------------------------------ candidate table
@needs_reports
def test_candidate_table_shape_definitions_and_storage(table):
    rows, meta, layers = table
    assert len(layers) == 53 and len(rows) == meta["candidates"] == 53 * 30 and meta["quantized_weights_N"] == 2_202_560
    assert all(float(r["Q_ce_loss_change"]) >= 0 for r in rows)
    by = {(r["layer_name"], r["option"]): r for r in rows}
    # unprotected storage = Stage 2 totals; protection overheads = Stage 5 actual bytes
    tot = lambda o: sum(int(by[(l, o)]["S_total_bytes"]) for l in layers)
    assert tot("fp16") == 4_405_120 and tot("int8_pt") == 2_202_772 and tot("int4_pt") == 1_101_492
    assert tot("fp16+secded") == 5_368_740 and tot("int4_pt+secded") == 1_342_452 and tot("fp8_e4m3fn+parity") == 2_271_391
    assert tot("int8_pt+scale_tmr") == 2_202_772 + 8 * 53 and tot("int4_pt+dmr") == 2_202_984
    # residual risk: SECDED removes the single-bit weight risk (measured residual rate 0), parity / DMR / TMR(payload) do not
    for l in layers:
        for f in S6.FORMATS:
            r0 = float(by[(l, f)]["R_exposure_pp"])
            assert float(by[(l, S6.option_token(f, "secded"))]["R_exposure_pp"]) == 0.0
            assert float(by[(l, S6.option_token(f, "parity"))]["R_exposure_pp"]) == pytest.approx(r0) == pytest.approx(float(by[(l, S6.option_token(f, "dmr"))]["R_exposure_pp"]))
    # scale-word risk exists only for INT; SECDED and TMR remove it
    assert all(float(r["R_scale_pp"]) == 0.0 for r in rows if S3.SPECS[r["datatype"]]["kind"] != "int")
    assert all(float(r["R_scale_pp"]) == 0.0 for r in rows if r["protection"] in ("secded", S6.TMR))
    assert any(float(r["R_scale_pp"]) > 0 for r in rows if r["option"] in ("int8_pt", "int4_pt"))
    # the declared-distribution exposure of the unprotected formats adds up to the Stage 4 magnitude (8 random flips over the whole model)
    exp = {f: sum(float(by[(l, f)]["R_exposure_pp"]) for l in layers) for f in S6.FORMATS}
    assert exp["fp16"] > 1.0 and exp["int8_pt"] < exp["fp16"] and exp["int4_pt"] < exp["fp16"]


@needs_reports
def test_problem_sweep_and_constraints(table):
    rows, meta, layers = table
    pj, nw = S6.rows_to_problem(rows, "joint")
    pd, _ = S6.rows_to_problem(rows, "datatype_only")
    assert len(pj.formats) == 30 and len(pd.formats) == 7 and int(nw.sum()) == 2_202_560
    sweep = S6.run_sweep({"joint": (pj, nw), "datatype_only": (pd, nw)}, [0, 1], [8, 16], [0.5], ["minmax"], 1.0)
    assert len(sweep) == 2 * 1 * 2 * 2 * 2
    ok = [r for r in sweep if r["status"] == "optimal"]
    assert ok
    for r in ok:
        assert r["storage_bytes"] <= r["B_bytes"] + 1e-6
        if r["tau_fault_pred_drop_pp"] != "":
            assert r["pred_weight_fault_drop_pp"] <= r["tau_fault_pred_drop_pp"] + 1e-9
        if r["mode"] == "joint":
            assert r["pred_max_scale_word_risk_pp"] <= 1.0 + 1e-9
        ch = [pj.formats.index(S6.long_token(s)) for s in r["assignment"].split(";")]
        assert len(ch) == 53
    d = S6.distinct_assignments(sweep)
    assert d and sum(x["n_settings"] for x in d) == len(ok)


def test_milp_matches_brute_force_on_a_tiny_problem():
    rng = np.random.default_rng(3)
    L, D = 4, 5
    p = S5.Problem([f"l{i}" for i in range(L)], [f"o{j}" for j in range(D)], rng.random((L, D)), rng.random((L, D)), rng.integers(10, 100, (L, D)).astype(float), Rscale=rng.random((L, D)))
    for lam in (0.0, 1.0, 5.0):
        for B in (150.0, 250.0):
            s = S5.solve(p, lam, "minmax", "weights", max_bytes=B, max_sum_R=1.5, tau_scale=0.9)
            bf = S5.brute_force(p, lam, "minmax", "weights", max_bytes=B, max_sum_R=1.5, tau_scale=0.9)
            assert (s["choice"] is None) == (bf is None)
            if bf is not None:
                assert s["objective"] == pytest.approx(bf[0], abs=1e-9)


def test_measured_selection_rule():
    def item(aid, drop, w, bits, obj, sc=None, bytes_=1000):
        fe = {"weights_k8": {"drop_pp_mean": w}}
        if sc is not None:
            fe["scales_k1"] = {"drop_pp_mean": sc}
        return {"assignment_id": aid, "clean_validation": {"drop_pp_vs_fp32": drop}, "fault_end_to_end": fe, "storage": {"avg_bits_per_weight": bits, "actual_stored_bytes": bytes_},
                "objective_value": obj}
    items = [item("a", 0.2, 0.1, 7.0, 0.5), item("b", 0.4, 0.2, 6.0, 0.3), item("c", 0.9, 0.0, 5.0, 0.1), item("d", 0.1, 0.9, 7.0, 0.2), item("e", 0.1, 0.1, 9.0, 0.0),
             item("f", 0.1, 0.1, 7.0, 0.05, sc=3.0), {"assignment_id": "g", "clean_validation": {"drop_pp_vs_fp32": 0.0}, "fault_end_to_end": {}, "storage": {"avg_bits_per_weight": 1.0, "actual_stored_bytes": 1}, "objective_value": -1}]
    assert S6.best_item(items, 0.5, 0.5, 8.0, 1.0)["assignment_id"] == "b"           # c fails delta, d fails tau, e fails B, f fails tau_scale, g has no fault measurement
    assert S6.best_item(items, 0.5, 0.5, 8.0, None)["assignment_id"] == "f"          # without the scale limit f is feasible and cheapest in the objective
    assert S6.best_item(items, 0.05, 0.5, 8.0, 1.0) is None


# ------------------------------------------------------------------------------------------------ protected faults on an assembled model
@pytest.fixture(scope="module")
def assembled():
    torch.manual_seed(0)
    model = build_mobilenetv2_cifar(10).eval()
    mods = quantizable_modules(model)
    reg = S2.codec_registry()
    stores = {f: S3.EncodedStore(f, OrderedDict((f"{n}.weight", reg[f].encode(m.weight.detach().numpy())) for n, m in mods)) for f in S6.FORMATS}
    return model, mods, stores, reg


def make(assembled, opts):
    model, mods, stores, reg = assembled
    ms = E2.MixedStore([S6.base_token(t) for t in opts], stores)
    prots = [S6.parse_option(t)[1] for t in opts]
    return ms, S6.ProtectedFaults(ms, prots), prots


def test_stored_bytes_match_the_table_arithmetic(assembled):
    L = 53
    for o in ("fp16", "fp16+secded", "int8_pt+scale_tmr", "int4_pt+parity", "fp8_e5m2+dmr", "int4_pt+secded"):
        ms, pf, prots = make(assembled, [o] * L)
        sb = S6.stored_bytes(ms, prots)
        f, p = S6.parse_option(o)
        expect = 0
        for l in range(L):
            payload = int(ms.words[l].nbytes)
            scale = 4 if S3.SPECS[f]["kind"] == "int" else 0
            expect += payload + scale + 8 if p == S6.TMR else ecc.storage_bytes(p, payload, scale)["total_bytes"]
        assert sb["total_bytes"] == expect


def _flip_in_layer(ms, l, element, bit):
    return [(l, element, bit)]


def test_secded_removes_single_data_flips_and_keeps_double_flips_in_one_word(assembled):
    L = 53
    for fmt in ("fp16", "int4_pt"):
        ms, pf, _ = make(assembled, [S6.option_token(fmt, "secded")] * L)
        w = S3.SPECS[fmt]["width"]
        assert pf.residual_from_data_flips([(5, 7, w - 1)]) == []                    # single flip corrected
        # two flips in the SAME 32-bit word: uncorrectable -> consumed as read -> both residual
        per = 32 // w
        e0 = 3 * per
        res = pf.residual_from_data_flips([(5, e0, 0), (5, e0 + per - 1, w - 1)])
        assert sorted(res) == sorted([(5, e0, 0), (5, e0 + per - 1, w - 1)])
        # two flips in different words: each corrected
        assert pf.residual_from_data_flips([(5, 0, 0), (5, 10 * per, 0)]) == []


def test_other_protections_leave_the_flip_in_place(assembled):
    L = 53
    flips = [(7, 100, 3), (20, 5, 0)]
    for o in ("fp16", "fp16+parity", "fp16+dmr"):
        ms, pf, _ = make(assembled, [o] * L)
        assert pf.residual_from_data_flips(flips) == sorted(flips)
    ms, pf, _ = make(assembled, ["int8_pt+scale_tmr"] * L)
    assert pf.residual_from_data_flips(flips) == sorted(flips)                        # TMR protects the scale word only


def test_stored_flips_on_redundancy_bits_never_reach_the_weights(assembled):
    L = 53
    ms, pf, _ = make(assembled, ["fp16+secded"] * L)
    for pos in ecc.check_positions("secded"):
        assert pf.residual_from_stored_flips([(4, 2, pos)]) == []
    ms, pf, _ = make(assembled, ["fp16+parity"] * L)
    assert pf.residual_from_stored_flips([(4, 2, 32)]) == []                           # the parity bit itself
    assert len(pf.residual_from_stored_flips([(4, 2, 5)])) == 1                       # a data bit: detected only, still residual
    ms, pf, _ = make(assembled, ["fp16+dmr"] * L)
    assert pf.residual_from_stored_flips([(4, 2, 40)]) == []                           # copy B flipped: the system reads copy A


def test_stored_bit_distribution_covers_data_and_redundancy_without_padding(assembled):
    L = 53
    ms, pf, _ = make(assembled, ["int4_pt+secded"] * L)
    rng = np.random.default_rng(1)
    seen_check = seen_data = 0
    for _ in range(300):
        for (l, w, pos) in pf.draw_stored_flips(rng, 8):
            assert 0 <= w < pf.nwords[l] and 0 <= pos < 39
            valid = pf.nbits[l] - 32 * w
            if pos in ecc.check_positions("secded"):
                seen_check += 1
            else:
                seen_data += 1
                data_bits = {ecc.stored_position("secded", b) for b in range(min(32, valid))}
                assert pos in data_bits                                                # never a padding bit
    assert seen_check > 0 and seen_data > 0
    assert 0.1 < seen_check / (seen_check + seen_data) < 0.25                          # 7 of 39 stored bits are check bits (~18 %)
    d = pf.draw_stored_flips(np.random.default_rng(2), 8)
    assert len(set(d)) == 8


def test_scale_words_under_each_protection(assembled):
    L = 53
    for o, survives in (("int8_pt", True), ("int8_pt+parity", True), ("int8_pt+dmr", True), ("int8_pt+secded", False), ("int8_pt+scale_tmr", False)):
        ms, pf, _ = make(assembled, [o] * L)
        pos = pf.scale_positions()
        assert len({l for l, _, _ in pos}) == 53
        l, c, b = [x for x in pos if x[0] == 3][0]                                     # a stored scale bit of layer 3 (a data bit)
        res = pf.residual_from_scale_flips([(l, c, b)])
        if o.endswith("scale_tmr"):
            assert len(pos) == 53 * 96
            assert res == [(3, 0, 0)]                                                  # handed to the store, whose majority vote corrects it (see next test)
        else:
            assert (len(res) == 1) == survives


def test_scale_tmr_majority_corrects_every_single_scale_bit_in_the_assembled_store(assembled):
    ms, pf, _ = make(assembled, ["int8_pt+scale_tmr"] * 53)
    ref = np.array(ms.scales[3], copy=True)
    for c in range(3):
        for b in range(32):
            old = ms.flip_scale(3, c, b)
            assert ms.scales[3] == ref
            ms.set_scale_copy(3, c, old)


def test_protected_trial_end_to_end_is_exact_and_records_both_fault_sets(assembled):
    model, mods, stores, reg = assembled
    torch.manual_seed(1)
    X, y = torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,))
    from fr import stage3 as S3_
    ctx = {"experiment_id": "t", "run_id": "r", "model": "m", "dataset": "d", "checkpoint_hash": "c", "codec_version": "v"}
    for o in ("int8_pt+secded", "fp16", "int4_pt+scale_tmr", "fp8_e4m3fn+parity"):
        opts = [o] * 53
        ms, pf, prots = make(assembled, opts)
        inj = E2.MixedFaultInjector(model, ms, reg, torch.device("cpu"))
        clean = S3_.evaluate(inj.model, X, y)
        h0 = ms.hash()
        for dist in S6.DISTRIBUTIONS:
            if dist == "scales_k1" and S3.SPECS[S6.parse_option(o)[0]]["kind"] != "int":
                continue
            for t in range(3):
                rec = S6.run_protected_trial(inj, pf, X, y, clean, ctx, assignment_id="T", dist=dist, trial=t, base_seed=42)
                assert rec["restored_exactly"] and rec["flip_applied_to_encoded_bytes"] and rec["distribution"] == dist and rec["requested_k"] == S6.DIST_K[dist]
                assert rec["trial_id"] == f"T|{dist}|k{S6.DIST_K[dist]}|t{t}" and "stored_faults" in rec
                if o.endswith("secded") and dist == "weights_k8":
                    assert rec["residual_fault_count"] == 0 and rec["fault_accuracy"] == clean["accuracy"]      # eight single flips in different words: all corrected
                if dist == "weights_k8" and o in ("fp16", "int4_pt+scale_tmr", "fp8_e4m3fn+parity"):
                    assert rec["residual_fault_count"] == 8
        assert ms.hash() == h0


def test_trials_are_reproducible_from_the_seed(assembled):
    model, mods, stores, reg = assembled
    ms, pf, _ = make(assembled, ["fp16+secded"] * 53)
    a = S6.draw_trial_faults(pf, "stored_k8", 12345, 8)
    b = S6.draw_trial_faults(pf, "stored_k8", 12345, 8)
    assert a == b
