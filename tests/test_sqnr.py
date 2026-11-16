import math
from collections import OrderedDict

import numpy as np
import pytest
import torch

from fr import weight_sqnr as Q
from fr import stage2 as S2
from fr import stage3 as S3
from fr import stage5_e2e as E2
from fr import stage6 as S6
from fr.model import build_mobilenetv2_cifar, build_resnet18_cifar, quantizable_modules

FMTS = ["e4m11_sn", "fp16", "fp8_e4m3fn", "int8_pt", "int4_pt"]
CODECS = S2.codec_registry()


def make(net, fmts=FMTS):
    mods = quantizable_modules(net)
    stores = {}
    for f in fmts:
        enc = OrderedDict((f"{n}.weight", CODECS[f].encode(m.weight.detach().numpy())) for n, m in mods)
        stores[f] = S3.EncodedStore(f, enc)
    return mods, stores


def test_sqnr_db_edge_cases():
    assert Q.sqnr_db(4.0, 0.0) == float("inf")
    assert Q.sqnr_db(4.0, float("nan")) == float("-inf")
    assert Q.sqnr_db(100.0, 1.0) == pytest.approx(20.0)


@pytest.fixture(scope="module")
def resnet():
    torch.manual_seed(0)
    net = build_resnet18_cifar().eval()
    return (net,) + make(net)


def test_clean_sqnr_matches_direct_computation_for_uniform_formats(resnet):
    net, mods, stores = resnet
    for f in ("int8_pt", "fp8_e4m3fn"):
        ms = E2.MixedStore([f] * len(mods), stores)
        inj = E2.MixedFaultInjector(net, ms, CODECS, "cpu")
        meter = Q.SqnrMeter([m.weight for _, m in mods], inj)
        sig = sum(float(m.weight.detach().double().pow(2).sum()) for _, m in mods)
        err = sum(float((m.weight.detach().double().reshape(-1) - inj.flat[l].double()).pow(2).sum()) for l, (_, m) in enumerate(mods))
        assert meter.clean_db() == pytest.approx(10 * math.log10(sig / err), abs=1e-9)


def test_fp32_assembled_model_would_be_infinite_and_faults_never_improve_beyond_noise(resnet):
    net, mods, stores = resnet
    opts = [FMTS[i % 5] for i in range(len(mods))]
    ms = E2.MixedStore(opts, stores)
    inj = E2.MixedFaultInjector(net, ms, CODECS, "cpu")
    pf = S6.ProtectedFaults(ms, ["none"] * len(opts))
    meter = Q.SqnrMeter([m.weight for _, m in mods], inj)
    h0 = ms.hash()
    rows = Q.run_assignment(meter, pf, ms, "T", ["weights_k8"], {"weights_k8": 8}, 12, 0, "e2")
    assert ms.hash() == h0 and all(r["restored_exactly"] for r in rows)
    assert all(r["n_residual_flips"] == 8 for r in rows)
    clean = meter.clean_db()
    assert all(r["sqnr_db"] <= clean + 1e-6 for r in rows)                     # extra bit errors can only add error (a flip may rarely hit an already-wrong weight toward w)
    assert meter.clean_db() == pytest.approx(clean)                              # state is back to clean after the trials
    s = Q.summarize(rows, clean)["weights_k8"]
    assert s["n_trials"] == 12 and s["restored_all"]


def test_scale_flip_on_unprotected_int_layer_collapses_sqnr_and_tmr_prevents_it(resnet):
    net, mods, stores = resnet
    L = len(mods)
    plain = E2.MixedStore(["int8_pt"] * L, stores)
    tmr = E2.MixedStore(["int8_pt+scale_tmr"] * L, stores)
    res = {}
    for name, ms in (("plain", plain), ("tmr", tmr)):
        inj = E2.MixedFaultInjector(net, ms, CODECS, "cpu")
        pf = S6.ProtectedFaults(ms, ["none" if name == "plain" else "scale_tmr"] * L)
        meter = Q.SqnrMeter([m.weight for _, m in mods], inj)
        rows = Q.run_assignment(meter, pf, ms, "S", ["scales_k1"], {"scales_k1": 1}, 40, 0, "e2")
        res[name] = (meter.clean_db(), min(r["sqnr_db"] for r in rows), rows)
    assert res["plain"][1] < res["plain"][0] - 3                                  # some single scale bit costs several dB or more (exponent bits)
    assert all(abs(r["sqnr_db"] - res["tmr"][0]) < 1e-9 for r in res["tmr"][2])  # TMR: every single flipped scale bit is voted out


def test_mobilenet_secded_removes_all_weight_flips_so_sqnr_is_unchanged():
    torch.manual_seed(0)
    net = build_mobilenetv2_cifar().eval()
    mods, stores = make(net, ["e4m11_sn", "int8_pt"])
    opts = ["e4m11_sn" if i % 2 == 0 else "int8_pt" for i in range(len(mods))]
    ms = E2.MixedStore(opts, stores)
    inj = E2.MixedFaultInjector(net, ms, CODECS, "cpu")
    pf = S6.ProtectedFaults(ms, ["secded"] * len(opts))
    meter = Q.SqnrMeter([m.weight for _, m in mods], inj)
    rows = Q.run_assignment(meter, pf, ms, "M", ["weights_k8"], {"weights_k8": 8}, 6, 42, "s6")
    assert all(r["restored_exactly"] for r in rows)
    assert all(r["n_residual_flips"] == 0 for r in rows)                          # single flips in distinct words: SECDED corrects them
    assert all(r["sqnr_db"] == pytest.approx(meter.clean_db()) for r in rows)
