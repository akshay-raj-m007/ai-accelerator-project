from collections import OrderedDict

import numpy as np
import pytest
import torch

from fr import stage2 as S2
from fr import stage3 as S3
from fr import stage5_e2e as E2
from fr.model import build_resnet18_cifar, quantizable_modules

FMTS = ["e4m11_sn", "fp16", "fp8_e4m3fn", "int8_pt", "int4_pt"]
CODECS = S2.make_codecs(True)
CTX = {k: "x" for k in ("experiment_id", "run_id", "model", "dataset", "checkpoint_hash", "codec_version")}


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    net = build_resnet18_cifar().eval()
    mods = quantizable_modules(net)
    stores = {}
    for f in FMTS:
        enc = OrderedDict((f"{n}.weight", CODECS[f].encode(m.weight.detach().numpy())) for n, m in mods)
        stores[f] = S3.EncodedStore(f, enc)
    g = torch.Generator().manual_seed(1)
    X, y = torch.randn(16, 3, 32, 32, generator=g), torch.randint(0, 10, (16,), generator=g)
    return net, stores, X, y


def _assign():
    return [FMTS[i % 5] for i in range(21)]


def test_uniform_assembly_equals_single_format_model(setup):
    net, stores, X, y = setup
    for f in ("int4_pt", "fp8_e4m3fn"):
        ms = E2.MixedStore([f] * 21, stores)
        mi = E2.MixedFaultInjector(net, ms, CODECS, "cpu")
        ref = S3.EncodedFaultInjector(net, stores[f], CODECS[f], "cpu")
        a = S3.evaluate(mi.model, X, y, return_logits=True)["logits"]
        b = S3.evaluate(ref.model, X, y, return_logits=True)["logits"]
        assert torch.equal(a, b)
        assert ms.stored_bytes()["total_bytes"] == sum(S2.storage_row(CODECS[f], e)["total_bytes"] for e in stores[f].to_encoded().values())


def test_mixed_assembly_no_fault_control_and_restoration(setup):
    net, stores, X, y = setup
    asg = _assign()
    ms = E2.MixedStore(asg, stores)
    inj = E2.MixedFaultInjector(net, ms, CODECS, "cpu")
    clean = S3.evaluate(inj.model, X, y, return_logits=True)
    h0 = ms.hash()
    r0 = E2.run_mixed_trial(inj, X, y, clean, CTX, assignment_id="T", dist="weights_k8", k=0, trial=0, base_seed=0)
    assert r0["fault_count"] == 0 and r0["fault_accuracy"] == clean["accuracy"] and r0["restored_exactly"] and not r0["failure"]
    for t in range(6):
        r = E2.run_mixed_trial(inj, X, y, clean, CTX, assignment_id="T", dist="weights_k8", k=8, trial=t, base_seed=0)
        assert r["restored_exactly"] and r["encoded_hash_before"] == r["encoded_hash_after_restore"] == h0
        assert r["fault_count"] == 8 and len({(f[1], f[2]) for f in r["flips"]}) == 8            # distinct weights
        assert all(0 <= f[3] < S3.SPECS[f[4]]["width"] and f[4] == asg[f[1]] for f in r["flips"])   # bit inside the layer's own encoding
        assert r["flip_applied_to_encoded_bytes"]
    assert ms.hash() == h0
    after = S3.evaluate(inj.model, X, y, return_logits=True)["logits"]
    assert torch.equal(after, clean["logits"])


def test_flip_hits_the_intended_layer_format_element_and_bit(setup):
    net, stores, X, y = setup
    asg = _assign()                                   # layer 3 -> int8, layer 4 -> int4, layer 1 -> fp16
    ms = E2.MixedStore(asg, stores)
    inj = E2.MixedFaultInjector(net, ms, CODECS, "cpu")
    before1, before3, before4 = (inj.flat[l][5].item() for l in (1, 3, 4))
    inj.inject([(1, 5, 14), (3, 5, 7), (4, 5, 3)])          # fp16 exponent MSB, int8 sign, int4 sign
    try:
        assert abs(inj.flat[1][5].item()) != abs(before1)
        assert abs(inj.flat[3][5].item() - before3) == pytest.approx(128 * float(ms.scales[3]), rel=1e-5)
        assert abs(inj.flat[4][5].item() - before4) == pytest.approx(8 * float(ms.scales[4]), rel=1e-5)
        assert inj.flat[1][4].item() == inj.flat[1][4].item()
    finally:
        inj.restore()
    assert (inj.flat[1][5].item(), inj.flat[3][5].item(), inj.flat[4][5].item()) == (before1, before3, before4)
    with pytest.raises(ValueError):
        ms.flip(3, 0, 8)                               # int8 has only 8 bits (0..7)
    with pytest.raises(ValueError):
        ms.flip(4, 0, 4)                               # int4 has only 4 bits


def test_scale_distribution_requires_int_layers_and_restores(setup):
    net, stores, X, y = setup
    assert not E2.has_scale_words(["fp16"] * 21) and E2.has_scale_words(_assign())
    ms = E2.MixedStore(["fp16"] * 21, stores)
    with pytest.raises(ValueError):
        E2.draw_scale_flips(np.random.default_rng(0), ms, 1)
    ms = E2.MixedStore(_assign(), stores)
    inj = E2.MixedFaultInjector(net, ms, CODECS, "cpu")
    clean = S3.evaluate(inj.model, X, y)
    for t in range(4):
        r = E2.run_mixed_trial(inj, X, y, clean, CTX, assignment_id="T", dist="scales_k1", k=1, trial=t, base_seed=0)
        assert r["restored_exactly"] and r["fault_count"] == 1 and r["flips"][0][0] == "s" and r["flips"][0][4] in ("int8_pt", "int4_pt")


def test_summarize_trials_and_pareto():
    recs = [{"clean_accuracy": 96.0, "fault_accuracy": fa, "clean_loss": 0.1, "fault_loss": fl, "sample_count": 1000, "requested_k": 8,
             "output_flag": fg, "failure": fg != "none"} for fa, fl, fg in
            [(96.0, 0.1, "none"), (90.0, 0.5, "none"), (10.0, None, "logits_nonfinite,loss_nonfinite"), (95.0, 0.2, "none")]]
    s = E2.summarize_trials(recs)
    d = np.array([0, 6, 86, 1.0])
    assert s["n_trials"] == 4 and s["drop_pp_mean"] == pytest.approx(d.mean()) and s["drop_pp_std"] == pytest.approx(d.std(ddof=1))
    assert s["drop_pp_p90"] == pytest.approx(np.percentile(d, 90)) and s["failures"] == 1 and s["fail_loss_nonfinite"] == 1 and s["n_loss_finite"] == 3
    assert E2.pareto_front([[1, 5], [2, 2], [3, 1], [3, 3], [2, 2]]) == [0, 1, 2, 4]
