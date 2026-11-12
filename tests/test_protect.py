from collections import OrderedDict

import numpy as np
import pytest
import torch

from fr import protect as P
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


def test_vote_corrects_every_single_flip_and_documents_the_limit():
    rng = np.random.default_rng(0)
    for _ in range(50):
        v = int(rng.integers(0, 2 ** 32, dtype=np.uint64))
        for c in range(3):
            for b in range(32):
                cp = np.array([v, v, v], dtype=np.uint32)
                cp[c] ^= np.uint32(1 << b)
                assert int(P.vote_u32(*cp)) == v                       # any ONE flipped bit in any copy
        a = np.array([v, v, v], dtype=np.uint32)
        a[0] ^= np.uint32(1 << 5)
        a[1] ^= np.uint32(1 << 17)
        assert int(P.vote_u32(*a)) == v                               # two flips at DIFFERENT bit positions: still corrected
        a = np.array([v, v, v], dtype=np.uint32)
        a[0] ^= np.uint32(1 << 5)
        a[2] ^= np.uint32(1 << 5)
        assert int(P.vote_u32(*a)) == v ^ (1 << 5)                    # same bit in two copies: NOT corrected (documented limit)
    assert P.parse_option("int8_pt") == ("int8_pt", "none") and P.parse_option("int4_pt+scale_tmr") == ("int4_pt", "scale_tmr")
    assert P.option_token("int8_pt", "none") == "int8_pt" and P.option_token("int8_pt", "scale_tmr") == "int8_pt+scale_tmr"
    with pytest.raises(ValueError):
        P.parse_option("int8_pt+ecc")
    assert P.effective_scale(np.array([np.float32(0.25).view(np.uint32)])) == np.float32(0.25)


def test_tmr_storage_accounting_and_restriction_to_int(setup):
    net, stores, X, y = setup
    plain = E2.MixedStore(["int8_pt"] * 21, stores)
    prot = E2.MixedStore(["int8_pt+scale_tmr"] * 21, stores)
    a, b = plain.stored_bytes(), prot.stored_bytes()
    assert a["protection_overhead_bytes"] == 0 and b["protection_overhead_bytes"] == 21 * P.TMR_OVERHEAD_BYTES == 168
    assert b["total_bytes"] - a["total_bytes"] == 168 and b["payload_bytes"] == a["payload_bytes"]
    assert plain.hash() != prot.hash()
    with pytest.raises(ValueError):
        E2.MixedStore(["fp16+scale_tmr"] * 21, stores)               # nothing to protect in a float format
    assert [float(s) for s in plain.scales] == [float(s) for s in prot.scales]


def test_every_single_scale_flip_is_corrected_exactly_and_restored(setup):
    net, stores, X, y = setup
    asg = [("int8_pt+scale_tmr" if i % 2 == 0 else "int4_pt+scale_tmr") for i in range(21)]
    ms = E2.MixedStore(asg, stores)
    inj = E2.MixedFaultInjector(net, ms, CODECS, "cpu")
    ref = [f.clone() for f in inj.flat]
    h0 = ms.hash()
    for l in range(21):
        for c in range(3):
            for b in range(32):
                inj.inject([], [(l, c, b)])
                try:
                    assert float(ms.scales[l]) == float(stores[asg[l].split("+")[0]].scales[l])          # voted scale unchanged
                    assert torch.equal(inj.flat[l], ref[l])
                finally:
                    inj.restore()
    assert ms.hash() == h0 and all(torch.equal(a, b) for a, b in zip(inj.flat, ref))


def test_unprotected_scale_flip_changes_the_weights_protected_does_not(setup):
    net, stores, X, y = setup
    for tok, same in (("int8_pt", False), ("int8_pt+scale_tmr", True)):
        ms = E2.MixedStore([tok] * 21, stores)
        inj = E2.MixedFaultInjector(net, ms, CODECS, "cpu")
        clean = S3.evaluate(inj.model, X, y, return_logits=True)
        inj.inject([], [(3, 0, 30)])                                  # a high exponent bit of copy 0
        try:
            out = S3.evaluate(inj.model, X, y, return_logits=True)
            changed = not torch.equal(out["logits"], clean["logits"])
        finally:
            inj.restore()
        assert changed == (not same)
        assert torch.equal(S3.evaluate(inj.model, X, y, return_logits=True)["logits"], clean["logits"])


def test_scale_bit_draw_is_uniform_over_all_stored_scale_bits(setup):
    net, stores, X, y = setup
    asg = ["int8_pt+scale_tmr"] + ["int8_pt"] * 2 + ["fp16"] * 18
    ms = E2.MixedStore(asg, stores)
    pos = E2.scale_bit_positions(ms)
    assert len(pos) == 3 * 32 + 2 * 32 and {p[0] for p in pos} == {0, 1, 2}
    rng = np.random.default_rng(0)
    counts = np.zeros(3)
    for _ in range(6000):
        l, c, b = E2.draw_scale_flips(rng, ms, 1)[0]
        assert 0 <= c < (3 if l == 0 else 1) and 0 <= b < 32
        counts[l] += 1
    assert counts[0] / counts.sum() == pytest.approx(96 / 160, abs=0.03)                  # protected layer holds 3x the bits
    assert len(set(E2.draw_scale_flips(rng, ms, 5))) == 5                                  # distinct positions


def test_trial_record_with_protected_scale_flip(setup):
    net, stores, X, y = setup
    ms = E2.MixedStore(["int4_pt+scale_tmr"] * 21, stores)
    inj = E2.MixedFaultInjector(net, ms, CODECS, "cpu")
    clean = S3.evaluate(inj.model, X, y)
    for t in range(4):
        r = E2.run_mixed_trial(inj, X, y, clean, CTX, assignment_id="T", dist="scales_k1", k=1, trial=t, base_seed=0)
        assert r["restored_exactly"] and r["flip_applied_to_encoded_bytes"] and r["fault_accuracy"] == clean["accuracy"] and not r["failure"]
        assert r["flips"][0][0] == "s" and 0 <= r["flips"][0][2] < 3
