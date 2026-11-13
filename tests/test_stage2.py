import math

import numpy as np
import pytest
import torch

from fr import stage2 as S2
from fr.model import build_resnet18_cifar
from fr.model import quantizable_modules


def test_all_codec_selftests_pass():
    res = S2.run_codec_selftests()
    failed = [r for r in res if not r["passed"]]
    assert not failed, failed
    names = " ".join(r["test"] for r in res)
    assert "equals torch.float8_e4m3fn casting" in names and "equal torch.float8_e4m3fn" in names   # cross-checks ran


def test_nibble_packing_and_bit_location():
    c = np.array([1, 2, 3], dtype=np.uint8)
    assert list(S2.pack_nibbles(c)) == [0x21, 0x03]
    assert list(S2.unpack_nibbles(S2.pack_nibbles(c), 3)) == [1, 2, 3]
    # flipping bit b of element i in the packed byte changes exactly that code bit
    packed = S2.pack_nibbles(np.array([0x0, 0x0, 0x0, 0x0], dtype=np.uint8))
    byte, pos = S2.int4_bit_location(3, 2)
    packed[byte] ^= np.uint8(1 << pos)
    assert list(S2.unpack_nibbles(packed, 4)) == [0, 0, 0, 4]
    with pytest.raises(ValueError):
        S2.int4_bit_location(0, 4)


@pytest.fixture(scope="module")
def weights():
    torch.manual_seed(0)
    net = build_resnet18_cifar().eval()
    return {f"{n}.weight": m.weight.detach().numpy().astype(np.float32) for n, m in quantizable_modules(net)}


def test_encode_decode_from_stored_words_only(weights):
    w = weights["layer1.0.conv1.weight"]
    for key, codec in S2.make_codecs().items():
        enc = codec.encode(w)
        assert enc["words"].dtype in (np.uint8, np.uint16) and enc["shape"] == w.shape and enc["n"] == w.size
        what = codec.decode(enc)
        assert what.shape == w.shape and what.dtype == np.float32 and np.isfinite(what).all()
        row = S2.roundtrip_row(w, what)
        assert row["sqnr_db"] > 0, key
    # encoded payload sizes are the real bytes
    c = S2.make_codecs()
    assert c["int4_pt"].encode(w)["words"].nbytes == (w.size + 1) // 2
    assert c["int8_pt"].encode(w)["words"].nbytes == w.size
    assert c["fp8_e4m3fn"].encode(w)["words"].nbytes == w.size
    assert c["fp16"].encode(w)["words"].nbytes == 2 * w.size


def test_formats_are_independent_and_deterministic(weights):
    w = weights["layer2.0.conv1.weight"]
    c = S2.make_codecs()["int8_pt"]
    a, b = c.encode(w), c.encode(w.copy())
    assert np.array_equal(a["words"], b["words"]) and float(a["scale"]) == float(b["scale"])
    assert S2.content_sha256({"x": a}) == S2.content_sha256({"x": b})
    flipped = {"x": dict(a, words=a["words"] ^ np.uint8(1))}
    assert S2.content_sha256(flipped) != S2.content_sha256({"x": a})


def test_storage_accounting(weights):
    c = S2.make_codecs()
    w = np.arange(7, dtype=np.float32) * 0.01                 # odd count -> INT4 padding nibble
    e4 = c["int4_pt"].encode(w)
    st = S2.storage_row(c["int4_pt"], e4)
    assert st == {"theoretical_bits": 28, "payload_bytes": 4, "padding_bits": 4, "scale_bytes": 4,
                  "codec_metadata_bytes": 0, "total_bytes": 8}
    st8 = S2.storage_row(c["int8_pt"], c["int8_pt"].encode(w))
    assert st8["payload_bytes"] == 7 and st8["scale_bytes"] == 4 and st8["padding_bits"] == 0
    assert S2.storage_row(c["fp16"], c["fp16"].encode(w))["scale_bytes"] == 0


def test_save_load_roundtrip_and_hash(weights, tmp_path):
    names = ["conv1.weight", "layer1.0.conv1.weight", "fc.weight"]
    for key, codec in S2.make_codecs().items():
        enc = {n: codec.encode(weights[n]) for n in names}
        S2.save_encoded(tmp_path / f"{key}.npz", enc)
        back = S2.load_encoded(tmp_path / f"{key}.npz", names)
        assert S2.content_sha256(back) == S2.content_sha256(enc)
        for n in names:
            assert np.array_equal(codec.decode(back[n]), codec.decode(enc[n]), equal_nan=True)


def test_pooled_row_matches_direct_computation(weights):
    c = S2.make_codecs()["fp8_e4m3fn"]
    parts, ws, hs = [], [], []
    for n in ("conv1.weight", "layer1.0.conv1.weight", "fc.weight"):
        w = weights[n]
        h = c.decode(c.encode(w))
        parts.append(S2.roundtrip_row(w, h))
        ws.append(w.reshape(-1).astype(np.float64))
        hs.append(h.reshape(-1).astype(np.float64))
    g = S2.pooled_row(parts)
    W, H = np.concatenate(ws), np.concatenate(hs)
    assert g["sqnr_db"] == pytest.approx(10 * math.log10((W @ W) / ((W - H) @ (W - H))), rel=1e-9)
    assert g["max_abs_error"] == pytest.approx(np.abs(W - H).max())


def test_fp8_e4m3fn_flushes_tiny_weights_and_reports_it():
    c = S2.make_codecs()["fp8_e4m3fn"]
    w = np.array([1e-4, -1e-4, 0.5, 0.0], dtype=np.float32)      # 1e-4 < 2^-10 / ... rounds to zero
    r = S2.roundtrip_row(w, c.decode(c.encode(w)))
    assert r["zeroed_from_nonzero"] == 2


# ------------------------------------------------------------------------------------ MobileNetV2 protocol additions
def test_protocol_format_set_order_roles_and_policy():
    c = S2.make_codecs_protocol("per_tensor")
    assert list(c) == ["fp16", "bf16", "e4m11_sn", "fp8_e4m3fn", "fp8_e5m2", "int8_pt", "int4_pt", "int8_pc", "int4_pc", "e4m11_ftz"]
    assert [k for k, v in c.items() if v.role == "primary"] == list(c)[:7]
    assert [k for k, v in c.items() if v.role == "comparison"] == ["int8_pc", "int4_pc", "e4m11_ftz"]
    p = S2.make_codecs_protocol("per_channel", include_comparison=False)
    assert list(p) == ["fp16", "bf16", "e4m11_sn", "fp8_e4m3fn", "fp8_e5m2", "int8_pc", "int4_pc"] and all(v.role == "primary" for v in p.values())
    assert set(S2.codec_registry()) >= set(c)
    with pytest.raises(ValueError):
        S2.make_codecs_protocol("per_row")
    assert list(S2.make_codecs()) == ["e4m11_sn", "fp16", "fp8_e4m3fn", "int8_pt", "int4_pt", "e4m11_ftz"]    # ResNet-18 set unchanged


@pytest.fixture(scope="module")
def mw():
    """A few real MobileNetV2 weight tensors: stem, depthwise, pointwise, classifier."""
    from fr.model import build_mobilenetv2_cifar
    torch.manual_seed(0)
    net = build_mobilenetv2_cifar(10).eval()
    names = ["features.0.0", "features.2.conv.1.0", "features.2.conv.2", "features.18.0", "classifier.1"]
    mods = dict(quantizable_modules(net))
    return {f"{n}.weight": mods[n].weight.detach().numpy().astype(np.float32) for n in names}


def test_all_seven_formats_roundtrip_and_sizes(mw):
    codecs = S2.make_codecs_protocol("per_channel")
    for name, w in mw.items():
        for key, c in codecs.items():
            enc = c.encode(w)
            dec = c.decode(enc)
            assert dec.shape == w.shape and dec.dtype == np.float32 and np.isfinite(dec).all(), (name, key)
            assert S2.roundtrip_row(w, dec)["sqnr_db"] > 3.0, (name, key)
    w = mw["features.2.conv.1.0.weight"]                                    # depthwise [144, 1, 3, 3] -> 144 channels of 9 weights
    n, ch = w.size, w.shape[0]
    sizes = {k: (c.encode(w)["words"].nbytes, 0 if c.encode(w)["scale"] is None else c.encode(w)["scale"].nbytes) for k, c in codecs.items()}
    assert sizes["bf16"] == (2 * n, 0) and sizes["fp8_e5m2"] == (n, 0) and sizes["fp16"] == (2 * n, 0)
    assert sizes["int8_pc"] == (n, 4 * ch) and sizes["int4_pc"] == ((n + 1) // 2, 4 * ch)


def test_per_channel_scales_follow_axis_0(mw):
    c = S2.IntPerChannel(8)
    w = mw["features.2.conv.1.0.weight"]
    enc = c.encode(w)
    assert enc["scale"].shape == (w.shape[0],)
    np.testing.assert_allclose(enc["scale"], np.abs(w.reshape(w.shape[0], -1)).max(axis=1) / 127, rtol=1e-6)
    # a per-channel scale never loses to the per-tensor one on a tensor whose channels have very different ranges
    w2 = w.copy()
    w2[0] *= 50.0
    pc = S2.roundtrip_row(w2, c.decode(c.encode(w2)))["sqnr_db"]
    pt = S2.roundtrip_row(w2, S2.IntPerTensor(8).decode(S2.IntPerTensor(8).encode(w2)))["sqnr_db"]
    assert pc > pt
    # flipping one bit of channel k's scale only changes channel k's decoded weights
    bad = dict(enc, scale=enc["scale"].copy())
    bad["scale"].view(np.uint32)[3] ^= np.uint32(1 << 22)
    diff = (c.decode(bad) != c.decode(enc)).reshape(w.shape[0], -1).any(axis=1)
    assert np.flatnonzero(diff).tolist() == [3]


def test_reference_decode_matches_codecs_exactly(mw):
    for key, c in S2.make_codecs_protocol("per_channel").items():
        for name, w in mw.items():
            ref = S2.reference_decode(key, w)
            if key.startswith("e4m11"):
                assert ref is None
            else:
                assert np.array_equal(ref, c.decode(c.encode(w))), (key, name)
    w = mw["features.18.0.weight"]
    for key in ("int8_pt", "int4_pt"):
        c = S2.codec_registry()[key]
        assert np.array_equal(S2.reference_decode(key, w), c.decode(c.encode(w)))


def test_zero_rate_fields_and_pooling():
    w = np.array([0.0, 1e-9, 0.5, -0.25], dtype=np.float32)
    c = S2.FP8E5M2()                                         # 1e-9 < 2^-17 -> flushed to zero
    row = S2.roundtrip_row(w, c.decode(c.encode(w)))
    assert row["zero_count"] == 2 and row["fp32_zero_count"] == 1 and row["zeroed_from_nonzero"] == 1 and row["zero_fraction"] == 0.5
    g = S2.pooled_row([row, row])
    assert g["zero_count"] == 4 and g["zero_fraction"] == 0.5 and g["fp32_zero_count"] == 2


def test_special_codes_are_reachable_only_through_decode():
    b = S2.BF16()
    dec = b.decode({"words": np.array([0x7F80, 0xFF80, 0x7FC0], dtype=np.uint16), "shape": (3,)})
    assert np.isposinf(dec[0]) and np.isneginf(dec[1]) and np.isnan(dec[2])
    e = S2.FP8E5M2().decode({"words": np.array([0x7C, 0xFC, 0x7D, 0x7B], dtype=np.uint8), "shape": (4,)})
    assert np.isposinf(e[0]) and np.isneginf(e[1]) and np.isnan(e[2]) and e[3] == 57344.0
    with pytest.raises(ValueError):
        b.encode(np.array([np.inf], dtype=np.float32))
    with pytest.raises(ValueError):
        S2.FP8E5M2().encode(np.array([np.nan], dtype=np.float32))
