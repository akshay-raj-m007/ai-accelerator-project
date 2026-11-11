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
