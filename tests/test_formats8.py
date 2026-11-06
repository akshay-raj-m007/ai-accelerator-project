import numpy as np
import pytest

from fr import e4m11 as E
from fr import formats8 as F


def test_fp8_table_basics():
    t = F.fp_table(4, 3, 7)
    assert len(t) == 15 * 8 == 120
    assert t[0] == 0.0 and t[1] == 2.0 ** -9                       # subnormal step
    assert t[8] == 2.0 ** -6                                       # first normal (es=1, m=0)
    assert t[-1] == 240.0 == F.FP8_MAX                             # paper Table 2: F8E4M3 max ~240
    assert np.all(np.diff(t) > 0)                                  # word order == value order


def test_fp8_all_words_roundtrip_and_sign():
    words = np.array([(s << 7) | i for s in (0, 1) for i in range(120)], dtype=np.uint8)
    vals = F.fp8_decode(words)
    assert np.array_equal(F.fp8_encode(vals), words)               # idempotent, -0.0 keeps its sign bit
    assert np.all(vals[:120] >= 0) and np.all(vals[120:] <= 0)


def test_fp8_is_nearest_with_ties_to_even():
    t = F.fp_table(4, 3, 7)
    rng = np.random.default_rng(0)
    x = np.concatenate([np.exp(rng.uniform(np.log(1e-4), np.log(300), 20000)),
                        rng.uniform(0, 0.05, 20000)])
    w = F.fp8_encode(x)
    got = F.fp8_decode(w).astype(np.float64)
    best = np.abs(t[None, :] - np.minimum(x, 240.0)[:, None]).min(axis=1)
    np.testing.assert_allclose(np.abs(got - np.minimum(x, 240.0)), best, atol=1e-12)
    mids = (t[:-1] + t[1:]) / 2
    wm = F.fp8_encode(mids).astype(np.int64)
    assert np.all(wm % 2 == 0)                                     # every exact tie lands on the even word


def test_fp8_saturation_and_nonfinite():
    assert F.fp8_encode(np.array([1000.0]))[0] == 119
    assert F.fp8_encode(np.array([-1000.0]))[0] == 0x80 | 119
    with pytest.raises(ValueError):
        F.fp8_encode(np.array([np.nan]))


def test_fp8_all_ones_exponent_policy():
    w = np.array([0x78, 0x7F], dtype=np.uint8)                     # es=15, m=0 / m=7
    assert list(F.fp8_decode(w, "clamp")) == [128.0, 240.0]        # read as es=14
    assert list(F.fp8_decode(w, "literal")) == [256.0, 480.0]      # Eq. 2 unmodified
    with pytest.raises(ValueError):
        F.fp8_decode(w, "ieee")


def test_generic_table_codec_agrees_with_e4m11_subnormal_mode():
    """Independent cross-check: nearest-value search on the full E4M11 table == the closed-form encoder."""
    t = F.fp_table(4, 11, 7)
    assert t[-1] == E.MAX_FINITE and t[1] == E.SUBNORMAL_STEP
    rng = np.random.default_rng(1)
    x = np.concatenate([np.exp(rng.uniform(np.log(1e-7), np.log(255.9), 30000)),
                        (t[:-1] + t[1:])[rng.integers(0, len(t) - 1, 5000)] / 2])   # incl. exact ties
    ref = E.encode(x, "subnormal").astype(np.int64)
    got = F.fp_encode_magnitude(x, t)
    assert np.array_equal(ref, got)


def test_int8_per_tensor_spec():
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((16, 8, 3, 3)) * 0.05).astype(np.float32)
    q = F.quantize8(w, "int8_pt")
    s = np.float32(np.abs(w).max() / 127.0)
    assert float(q["scale"]) == pytest.approx(float(s), rel=1e-6)
    ints = q["words"].view(np.int8)
    assert ints.min() >= -127 and ints.max() <= 127 and np.abs(ints).max() == 127   # max|w| -> +-127
    assert q["dequantized"].shape == w.shape and q["dequantized"].dtype == np.float32
    assert np.abs(q["dequantized"] - w).max() <= float(s) / 2 + 1e-7              # nearest rounding
    assert q["metadata_bits"] == 32 and q["bits"] == 8


def test_int8_per_channel_better_than_per_tensor_and_exact_max():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((8, 32)).astype(np.float32) * np.linspace(0.01, 1.0, 8, dtype=np.float32)[:, None]
    pt, pc = F.quantize8(w, "int8_pt"), F.quantize8(w, "int8_pc")
    assert pc["scale"].shape == (8,) and pc["metadata_bits"] == 32 * 8
    assert np.mean((pc["dequantized"] - w) ** 2) < np.mean((pt["dequantized"] - w) ** 2)
    ints = pc["words"].view(np.int8).reshape(8, 32)
    assert np.all(np.abs(ints).max(axis=1) == 127)                                # each channel uses its range


def test_int8_zero_tensor_and_nonfinite():
    q = F.quantize8(np.zeros((4, 4), np.float32), "int8_pt")
    assert np.all(q["words"] == 0) and np.all(q["dequantized"] == 0)
    q = F.quantize8(np.zeros((4, 4), np.float32), "int8_pc")
    assert np.all(q["dequantized"] == 0)
    with pytest.raises(ValueError):
        F.quantize8(np.array([[np.inf]], np.float32), "int8_pt")
    with pytest.raises(ValueError):
        F.quantize8(np.ones((2, 2), np.float32), "int4")


def test_fp8_scaled_exponent_and_benefit_on_small_weights():
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((64, 64)) * 0.001).astype(np.float32)              # |w| ~ 1e-3: mostly below the FP8 subnormal step
    amax = float(np.abs(w).max())
    k = F.pow2_exponent(amax)
    assert amax / 2.0 ** k <= 240.0 < amax / 2.0 ** (k - 1)                      # smallest sufficient k
    un, sc = F.quantize8(w, "fp8"), F.quantize8(w, "fp8_scaled")
    assert int(sc["scale"]) == k and sc["metadata_bits"] == 8 and un["metadata_bits"] == 0
    err_un = np.sum((un["dequantized"] - w) ** 2) / np.sum(w ** 2)
    err_sc = np.sum((sc["dequantized"] - w) ** 2) / np.sum(w ** 2)
    assert err_sc < err_un / 100                                                 # measured: ~7e-4 vs ~0.32
    assert err_sc < 1e-3                                                         # ~3 mantissa bits
    assert F.pow2_exponent(0.0) == 0


@pytest.mark.parametrize("fmt", F.FORMATS8)
def test_dequantize_from_stored_words_matches(fmt):
    rng = np.random.default_rng(3)
    w = (rng.standard_normal((6, 5, 3, 3)) * 0.1).astype(np.float32)
    q = F.quantize8(w, fmt)
    back = F.dequantize8(q["words"], q["scale"], q["shape"], fmt)
    np.testing.assert_array_equal(back, q["dequantized"])
    # deterministic
    np.testing.assert_array_equal(F.quantize8(w, fmt)["words"], q["words"])
