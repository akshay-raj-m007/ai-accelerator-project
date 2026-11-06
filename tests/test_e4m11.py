import numpy as np
import pytest
import torch

from fr import e4m11 as E

U16 = np.uint16


# --- literal copy of the original notebook code (the "legacy" ground truth) -------------
def _nb_encode(values):
    x = np.asarray(values, dtype=np.float64)
    sign = np.signbit(x).astype(np.uint16)
    magnitude = np.minimum(np.abs(x), (2.0 - 2.0 ** -11) * 2.0 ** 7)
    encoded = sign << 15
    normal = magnitude >= 2.0 ** -7
    if np.any(normal):
        a = magnitude[normal]
        exponent = np.floor(np.log2(a)).astype(np.int32)
        exponent = np.clip(exponent, -6, 7)
        fraction = np.ldexp(a, -exponent) - 1.0
        fraction = np.clip(fraction, 0.0, 1.0 - 2.0 ** -11)
        mantissa = np.rint(fraction * (2 ** 11))
        mantissa = np.clip(mantissa, 0, (1 << 11) - 1).astype(np.uint16)
        stored_exponent = (exponent + 7).astype(np.uint16)
        encoded[normal] |= (stored_exponent << 11) | mantissa
    return encoded.astype(np.uint16)


def _nb_decode(words):
    words = np.asarray(words, dtype=np.uint16)
    sign = ((words >> 15) & 1).astype(np.int8)
    exponent = ((words >> 11) & 0xF).astype(np.int32)
    mantissa = (words & 0x7FF).astype(np.float64)
    result = np.zeros(words.shape, dtype=np.float64)
    normal = exponent != 0
    actual = np.clip(exponent[normal], 1, 14) - 7
    result[normal] = (1.0 + mantissa[normal] / (2 ** 11)) * np.exp2(actual)
    result[sign == 1] *= -1.0
    return result.astype(np.float32)


def _sample_values(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.concatenate([
        rng.normal(0, 0.05, n),
        rng.normal(0, 1.0, n // 4),
        np.exp(rng.uniform(np.log(1e-5), np.log(400), n // 4)) * rng.choice([-1, 1], n // 4),
        # just below / at / just above powers of two, both signs
        np.concatenate([[s * (2.0 ** k) * f for f in (1 - 2 ** -13, 1 - 2 ** -12, 1.0, 1 + 2 ** -13)]
                        for k in range(-9, 9) for s in (1, -1)]).ravel(),
        [0.0, -0.0, 255.9375, 255.99, 1000.0, -1000.0],
    ])
    return vals.astype(np.float32)


def test_paper_example_3_14():
    w = E.encode(np.float32(3.14))
    assert int(w) == 0x448F
    assert E.decode(w) == pytest.approx(3.1396, abs=5e-5)
    assert int(E.encode(np.float32(3.14), "legacy")) == 0x448F


def test_max_finite_and_min_normal():
    assert E.MAX_FINITE == 255.9375
    assert float(E.decode(U16(0x77FF))) == 255.9375          # es=14, m=2047
    assert int(E.encode(255.9375)) == 0x77FF
    assert float(E.decode(E.encode(1e6))) == 255.9375          # clipped, never inf
    assert float(E.decode(U16(0x0800))) == 2.0 ** -6           # es=1, m=0
    assert int(E.encode(2.0 ** -6)) == 0x0800
    assert E.MIN_NORMAL == 2.0 ** -6


def test_flush_rule():
    assert int(E.encode(2.0 ** -7 * 0.999)) == 0x0000
    assert int(E.encode(-(2.0 ** -7) * 0.999)) == 0x8000        # signed zero
    assert int(E.encode(2.0 ** -7)) == 0x0800                   # rounds up to min normal
    assert int(E.encode(2.0 ** -7, "legacy")) == 0x0800


def test_carry_paper_vs_legacy():
    x = np.float64(1.99999)
    assert int(E.encode(x, "paper")) == 0x4000
    assert float(E.decode(E.encode(x, "paper"))) == 2.0
    legacy = E.encode(x, "legacy")
    assert int(legacy) == 0x3FFF                                # m clipped to 2047, e stays 0
    assert float(E.decode(legacy, "legacy")) == pytest.approx(2.0 - 2.0 ** -11)
    # carry at the exponent boundary of the top binade stays finite
    assert int(E.encode(255.99)) == 0x77FF


@pytest.mark.parametrize("bit,expected", [(11, 6.0), (12, 12.0), (13, 48.0), (14, 0.0),
                                          (15, -3.0), (10, 2.0)])
def test_single_bit_flips_of_3_legacy(bit, expected):
    w = E.encode(np.float32(3.0), "legacy")
    flipped = (w ^ U16(1 << bit)).astype(np.uint16)
    assert float(E.decode(flipped, "legacy")) == expected


def test_idempotent_all_normal_words():
    words = np.array([(s << 15) | (es << 11) | m for s in (0, 1) for es in range(1, 15)
                      for m in range(0, 2048)], dtype=np.uint16)
    for mode in ("paper", "legacy"):
        np.testing.assert_array_equal(E.encode(E.decode(words, mode), mode), words)


def test_idempotent_on_quantized_values():
    x = _sample_values()
    for mode in ("paper", "legacy"):
        q = E.quantize_dequantize(x, mode)
        np.testing.assert_array_equal(E.quantize_dequantize(q, mode), q)


def test_legacy_matches_original_notebook_exactly():
    x = _sample_values()
    np.testing.assert_array_equal(E.encode(x, "legacy"), _nb_encode(x))
    all_words = np.arange(65536, dtype=np.uint16)
    np.testing.assert_array_equal(E.decode(all_words, "legacy"), _nb_decode(all_words))
    # the paper-mode decoder is Algorithm 2 = the same function
    np.testing.assert_array_equal(E.decode(all_words, "paper"), _nb_decode(all_words))


def test_paper_differs_from_legacy_only_by_carry():
    x = _sample_values()
    wp, wl = E.encode(x, "paper"), E.encode(x, "legacy")
    diff = wp != wl
    assert diff.any()
    # every difference is a carry: paper value is exactly a power of two, legacy is the
    # largest mantissa of the binade below
    dp = E.decode(wp[diff], "paper")
    assert np.all((wp[diff] & 0x7FF) == 0)
    assert np.all(np.log2(np.abs(dp)) == np.round(np.log2(np.abs(dp))))
    assert np.all((wl[diff] & 0x7FF) == 0x7FF)


def test_paper_error_bound():
    x = np.random.default_rng(1).uniform(2.0 ** -6, 255.0, 100_000).astype(np.float32)
    q = E.quantize_dequantize(x, "paper")
    assert np.max(np.abs(q - x) / x) <= 2.0 ** -12 * 1.0001      # half an ulp, relative


def test_shapes_and_dtypes():
    x = np.zeros((3, 4, 5), np.float32)
    assert E.encode(x).shape == x.shape and E.encode(x).dtype == np.uint16
    assert E.decode(E.encode(x)).dtype == np.float32
    assert E.encode(np.float32(1.0)).shape == ()
    with pytest.raises(ValueError):
        E.encode(np.array([np.nan]))
    with pytest.raises(ValueError):
        E.encode(np.array([np.inf]))


def test_all_ones_policies():
    w = U16(0x7800 | 5)                                      # es=15, m=5
    clamp = float(E.decode(w, all_ones="clamp"))
    assert clamp == pytest.approx((1 + 5 / 2048) * 2.0 ** 7)
    assert float(E.decode(w, all_ones="literal")) == pytest.approx((1 + 5 / 2048) * 2.0 ** 8)
    assert np.isnan(E.decode(w, all_ones="ieee"))
    assert np.isposinf(E.decode(U16(0x7800), all_ones="ieee"))
    assert np.isneginf(E.decode(U16(0xF800), all_ones="ieee"))
    assert float(E.decode(w, "legacy", all_ones="ieee")) == clamp   # legacy ignores the option


@pytest.mark.parametrize("policy", ["clamp", "ieee", "literal"])
def test_torch_decoder_matches_numpy(policy):
    words = np.arange(65536, dtype=np.uint16)
    a = E.decode(words, all_ones=policy)
    b = E.decode_torch(torch.from_numpy(words.astype(np.int32)), all_ones=policy).numpy()
    np.testing.assert_array_equal(a, b)                       # also equal for nan/inf positions
    t = E.decode_to_tensor(words, all_ones=policy)
    np.testing.assert_array_equal(t.numpy(), a)


# ------------------------------------------------------------------ subnormal mode
def test_subnormal_mode_basics():
    step = 2.0 ** -17
    assert E.SUBNORMAL_STEP == step
    # tiny value is kept (paper/legacy flush it to zero)
    x = np.float32(0.001)
    assert float(E.quantize_dequantize(x, "paper")) == 0.0
    q = float(E.quantize_dequantize(x, "subnormal"))
    assert abs(q - 0.001) <= step / 2 + 1e-12
    assert int(E.encode(np.float32(step), "subnormal")) == 0x0001
    assert int(E.encode(np.float32(-step), "subnormal")) == 0x8001
    assert float(E.decode(U16(0x0001), "subnormal")) == step
    assert float(E.decode(U16(0x07FF), "subnormal")) == 2047 * step
    # rounding up into the normal range: just below 2^-6 -> exactly 2^-6 (es=1, m=0)
    assert int(E.encode(np.float64(2.0 ** -6 * (1 - 2 ** -14)), "subnormal")) == 0x0800
    # values >= 2^-6 are identical to paper mode
    x = _sample_values()
    big = np.abs(x) >= 2.0 ** -6
    np.testing.assert_array_equal(E.encode(x[big], "subnormal"), E.encode(x[big], "paper"))
    assert int(E.encode(np.float32(0.0), "subnormal")) == 0
    assert int(E.encode(np.float32(-0.0), "subnormal")) == 0x8000


def test_subnormal_idempotent_all_words():
    words = np.array([(s << 15) | (es << 11) | m for s in (0, 1) for es in range(0, 15)
                      for m in range(0, 2048)], dtype=np.uint16)
    dec = E.decode(words, "subnormal")
    re = E.encode(dec, "subnormal")
    # (+-0 and es=0,m=0 map to the same signed zero; every other word is stable)
    np.testing.assert_array_equal(re, words)


def test_subnormal_absolute_error_bound_in_small_range():
    x = np.random.default_rng(2).uniform(-2.0 ** -6, 2.0 ** -6, 200_000)
    q = E.quantize_dequantize(x, "subnormal")
    assert np.max(np.abs(q - x)) <= 2.0 ** -18 * 1.0001


def test_subnormal_torch_decoder_matches_numpy():
    words = np.arange(65536, dtype=np.uint16)
    a = E.decode(words, "subnormal")
    b = E.decode_torch(torch.from_numpy(words.astype(np.int32)), mode="subnormal").numpy()
    np.testing.assert_array_equal(a, b)


# ------------------------------------------------------------------ bias option
def test_default_bias_is_paper_bias():
    x = _sample_values()
    for mode in E.MODES:
        np.testing.assert_array_equal(E.encode(x, mode), E.encode(x, mode, bias=7))
    assert E.max_finite(7) == 255.9375 and E.max_finite(11) == (2 - 2.0 ** -11) * 2.0 ** 3


@pytest.mark.parametrize("bias", [9, 11, 14, 16])
@pytest.mark.parametrize("mode", ["paper", "subnormal"])
def test_bias_roundtrip_and_ranges(bias, mode):
    lo_stored, hi_stored = (0 if mode == "subnormal" else 1), 14
    words = np.array([(s << 15) | (es << 11) | m for s in (0, 1) for es in range(lo_stored, hi_stored + 1)
                      for m in range(0, 2048)], dtype=np.uint16)
    dec = E.decode(words, mode, bias=bias)
    np.testing.assert_array_equal(E.encode(dec, mode, bias), words)
    assert float(np.max(dec)) == E.max_finite(bias)
    # min normal moves with the bias
    assert float(E.decode(U16(0x0800), mode, bias=bias)) == 2.0 ** (1 - bias)
    # clipping at the new max
    assert float(E.quantize_dequantize(np.float32(1e6), mode, bias)) == pytest.approx(E.max_finite(bias))


def test_bias_flush_threshold_and_subnormal_step():
    b = 12
    assert float(E.quantize_dequantize(np.float64(2.0 ** -b * 0.99), "paper", b)) == 0.0
    s = float(E.quantize_dequantize(np.float64(2.0 ** -b * 0.99), "subnormal", b))
    assert abs(s - 2.0 ** -b * 0.99) <= 2.0 ** (1 - b - 11) / 2 * 1.0001


def test_torch_decoder_with_bias():
    words = np.arange(65536, dtype=np.uint16)
    for mode in ("paper", "subnormal"):
        a = E.decode(words, mode, bias=11)
        b = E.decode_torch(torch.from_numpy(words.astype(np.int32)), mode=mode, bias=11).numpy()
        np.testing.assert_array_equal(a, b)
