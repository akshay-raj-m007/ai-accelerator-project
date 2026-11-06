"""E4M11 (1 sign, 4 exponent, 11 mantissa bits, bias 7) encode/decode.

Two encoder modes:

``paper`` (default)
    Eq. 4 of the paper including the mantissa carry
    ("if m >= 2^11: set e <- e+1, m <- 0"), combined with the clipping/flush
    rules of Algorithm 1 (magnitude clipped to vmax, |w| < 2^-7 flushed to
    signed zero, exponent clamped to [-6, 7], fraction lower-clipped at 0 so
    that [2^-7, 2^-6) rounds up to the minimum normal 2^-6).

``legacy``
    Bit-exact reproduction of the original notebook: the fraction is clipped to
    [0, 1 - 2^-11] and the mantissa to [0, 2^11 - 1] (this equals Algorithm 1
    steps 7-8 literally) and there is **no carry**.

``subnormal``  (NOT in the paper; exploratory)
    Same 16-bit layout and bias 7, but es == 0 stores a subnormal
    ``(-1)^s * m * 2^-17`` (= m/2^11 * 2^-6) instead of flushing to zero, as in
    IEEE formats. Normals (|w| >= 2^-6) are encoded exactly as in ``paper``
    mode. This contradicts the paper's Sec. 3 "subnormal numbers are flushed to
    zero" and exists to test whether the flush rule alone explains the accuracy
    collapse on small trained weights.

The decoder is Algorithm 2 (exponent clamped to [1, 14], es == 0 -> zero).
``all_ones`` selects how es == 15 is read (only reachable via bit flips):

* ``clamp``   (default) Algorithm 2 step 4: treat as es = 14.
* ``ieee``    Sec. 3: "the all-ones exponent pattern is reserved for +-inf/NaN".
* ``literal`` Eq. 5 unmodified: 2^(15-7).

All functions are pure and vectorised: uint16 words in/out, float32 values out.
"""
from __future__ import annotations

import numpy as np

E_BITS = 4
M_BITS = 11
BIAS = 7
MANT_MASK = (1 << M_BITS) - 1                      # 0x7FF
EXP_MAX_STORED = (1 << E_BITS) - 2                 # 14 (15 = all ones, reserved)
MAX_FINITE = (2.0 - 2.0 ** -M_BITS) * 2.0 ** BIAS  # 255.9375
MIN_NORMAL = 2.0 ** (1 - BIAS)                     # 2^-6
FLUSH_THRESHOLD = 2.0 ** -BIAS                     # 2^-7 (Alg. 1 step 3)
SUBNORMAL_STEP = 2.0 ** (1 - BIAS - M_BITS)        # 2^-17

MODES = ("paper", "legacy", "subnormal")
ALL_ONES = ("clamp", "ieee", "literal")


def _as_f64(values) -> np.ndarray:
    if hasattr(values, "detach"):  # torch tensor
        values = values.detach().cpu().numpy()
    x = np.asarray(values, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("E4M11 encoder supports finite values only.")
    return x


def max_finite(bias: int = BIAS) -> float:
    return (2.0 - 2.0 ** -M_BITS) * 2.0 ** (EXP_MAX_STORED - bias)


def encode(values, mode: str = "paper", bias: int = BIAS) -> np.ndarray:
    """FP32/FP64 array -> uint16 E4M11 words (same shape).

    ``bias`` (default 7 = paper) shifts the 14 usable binades to [2^(1-bias), 2^(15-bias)); every
    other rule (flush below 2^-bias, clip to max_finite(bias), carry) is the paper rule re-expressed
    in terms of the bias. bias != 7 is a deviation from the paper (Table 1).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    x = _as_f64(values)
    shape = x.shape
    x = x.reshape(-1)
    sign = np.signbit(x).astype(np.uint16)
    a = np.minimum(np.abs(x), max_finite(bias))      # Alg.1 step 2
    word = (sign << 15).astype(np.uint16)
    if mode == "subnormal":
        sub = a < 2.0 ** (1 - bias)
        ms = np.rint(a[sub] / 2.0 ** (1 - bias - M_BITS)).astype(np.int64)    # 0..2048
        # m == 2048 is exactly 2^-6: es=1, m=0 (carry into the normal range)
        word[sub] |= np.where(ms >= (1 << M_BITS), 1 << M_BITS, ms).astype(np.uint16)
        normal = ~sub
    else:
        normal = a >= 2.0 ** -bias                   # Alg.1 step 3 (flush below 2^-bias)
    if not normal.any():
        return word.reshape(shape)
    v = a[normal]
    if mode == "legacy":
        e = np.floor(np.log2(v)).astype(np.int64)    # as in the notebook
    else:
        e = np.frexp(v)[1].astype(np.int64) - 1      # exact floor(log2 v)
    e = np.clip(e, 1 - bias, EXP_MAX_STORED - bias)  # Alg.1 step 5: [-6, 7] for bias 7
    frac = np.ldexp(v, -e) - 1.0
    if mode == "legacy":
        frac = np.clip(frac, 0.0, 1.0 - 2.0 ** -M_BITS)
        m = np.clip(np.rint(frac * (1 << M_BITS)), 0, MANT_MASK).astype(np.int64)
    else:
        frac = np.maximum(frac, 0.0)                 # [2^-7, 2^-6) -> 2^-6 (paper mode)
        m = np.rint(frac * (1 << M_BITS)).astype(np.int64)
        carry = m >= (1 << M_BITS)                   # Eq. 4 carry
        e = e + carry
        m = np.where(carry, 0, m)
        over = e > EXP_MAX_STORED - bias             # unreachable after the vmax clip
        e = np.where(over, EXP_MAX_STORED - bias, e)
        m = np.where(over, MANT_MASK, m)
    es = (e + bias).astype(np.uint16)
    word[normal] |= (es << M_BITS) | m.astype(np.uint16)
    return word.reshape(shape)


def decode(words, mode: str = "paper", all_ones: str = "clamp", bias: int = BIAS) -> np.ndarray:
    """uint16 E4M11 words -> float32 array (same shape)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if all_ones not in ALL_ONES:
        raise ValueError(f"all_ones must be one of {ALL_ONES}, got {all_ones!r}")
    if mode == "legacy":
        all_ones = "clamp"
    w = np.asarray(words).astype(np.uint16)
    sign = ((w >> 15) & 1).astype(bool)
    es = ((w >> M_BITS) & 0xF).astype(np.int64)
    m = (w & MANT_MASK).astype(np.float64)
    if all_ones == "literal":
        e_used = np.where(es == 0, 1, es)
    else:
        e_used = np.clip(es, 1, EXP_MAX_STORED)
    val = (1.0 + m / (1 << M_BITS)) * np.exp2(e_used - bias)
    if mode == "subnormal":
        val = np.where(es == 0, m * 2.0 ** (1 - bias - M_BITS), val)
    else:
        val = np.where(es == 0, 0.0, val)            # Alg.2 step 6
    if all_ones == "ieee":
        val = np.where(es == 15, np.where(m == 0, np.inf, np.nan), val)
    val = np.where(sign, -val, val)
    return val.astype(np.float32)


def quantize_dequantize(values, mode: str = "paper", bias: int = BIAS) -> np.ndarray:
    """Round-trip FP -> E4M11 -> float32."""
    return decode(encode(values, mode, bias), mode, bias=bias)


# ----------------------------------------------------------------- torch helpers
def encode_tensor(t, mode: str = "paper", bias: int = BIAS) -> np.ndarray:
    """torch tensor -> uint16 numpy words (torch has no usable uint16 ops)."""
    return encode(t.detach().float().cpu().numpy(), mode, bias)


def decode_to_tensor(words, dtype=None, device=None, mode: str = "paper",
                     all_ones: str = "clamp", bias: int = BIAS):
    """uint16 words -> torch tensor (float32 by default)."""
    import torch
    out = torch.from_numpy(decode(words, mode, all_ones, bias))
    return out.to(device=device, dtype=dtype or torch.float32)


def decode_torch(words_int, all_ones: str = "clamp", mode: str = "paper", bias: int = BIAS):
    """Pure-torch decoder for int32/int64 tensors holding the 16-bit words.

    Runs on any device (useful for fault injection on GPU later). Equivalent to
    :func:`decode` (paper mode) for the same ``all_ones`` policy.
    """
    import torch
    w = words_int.to(torch.int64) & 0xFFFF
    sign = (w >> 15) & 1
    es = (w >> M_BITS) & 0xF
    m = (w & MANT_MASK).to(torch.float64)
    e_used = es.clamp(1, EXP_MAX_STORED) if all_ones != "literal" else es.clamp(min=1)
    val = (1.0 + m / (1 << M_BITS)) * torch.exp2((e_used - bias).to(torch.float64))
    zero_val = m * 2.0 ** (1 - bias - M_BITS) if mode == "subnormal" else torch.zeros_like(val)
    val = torch.where(es == 0, zero_val, val)
    if all_ones == "ieee":
        special = torch.where(m == 0, torch.full_like(val, float("inf")),
                              torch.full_like(val, float("nan")))
        val = torch.where(es == 15, special, val)
    val = torch.where(sign == 1, -val, val)
    return val.to(torch.float32)
