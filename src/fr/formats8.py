"""8-bit weight formats for the mixed-precision experiments (NOT part of the fault-robustness paper's
E4M11 baseline; these are the candidate formats for low-sensitivity layers).

Four formats (``FORMATS8``):

``int8_pt``      INT8, per-tensor symmetric. Paper Eq. 3: ``w_q = clip(round(w / s), -2^(b-1), 2^(b-1)-1)``,
                 ``s = max|w| / (2^(b-1) - 1)``; dequantization ``w^ = w_q * s``. The scale is stored as
                 float32 metadata (32 bits per tensor).
``int8_pc``      Same but one scale per output channel (axis 0). 32 bits of metadata per channel.
``fp8``          FP8 E4M3 as in the paper's Table 2 (1 sign, 4 exponent, 3 mantissa bits, bias 7, max finite
                 240) but WITH subnormals (stored exponent 0 holds ``m * 2^-9``). The paper flushes
                 subnormals to zero; with weights this small that would destroy most of them (same effect
                 as for E4M11, see docs/CODE_SUMMARY.md D10). No scale.
``fp8_scaled``   ``fp8`` plus a per-tensor power-of-two scale ``2^k`` chosen so that ``max|w| / 2^k`` fits the
                 format (``k = ceil(log2(max|w| / 240))``): equivalent to a per-tensor exponent bias. ``k`` is
                 stored as 8 bits of metadata and assumed fault-free.

Rounding is round-to-nearest, ties to even, everywhere. FP8 words are built from the table of all finite
magnitudes (``fp_table``): the nearest representable value is chosen (ties to the even word, which is
IEEE round-to-nearest-even because the mantissa LSB is the word LSB). Magnitudes above the maximum finite
value saturate. The all-ones exponent (15) is reserved and never produced by the encoder; the decoder reads
it as exponent 14 (``clamp``), the same policy as ``e4m11.decode`` (relevant only for fault injection).
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

FORMATS8 = ("int8_pt", "int8_pc", "fp8", "fp8_scaled")

E8, M8, BIAS8 = 4, 3, 7


# ----------------------------------------------------------------------------------------------
# generic small floating-point table codec (also used to cross-check e4m11 in the tests)
# ----------------------------------------------------------------------------------------------
def fp_table(e_bits: int, m_bits: int, bias: int) -> np.ndarray:
    """Finite non-negative magnitudes indexed by word (sign=0), exponent field 0..2^E-2, subnormals on."""
    n_exp = (1 << e_bits) - 1                      # exponent fields 0 .. 2^E - 2
    es = np.repeat(np.arange(n_exp), 1 << m_bits).astype(np.float64)
    m = np.tile(np.arange(1 << m_bits), n_exp).astype(np.float64)
    sub = es == 0
    val = np.where(sub, m * 2.0 ** (1 - bias - m_bits),
                   (1.0 + m / (1 << m_bits)) * 2.0 ** (es - bias))
    return val


def fp_encode_magnitude(a: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Word index of the nearest table entry (ties to even word); saturates at the largest entry."""
    a = np.minimum(np.abs(a), table[-1])
    mid = (table[:-1] + table[1:]) / 2.0
    idx = np.searchsorted(mid, a, side="left")
    tie = (idx < len(mid)) & (a == mid[np.minimum(idx, len(mid) - 1)]) & (idx % 2 == 1)
    return np.where(tie, idx + 1, idx)


_T8 = fp_table(E8, M8, BIAS8)
FP8_MAX = float(_T8[-1])                           # 240.0


def fp8_encode(x: np.ndarray) -> np.ndarray:
    """float -> uint8 words (sign << 7 | es << 3 | m), saturating, subnormals kept."""
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("fp8 encoder supports finite values only.")
    idx = fp_encode_magnitude(x.reshape(-1), _T8)
    sign = (np.signbit(x.reshape(-1))).astype(np.uint8)
    return ((sign << 7) | idx.astype(np.uint8)).astype(np.uint8)


def fp8_decode(words: np.ndarray, all_ones: str = "clamp") -> np.ndarray:
    """uint8 words -> float32. ``all_ones``: clamp (es=15 read as 14) | literal (Eq. 2: 2^(15-7))."""
    w = np.asarray(words, dtype=np.uint8).astype(np.int64)
    s = (w >> 7) & 1
    es = (w >> M8) & ((1 << E8) - 1)
    m = w & ((1 << M8) - 1)
    if all_ones == "clamp":
        es_c = np.minimum(es, (1 << E8) - 2)
    elif all_ones == "literal":
        es_c = es
    else:
        raise ValueError("all_ones must be 'clamp' or 'literal' for fp8")
    val = np.where(es_c == 0, m * 2.0 ** (1 - BIAS8 - M8),
                   (1.0 + m / (1 << M8)) * 2.0 ** (es_c - BIAS8))
    return (np.where(s == 1, -val, val)).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# INT8
# ----------------------------------------------------------------------------------------------
def _int8_scale(absmax: np.ndarray) -> np.ndarray:
    s = (absmax / 127.0).astype(np.float32)
    return np.where(s == 0, np.float32(1.0), s).astype(np.float32)


def int8_encode(w: np.ndarray, per_channel: bool) -> Tuple[np.ndarray, np.ndarray]:
    """-> (uint8 words in two's complement, float32 scale(s)). Scale shape: () or (out_channels,)."""
    w = np.asarray(w, dtype=np.float32)
    if not np.isfinite(w).all():
        raise ValueError("int8 encoder supports finite values only.")
    if per_channel:
        flat = w.reshape(w.shape[0], -1)
        scale = _int8_scale(np.abs(flat).max(axis=1))
        q = np.rint(flat.astype(np.float64) / scale.astype(np.float64)[:, None])
    else:
        scale = _int8_scale(np.abs(w).max(initial=0.0))
        q = np.rint(w.reshape(-1).astype(np.float64) / float(scale))
    q = np.clip(q, -128, 127).astype(np.int8)
    return q.reshape(-1).view(np.uint8), np.asarray(scale, dtype=np.float32)


def int8_decode(words: np.ndarray, scale: np.ndarray, shape) -> np.ndarray:
    q = np.asarray(words, dtype=np.uint8).view(np.int8).astype(np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    if scale.ndim == 0:
        return (q * scale).reshape(shape).astype(np.float32)
    return (q.reshape(shape[0], -1) * scale[:, None]).reshape(shape).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# unified interface
# ----------------------------------------------------------------------------------------------
def pow2_exponent(absmax: float) -> int:
    """Smallest k with absmax / 2^k <= FP8_MAX (0 for an all-zero tensor)."""
    if absmax == 0.0:
        return 0
    k = math.ceil(math.log2(absmax / FP8_MAX))
    while absmax / 2.0 ** k > FP8_MAX:               # guard against log2 rounding
        k += 1
    return int(k)


def quantize8(w: np.ndarray, fmt: str) -> Dict[str, object]:
    """Quantize one weight tensor. Returns words (uint8, flat), scale metadata, the dequantized tensor
    (float32, ``w``'s shape) and ``metadata_bits``."""
    if fmt not in FORMATS8:
        raise ValueError(f"fmt must be one of {FORMATS8}, got {fmt!r}")
    w = np.asarray(w, dtype=np.float32)
    shape = tuple(w.shape)
    if fmt in ("int8_pt", "int8_pc"):
        words, scale = int8_encode(w, per_channel=(fmt == "int8_pc"))
        deq = int8_decode(words, scale, shape)
        meta = 32 * int(np.size(scale))
    elif fmt == "fp8":
        words = fp8_encode(w.reshape(-1))
        scale = None
        deq = fp8_decode(words).reshape(shape)
        meta = 0
    else:                                            # fp8_scaled
        k = pow2_exponent(float(np.abs(w).max(initial=0.0)))
        if not -127 <= k <= 127:
            raise ValueError(f"power-of-two scale exponent {k} does not fit 8 bits")
        words = fp8_encode(w.astype(np.float64).reshape(-1) / 2.0 ** k)
        scale = np.int8(k)
        deq = (fp8_decode(words).astype(np.float64) * 2.0 ** k).astype(np.float32).reshape(shape)
        meta = 8
    return {"format": fmt, "bits": 8, "words": words, "scale": scale, "shape": shape,
            "dequantized": deq, "metadata_bits": meta}


def dequantize8(words: np.ndarray, scale, shape, fmt: str, all_ones: str = "clamp") -> np.ndarray:
    """Decode stored words (possibly corrupted) back to float32; inverse of ``quantize8``'s storage."""
    if fmt in ("int8_pt", "int8_pc"):
        return int8_decode(words, scale, shape)
    deq = fp8_decode(words, all_ones).astype(np.float64)
    if fmt == "fp8_scaled":
        deq = deq * 2.0 ** int(scale)
    return deq.astype(np.float32).reshape(shape)
