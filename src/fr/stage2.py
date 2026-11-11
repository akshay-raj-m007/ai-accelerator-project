"""J-FAMP runbook Stage 2: candidate weight formats stored as REAL encoded words, round-trip metrics, storage accounting.

Primary formats (INT3 is deferred, runbook 4.1):

  e4m11_sn   E4M11-SN   1+4+11 bits, bias 7, subnormals enabled (``e4m11`` codec, mode "subnormal")
  fp16       FP16       IEEE 754 binary16 (E5M10), bias 15
  fp8_e4m3fn FP8 E4M3FN OCP E4M3FN: 1+4+3 bits, bias 7, no infinities, one NaN pattern (S.1111.111), max finite 448,
                        subnormals enabled
  int8_pt    INT8       symmetric, per-tensor (the paper's Eq. 3)
  int4_pt    INT4       symmetric, per-tensor, packed two codes per byte
Comparison only (supported by the existing codec, runbook 4.2):
  e4m11_ftz  E4M11-FTZ  same layout, subnormals flushed to zero (``e4m11`` mode "paper")

Every codec encodes ONE tensor independently from the original FP32 weights and decodes from the stored words only:

  enc = codec.encode(w)        {"words": ndarray (the stored bytes), "scale": float32 ndarray | None, "shape", "n", "n_clipped"}
  w_hat = codec.decode(enc)    float32, same shape

No format here is scaled except the INT formats (their float32 scale is stored with the tensor and counted as metadata).
Nothing in this module reads data or the test set.
"""
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from . import e4m11, formats8
from .sqnr import sqnr_db

FP16_MAX = 65504.0
FP8FN_MAX = 448.0


# ---------------------------------------------------------------------------------------------- packing (INT4)
def pack_nibbles(codes: np.ndarray) -> np.ndarray:
    """uint8 codes (only the low nibble used) -> packed bytes. Element 2j goes to the LOW nibble (bits 0-3) of byte j,
    element 2j+1 to the HIGH nibble (bits 4-7). An odd count is padded with one zero nibble in the high half of the
    last byte."""
    c = (np.asarray(codes, dtype=np.uint8).reshape(-1)) & np.uint8(0xF)
    if c.size % 2:
        c = np.concatenate([c, np.zeros(1, dtype=np.uint8)])
    return (c[0::2] | (c[1::2] << np.uint8(4))).astype(np.uint8)


def unpack_nibbles(packed: np.ndarray, n: int) -> np.ndarray:
    """Inverse of :func:`pack_nibbles` for ``n`` elements (drops the padding nibble)."""
    p = np.asarray(packed, dtype=np.uint8).reshape(-1)
    out = np.empty(p.size * 2, dtype=np.uint8)
    out[0::2] = p & np.uint8(0xF)
    out[1::2] = p >> np.uint8(4)
    return out[:n]


def int4_bit_location(element: int, bit: int) -> Tuple[int, int]:
    """(byte index, bit position within the byte) of bit ``bit`` (0 = LSB) of INT4 element ``element``."""
    if not 0 <= bit < 4:
        raise ValueError("bit must be 0..3")
    return element // 2, (element % 2) * 4 + bit


# ---------------------------------------------------------------------------------------------- codecs
def _finite(w: np.ndarray, name: str) -> np.ndarray:
    w = np.asarray(w, dtype=np.float32)
    if not np.isfinite(w).all():
        raise ValueError(f"{name} encoder supports finite values only")
    return w


class Codec:
    key = ""
    label = ""                       # exact variant name used in result tables
    bits = 0                         # stored bits per weight (payload)
    role = "primary"                 # primary | comparison
    has_scale = False

    def encode(self, w: np.ndarray) -> dict:
        raise NotImplementedError

    def decode(self, enc: dict) -> np.ndarray:
        raise NotImplementedError

    def config(self) -> dict:
        raise NotImplementedError


class E4M11(Codec):
    bits = 16

    def __init__(self, mode: str):
        self.mode = mode
        self.key = "e4m11_sn" if mode == "subnormal" else "e4m11_ftz"
        self.label = "E4M11-SN" if mode == "subnormal" else "E4M11-FTZ"
        self.role = "primary" if mode == "subnormal" else "comparison"

    def encode(self, w):
        w = _finite(w, self.key)
        words = e4m11.encode(w, self.mode, e4m11.BIAS).reshape(-1)
        return {"words": words, "scale": None, "shape": tuple(w.shape), "n": int(w.size),
                "n_clipped": int((np.abs(w) > e4m11.max_finite()).sum())}

    def decode(self, enc):
        return e4m11.decode(enc["words"], self.mode, "clamp", e4m11.BIAS).reshape(enc["shape"])

    def config(self):
        sn = self.mode == "subnormal"
        return {
            "variant": self.label, "bit_layout": "S(1) EEEE(4) M(11), word = (s<<15)|(es<<11)|m, uint16",
            "exponent_bias": 7, "stored_exponent_range_normals": "1..14 (value (1+m/2^11)*2^(es-7))",
            "subnormal_behavior": ("enabled: es=0 stores (-1)^s * m * 2^-17 (= m/2^11 * 2^-6)" if sn else
                                   "flush-to-zero: |w| < 2^-7 -> signed zero; [2^-7, 2^-6) rounds up to 2^-6 (paper Algorithm 1)"),
            "rounding": "round-to-nearest, ties to even (np.rint on the integer mantissa); mantissa carry increments the exponent",
            "saturation_clipping": f"|w| clipped to max finite {e4m11.max_finite()} before encoding",
            "scale": "none", "zero_point": "none",
            "nan_inf": "encoder rejects non-finite input; decoder reads the reserved all-ones exponent (es=15) as es=14 "
                       "('clamp'), reachable only through bit flips",
            "granularity": "n/a (float format)", "calibration": "none (data-free)",
            "storage": "actual encoded uint16 words; decoded to FP32 for evaluation (weight-only)",
            "codec_source": "src/fr/e4m11.py (mode=%r)" % self.mode}


class FP16(Codec):
    key, label, bits = "fp16", "FP16 (IEEE 754 binary16, E5M10)", 16

    def encode(self, w):
        w = _finite(w, self.key)
        clipped = int((np.abs(w) > FP16_MAX).sum())
        words = np.clip(w, -FP16_MAX, FP16_MAX).astype(np.float16).view(np.uint16).reshape(-1)
        return {"words": words, "scale": None, "shape": tuple(w.shape), "n": int(w.size), "n_clipped": clipped}

    def decode(self, enc):
        return enc["words"].view(np.float16).astype(np.float32).reshape(enc["shape"])

    def config(self):
        return {"variant": self.label, "bit_layout": "S(1) EEEEE(5) M(10), uint16", "exponent_bias": 15,
                "subnormal_behavior": "enabled (IEEE): es=0 stores m * 2^-24",
                "rounding": "round-to-nearest, ties to even (NumPy float32 -> float16 cast)",
                "saturation_clipping": f"|w| clipped to +-{FP16_MAX} before the cast (no overflow to inf)",
                "scale": "none", "zero_point": "none",
                "nan_inf": "encoder rejects non-finite input; the all-ones exponent (inf/NaN) is IEEE-reserved and is "
                           "decoded as inf/NaN (reachable only through bit flips)",
                "granularity": "n/a (float format)", "calibration": "none (data-free)",
                "storage": "actual encoded uint16 words; decoded to FP32 for evaluation (weight-only)",
                "codec_source": "src/fr/stage2.py (NumPy float16)"}


def _fp8fn_magnitudes() -> np.ndarray:
    w = np.arange(127)                                    # words 0..126 (127 = NaN)
    e, m = w >> 3, w & 7
    return np.where(e == 0, m * 2.0 ** -9, (1.0 + m / 8.0) * 2.0 ** (e - 7.0))


_T_FN = _fp8fn_magnitudes()
_DEC_FN = np.concatenate([_T_FN, [np.nan]])               # index = word & 127


class FP8E4M3FN(Codec):
    key, label, bits = "fp8_e4m3fn", "FP8 E4M3FN (OCP, bias 7, max 448)", 8

    def encode(self, w):
        w = _finite(w, self.key)
        x = w.astype(np.float64).reshape(-1)
        idx = formats8.fp_encode_magnitude(x, _T_FN)
        words = ((np.signbit(x).astype(np.uint8) << 7) | idx.astype(np.uint8)).astype(np.uint8)
        return {"words": words, "scale": None, "shape": tuple(w.shape), "n": int(w.size),
                "n_clipped": int((np.abs(x) > FP8FN_MAX).sum())}

    def decode(self, enc):
        u = enc["words"].astype(np.int64)
        val = _DEC_FN[u & 127]
        return np.where(u >> 7 == 1, -val, val).astype(np.float32).reshape(enc["shape"])

    def config(self):
        return {"variant": self.label, "bit_layout": "S(1) EEEE(4) MMM(3), word = (s<<7)|(es<<3)|m, uint8",
                "exponent_bias": 7,
                "subnormal_behavior": "enabled: es=0 stores m * 2^-9 (smallest nonzero magnitude 2^-9 = 0.001953)",
                "finite_nan_handling": "no infinities; exactly one NaN magnitude pattern S.1111.111 (0x7F / 0xFF); "
                                       "all other 254 words are finite; max finite 448 (S.1111.110); min normal 2^-6",
                "rounding": "round-to-nearest, ties to the even word (= IEEE round-to-nearest-even)",
                "saturation_clipping": "|w| > 448 saturates to 448 (the encoder never produces the NaN pattern)",
                "scale": "none (UNSCALED: weights smaller than 2^-10 round to zero)", "zero_point": "none",
                "nan_inf": "encoder rejects non-finite input; decoder maps the NaN pattern to NaN",
                "granularity": "n/a (float format)", "calibration": "none (data-free)",
                "storage": "actual encoded uint8 words; decoded to FP32 for evaluation (weight-only)",
                "codec_source": "src/fr/stage2.py (table-based); cross-checked against torch.float8_e4m3fn in the self-tests"}


class IntPerTensor(Codec):
    has_scale = True

    def __init__(self, bits: int):
        self.bits = bits
        self.key = f"int{bits}_pt"
        self.label = f"INT{bits} (symmetric, per-tensor)"

    def encode(self, w):
        w = _finite(w, self.key)
        words, scale = formats8.int_encode(w, self.bits, per_channel=False)
        raw = np.rint(w.astype(np.float64).reshape(-1) / float(scale))
        clipped = int(((raw < -(1 << (self.bits - 1))) | (raw > (1 << (self.bits - 1)) - 1)).sum())
        stored = pack_nibbles(words) if self.bits == 4 else words
        return {"words": stored, "scale": np.asarray(scale, dtype=np.float32).reshape(()), "shape": tuple(w.shape),
                "n": int(w.size), "n_clipped": clipped}

    def decode(self, enc):
        words = unpack_nibbles(enc["words"], enc["n"]) if self.bits == 4 else enc["words"]
        return formats8.int_decode(words, enc["scale"], enc["shape"], self.bits)

    def config(self):
        b = self.bits
        qmax = (1 << (b - 1)) - 1
        c = {"variant": self.label,
             "bit_layout": f"{b}-bit two's complement signed integer code" + (
                 "; two codes per byte, element 2j in the LOW nibble (bits 0-3) of byte j, element 2j+1 in the HIGH "
                 "nibble (bits 4-7); an odd element count is padded with one zero nibble" if b == 4 else "; one code per byte"),
             "mapping": "symmetric (paper Eq. 3)", "zero_point": "none (symmetric, zero maps to zero)",
             "scale": f"s = max|w| / {qmax} computed once per tensor (per layer), stored as one float32 (4 bytes); "
                      "an all-zero tensor gets s = 1",
             "granularity": "per-tensor (one scale per layer's weight tensor)", "scale_computed": "per layer",
             "calibration": "max-abs of the weights; data-free (no calibration data)",
             "rounding": "round-to-nearest, ties to even (np.rint) of w/s computed in float64 with the float32 scale",
             "signed_range": f"[{-(1 << (b - 1))}, {qmax}] (clip); with s = max|w|/{qmax} the code -{1 << (b - 1)} is not reached",
             "saturation_clipping": f"codes clipped to [{-(1 << (b - 1))}, {qmax}]",
             "subnormal_behavior": "n/a", "nan_inf": "encoder rejects non-finite input; no NaN/Inf codes exist; a flipped "
                                                    "float32 scale can be non-finite (not touched in Stage 2)",
             "storage": "actual encoded bytes" + (" (packed nibbles)" if b == 4 else "") +
                        " + float32 scale; decoded to FP32 for evaluation (weight-only)",
             "codec_source": "src/fr/formats8.py (int_encode / int_decode)" + (" + stage2.pack_nibbles" if b == 4 else "")}
        return c


def make_codecs(include_comparison: bool = True) -> "OrderedDict[str, Codec]":
    cs = [E4M11("subnormal"), FP16(), FP8E4M3FN(), IntPerTensor(8), IntPerTensor(4)]
    if include_comparison:
        cs.append(E4M11("paper"))
    return OrderedDict((c.key, c) for c in cs)


# ---------------------------------------------------------------------------------------------- round trip
def roundtrip_row(w: np.ndarray, w_hat: np.ndarray) -> dict:
    """Reconstruction metrics of one tensor. Signal = the FP32 weights, noise = w - w_hat (float64 sums)."""
    w64 = np.asarray(w, dtype=np.float64).reshape(-1)
    h64 = np.asarray(w_hat, dtype=np.float64).reshape(-1)
    nonfinite = int((~np.isfinite(h64)).sum())
    e = w64 - h64
    sig, noise = float(np.dot(w64, w64)), float(np.dot(e, e))
    n = w64.size
    return {"mse": noise / n, "rmse": math.sqrt(noise / n), "mae": float(np.abs(e).mean()),
            "max_abs_error": float(np.abs(e).max()), "relative_l2": math.sqrt(noise / sig) if sig > 0 else float("nan"),
            "sqnr_db": sqnr_db(sig, noise), "signal_power_sum": sig, "noise_power_sum": noise, "n": n,
            "decoded_nonfinite": nonfinite,
            "zeroed_from_nonzero": int(((h64 == 0) & (w64 != 0)).sum())}


def pooled_row(parts: List[dict]) -> dict:
    """Global metrics pooled over all layers (sums of signal and noise power, as in paper Eq. 6)."""
    n = sum(p["n"] for p in parts)
    sig = sum(p["signal_power_sum"] for p in parts)
    noise = sum(p["noise_power_sum"] for p in parts)
    return {"mse": noise / n, "rmse": math.sqrt(noise / n),
            "mae": sum(p["mae"] * p["n"] for p in parts) / n,
            "max_abs_error": max(p["max_abs_error"] for p in parts),
            "relative_l2": math.sqrt(noise / sig), "sqnr_db": sqnr_db(sig, noise), "signal_power_sum": sig,
            "noise_power_sum": noise, "n": n, "decoded_nonfinite": sum(p["decoded_nonfinite"] for p in parts),
            "zeroed_from_nonzero": sum(p["zeroed_from_nonzero"] for p in parts)}


# ---------------------------------------------------------------------------------------------- storage
def storage_row(codec: Codec, enc: dict) -> dict:
    """Actual and theoretical storage of one encoded tensor."""
    n = enc["n"]
    payload = int(enc["words"].nbytes)
    scale_bytes = int(enc["scale"].nbytes) if enc["scale"] is not None else 0
    theoretical_bits = n * codec.bits
    padding_bits = payload * 8 - theoretical_bits
    return {"theoretical_bits": theoretical_bits, "payload_bytes": payload, "padding_bits": padding_bits,
            "scale_bytes": scale_bytes, "codec_metadata_bytes": 0, "total_bytes": payload + scale_bytes}


def content_sha256(encoded: Mapping[str, dict]) -> str:
    """Hash of the stored content (layer names, shapes, words, scales), independent of the container file format."""
    h = hashlib.sha256()
    for name, enc in encoded.items():
        h.update(name.encode())
        h.update(str(tuple(enc["shape"])).encode())
        h.update(np.ascontiguousarray(enc["words"]).tobytes())
        if enc["scale"] is not None:
            h.update(np.ascontiguousarray(enc["scale"]).tobytes())
    return h.hexdigest()


def save_encoded(path, encoded: Mapping[str, dict]) -> None:
    arrays = {}
    for name, enc in encoded.items():
        arrays[f"{name}|words"] = enc["words"]
        arrays[f"{name}|shape"] = np.asarray(enc["shape"], dtype=np.int64)
        arrays[f"{name}|n"] = np.asarray(enc["n"], dtype=np.int64)
        if enc["scale"] is not None:
            arrays[f"{name}|scale"] = enc["scale"]
    np.savez(path, **arrays)


def load_encoded(path, layer_names) -> "OrderedDict[str, dict]":
    z = np.load(path)
    out = OrderedDict()
    for name in layer_names:
        out[name] = {"words": z[f"{name}|words"], "shape": tuple(int(v) for v in z[f"{name}|shape"]),
                     "n": int(z[f"{name}|n"]), "scale": z[f"{name}|scale"] if f"{name}|scale" in z.files else None,
                     "n_clipped": 0}
    return out


# ---------------------------------------------------------------------------------------------- self-tests
def _t(name: str, passed: bool, detail: str = "") -> dict:
    return {"test": name, "passed": bool(passed), "detail": detail}


def run_codec_selftests() -> List[dict]:
    """Encode/decode tests whose outcome is recorded in the quantizer manifest (runbook 4.2)."""
    R: List[dict] = []
    rng = np.random.default_rng(0)

    # ---- E4M11 (SN and FTZ)
    for mode, key in (("subnormal", "e4m11_sn"), ("paper", "e4m11_ftz")):
        words = np.array([(s << 15) | (es << 11) | m for s in (0, 1) for es in range(1, 15) for m in range(2048)], dtype=np.uint16)
        if mode == "subnormal":
            words = np.concatenate([words, np.array([(s << 15) | m for s in (0, 1) for m in range(2048)], dtype=np.uint16)])
        dec = e4m11.decode(words, mode, "clamp", 7)
        R.append(_t(f"{key}: decode->encode returns the same word for all {words.size} non-reserved words",
                    np.array_equal(e4m11.encode(dec, mode, 7), words)))
    enc = E4M11("subnormal").encode(np.array([3.14], dtype=np.float32))["words"][0]
    R.append(_t("e4m11_sn: paper example w=3.14 -> 0x448F", int(enc) == 0x448F, f"got 0x{int(enc):04X}"))
    tiny = np.float32(2.0 ** -10)
    ws = E4M11("subnormal").encode(np.array([tiny]))["words"][0]
    R.append(_t("e4m11_sn: 2^-10 is a subnormal (es=0, m=128) and decodes exactly",
                int(ws) == 128 and float(e4m11.decode(np.array([ws]), "subnormal", "clamp", 7)[0]) == float(tiny)))
    wf = E4M11("paper").encode(np.array([tiny]))["words"][0]
    R.append(_t("e4m11_ftz: 2^-10 flushes to zero", int(wf) == 0))
    big = E4M11("subnormal").encode(np.array([1000.0], dtype=np.float32))["words"][0]
    R.append(_t("e4m11_sn: saturation at max finite 255.9375",
                float(e4m11.decode(np.array([big]), "subnormal", "clamp", 7)[0]) == 255.9375))
    t1 = E4M11("subnormal").encode(np.array([1 + 2.0 ** -12, 1 + 3 * 2.0 ** -12], dtype=np.float32))["words"]
    R.append(_t("e4m11_sn: round-to-nearest-even on exact ties (1+2^-12 -> m=0, 1+3*2^-12 -> m=2)",
                [int(t1[0]) & 0x7FF, int(t1[1]) & 0x7FF] == [0, 2]))

    # ---- FP16
    c16 = FP16()
    e = c16.encode(np.array([1.0, 65504.0, 2.0 ** -24, 1e9, -0.5], dtype=np.float32))
    R.append(_t("fp16: 1.0 -> 0x3C00, 65504 -> 0x7BFF, 2^-24 -> 0x0001 (smallest subnormal), 1e9 saturates to 65504, -0.5 -> 0xB800",
                [int(v) for v in e["words"]] == [0x3C00, 0x7BFF, 0x0001, 0x7BFF, 0xB800], str([hex(int(v)) for v in e["words"]])))
    allw = np.arange(65536, dtype=np.uint16)
    fin = np.isfinite(allw.view(np.float16))
    back = c16.encode(c16.decode({"words": allw, "shape": (65536,)}).astype(np.float32)[fin])["words"]
    R.append(_t("fp16: decode->encode returns the same word for every finite word (-0 and +0 included)",
                np.array_equal(back, allw[fin])))

    # ---- FP8 E4M3FN
    c8 = FP8E4M3FN()
    allb = np.arange(256, dtype=np.uint8)
    dec8 = c8.decode({"words": allb, "shape": (256,)})
    R.append(_t("fp8_e4m3fn: exactly 2 NaN words (0x7F, 0xFF), no infinities, max finite 448, min positive 2^-9",
                set(np.where(np.isnan(dec8))[0].tolist()) == {127, 255} and not np.isinf(dec8).any()
                and float(np.nanmax(dec8)) == 448.0 and float(dec8[1]) == 2.0 ** -9))
    fin8 = ~np.isnan(dec8)
    back8 = c8.encode(dec8[fin8])["words"]
    R.append(_t("fp8_e4m3fn: decode->encode returns the same word for all 254 finite words", np.array_equal(back8, allb[fin8])))
    sat = c8.encode(np.array([1000.0, -1000.0], dtype=np.float32))["words"]
    R.append(_t("fp8_e4m3fn: values above 448 saturate to +-448 (0x7E / 0xFE), never to the NaN pattern",
                [int(v) for v in sat] == [0x7E, 0xFE]))
    try:
        import torch
        tt = getattr(torch, "float8_e4m3fn", None)
        if tt is None:
            raise AttributeError
        ref = torch.tensor(allb.astype(np.uint8)).view(tt).float().numpy()
        same = np.array_equal(np.nan_to_num(ref, nan=-1.0), np.nan_to_num(dec8, nan=-1.0))
        R.append(_t("fp8_e4m3fn: all 256 decoded values equal torch.float8_e4m3fn", same))
        x = np.exp(rng.uniform(math.log(2.0 ** -12), math.log(448.0), 200000)).astype(np.float32) * rng.choice([-1, 1], 200000)
        mids = (_T_FN[:-1] + _T_FN[1:]) / 2.0                       # exact ties between neighbours
        x = np.concatenate([x, mids.astype(np.float32), -mids.astype(np.float32)])
        mine = c8.encode(x)["words"]
        theirs = torch.tensor(x).to(tt).view(torch.uint8).numpy()
        R.append(_t("fp8_e4m3fn: encoder rounding equals torch.float8_e4m3fn casting (200k random values + all exact ties)",
                    np.array_equal(mine, theirs), f"{int((mine != theirs).sum())} mismatches"))
        x16 = (rng.standard_normal(100000) * 0.01).astype(np.float32)
        R.append(_t("fp16: encoder equals torch.float16 casting on 100k N(0, 0.01) values",
                    np.array_equal(c16.encode(x16)["words"], torch.tensor(x16).to(torch.float16).view(torch.int16).numpy().view(np.uint16))))
    except (ImportError, AttributeError):
        R.append(_t("fp8_e4m3fn / fp16: cross-check with torch", False, "torch.float8_e4m3fn not available"))

    # ---- INT8 / INT4
    for bits in (8, 4):
        ci = IntPerTensor(bits)
        w = (rng.standard_normal((16, 9)) * 0.02).astype(np.float32)
        enc_i = ci.encode(w)
        s = float(np.abs(w).max()) / ((1 << (bits - 1)) - 1)
        R.append(_t(f"int{bits}_pt: scale = max|w|/{(1 << (bits - 1)) - 1} stored as one float32",
                    enc_i["scale"].dtype == np.float32 and enc_i["scale"].shape == () and abs(float(enc_i["scale"]) - s) < 1e-9 * max(s, 1)))
        q = np.rint(w.astype(np.float64) / float(enc_i["scale"]))
        R.append(_t(f"int{bits}_pt: decode equals round(w/s)*s with codes in [{-(1 << (bits - 1))}, {(1 << (bits - 1)) - 1}]",
                    np.allclose(ci.decode(enc_i), (q * float(enc_i["scale"])).reshape(w.shape), rtol=0, atol=1e-7)
                    and q.min() >= -(1 << (bits - 1)) and q.max() <= (1 << (bits - 1)) - 1))
        z = ci.encode(np.zeros((3, 3), dtype=np.float32))
        R.append(_t(f"int{bits}_pt: all-zero tensor uses scale 1 and decodes to zeros",
                    float(z["scale"]) == 1.0 and not ci.decode(z).any()))
    for n in (1, 2, 7, 8, 1001):
        codes = rng.integers(0, 16, n).astype(np.uint8)
        p = pack_nibbles(codes)
        R.append(_t(f"int4 packing: pack/unpack round trip for n={n}", np.array_equal(unpack_nibbles(p, n), codes) and p.size == (n + 1) // 2))
    p = pack_nibbles(np.array([0x1, 0x2, 0x3], dtype=np.uint8))
    R.append(_t("int4 packing: element 0 -> low nibble, element 1 -> high nibble, odd count pads a zero high nibble",
                [int(v) for v in p] == [0x21, 0x03]))
    R.append(_t("int4 bit location: element 3 bit 2 -> byte 1, bit position 6", int4_bit_location(3, 2) == (1, 6)))
    c4 = IntPerTensor(4)
    w4 = np.array([-0.7, 0.7, 0.1], dtype=np.float32)
    e4 = c4.encode(w4)
    R.append(_t("int4_pt: two's complement nibbles (-7 -> 0x9, 7 -> 0x7, 1 -> 0x1) packed as [0x79, 0x01]",
                [int(v) for v in e4["words"]] == [0x79, 0x01], str([hex(int(v)) for v in e4["words"]])))
    return R
