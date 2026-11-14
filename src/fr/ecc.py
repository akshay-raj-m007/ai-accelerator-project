"""Word-level protection codes for the protocol Section 5.3 protection analysis (None / Parity / SECDED / DMR).

The protected unit is one 32-bit LOGICAL WORD of the serialized encoded payload (little-endian view of the stored bytes, the last
word zero-padded). Every strategy is really encoded, bits of the STORED word are flipped, and the word is read back through the
strategy's own decoder; nothing is computed from a formula.

  none    stored = 32 data bits.
  parity  stored = 33 bits: data bits 0-31 + one EVEN parity bit (bit 32). Detects an odd number of flipped bits; never corrects.
  secded  stored = 39 bits: extended Hamming (39,32). Stored bit j (0..37) is Hamming position j+1; positions 1,2,4,8,16,32 hold
          the six check bits, the other 32 positions hold data bit i at position DATA_POS[i]; stored bit 38 is the overall EVEN
          parity of bits 0..37. Single-bit errors are corrected, double-bit errors are detected (data returned as read).
  dmr     stored = 64 bits: copy A (bits 0-31, the copy the system uses) and copy B (bits 32-63). A disagreement is DETECTED, but
          with two copies nothing says which is right, so the data of copy A is returned (detect-only).

Reaction policy (the same for every strategy): a word that is detected but not corrected is consumed as read, so the error remains
RESIDUAL. Faults are injected into the stored word of ONE copy (copy A for DMR).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

WORD_BITS = 32
STRATEGIES = ("none", "parity", "secded", "dmr")
STORED_BITS = {"none": 32, "parity": 33, "secded": 39, "dmr": 64}
CHECK_BITS = {"none": 0, "parity": 1, "secded": 7, "dmr": 32}             # redundancy bits per 32-bit word

CHECK_POS = (1, 2, 4, 8, 16, 32)                                           # Hamming check-bit positions
DATA_POS = tuple(p for p in range(1, 39) if p not in CHECK_POS)           # 32 data positions
assert len(DATA_POS) == 32
SECDED_OVERALL_BIT = 38

_U64 = np.uint64
_MASK32 = _U64(0xFFFFFFFF)
CHECK_MASK = tuple(sum(1 << (p - 1) for p in range(1, 39) if p & (1 << c)) for c in range(6))   # stored bits a check covers


def parity64(x: np.ndarray) -> np.ndarray:
    """Parity (0/1, uint64) of the set bits of each uint64 element."""
    x = np.asarray(x, dtype=_U64).copy()
    for s in (32, 16, 8, 4, 2, 1):
        x ^= x >> _U64(s)
    return x & _U64(1)


def popcount64(x: np.ndarray) -> np.ndarray:
    """Number of set bits (SWAR, independent of the NumPy version)."""
    x = np.asarray(x, dtype=_U64).copy()
    x = x - ((x >> _U64(1)) & _U64(0x5555555555555555))
    x = (x & _U64(0x3333333333333333)) + ((x >> _U64(2)) & _U64(0x3333333333333333))
    x = (x + (x >> _U64(4))) & _U64(0x0F0F0F0F0F0F0F0F)
    return ((x * _U64(0x0101010101010101)) >> _U64(56)).astype(np.int64)


# ------------------------------------------------------------------------------------------------ SECDED
def secded_encode(data: np.ndarray) -> np.ndarray:
    d = np.asarray(data, dtype=_U64) & _MASK32
    cw = np.zeros(d.shape, dtype=_U64)
    for i, p in enumerate(DATA_POS):
        cw |= ((d >> _U64(i)) & _U64(1)) << _U64(p - 1)
    for c, p in enumerate(CHECK_POS):
        cw |= parity64(cw & _U64(CHECK_MASK[c])) << _U64(p - 1)
    cw |= parity64(cw) << _U64(SECDED_OVERALL_BIT)
    return cw


def secded_decode(cw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (data, status, corrected_bit). status 0 = clean, 1 = single error corrected (``corrected_bit`` = stored bit flipped back),
    2 = uncorrectable error detected (data returned as read, corrected_bit = -1)."""
    cw = np.asarray(cw, dtype=_U64).copy()
    s = np.zeros(cw.shape, dtype=np.int64)
    for c in range(6):
        s |= parity64(cw & _U64(CHECK_MASK[c])).astype(np.int64) << c
    p = parity64(cw).astype(np.int64)
    status = np.zeros(cw.shape, dtype=np.int8)
    bit = np.full(cw.shape, -1, dtype=np.int64)
    single = p == 1
    ok_pos = single & (s <= 38)                                             # s == 0: the overall bit itself; 1..38: that position
    bit = np.where(ok_pos & (s == 0), SECDED_OVERALL_BIT, bit)
    bit = np.where(ok_pos & (s > 0), s - 1, bit)
    status[ok_pos] = 1
    status[(single & (s > 38)) | (~single & (s != 0))] = 2                  # impossible single position / even-weight error
    fix = bit >= 0
    cw[fix] ^= _U64(1) << bit[fix].astype(_U64)
    data = np.zeros(cw.shape, dtype=_U64)
    for i, pos in enumerate(DATA_POS):
        data |= ((cw >> _U64(pos - 1)) & _U64(1)) << _U64(i)
    return data, status, bit


# ------------------------------------------------------------------------------------------------ generic strategy interface
def protect(strategy: str, data: np.ndarray) -> np.ndarray:
    """32-bit data words -> stored words (uint64)."""
    d = np.asarray(data, dtype=_U64) & _MASK32
    if strategy == "none":
        return d
    if strategy == "parity":
        return d | (parity64(d) << _U64(32))
    if strategy == "secded":
        return secded_encode(d)
    if strategy == "dmr":
        return d | (d << _U64(32))
    raise ValueError(strategy)


def read(strategy: str, stored: np.ndarray) -> Dict[str, np.ndarray]:
    """Read stored words back. -> data (uint64), detected (bool), corrected_action (bool: the code changed the data bits),
    uncorrectable (bool: error signalled and not repaired)."""
    st = np.asarray(stored, dtype=_U64)
    n = st.shape
    f = np.zeros(n, dtype=bool)
    if strategy == "none":
        return {"data": st & _MASK32, "detected": f, "action": f.copy(), "uncorrectable": f.copy()}
    if strategy == "parity":
        det = parity64(st & _U64((1 << 33) - 1)) == 1
        return {"data": st & _MASK32, "detected": det, "action": f.copy(), "uncorrectable": det}
    if strategy == "secded":
        data, status, _ = secded_decode(st)
        return {"data": data, "detected": status > 0, "action": status == 1, "uncorrectable": status == 2}
    if strategy == "dmr":
        a, b = st & _MASK32, (st >> _U64(32)) & _MASK32
        det = a != b
        return {"data": a, "detected": det, "action": f.copy(), "uncorrectable": det}
    raise ValueError(strategy)


def stored_position(strategy: str, data_bit: int) -> int:
    """Position inside the STORED word of data bit ``data_bit`` (0..31) (copy A for DMR)."""
    if strategy == "secded":
        return DATA_POS[data_bit] - 1
    return int(data_bit)


def check_positions(strategy: str) -> List[int]:
    """Stored-bit positions that are redundancy (not data) bits of one copy."""
    if strategy == "parity":
        return [32]
    if strategy == "secded":
        return [p - 1 for p in CHECK_POS] + [SECDED_OVERALL_BIT]
    return []


def inject(stored: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """XOR the stored-bit ``positions`` (N x m int array, -1 = no flip) into the stored words."""
    st = np.asarray(stored, dtype=_U64).copy()
    pos = np.asarray(positions, dtype=np.int64)
    for j in range(pos.shape[1]):
        m = pos[:, j] >= 0
        st[m] ^= _U64(1) << pos[m, j].astype(_U64)
    return st


def outcome(strategy: str, data: np.ndarray, positions: np.ndarray) -> Dict[str, np.ndarray]:
    """Encode ``data``, flip the stored bits at ``positions``, read back and classify.

    detected          the code signalled an error (parity mismatch / non-zero SECDED syndrome / copy disagreement)
    corrected         the code repaired the word: the data read back equals the original after a correcting action
    uncorrectable     an error was signalled but the data was not repaired
    miscorrected      the SECDED decoder 'corrected' a bit and the data is still wrong
    silent            wrong data and no signal
    residual_word     the data read back differs from the original
    residual_bits     number of data bits that differ
    """
    d = np.asarray(data, dtype=_U64) & _MASK32
    out = read(strategy, inject(protect(strategy, d), positions))
    wrong = out["data"] != d
    return {"detected": out["detected"], "corrected": out["action"] & ~wrong, "uncorrectable": out["uncorrectable"],
            "miscorrected": out["action"] & wrong, "silent": ~out["detected"] & wrong, "residual_word": wrong,
            "residual_bits": popcount64(out["data"] ^ d), "data_out": out["data"]}


# ------------------------------------------------------------------------------------------------ storage accounting
def storage_bytes(strategy: str, payload_bytes: int, scale_bytes: int) -> Dict[str, int]:
    """Actual bytes of one layer's weight store under ``strategy`` (payload zero-padded to whole 32-bit words; the float32 scale
    word, if any, is one more protected word). Redundancy bits are packed densely and rounded up to whole bytes per layer; DMR keeps
    a full second copy of the stored bytes."""
    unprotected = payload_bytes + scale_bytes
    if strategy == "none":
        return {"unprotected_bytes": unprotected, "alignment_padding_bytes": 0, "redundancy_bytes": 0, "total_bytes": unprotected,
                "protected_words": 0}
    n_words = -(-payload_bytes // 4) + scale_bytes // 4
    pad = (-payload_bytes) % 4
    if strategy == "dmr":
        red = unprotected
        return {"unprotected_bytes": unprotected, "alignment_padding_bytes": 0, "redundancy_bytes": red, "total_bytes": unprotected + red,
                "protected_words": n_words}
    red = -(-n_words * CHECK_BITS[strategy] // 8)
    return {"unprotected_bytes": unprotected, "alignment_padding_bytes": pad, "redundancy_bytes": red,
            "total_bytes": unprotected + pad + red, "protected_words": n_words}
