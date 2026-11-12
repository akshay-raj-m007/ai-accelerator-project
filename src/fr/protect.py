"""Protection option ``scale_tmr``: triple modular redundancy of the float32 scale word of an integer layer.

Why this word: an INT8/INT4 layer is stored as integer codes plus ONE float32 scale (4 bytes per layer, 84 bytes for all 21 layers). One flipped bit in that word
rescales every weight of the layer (Stage 4: a flipped exponent bit collapses the network, 8 - 30 pp on average per flipped scale word).

Method (implemented, tested, measured - nothing else is claimed):
  * the scale is stored THREE times (3 x 4 bytes, +8 bytes per protected layer, 4-byte aligned, no padding);
  * on every read the effective scale is the BIT-WISE MAJORITY of the three stored 32-bit patterns:  vote(a, b, c) = (a & b) | (a & c) | (b & c).
Guarantee: any set of flips that touches at most one copy at each bit position is corrected, in particular every single flipped bit (3 x 32 positions). It is NOT
corrected if the same bit position is flipped in two of the three copies (two flips needed); that residual case is measured / reported, not hidden.
The integer weight codes are NOT protected by this option (their flips are the weight-fault part of R).
"""
from __future__ import annotations

import numpy as np

COPIES = 3
PROTECTION_NONE = "none"
PROTECTION_TMR = "scale_tmr"
TMR_OVERHEAD_BYTES = (COPIES - 1) * 4          # two extra float32 copies per protected layer


def vote_u32(a, b, c):
    """Bit-wise majority of three uint32 patterns (scalars or arrays)."""
    a, b, c = (np.asarray(x, dtype=np.uint32) for x in (a, b, c))
    return (a & b) | (a & c) | (b & c)


def effective_scale(copies: np.ndarray) -> np.float32:
    """Effective float32 scale from the stored uint32 copies (1 copy: as stored; 3 copies: bit-wise majority)."""
    u = np.asarray(copies, dtype=np.uint32).reshape(-1)
    if u.size == 1:
        v = u[0]
    elif u.size == COPIES:
        v = vote_u32(u[0], u[1], u[2])
    else:
        raise ValueError(f"expected 1 or {COPIES} stored copies, got {u.size}")
    return np.array(v, dtype=np.uint32).reshape(1).view(np.float32)[0]


def parse_option(token: str):
    """'int8_pt' -> ('int8_pt', 'none');  'int8_pt+scale_tmr' -> ('int8_pt', 'scale_tmr')."""
    fmt, _, prot = token.partition("+")
    prot = prot or PROTECTION_NONE
    if prot not in (PROTECTION_NONE, PROTECTION_TMR):
        raise ValueError(f"unknown protection {prot!r}")
    return fmt, prot


def option_token(fmt: str, prot: str) -> str:
    return fmt if prot == PROTECTION_NONE else f"{fmt}+{prot}"
