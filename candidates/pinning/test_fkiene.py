#!/usr/bin/env python3
"""Host-only audit of QSB_FKIENE digit extraction.

The production decoder now uses a 64-bit funnel-right of two recode limbs
instead of `M[j]>>sh` plus a `sh>46` branch, and keeps the 15 codes in
registers instead of volatile shared. This file proves:

  1. funnel(lo, hi, sh) equals the bit-slice of the 256-bit integer M.
  2. The idx/neg packing matches the previous shared-path formula
     (top-bit sign, last window uses recode sign) on random recode states.

No GPU, no PTX. SPDX-License-Identifier: GPL-3.0-only
"""

from __future__ import annotations

import random


MASK64 = (1 << 64) - 1
GT_CHUNKS = 15


def funnel_r64(lo: int, hi: int, sh: int) -> int:
    lo &= MASK64
    hi &= MASK64
    if sh == 0:
        return lo
    return ((lo >> sh) | ((hi << (64 - sh)) & MASK64)) & MASK64


def extract_field(M, pos: int, bits: int) -> int:
    j, sh = pos >> 6, pos & 63
    lo = M[j]
    hi = M[j + 1] if j < 3 else 0
    return funnel_r64(lo, hi, sh) & ((1 << bits) - 1)


def bit_slice(M, pos: int, bits: int) -> int:
    v = M[0] | (M[1] << 64) | (M[2] << 128) | (M[3] << 192)
    return (v >> pos) & ((1 << bits) - 1)


def digit_code(M, negative: int, c: int) -> int:
    pos = 1 if c == 0 else 17 * c + 2
    bits = 18 if c == 0 else 17
    f = extract_field(M, pos, bits)
    tm = -negative if c == GT_CHUNKS - 1 else (f >> (bits - 1)) - 1
    idx = (f ^ (tm & 0xFFFFFFFF)) & ((1 << (bits - 1)) - 1)
    neg = 1 if tm < 0 else 0
    return idx | (neg << 31)


def old_extract(M, pos: int, bits: int) -> int:
    """Shared-path extractor: >> plus the sh>46 cross-limb or."""
    j, sh = pos // 64, pos % 64
    value = M[j] >> sh
    if j < 3 and sh > 46:
        value |= (M[j + 1] << (64 - sh)) & MASK64
    return value & ((1 << bits) - 1)


def audit_funnel(rng: random.Random, n: int = 200_000) -> int:
    for _ in range(n):
        M = [rng.randrange(1 << 64) for _ in range(4)]
        for c in range(GT_CHUNKS):
            pos = 1 if c == 0 else 17 * c + 2
            bits = 18 if c == 0 else 17
            a = extract_field(M, pos, bits)
            b = bit_slice(M, pos, bits)
            d = old_extract(M, pos, bits)
            if a != b or a != d:
                raise AssertionError(f"c={c} pos={pos} funnel={a} slice={b} old={d}")
    return n


def audit_codes(rng: random.Random, n: int = 50_000) -> int:
    for _ in range(n):
        M = [rng.randrange(1 << 64) for _ in range(4)]
        negative = rng.randrange(2)
        for c in range(GT_CHUNKS):
            code = digit_code(M, negative, c)
            pos = 1 if c == 0 else 17 * c + 2
            bits = 18 if c == 0 else 17
            f = bit_slice(M, pos, bits)
            tm = -negative if c == GT_CHUNKS - 1 else (f >> (bits - 1)) - 1
            idx = (f ^ (tm & 0xFFFFFFFF)) & ((1 << (bits - 1)) - 1)
            neg = 1 if tm < 0 else 0
            want = idx | (neg << 31)
            if code != want:
                raise AssertionError(f"c={c} code={code:#x} want={want:#x}")
    return n


def main() -> int:
    rng = random.Random(0xF1E4E)
    n1 = audit_funnel(rng)
    n2 = audit_codes(rng)
    print(f"test_fkiene: funnel==bit-slice==old-extract {n1} windows ok")
    print(f"test_fkiene: idx/neg packing {n2} recode-states x 15 ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
