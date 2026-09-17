#!/usr/bin/env python3
"""Bit-exact audit of the Fast 11-byte sparse SHA-256 schedule.

Compares the specialized first-16-rounds + first-WMIX schedule used by
_SHA256TransformFastTail11 against a generic SHA-256 compress on the same
padded block: W[0..2] live, W[3..14]=0, W[15]=9995*8. Also binds the
production call site and pad constants in pinning.cu.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = (ROOT / "pinning.cu").read_text()

K = [
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
]

L = 9995 * 8


def ror(x: int, n: int) -> int:
    x &= 0xFFFFFFFF
    return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF


def S0(x: int) -> int:
    return ror(x, 2) ^ ror(x, 13) ^ ror(x, 22)


def S1(x: int) -> int:
    return ror(x, 6) ^ ror(x, 11) ^ ror(x, 25)


def s0(x: int) -> int:
    return ror(x, 7) ^ ror(x, 18) ^ (x >> 3)


def s1(x: int) -> int:
    return ror(x, 17) ^ ror(x, 19) ^ (x >> 10)


def Ch(x: int, y: int, z: int) -> int:
    return z ^ (x & (y ^ z))


def Maj(x: int, y: int, z: int) -> int:
    return (x & y) | (z & (x | y))


def transform_generic(state: list[int], block16: list[int]) -> list[int]:
    w = list(block16)
    a, b, c, d, e, f, g, h = state

    def rnd(kbase: int) -> None:
        nonlocal a, b, c, d, e, f, g, h
        for i in range(16):
            t1 = (h + S1(e) + Ch(e, f, g) + K[kbase + i] + w[i]) & 0xFFFFFFFF
            t2 = (S0(a) + Maj(a, b, c)) & 0xFFFFFFFF
            h, g, f, e, d, c, b, a = (
                g,
                f,
                e,
                (d + t1) & 0xFFFFFFFF,
                c,
                b,
                a,
                (t1 + t2) & 0xFFFFFFFF,
            )

    def wmix() -> None:
        for i in range(16):
            w[i] = (
                w[i]
                + s1(w[(i + 14) % 16])
                + w[(i + 9) % 16]
                + s0(w[(i + 1) % 16])
            ) & 0xFFFFFFFF

    rnd(0)
    wmix()
    rnd(16)
    wmix()
    rnd(32)
    wmix()
    rnd(48)
    return [(state[i] + v) & 0xFFFFFFFF for i, v in enumerate([a, b, c, d, e, f, g, h])]


def transform_fast11(state: list[int], w0: int, w1: int, w2: int) -> list[int]:
    w = [w0, w1, w2] + [0] * 12 + [L]
    a, b, c, d, e, f, g, h = state

    def rnd_sparse() -> None:
        nonlocal a, b, c, d, e, f, g, h
        words = [w0, w1, w2] + [0] * 12 + [L]
        for i in range(16):
            t1 = (h + S1(e) + Ch(e, f, g) + K[i] + words[i]) & 0xFFFFFFFF
            t2 = (S0(a) + Maj(a, b, c)) & 0xFFFFFFFF
            h, g, f, e, d, c, b, a = (
                g,
                f,
                e,
                (d + t1) & 0xFFFFFFFF,
                c,
                b,
                a,
                (t1 + t2) & 0xFFFFFFFF,
            )

    def wmix_spec() -> None:
        w[0] = (w[0] + s0(w[1])) & 0xFFFFFFFF
        w[1] = (w[1] + s1(L) + s0(w[2])) & 0xFFFFFFFF
        w[2] = (w[2] + s1(w[0])) & 0xFFFFFFFF
        w[3] = s1(w[1]) & 0xFFFFFFFF
        w[4] = s1(w[2]) & 0xFFFFFFFF
        w[5] = s1(w[3]) & 0xFFFFFFFF
        w[6] = (s1(w[4]) + L) & 0xFFFFFFFF
        w[7] = (s1(w[5]) + w[0]) & 0xFFFFFFFF
        w[8] = (s1(w[6]) + w[1]) & 0xFFFFFFFF
        w[9] = (s1(w[7]) + w[2]) & 0xFFFFFFFF
        w[10] = (s1(w[8]) + w[3]) & 0xFFFFFFFF
        w[11] = (s1(w[9]) + w[4]) & 0xFFFFFFFF
        w[12] = (s1(w[10]) + w[5]) & 0xFFFFFFFF
        w[13] = (s1(w[11]) + w[6]) & 0xFFFFFFFF
        w[14] = (s1(w[12]) + w[7] + s0(L)) & 0xFFFFFFFF
        w[15] = (L + s1(w[13]) + w[8] + s0(w[0])) & 0xFFFFFFFF

    def wmix() -> None:
        for i in range(16):
            w[i] = (
                w[i]
                + s1(w[(i + 14) % 16])
                + w[(i + 9) % 16]
                + s0(w[(i + 1) % 16])
            ) & 0xFFFFFFFF

    def rnd(kbase: int) -> None:
        nonlocal a, b, c, d, e, f, g, h
        for i in range(16):
            t1 = (h + S1(e) + Ch(e, f, g) + K[kbase + i] + w[i]) & 0xFFFFFFFF
            t2 = (S0(a) + Maj(a, b, c)) & 0xFFFFFFFF
            h, g, f, e, d, c, b, a = (
                g,
                f,
                e,
                (d + t1) & 0xFFFFFFFF,
                c,
                b,
                a,
                (t1 + t2) & 0xFFFFFFFF,
            )

    rnd_sparse()
    wmix_spec()
    rnd(16)
    wmix()
    rnd(32)
    wmix()
    rnd(48)
    return [(state[i] + v) & 0xFFFFFFFF for i, v in enumerate([a, b, c, d, e, f, g, h])]


def check_source_shape() -> None:
    assert "_SHA256TransformFastTail11" in SRC
    assert re.search(
        r"_SHA256TransformFastTail11\s*\(\s*state\s*,\s*w0\s*,\s*w1\s*,\s*w2\s*\)",
        SRC,
    )
    assert "const uint32_t L = 9995u * 8u" in SRC
    # Fast path no longer materializes a 16-word generic block for the tail.
    assert "0,0,0,0,0,0,0,0,0,0,0,0,9995u*8u" not in SRC
    # The second SHA256d compression and the finish key hash now use their own
    # audited sparse-schedule transforms (see audit_sparse_pads.py); the
    # generic 16-word pads must be gone from both call sites.
    assert "b2[15]=256" not in SRC and "b2[15] = 256" not in SRC
    assert "_SHA256TransformDigest32(s2, state);" in SRC
    assert "_SHA256TransformPubkey33(hs,pb);" in SRC
    assert "_SHA256Transform(hs,pb)" not in SRC and "_SHA256Transform(hs, pb)" not in SRC


def main() -> None:
    check_source_shape()
    rng = random.Random(0xFA5711)
    cases = 0
    for _ in range(50_000):
        st = [rng.getrandbits(32) for _ in range(8)]
        w0, w1, w2 = (rng.getrandbits(32) for _ in range(3))
        blk = [w0, w1, w2] + [0] * 12 + [L]
        assert transform_generic(st, blk) == transform_fast11(st, w0, w1, w2)
        cases += 1

    I = [
        0x6A09E667,
        0xBB67AE85,
        0x3C6EF372,
        0xA54FF53A,
        0x510E527F,
        0x9B05688C,
        0x1F83D9AB,
        0x5BE0CD19,
    ]
    edges = [0, 1, 255, 256, 65535, 65536, 500000000, 0xFFFFFFFF]
    for lt in edges:
        w0 = 0xABCDEF00 | (lt & 255)
        w1 = (
            (((lt >> 8) & 255) << 24)
            | (((lt >> 16) & 255) << 16)
            | (((lt >> 24) & 255) << 8)
            | 0x11
        )
        w2 = 0x22334480
        blk = [w0, w1, w2] + [0] * 12 + [L]
        assert transform_generic(I, blk) == transform_fast11(I, w0, w1, w2)
        cases += 1

    # Zero midstate / zero words / max words
    for st in ([0] * 8, [0xFFFFFFFF] * 8, I):
        for w0, w1, w2 in ((0, 0, 0), (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF), (1, 2, 3)):
            blk = [w0, w1, w2] + [0] * 12 + [L]
            assert transform_generic(st, blk) == transform_fast11(st, w0, w1, w2)
            cases += 1

    assert L == 79960
    print(
        f"PASS: sparse FastTail11 schedule; cases={cases}; "
        f"L={L}; source call site and pad constants bound"
    )


if __name__ == "__main__":
    main()
