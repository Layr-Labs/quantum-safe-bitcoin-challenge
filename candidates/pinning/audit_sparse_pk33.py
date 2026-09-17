#!/usr/bin/env python3
"""Bit-exact audit of the 33-byte compressed-key sparse SHA-256 schedule.

Compares _SHA256TransformPk33 (fresh IV, W[0..8] live, W[9..14]=0, W[15]=264)
against a generic SHA-256 compress on the same padded block. Also binds the
production call site in pinning.cu.
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

IV = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
]

L = 0x108  # 33*8 = 264


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
    return [
        (state[0] + a) & 0xFFFFFFFF,
        (state[1] + b) & 0xFFFFFFFF,
        (state[2] + c) & 0xFFFFFFFF,
        (state[3] + d) & 0xFFFFFFFF,
        (state[4] + e) & 0xFFFFFFFF,
        (state[5] + f) & 0xFFFFFFFF,
        (state[6] + g) & 0xFFFFFFFF,
        (state[7] + h) & 0xFFFFFFFF,
    ]


def transform_pk33(in9: list[int]) -> list[int]:
    """Mirror of _SHA256TransformPk33 first-16 + first-WMIX, then generic."""
    w = list(in9) + [0] * 6 + [L]
    a, b, c, d, e, f, g, h = IV

    def round_add(wi: int, ki: int) -> None:
        nonlocal a, b, c, d, e, f, g, h
        t1 = (h + S1(e) + Ch(e, f, g) + K[ki] + wi) & 0xFFFFFFFF
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

    for i in range(9):
        round_add(w[i], i)
    for i in range(9, 15):
        round_add(0, i)
    round_add(L, 15)

    # Specialized first WMIX (in-place), matching production
    w[0] = (w[0] + s0(w[1])) & 0xFFFFFFFF
    w[1] = (w[1] + s1(L) + s0(w[2])) & 0xFFFFFFFF
    w[2] = (w[2] + s1(w[0]) + s0(w[3])) & 0xFFFFFFFF
    w[3] = (w[3] + s1(w[1]) + s0(w[4])) & 0xFFFFFFFF
    w[4] = (w[4] + s1(w[2]) + s0(w[5])) & 0xFFFFFFFF
    w[5] = (w[5] + s1(w[3]) + s0(w[6])) & 0xFFFFFFFF
    w[6] = (w[6] + s1(w[4]) + L + s0(w[7])) & 0xFFFFFFFF
    w[7] = (w[7] + s1(w[5]) + w[0] + s0(w[8])) & 0xFFFFFFFF
    w[8] = (w[8] + s1(w[6]) + w[1]) & 0xFFFFFFFF
    w[9] = (s1(w[7]) + w[2]) & 0xFFFFFFFF
    w[10] = (s1(w[8]) + w[3]) & 0xFFFFFFFF
    w[11] = (s1(w[9]) + w[4]) & 0xFFFFFFFF
    w[12] = (s1(w[10]) + w[5]) & 0xFFFFFFFF
    w[13] = (s1(w[11]) + w[6]) & 0xFFFFFFFF
    w[14] = (s1(w[12]) + w[7] + s0(L)) & 0xFFFFFFFF
    w[15] = (w[15] + s1(w[13]) + w[8] + s0(w[0])) & 0xFFFFFFFF

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

    rnd(16)
    wmix()
    rnd(32)
    wmix()
    rnd(48)
    return [
        (IV[0] + a) & 0xFFFFFFFF,
        (IV[1] + b) & 0xFFFFFFFF,
        (IV[2] + c) & 0xFFFFFFFF,
        (IV[3] + d) & 0xFFFFFFFF,
        (IV[4] + e) & 0xFFFFFFFF,
        (IV[5] + f) & 0xFFFFFFFF,
        (IV[6] + g) & 0xFFFFFFFF,
        (IV[7] + h) & 0xFFFFFFFF,
    ]


def bind_source() -> None:
    assert "_SHA256TransformPk33" in SRC
    assert re.search(
        r"_SHA256TransformPk33\s*\(\s*hs\s*,\s*pb\s*\)",
        SRC,
    ), "finish call site missing"
    assert "uint32_t pb[9]" in SRC
    assert "pb[15]=0x108" not in SRC or SRC.count("pb[15]=0x108") == 0
    # FastTail11 must remain
    assert "_SHA256TransformFastTail11" in SRC
    # Pad constant in helper
    assert "0x108u" in SRC
    # No fused X3
    assert "_SHA256TransformFused" not in SRC


def main() -> None:
    bind_source()
    rng = random.Random(20260917)
    cases = 0
    for _ in range(50000):
        in9 = [rng.randrange(1 << 32) for _ in range(9)]
        block = in9 + [0] * 6 + [L]
        got = transform_pk33(in9)
        ref = transform_generic(list(IV), block)
        assert got == ref, (got, ref, in9)
        cases += 1
    # Boundary: all-zero data words, all-ones, alternating
    for in9 in (
        [0] * 9,
        [0xFFFFFFFF] * 9,
        [0xA5A5A5A5, 0x5A5A5A5A] * 4 + [0x12345678],
    ):
        block = in9 + [0] * 6 + [L]
        assert transform_pk33(in9) == transform_generic(list(IV), block)
        cases += 1
    print(
        f"PASS: sparse Pk33 schedule; cases={cases}; "
        f"L={L}; call site + FastTail11 retained"
    )


if __name__ == "__main__":
    main()
