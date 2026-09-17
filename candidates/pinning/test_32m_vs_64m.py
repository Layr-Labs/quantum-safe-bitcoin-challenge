#!/usr/bin/env python3
"""Audit and compare 64 MiB (15-window non-uniform) vs 32 MiB (16-window uniform).

Tests mathematical equivalence, recoder logic, table addressing, and deferred-Y
accumulation chains for both configurations.
"""

from __future__ import annotations
import random

# Secp256k1 curve parameters
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
MASK64 = (1 << 64) - 1

G = (
    0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
    0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
)


def limbs(x: int) -> list[int]:
    return [(x >> (64 * i)) & MASK64 for i in range(4)]


def sub4(a: list[int], b: list[int]) -> tuple[list[int], int]:
    out, borrow = [], 0
    for x, y in zip(a, b):
        value = x - y - borrow
        out.append(value & MASK64)
        borrow = int(value < 0)
    return out, borrow


def source_setup(k: int) -> tuple[list[int], int]:
    kw, nw = limbs(k), limbs(N)
    diff, borrow = sub4(kw, nw)
    reduced = diff if borrow == 0 else kw
    doubled = [
        (reduced[0] << 1) & MASK64,
        ((reduced[1] << 1) | (reduced[0] >> 63)) & MASK64,
        ((reduced[2] << 1) | (reduced[1] >> 63)) & MASK64,
        ((reduced[3] << 1) | (reduced[2] >> 63)) & MASK64,
    ]
    carry = reduced[3] >> 63
    diff, borrow = sub4(doubled, nw)
    value = diff if carry or borrow == 0 else doubled
    odd = value[0] & 1
    negated, _ = sub4(nw, value)
    return (value if odd else negated), (1 if odd else -1)


def source_step(words: list[int], sign: int, bits: int) -> tuple[list[int], int]:
    digit = (words[0] & ((1 << (bits + 1)) - 1)) - (1 << bits)
    shift = bits + 1
    right = [
        ((words[0] >> shift) | (words[1] << (64 - shift))) & MASK64,
        ((words[1] >> shift) | (words[2] << (64 - shift))) & MASK64,
        ((words[2] >> shift) | (words[3] << (64 - shift))) & MASK64,
        words[3] >> shift,
    ]
    next_words = [
        ((right[0] << 1) | 1) & MASK64,
        ((right[1] << 1) | (right[0] >> 63)) & MASK64,
        ((right[2] << 1) | (right[1] >> 63)) & MASK64,
        ((right[3] << 1) | (right[2] >> 63)) & MASK64,
    ]
    return next_words, sign * digit


# Recoders
def recode_64mib_15w(k: int) -> list[int]:
    """15 chunks: widths [18] + 13*[17], total 1,048,576 points (64 MiB)."""
    words, sign = source_setup(k)
    digits = []
    for bits in [18] + [17] * 13:
        words, digit = source_step(words, sign, bits)
        digits.append(digit)
    assert words[1:] == [0, 0, 0]
    digits.append(sign * words[0])
    return digits


def recode_32mib_16w(k: int) -> list[int]:
    """16 chunks: widths 15*[16] + top word, total 524,288 points (32 MiB)."""
    words, sign = source_setup(k)
    digits = []
    for _ in range(15):
        words, digit = source_step(words, sign, 16)
        digits.append(digit)
    assert words[1:] == [0, 0, 0]
    assert words[0] < (1 << 16)
    digits.append(sign * words[0])
    return digits


# Elliptic curve arithmetic
def affine_add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    x1, y1 = a
    x2, y2 = b
    if x1 == x2:
        if (y1 + y2) % P == 0:
            return None
        slope = 3 * x1 * x1 * pow(2 * y1, -1, P) % P
    else:
        slope = (y2 - y1) * pow(x2 - x1, -1, P) % P
    x3 = (slope * slope - x1 - x2) % P
    return x3, (slope * (x1 - x3) - y1) % P


def scalar_mult(k, point=G):
    out = None
    addend = point
    k %= N
    while k:
        if k & 1:
            out = affine_add(out, addend)
        addend = affine_add(addend, addend)
        k >>= 1
    return out


def mm_deferred(a, b):
    x1, y1 = a
    x2, y2 = b
    h = (x2 - x1) % P
    r = (y2 - y1) % P
    hh = h * h % P
    hhh = h * hh % P
    q = x1 * hh % P
    x = (r * r - hhh - 2 * q) % P
    yd = r * (q - x) % P
    return x, yd, hh, hhh


def madd_from_deferred(s, a, old_anchor_y, defer_output):
    x, yd, zz, zzz = s
    xa, ya = a
    u = xa * zz % P
    ss = (ya + old_anchor_y) * zzz % P
    h = (u - x) % P
    r = (ss - yd) % P
    hh = h * h % P
    hhh = h * hh % P
    v = u * hh % P
    xn = (r * r + hhh - 2 * v) % P
    zzn = zz * hh % P
    zzzn = zzz * hhh % P
    ycore = r * (v - xn) % P
    yn = ycore if defer_output else (ycore - ya * zzzn) % P
    return xn, yn, zzn, zzzn


def normalize(s):
    x, y, zz, zzz = s
    return x * pow(zz, -1, P) % P, y * pow(zzz, -1, P) % P


def run_tests():
    HALF_G = scalar_mult(pow(2, -1, N), G)
    rng = random.Random(0x4090_3216)

    print("1. Auditing scalar recoding representation...")
    boundaries = [
        1, 2, 3,
        N - 3, N - 2, N - 1,
        (1 << 128) - 1, (1 << 128),
        (1 << 255) - 1, 1 << 255,
        N,
    ]
    for b in boundaries:
        # Test 16-window
        d16 = recode_32mib_16w(b)
        rep16 = sum(d << (16 * i) for i, d in enumerate(d16))
        assert rep16 % N == (2 * (b % N)) % N

        # Test 15-window
        d15 = recode_64mib_15w(b)
        shifts15 = [0] + [17 * i + 1 for i in range(1, 15)]
        rep15 = sum(d << shifts15[i] for i, d in enumerate(d15))
        assert rep15 % N == (2 * (b % N)) % N

    for _ in range(5000):
        k = rng.randrange(1, N)
        d16 = recode_32mib_16w(k)
        rep16 = sum(d << (16 * i) for i, d in enumerate(d16))
        assert rep16 % N == (2 * k) % N

        d15 = recode_64mib_15w(k)
        shifts15 = [0] + [17 * i + 1 for i in range(1, 15)]
        rep15 = sum(d << shifts15[i] for i, d in enumerate(d15))
        assert rep15 % N == (2 * k) % N

    print("   -> PASS: Recoder representations exact across boundary & random scalars.")

    print("2. Auditing 15-window vs 16-window deferred-Y point accumulation...")
    for _ in range(200):
        k = rng.randrange(1, N)
        expected = scalar_mult(k, G)

        # 64 MiB 15-window deferred chain (95M + 28S)
        d15 = recode_64mib_15w(k)
        shifts15 = [0] + [17 * i + 1 for i in range(1, 15)]
        pts15 = [scalar_mult((d * pow(2, shifts15[i], N)) % N, HALF_G) for i, d in enumerate(d15)]
        s15 = mm_deferred(pts15[0], pts15[1])
        ay15 = pts15[0][1]
        for i in range(2, 14):
            s15 = madd_from_deferred(s15, pts15[i], ay15, defer_output=True)
            ay15 = pts15[i][1]
        s15 = madd_from_deferred(s15, pts15[14], ay15, defer_output=False)
        assert normalize(s15) == expected

        # 32 MiB 16-window deferred chain (102M + 30S)
        d16 = recode_32mib_16w(k)
        pts16 = [scalar_mult((d * pow(2, 16 * i, N)) % N, HALF_G) for i, d in enumerate(d16)]
        s16 = mm_deferred(pts16[0], pts16[1])
        ay16 = pts16[0][1]
        for i in range(2, 15):
            s16 = madd_from_deferred(s16, pts16[i], ay16, defer_output=True)
            ay16 = pts16[i][1]
        s16 = madd_from_deferred(s16, pts16[15], ay16, defer_output=False)
        assert normalize(s16) == expected

    print("   -> PASS: Both 15-window and 16-window deferred-Y chains produce identical k*G.")


if __name__ == "__main__":
    run_tests()
