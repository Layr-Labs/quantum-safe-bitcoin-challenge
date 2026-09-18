#!/usr/bin/env python3
"""Audit the direct fixed-base table recoder against the streamed source path.

This is a selected-track port of the independent lane217 extracted-C++ oracle:
it mirrors gt_recode_setup, gt_mixed_step, gt_field_bits_v, and gt_direct_digit,
then checks table indices/signs at every 15-window boundary, including raw
scalars above the secp256k1 group order.
"""

from __future__ import annotations

from pathlib import Path
import random


N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
G = (
    0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
    0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
)
MASK64 = (1 << 64) - 1
MASK256 = (1 << 256) - 1
CHUNKS = 15


def limbs(x: int) -> list[int]:
    return [(x >> (64 * i)) & MASK64 for i in range(4)]


def from_limbs(words: list[int]) -> int:
    return sum(w << (64 * i) for i, w in enumerate(words))


def sub4(a: list[int], b: list[int]) -> tuple[list[int], int]:
    out, borrow = [], 0
    for x, y in zip(a, b):
        value = x - y - borrow
        out.append(value & MASK64)
        borrow = int(value < 0)
    return out, borrow


def setup(k: int) -> tuple[list[int], int]:
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


def shift_for(chunk: int) -> int:
    return 0 if chunk == 0 else 17 * chunk + 1


def offset_for(chunk: int) -> int:
    return 0 if chunk == 0 else (chunk + 1) << 16


def entries_for(chunk: int) -> int:
    return 1 << (17 if chunk == 0 else 16)


def mixed_step(words: list[int], sign: int, bits: int) -> tuple[list[int], int]:
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


def digit_idx(digit: int) -> tuple[int, int]:
    return ((abs(digit) - 1) >> 1, int(digit < 0))


def streamed(k: int) -> list[tuple[int, int, int, int]]:
    words, sign = setup(k)
    out = []
    for c in range(CHUNKS):
        if c == 0:
            words, digit = mixed_step(words, sign, 18)
        elif c < CHUNKS - 1:
            words, digit = mixed_step(words, sign, 17)
        else:
            digit = sign * from_limbs(words)
        idx, neg = digit_idx(digit)
        out.append((offset_for(c), idx, neg, digit))
    return out


def direct_digit(words: list[int], sign: int, chunk: int) -> tuple[int, int, int]:
    width = 18 if chunk == 0 else 17
    f = (from_limbs(words) >> (shift_for(chunk) + 1)) & ((1 << width) - 1)
    t = f >> (width - 1)
    low_mask = (1 << (width - 1)) - 1
    if chunk == CHUNKS - 1:
        idx = f & low_mask
        neg = 0
        digit = (f << 1) | 1
    else:
        idx = (f ^ (t - 1)) & low_mask
        neg = t ^ 1
        digit = (f << 1) + 1 - (1 << width)
    if sign < 0:
        neg ^= 1
        digit = -digit
    return idx, neg, digit


def direct(k: int) -> list[tuple[int, int, int, int]]:
    words, sign = setup(k)
    return [
        (offset_for(c), *direct_digit(words, sign, c))
        for c in range(CHUNKS)
    ]


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
        lam = 3 * x1 * x1 * pow(2 * y1, -1, P) % P
    else:
        lam = (y2 - y1) * pow(x2 - x1, -1, P) % P
    x3 = (lam * lam - x1 - x2) % P
    return x3, (lam * (x1 - x3) - y1) % P


def scalar_mult(k: int, point=G):
    out = None
    addend = point
    k %= N
    while k:
        if k & 1:
            out = affine_add(out, addend)
        addend = affine_add(addend, addend)
        k >>= 1
    return out


def reconstruct(digits: list[int]) -> int:
    acc = 0
    for c, digit in enumerate(digits):
        acc = (acc + digit * (1 << shift_for(c))) % N
    return acc


def add_case(cases: set[int], value: int) -> None:
    if 0 <= value <= MASK256:
        cases.add(value)


def source_audit() -> None:
    source = (Path(__file__).resolve().parent / "pinning.cu").read_text()
    required = [
        "#define QSB_DIRECT_RECODE 1",
        "__device__ __forceinline__ uint32_t gt_field_bits_v",
        "__device__ __forceinline__ void gt_direct_digit",
        "gt_direct_digit(M,sflag,(unsigned)gt_shift(0)+1u,18u,false,&idx,&neg);",
        "gt_direct_digit(M,sflag,(unsigned)gt_shift(1)+1u,17u,false,&idx,&neg);",
        "gt_direct_digit(M,sflag,(unsigned)gt_shift(c)+1u,17u,false,&idx,&neg);",
        "gt_direct_digit(M,sflag,(unsigned)gt_shift(GT_CHUNKS-1)+1u,17u,true,&idx,&neg);",
        "#elif QSB_PREFETCH == 1",
        "#elif QSB_S0_SHM",
        "#elif QSB_FINAL_TEMPLATE",
        "_PointAddXYZZT<false>(X,Y,ZZ,ZZZ, cx,cy, y0);",
        "#define QSB_SPARSE_TAIL 1",
        "#define QSB_FINAL_TEMPLATE 1",
        "#define QSB_SPARSE_D 1",
        "#define QSB_CONST_MIDSTATE 1",
    ]
    for token in required:
        assert token in source, token


def audit_scalar(k: int, ec: bool = False) -> None:
    old = streamed(k)
    new = direct(k)
    assert [(b, i, n) for b, i, n, _ in old] == [(b, i, n) for b, i, n, _ in new], (k, old, new)
    digits = [d for _, _, _, d in new]
    for c, (base, idx, neg, digit) in enumerate(new):
        assert base == offset_for(c)
        assert base + idx < (1 << 20)
        assert idx < entries_for(c)
        assert digit & 1
        if c != CHUNKS - 1:
            assert abs(digit) < (1 << (18 if c == 0 else 17))
        assert digit_idx(digit) == (idx, neg)
    assert reconstruct(digits) == (2 * (k % N)) % N
    if ec:
        inv2 = (N + 1) // 2
        assert scalar_mult((reconstruct(digits) * inv2) % N) == scalar_mult(k)


def cases() -> set[int]:
    out = {0, 1, 2, 3, N - 2, N - 1, N, N + 1, N + 2, MASK256 - 1, MASK256}
    pivots = [N // 2, (N + 1) // 2, N, 2 * N - 1, MASK256]
    for pivot in pivots:
        for delta in range(-64, 65):
            add_case(out, pivot + delta)
    for bit in range(256):
        pivot = 1 << bit
        for delta in range(-3, 4):
            add_case(out, pivot + delta)
    for c in range(CHUNKS):
        width = 18 if c == 0 else 17
        for bit in (shift_for(c), shift_for(c) + width):
            if bit < 256:
                for delta in range(-5, 6):
                    add_case(out, (1 << bit) + delta)
    rng = random.Random(0x217D1A3C7)
    for _ in range(75000):
        out.add(rng.getrandbits(256))
    return out


def main() -> None:
    source_audit()
    scalars = cases()
    for scalar in scalars:
        audit_scalar(scalar)
    rng = random.Random(0x217EC)
    ec_cases = sorted(scalars)[:32] + [rng.getrandbits(256) for _ in range(96)]
    for scalar in ec_cases:
        audit_scalar(scalar, ec=True)
    print(
        "PASS direct recode audit: "
        f"{len(scalars)} scalar/index cases; {len(ec_cases)} sampled EC equalities; "
        "all 15 window boundaries and raw scalars >= order covered"
    )


if __name__ == "__main__":
    main()
