#!/usr/bin/env python3
"""Audit the default 2k/A/2 path and the gated single-k/A experiment.

The inherited direct-recode audits intentionally keep their hardcoded 2k
invariant. This file parameterizes that invariant for both values of
QSB_SINGLE_K_FULLBASE and checks that the scalar setup, direct digit extraction,
table-base scalar, table address geometry, and table-builder exceptional cases
stay coherent under the single numeric gate.
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
TOTAL = 1 << 20
NRI = 0x4A1D1E55AA17E4C8E6D7B31C0F123456789ABCDEFFEDCBA98765432100112233 % N


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


def setup(k: int, single_k: bool) -> tuple[list[int], int]:
    kw, nw = limbs(k), limbs(N)
    diff, borrow = sub4(kw, nw)
    reduced = diff if borrow == 0 else kw
    if single_k:
        value = reduced
    else:
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


def materialized(k: int, single_k: bool) -> list[int]:
    words, sign = setup(k, single_k)
    out = []
    for bits in [18] + [17] * 13:
        words, digit = mixed_step(words, sign, bits)
        out.append(digit)
    assert words[1:] == [0, 0, 0]
    out.append(sign * words[0])
    return out


def digit_idx(digit: int) -> tuple[int, int]:
    return ((abs(digit) - 1) >> 1, int(digit < 0))


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


def direct(k: int, single_k: bool) -> list[tuple[int, int, int]]:
    words, sign = setup(k, single_k)
    return [direct_digit(words, sign, c) for c in range(CHUNKS)]


def reconstruct(digits: list[int]) -> int:
    acc = 0
    for chunk, digit in enumerate(digits):
        acc = (acc + digit * (1 << shift_for(chunk))) % N
    return acc


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


def base_scalar(single_k: bool) -> int:
    return NRI if single_k else (NRI * ((N + 1) // 2)) % N


def target_scalar(k: int, single_k: bool) -> int:
    return (k % N) if single_k else (2 * (k % N)) % N


def table_scalar(chunk: int, index: int, single_k: bool) -> int:
    return ((2 * index + 1) * pow(2, shift_for(chunk), N) * base_scalar(single_k)) % N


def add_case(cases: set[int], value: int) -> None:
    if 0 <= value <= MASK256:
        cases.add(value)


def scalar_cases() -> set[int]:
    out = {
        0,
        1,
        2,
        3,
        N // 2 - 1,
        N // 2,
        N // 2 + 1,
        (N + 1) // 2 - 1,
        (N + 1) // 2,
        (N + 1) // 2 + 1,
        N - 2,
        N - 1,
        N,
        N + 1,
        N + 2,
        MASK256 - 2,
        MASK256 - 1,
        MASK256,
    }
    for value in (2 * N - 3, 2 * N - 2, 2 * N - 1):
        add_case(out, value)
    for pivot in (N // 2, (N + 1) // 2, N, 2 * N - 1, MASK256):
        for delta in range(-96, 97):
            add_case(out, pivot + delta)
    for bit in range(256):
        for delta in range(-4, 5):
            add_case(out, (1 << bit) + delta)
    for chunk in range(CHUNKS):
        width = 18 if chunk == 0 else 17
        for bit in (shift_for(chunk), shift_for(chunk) + width):
            if bit < 256:
                for delta in range(-7, 8):
                    add_case(out, (1 << bit) + delta)
    rng = random.Random(0x25251E7A)
    out.update(rng.getrandbits(256) for _ in range(80_000))
    return out


def audit_source() -> None:
    source = (Path(__file__).resolve().parent / "pinning.cu").read_text()
    required = [
        "#define QSB_SINGLE_K_FULLBASE 1",
        "#if QSB_SINGLE_K_FULLBASE != 0 && QSB_SINGLE_K_FULLBASE != 1",
        "#if QSB_SINGLE_K_FULLBASE",
        "BN_copy(bscal, nri);",
        "BN_copy(half_nri, nri);",
        "BN_mod_mul(bscal, inv2, nri, order, ctx);",
        "BN_mod_mul(half_nri, inv2, nri, order, ctx);",
        "gt_direct_digit(M,sflag,(unsigned)gt_shift(0)+1u,18u,false,&idx,&neg);",
        "gt_direct_digit(M,sflag,(unsigned)gt_shift(GT_CHUNKS-1)+1u,17u,true,&idx,&neg);",
        "#define GT_TOTAL_ENTRIES (1u << 20)",
        "QSB_SPARSE_TAIL 1",
        "QSB_FINAL_TEMPLATE 1",
        "QSB_SPARSE_D 1",
    ]
    for token in required:
        assert token in source, token
    assert source.count("#define QSB_SINGLE_K_FULLBASE 1") == 1


def audit_table_geometry() -> None:
    copied_identity = 0
    for thread in range(TOTAL):
        chunk = 0 if thread < (1 << 17) else 1 + ((thread - (1 << 17)) >> 16)
        index = thread - offset_for(chunk)
        entries = entries_for(chunk)
        assert 0 <= chunk < CHUNKS
        assert 0 <= index < entries
        assert offset_for(chunk) + index == thread
        multiplier = 2 * index + 1
        hi, lo = multiplier >> 8, multiplier & 255
        assert lo & 1
        assert hi < (1024 if chunk == 0 else 512)
        assert multiplier < (1 << 18)
        assert multiplier % N != 0
        if hi == 0:
            copied_identity += 1
        else:
            assert (hi * 256 - lo) % N != 0
            assert (hi * 256 + lo) % N != 0
        for single_k in (False, True):
            assert table_scalar(chunk, index, single_k) == (
                multiplier * pow(2, shift_for(chunk), N) * base_scalar(single_k)
            ) % N
    assert copied_identity == CHUNKS * 128
    assert offset_for(0) == 0
    assert offset_for(1) == 1 << 17
    table_base = offset_for(2)
    for chunk in range(2, CHUNKS):
        assert table_base == offset_for(chunk)
        assert table_base + entries_for(chunk) - 1 < TOTAL
        table_base += 1 << 16
    assert table_base == TOTAL


def audit_scalar(k: int, single_k: bool, ec: bool = False) -> None:
    words, sign = setup(k, single_k)
    assert from_limbs(words) & 1
    if single_k:
        reduced = k % N
        if reduced & 1:
            assert sign == 1 and from_limbs(words) == reduced
        else:
            assert sign == -1 and from_limbs(words) == N - reduced
    digits = materialized(k, single_k)
    direct_digits = direct(k, single_k)
    assert [d for _, _, d in direct_digits] == digits
    for chunk, (idx, neg, digit) in enumerate(direct_digits):
        assert digit and (digit & 1)
        assert digit_idx(digit) == (idx, neg)
        assert idx < entries_for(chunk)
        if chunk != CHUNKS - 1:
            assert abs(digit) < (1 << (18 if chunk == 0 else 17))
    recon = reconstruct(digits)
    assert recon == target_scalar(k, single_k)
    effective = (recon * base_scalar(single_k)) % N
    assert effective == ((k % N) * NRI) % N
    if ec:
        assert scalar_mult(effective) == scalar_mult((k % N) * NRI)


def main() -> None:
    audit_source()
    audit_table_geometry()
    cases = scalar_cases()
    for single_k in (False, True):
        for scalar in cases:
            audit_scalar(scalar, single_k)
        # EC checks are intentionally sampled; scalar/digit checks above cover the
        # full boundary/adversarial/random set for both gate values.
        ec_cases = [
            0,
            1,
            2,
            N // 2,
            (N + 1) // 2,
            N - 1,
            N,
            N + 1,
            MASK256,
        ]
        rng = random.Random(0xA2525 if single_k else 0x2525A)
        ec_cases.extend(rng.getrandbits(256) for _ in range(96))
        for scalar in ec_cases:
            audit_scalar(scalar, single_k, ec=True)
    zero_words, zero_sign = setup(0, True)
    assert zero_sign == -1 and from_limbs(zero_words) == N
    assert scalar_mult(reconstruct(materialized(0, True)) * base_scalar(True)) is None
    print(
        "PASS single-k/fullbase audit: both gates; "
        f"{len(cases)} scalar cases each; {TOTAL} table slots; "
        "raw 256-bit boundaries, signed reconstruction, EC equivalence, "
        "and H/L exceptional builder cases covered"
    )


if __name__ == "__main__":
    main()
