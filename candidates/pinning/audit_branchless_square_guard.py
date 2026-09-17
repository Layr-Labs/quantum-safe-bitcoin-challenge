#!/usr/bin/env python3
"""Audit the branchless signed representative used before recovery square."""

import random
from pathlib import Path


M = 1 << 256
C = (1 << 32) + 977
P = M - C


def dropped_carry_mult(left: int, right: int) -> int:
    product = left * right
    first = product % M + (product // M) * C
    second = first % M + (first // M) * C
    return second % M


def dropped_carry_square(value: int) -> int:
    return dropped_carry_mult(value, value)


def branch_rep(value: int) -> int:
    value %= P
    return P - value if value >> 255 else value


def mask_rep(value: int) -> int:
    value %= P
    neg = (P - value) % P
    mask = MASK256 if value >> 255 else 0
    return (neg & mask) | (value & (MASK256 ^ mask))


MASK256 = M - 1


def random_point(rng):
    while True:
        x = rng.randrange(P)
        rhs = (x * x * x + 7) % P
        if pow(rhs, (P - 1) // 2, P) == 1:
            y = pow(rhs, (P + 1) // 4, P)
            return x, y if rng.randrange(2) else (-y) % P


def recovery_u(point, recovery):
    x, _ = point
    a, b = recovery
    if x == a:
        return None
    return b * pow((a - x) % P, P - 2, P) % P


def audit_source():
    source = Path(__file__).with_name("pinning.cu").read_text()
    begin = source.index("uint32_t qsb_xyzz_finish_symmetric(")
    end = source.index("__device__ __constant__ uint32_t pin_tail_words", begin)
    finish = source[begin:end]
    assert "if(u[3] >> 63)" not in finish
    assert "bool square_high = (u[3] >> 63) != 0;" in finish
    assert "_ModNeg256(h, u);" in finish
    assert "h[limb]=square_high ? h[limb] : u[limb];" in finish
    assert finish.count("_ModSqr(f, h);") == 1
    assert "_ModMult(f, u, u)" not in finish


def main():
    audit_source()
    rng = random.Random(0xB12A6C4)

    values = [
        0, 1, 2, (1 << 255) - 1, 1 << 255, (1 << 255) + 1,
        P // 2 - 1, P // 2, P // 2 + 1, P - 1,
    ]
    values += [P - delta for delta in (2, 3, 7, 31, 977, 65537, 1 << 20, 1 << 32)]
    values += [rng.randrange(P) for _ in range(200_000)]

    high = 0
    for value in values:
        direct = branch_rep(value)
        selected = mask_rep(value)
        assert selected == direct
        assert dropped_carry_square(selected) % P == value * value % P
        high += value >> 255

    # Canonicalizing alone does not repair the inherited dropped carry. The
    # generic multiplication path uses the same double-fold convention, so
    # replacing the dedicated square with multiplication is not a repair.
    unsafe = [P - 65537, P - (1 << 20), P - (1 << 32)]
    for value in unsafe:
        assert 0 <= value < P
        assert dropped_carry_square(value) % P != value * value % P
        assert dropped_carry_mult(value, value) % P != value * value % P

    recovery = (
        0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
        0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
    )
    curve_cases = 0
    curve_high = 0
    for _ in range(10_000):
        u = recovery_u(random_point(rng), recovery)
        assert u is not None
        selected = mask_rep(u)
        assert selected == branch_rep(u)
        assert dropped_carry_square(selected) % P == u * u % P
        curve_high += u >> 255
        curve_cases += 1

    print(
        "PASS: branchless signed-square representative; "
        f"{len(values):,} boundary/random field values; {curve_cases:,} curve values; "
        f"high-half ratios field={high/len(values):.4f}, curve={curve_high/curve_cases:.4f}; "
        "canonical-only and generic-multiply alternatives rejected on targeted edges"
    )


if __name__ == "__main__":
    main()
