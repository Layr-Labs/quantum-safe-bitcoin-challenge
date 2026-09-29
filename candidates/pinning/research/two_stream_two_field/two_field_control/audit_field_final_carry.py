#!/usr/bin/env python3
"""Independent integer audit of the sparse secp256k1 double-fold reduction."""

import random


M = 1 << 256
C = (1 << 32) + 977
P = M - C


def double_fold(product: int) -> tuple[int, int]:
    """Return the low 256 bits and final carry used by GPUMath's schedule."""
    first = product % M + (product // M) * C
    second = first % M + (first // M) * C
    return second % M, second // M


def corrected(product: int) -> tuple[int, int]:
    low, carry = double_fold(product)
    value = low + carry * C
    assert value < M  # carry implies low <= C*C-1, so no fourth fold is needed.
    return value, carry


def check(a: int, b: int) -> tuple[bool, int]:
    product = a * b
    old, carry = double_fold(product)
    fixed, same_carry = corrected(product)
    assert carry == same_carry <= 1
    assert fixed % P == product % P
    return old % P == product % P, carry


def main() -> None:
    # For a,b<M, first<M*(C+1), its high part is at most C, and therefore
    # second<=M-1+C*C<M+2^66.  The last carry is at most one.  If it is one,
    # second-M<=C*C-1 and adding C remains far below M.
    assert C * C + C - 1 < M

    old_ok, carry = check(P - 65537, P - 65537)
    old, _ = double_fold((P - 65537) ** 2)
    fixed, _ = corrected((P - 65537) ** 2)
    assert not old_ok and carry == 1
    assert old == 0x1FC30
    assert fixed == 0x100020001

    old_failures = 0
    carries = 0
    # Dense near-modulus coverage finds the correlated edge that uniform random
    # sampling almost never reaches.
    deltas = [0, 1, 2, 3, 7, 31, 255, 256, 977, 65535, 65536, 65537,
              65538, (1 << 20) - 1, 1 << 20, C - 1, C, C + 1]
    values = [0, 1, 2, P - 1, P, M - 1] + [P - d for d in deltas if d <= P]
    for a in values:
        for b in values:
            ok, final_carry = check(a, b)
            old_failures += not ok
            carries += final_carry

    rng = random.Random(0xF1A1CA77)
    for _ in range(200_000):
        ok, final_carry = check(rng.randrange(M), rng.randrange(M))
        old_failures += not ok
        carries += final_carry

    assert old_failures > 0 and carries == old_failures
    print(
        "PASS: final-carry correction; "
        f"targeted+random cases={len(values) ** 2 + 200_000}, "
        f"old mismatches={old_failures}, final carries={carries}"
    )


if __name__ == "__main__":
    main()
