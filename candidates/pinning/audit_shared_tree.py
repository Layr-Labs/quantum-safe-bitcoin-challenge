#!/usr/bin/env python3
"""CPU audit of qsb_block_inverse's 256-leaf shared product tree."""

import random

P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
LEAVES = 256


def tree_inverse(values):
    products = [0] * (2 * LEAVES)
    inverses = [0] * LEAVES
    products[:LEAVES] = values
    offset = 0
    for count in (256, 128, 64, 32, 16, 8, 4, 2):
        half = count // 2
        for tid in range(half):
            products[offset + count + tid] = (
                products[offset + tid] * products[offset + half + tid] % P
            )
        offset += count
    assert offset == 510
    inverses[254] = pow(products[510], P - 2, P)
    offset = 508
    for count in (2, 4, 8, 16, 32, 64, 128):
        half = count // 2
        for tid in range(count):
            parent = offset + count - 256 + (tid & (half - 1))
            inverses[offset - 256 + tid] = (
                inverses[parent] * products[offset + (tid ^ half)] % P
            )
        offset -= 2 * count
    assert offset == 0
    return [
        inverses[tid & 127] * products[tid ^ 128] % P
        for tid in range(LEAVES)
    ]


def audit_case(raw, active):
    # This mirrors the production caller: zero or inactive denominators become 1.
    factors = [x if use and x != 0 else 1 for x, use in zip(raw, active)]
    got = tree_inverse(factors)
    want = [pow(x, P - 2, P) for x in factors]
    assert got == want
    assert all(x * y % P == 1 for x, y in zip(factors, got))


def main():
    rng = random.Random(0x515342)
    edge = [0, 1, 2, P - 2, P - 1]
    raw = [edge[i % len(edge)] for i in range(LEAVES)]
    audit_case(raw, [True] * LEAVES)
    for active_count in (0, 1, 31, 32, 33, 127, 128, 129, 255, 256):
        raw = [rng.randrange(P) for _ in range(LEAVES)]
        for i in range(0, LEAVES, 37):
            raw[i] = 0
        audit_case(raw, [i < active_count for i in range(LEAVES)])
    for _ in range(200):
        raw = [rng.randrange(P) for _ in range(LEAVES)]
        active = [rng.randrange(8) != 0 for _ in range(LEAVES)]
        for i in range(LEAVES):
            if rng.randrange(64) == 0:
                raw[i] = 0
        audit_case(raw, active)
    print("PASS: 211 product-tree cases, including zero and inactive identities")


if __name__ == "__main__":
    main()

