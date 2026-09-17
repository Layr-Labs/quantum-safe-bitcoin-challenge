#!/usr/bin/env python3
"""CPU audit of qsb_block_inverse's packed product tree at N=256 and N=128."""

import random

P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F


def counts_down(n):
    c = 2
    out = []
    while c < n:
        out.append(c)
        c *= 2
    return out


def tree_inverse(values):
    n = len(values)
    products = [0] * (2 * n)
    inverses = [0] * n
    products[:n] = values
    offset = 0
    count = n
    while count > 1:
        half = count // 2
        for tid in range(half):
            products[offset + count + tid] = (
                products[offset + tid] * products[offset + half + tid] % P
            )
        offset += count
        count //= 2
    assert offset == 2 * n - 2
    inverses[n - 2] = pow(products[2 * n - 2], P - 2, P)
    offset = 2 * n - 4
    for count in counts_down(n):
        half = count // 2
        for tid in range(count):
            parent = offset + count - n + (tid & (half - 1))
            inverses[offset - n + tid] = (
                inverses[parent] * products[offset + (tid ^ half)] % P
            )
        offset -= 2 * count
    assert offset == 0
    half = n // 2
    return [
        inverses[tid & (half - 1)] * products[tid ^ half] % P
        for tid in range(n)
    ]


def audit_case(raw, active):
    # This mirrors the production caller: zero or inactive denominators become 1.
    factors = [x if use and x != 0 else 1 for x, use in zip(raw, active)]
    got = tree_inverse(factors)
    want = [pow(x, P - 2, P) for x in factors]
    assert got == want
    assert all(x * y % P == 1 for x, y in zip(factors, got))


def audit_width(leaves, rng):
    edge = [0, 1, 2, P - 2, P - 1]
    raw = [edge[i % len(edge)] for i in range(leaves)]
    audit_case(raw, [True] * leaves)
    samples = [0, 1, leaves // 2 - 1, leaves // 2, leaves // 2 + 1, leaves - 1, leaves]
    for active_count in samples:
        raw = [rng.randrange(P) for _ in range(leaves)]
        for i in range(0, leaves, 37):
            raw[i] = 0
        audit_case(raw, [i < active_count for i in range(leaves)])
    for _ in range(100):
        raw = [rng.randrange(P) for _ in range(leaves)]
        active = [rng.randrange(8) != 0 for _ in range(leaves)]
        for i in range(leaves):
            if rng.randrange(64) == 0:
                raw[i] = 0
        audit_case(raw, active)


def main():
    rng = random.Random(0x515342)
    audit_width(256, rng)
    audit_width(128, rng)
    print("PASS: product-tree cases at N=256 and N=128, including zero and inactive identities")


if __name__ == "__main__":
    main()
