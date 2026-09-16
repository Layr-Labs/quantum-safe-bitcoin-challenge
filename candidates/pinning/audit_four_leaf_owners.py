#!/usr/bin/env python3
"""Audit packed-tree 2/4-leaf owner fusion against the exact 256-lane path."""

import hashlib
import random
import re
from pathlib import Path

P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
LEAVES = 256


def build_products(values):
    products = [0] * 511
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
    return products


def packed_leaf_sets():
    nodes = [frozenset((i,)) for i in range(LEAVES)] + [None] * 255
    offset = 0
    for count in (256, 128, 64, 32, 16, 8, 4, 2):
        half = count // 2
        for tid in range(half):
            nodes[offset + count + tid] = (
                nodes[offset + tid] | nodes[offset + half + tid]
            )
        offset += count
    return nodes


def expand_to_four_leaf_groups(products):
    inverses = [0] * 255
    inverses[254] = pow(products[510], P - 2, P)
    offset = 508
    for count in (2, 4, 8, 16, 32, 64):
        half = count // 2
        for tid in range(count):
            parent = offset + count - 256 + (tid & (half - 1))
            inverses[offset - 256 + tid] = (
                inverses[parent] * products[offset + (tid ^ half)] % P
            )
        offset -= 2 * count
    assert offset == 256
    return inverses


def original_leaves(products):
    inverses = expand_to_four_leaf_groups(products)
    offset = 256
    count = 128
    half = 64
    for tid in range(count):
        parent = offset + count - 256 + (tid & (half - 1))
        inverses[offset - 256 + tid] = (
            inverses[parent] * products[offset + (tid ^ half)] % P
        )
    return [
        inverses[tid & 127] * products[tid ^ 128] % P
        for tid in range(LEAVES)
    ]


def four_leaf_owners(products):
    inverses = expand_to_four_leaf_groups(products)
    out = [None] * LEAVES
    for owner in range(64):
        group_inv = inverses[128 + owner]
        left_pair = products[256 + owner]
        right_pair = products[320 + owner]
        left_pair_inv = group_inv * right_pair % P
        right_pair_inv = group_inv * left_pair % P
        for slot in range(4):
            leaf = owner + 64 * slot
            pair_inv = right_pair_inv if slot & 1 else left_pair_inv
            out[leaf] = pair_inv * products[leaf ^ 128] % P
    return out


def two_leaf_owners(products):
    inverses = expand_to_four_leaf_groups(products)
    out = [None] * LEAVES
    for owner in range(128):
        group = owner & 63
        side = owner >> 6
        other_pair = products[256 + group + ((side ^ 1) << 6)]
        owned_pair_inv = inverses[128 + group] * other_pair % P
        for slot in range(2):
            leaf = group + (side << 6) + (slot << 7)
            out[leaf] = owned_pair_inv * products[leaf ^ 128] % P
    return out


def finish_precomputed(C, Y, W, B, inv, x_r, y_r):
    yb = y_r * B % P
    h = B * inv % P
    delta = C * inv % P
    xs = (2 * x_r - delta) % P
    m1 = (yb - Y) * h % P
    x1 = (m1 * m1 - xs) % P
    y1 = (m1 * (x_r - x1) - y_r) % P
    m2 = (yb + Y) * h % P
    x2 = (m2 * m2 - xs) % P
    s2 = (m2 * (x_r - x2) - y_r) % P
    return x1, x2, y1 & 1, (s2 & 1) ^ 1


def leading_zero_bits(digest):
    return 256 - int.from_bytes(digest, "big").bit_length()


def candidate_hits(candidate, inverse, difficulty, double_hash):
    C, Y, W, B, x_r, y_r = candidate
    x1, x2, p1, p2 = finish_precomputed(C, Y, W, B, inverse, x_r, y_r)
    hits = []
    for recid, (x, parity) in enumerate(((x1, p1), (x2, p2))):
        first = hashlib.sha256(bytes((2 + parity,)) + x.to_bytes(32, "big")).digest()
        if leading_zero_bits(first) >= difficulty:
            hits.append((recid, 0, first))
            continue
        if double_hash:
            second = hashlib.sha256(first).digest()
            if leading_zero_bits(second) >= difficulty:
                hits.append((recid, 1, second))
    return hits


def audit_source():
    source = Path(__file__).with_name("pinning.cu").read_text()
    assert "#define QSB_FINISH_LEAVES_PER_OWNER 2" in source
    assert "for(int count=2;count<128;count<<=1)" in source
    assert "finish_inverses[limb][128+group]" in source
    assert "finish_products[limb][256+group]" in source
    assert "finish_products[limb][320+group]" in source
    assert "leaf_tid^128" in source
    assert "group+(owner_pair<<6)+(owner_slot<<7)" in source
    assert re.search(r"if\(owner>=64\) return;", source)
    assert re.search(r"if\(owner>=128\) return;", source)


def audit_mapping():
    nodes = packed_leaf_sets()
    assert nodes[510] == frozenset(range(256))
    for owner in range(64):
        expected = frozenset((owner, owner + 64, owner + 128, owner + 192))
        assert nodes[384 + owner] == expected
        assert nodes[256 + owner] == frozenset((owner, owner + 128))
        assert nodes[320 + owner] == frozenset((owner + 64, owner + 192))


def audit_tree_cases():
    rng = random.Random(0x344C454146)
    active_counts = (0, 1, 2, 31, 32, 33, 63, 64, 65, 127, 128,
                     129, 191, 192, 193, 255, 256)
    cases = []
    for active_count in active_counts:
        raw = [rng.randrange(P) for _ in range(LEAVES)]
        for i in range(0, LEAVES, 37):
            raw[i] = 0
        cases.append((raw, [i < active_count for i in range(LEAVES)]))
    for _ in range(100):
        raw = [rng.randrange(P) for _ in range(LEAVES)]
        active = [rng.randrange(8) != 0 for _ in range(LEAVES)]
        for i in range(LEAVES):
            if rng.randrange(64) == 0:
                raw[i] = 0
        cases.append((raw, active))

    for raw, active in cases:
        factors = [x if use and x else 1 for x, use in zip(raw, active)]
        products = build_products(factors)
        want = original_leaves(products)
        assert four_leaf_owners(products) == want
        assert two_leaf_owners(products) == want
        assert all(a * b % P == 1 for a, b in zip(factors, want))
    return len(cases)


def audit_hits():
    rng = random.Random(0x48495453494D)
    compared = 0
    for active_count in (0, 1, 63, 64, 65, 127, 128, 129, 255, 256):
        candidates = []
        factors = []
        for i in range(LEAVES):
            A, X, Y, B, x_r, y_r = (rng.randrange(P) for _ in range(6))
            d = (x_r * A - X) % P
            W = A * A % P * d % P
            C = A * d % P * d % P
            usable = i < active_count and W != 0
            candidates.append((C, Y, W, B, x_r, y_r))
            factors.append(W if usable else 1)
        products = build_products(factors)
        original = original_leaves(products)
        fused4 = four_leaf_owners(products)
        fused2 = two_leaf_owners(products)
        for difficulty in (0, 4, 8, 12, 24):
            for double_hash in (False, True):
                for i in range(active_count):
                    if candidates[i][2] == 0:
                        continue
                    want = candidate_hits(candidates[i], original[i], difficulty, double_hash)
                    assert candidate_hits(candidates[i], fused4[i], difficulty, double_hash) == want
                    assert candidate_hits(candidates[i], fused2[i], difficulty, double_hash) == want
                    compared += 1
    return compared


def main():
    audit_source()
    audit_mapping()
    tree_cases = audit_tree_cases()
    hit_cases = audit_hits()
    print(
        "PASS: packed node map; "
        f"{tree_cases} inverse blocks; {hit_cases} bit-identical hit simulations; "
        "2-leaf and 4-leaf owner variants"
    )


if __name__ == "__main__":
    main()
