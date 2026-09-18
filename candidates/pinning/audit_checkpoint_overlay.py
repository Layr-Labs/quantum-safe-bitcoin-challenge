#!/usr/bin/env python3
"""Audit the hybrid warp-register/checkpoint-overlay inverse tree."""

from pathlib import Path
import random

P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F


def mul(a, b):
    return a * b % P


def full_up(values):
    n = len(values)
    products = [0] * (2 * n - 1)
    products[:n] = values
    offset = 0
    count = n
    while count > 1:
        half = count // 2
        for tid in range(half):
            products[offset + count + tid] = mul(
                products[offset + tid], products[offset + half + tid]
            )
        offset += count
        count //= 2
    return products


def full_down(products, root_inverse):
    n = (len(products) + 1) // 2
    inverses = [0] * (n - 1)
    inverses[n - 2] = root_inverse
    offset = 2 * n - 4
    count = 2
    while count < n:
        half = count // 2
        for tid in range(count):
            parent = offset + count - n + (tid & (half - 1))
            inverses[offset - n + tid] = mul(
                inverses[parent], products[offset + (tid ^ half)]
            )
        offset -= 2 * count
        count *= 2
    return [
        mul(inverses[tid & (n // 2 - 1)], products[tid ^ (n // 2)])
        for tid in range(n)
    ]


def checkpoint_up(values):
    n = len(values)
    products = [0] * (2 * n)
    checkpoint = [None] * n
    products[:n] = values
    offset = 0
    count = n
    cur = [0] * 32
    while count > 32:
        half = count // 2
        for tid in range(half):
            node = offset + count + tid
            out = mul(products[offset + tid], products[offset + half + tid])
            products[node] = out
            if node < 2 * n - 2:
                checkpoint[node - n] = out
            if count == 64:
                cur[tid] = out
        offset += count
        count //= 2

    while count > 1:
        half = count // 2
        new = cur[:]
        for tid in range(half):
            node = offset + count + tid
            out = mul(cur[tid], cur[tid ^ half])
            new[tid] = out
            if node < 2 * n - 2:
                checkpoint[node - n] = out
        cur = new
        offset += count
        count //= 2

    assert all(x is not None for x in checkpoint[: n - 2])
    return products, checkpoint, cur[0]


def product_at(n, products, checkpoint, node):
    if node < n:
        return products[node]
    return checkpoint[node - n]


def checkpoint_down_overlay(products, checkpoint, root_inverse):
    n = (len(products)) // 2
    cur = [0] * 32
    cur[0] = root_inverse

    offset = 2 * n - 4
    count = 2
    while count <= 32:
        half = count // 2
        new = cur[:]
        level_products = [
            product_at(n, products, checkpoint, offset + tid)
            for tid in range(count)
        ]
        for tid in range(count):
            parent = cur[tid & (half - 1)]
            sibling = level_products[tid ^ half]
            new[tid] = mul(parent, sibling)
        cur = new
        offset -= 2 * count
        count *= 2

    for tid in range(32):
        products[2 * n - 64 + tid] = cur[tid]

    count = 64
    offset = 2 * n - 128
    while count < n:
        half = count // 2
        loads = []
        for tid in range(count):
            parent = products[offset + count + (tid & (half - 1))]
            sibling = products[offset + (tid ^ half)]
            loads.append((parent, sibling))
        for tid, (parent, sibling) in enumerate(loads):
            products[offset + tid] = mul(parent, sibling)
        offset -= 2 * count
        count *= 2

    return [
        mul(products[n + (tid & (n // 2 - 1))], products[tid ^ (n // 2)])
        for tid in range(n)
    ]


def audit_case(raw, active):
    values = [x % P if use and x % P else 1 for x, use in zip(raw, active)]
    full_products = full_up(values)
    want = full_down(full_products, pow(full_products[-1], P - 2, P))
    products, checkpoint, root = checkpoint_up(values)
    assert root == full_products[-1]
    got = checkpoint_down_overlay(products, checkpoint, pow(root, P - 2, P))
    assert got == want
    assert all(mul(x, y) == 1 for x, y in zip(values, got))


def audit_source():
    source = Path(__file__).with_name("pinning.cu").read_text()
    product = source[source.index("void qsb_block_product_checkpoint("):source.index("template<int N>\n__device__ __forceinline__ void qsb_block_product_checkpoint(\n    uint64_t *value, uint64_t *roots, uint64_t *checkpoint\n)", source.index("void qsb_block_product_checkpoint("))]
    finish = source[source.index("void qsb_block_inverse_checkpoint("):source.index("/* Batch the per-search-CTA roots")]
    assert "for(int count=N;count>32;count>>=1)" in product
    assert "if(count==64)" in product
    assert "if(count>64)__syncthreads();" in product
    assert "qsb_shfl_xor_u64(cur[k],half)" in product
    assert "__syncwarp" not in product
    assert "if(tid<N-64)" in finish
    assert "__shared__ uint64_t inverses" not in finish
    assert "products[k][2*N-64+tid]=cur_inv[k]" in finish
    assert "for(int count=64;count<N;count<<=1)" in finish
    assert "parent_inv[k]=products[k][offset+count+local_parent]" in finish
    assert "parent_inv[k]=products[k][N+(tid&(N/2-1))]" in finish
    assert "__syncwarp" not in finish
    assert "Lower overlay levels are count>=64" in finish


def main():
    audit_source()
    rng = random.Random(0x6531515342)
    cases = 0
    for n in (64, 128, 256):
        edge = [0, 1, 2, P - 2, P - 1]
        raw = [edge[i % len(edge)] for i in range(n)]
        audit_case(raw, [True] * n)
        cases += 1
        for active_count in (0, 1, 31, 32, 33, n // 2 - 1, n // 2, n // 2 + 1, n - 1, n):
            raw = [rng.randrange(P) for _ in range(n)]
            for i in range(0, n, 37):
                raw[i] = 0
            audit_case(raw, [i < active_count for i in range(n)])
            cases += 1
        for _ in range(120):
            raw = [rng.randrange(P) for _ in range(n)]
            active = [rng.randrange(8) != 0 for _ in range(n)]
            for i in range(n):
                if rng.randrange(64) == 0:
                    raw[i] = 0
            audit_case(raw, active)
            cases += 1
    print(f"PASS: {cases} hybrid checkpoint-overlay trees across N=64,128,256")


if __name__ == "__main__":
    main()
