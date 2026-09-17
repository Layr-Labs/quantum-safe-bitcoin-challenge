#!/usr/bin/env python3
"""CPU audit of the split root-inversion checkpoint layout and C/Y/W/ZZZ cut."""

import random
import re
from pathlib import Path

P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
LEAVES = 256


def audit_source():
    source = Path(__file__).with_name("pinning.cu").read_text()
    for name, value in (
        ("QSB_CHECKPOINT_NODES", "254"),
        ("QSB_CHECKPOINT_STRIDE", "256"),
    ):
        match = re.search(rf"^#define\s+{name}\s+(\d+)$", source, re.MULTILINE)
        assert match and match.group(1) == value, (name, match)

    assert "#define QSB_PIPE_THREADS 128" in source
    assert "#define QSB_PIPE_CHECKPOINT_STRIDE 128" in source

    up_begin = source.index("void qsb_block_product_checkpoint(")
    up_end = source.index("void qsb_block_inverse_checkpoint(", up_begin)
    up = source[up_begin:up_end]
    assert "for(int count=N;count>1;count>>=1)" in up
    assert "if(node<2*N-2)" in up
    assert "node-N" in up
    assert "products[k][2*N-2]" in up

    down_begin = up_end
    down_end = source.index("__global__ void __launch_bounds__(256,2) qsb_root_group_prepare", down_begin)
    down = source[down_begin:down_end]
    assert "if(tid<N-2)" in down
    assert "products[k][N+tid]" in down
    assert "for(int count=2;count<N;count<<=1)" in down
    assert "inverses[k][N-2]" in down
    assert "qsb_field_normalize(value);" in down
    assert "qsb_block_product_checkpoint<QSB_PIPE_THREADS>" in source
    assert "qsb_block_product_checkpoint<QSB_ROOT_THREADS>" in source

    root_begin = source.index("__device__ __forceinline__ void qsb_block_inverse(")
    root_end = source.index("#define QSB_CHECKPOINT_NODES", root_begin)
    root = source[root_begin:root_end]
    assert root.index("qsb_field_normalize(root);") < root.index("_ModInv(root);")

    assert "_ModSqr(qx,qx);" in source
    assert "_ModMult(qx,qzz);" in source
    assert "Load256(qzz,prod);" in source
    assert "_ModMult(C, inv);" in source
    assert "qsb_xyzz_finish_precomputed(" in source
    assert source.count("kernel_pinning_pipeline<FAST_TAIL,0>") == 1
    assert source.count("kernel_pinning_pipeline<FAST_TAIL,2>") == 1
    assert "GRDSZ*4u*QSB_PIPE_CHECKPOINT_STRIDE*sizeof(uint64_t)" in source
    assert "ROOT_GRDSZ*4u*QSB_CHECKPOINT_STRIDE*sizeof(uint64_t)" in source


def checkpoint_up(raw, active, n=LEAVES):
    products = [0] * (2 * n - 1)
    products[:n] = [x % P if use and x % P else 1 for x, use in zip(raw, active)]
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
    # CUDA stores internals n..2n-3; node 2n-2 uses the compact root array.
    return products[n:2 * n - 2], products[2 * n - 2], products[:n]


def checkpoint_down(saved_internal, root, leaves):
    n = len(leaves)
    assert len(saved_internal) == n - 2
    products = list(leaves) + list(saved_internal) + [None]
    inverses = [0] * (n - 1)
    # The root kernel is the only internal normalization boundary.
    inverses[n - 2] = pow(root % P, P - 2, P)
    offset = 2 * n - 4
    count = 2
    while count < n:
        half = count // 2
        for tid in range(count):
            parent = offset + count - n + (tid & (half - 1))
            inverses[offset - n + tid] = (
                inverses[parent] * products[offset + (tid ^ half)] % P
            )
        offset -= 2 * count
        count *= 2
    assert offset == 0
    half = n // 2
    # The CUDA leaf multiply is followed by the only per-leaf normalization.
    return [
        inverses[tid & (half - 1)] * products[tid ^ half] % P
        for tid in range(n)
    ]


def finish_original(X, Y, A, B, xR, yR):
    d = (xR * A - X) % P
    inv = pow(A * A % P * d % P, P - 2, P)
    yb = yR * B % P
    h = B * inv % P
    delta = d * d % P * A % P * inv % P
    xs = (2 * xR - delta) % P
    m1 = (yb - Y) * h % P
    x1 = (m1 * m1 - xs) % P
    y1 = (m1 * (xR - x1) - yR) % P
    m2 = (yb + Y) * h % P
    x2 = (m2 * m2 - xs) % P
    s2 = (m2 * (xR - x2) - yR) % P
    return x1, x2, y1 & 1, (s2 & 1) ^ 1


def finish_precomputed(X, Y, A, B, xR, yR):
    d = (xR * A - X) % P
    W = A * A % P * d % P
    C = A * d % P * d % P
    inv = pow(W, P - 2, P)
    yb = yR * B % P
    h = B * inv % P
    delta = C * inv % P
    xs = (2 * xR - delta) % P
    m1 = (yb - Y) * h % P
    x1 = (m1 * m1 - xs) % P
    y1 = (m1 * (xR - x1) - yR) % P
    m2 = (yb + Y) * h % P
    x2 = (m2 * m2 - xs) % P
    s2 = (m2 * (xR - x2) - yR) % P
    return x1, x2, y1 & 1, (s2 & 1) ^ 1


def main():
    audit_source()
    rng = random.Random(0x455854524F4F54)
    cases = 0
    for n in (128, 256):
        for active_count in (0, 1, n // 2 - 1, n // 2, n // 2 + 1, n - 1, n):
            raw = [rng.randrange(P) for _ in range(n)]
            for i in range(0, n, 37):
                raw[i] = 0
            active = [i < active_count for i in range(n)]
            internal, root, leaves = checkpoint_up(raw, active, n)
            got = checkpoint_down(internal, root, leaves)
            want = [pow(x, P - 2, P) for x in leaves]
            assert got == want
            cases += 1
        for _ in range(100):
            raw = [rng.randrange(P) for _ in range(n)]
            active = [rng.randrange(8) != 0 for _ in range(n)]
            for i in range(n):
                if rng.randrange(64) == 0:
                    raw[i] = 0
            internal, root, leaves = checkpoint_up(raw, active, n)
            assert checkpoint_down(internal, root, leaves) == [
                pow(x, P - 2, P) for x in leaves
            ]
            cases += 1

    finish_cases = 0
    while finish_cases < 10000:
        X, Y, A, B, xR, yR = (rng.randrange(P) for _ in range(6))
        if A == 0 or (xR * A - X) % P == 0:
            continue
        assert finish_precomputed(X, Y, A, B, xR, yR) == finish_original(
            X, Y, A, B, xR, yR
        )
        finish_cases += 1

    print(f"PASS: {cases} split trees and {finish_cases} C/W finish comparisons")


if __name__ == "__main__":
    main()
