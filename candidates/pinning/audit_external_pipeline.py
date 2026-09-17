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

    assert "for(int count=256;count>1;count>>=1)" in source
    assert "for(int count=2;count<256;count<<=1)" in source
    assert "products[k][510]" in source or "products[k][2*N-2]" in source
    assert "if(tid<QSB_CHECKPOINT_NODES)" in source or "if(tid<N-2)" in source
    assert "qsb_field_normalize(value);" in source

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
    assert "GRDSZ*4u*QSB_CHECKPOINT_STRIDE*sizeof(uint64_t)" in source


def checkpoint_up(raw, active):
    products = [0] * 511
    products[:LEAVES] = [x % P if use and x % P else 1 for x, use in zip(raw, active)]
    offset = 0
    for count in (256, 128, 64, 32, 16, 8, 4, 2):
        half = count // 2
        for tid in range(half):
            products[offset + count + tid] = (
                products[offset + tid] * products[offset + half + tid] % P
            )
        offset += count
    assert offset == 510
    # CUDA stores exactly nodes 256..509; node 510 uses the compact root array.
    return products[256:510], products[510], products[:256]


def checkpoint_down(saved_internal, root, leaves):
    assert len(saved_internal) == 254
    products = list(leaves) + list(saved_internal) + [None]
    inverses = [0] * 255
    # The root kernel is the only internal normalization boundary.
    inverses[254] = pow(root % P, P - 2, P)
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
    # The CUDA leaf multiply is followed by the only per-leaf normalization.
    return [
        inverses[tid & 127] * products[tid ^ 128] % P
        for tid in range(LEAVES)
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
    for active_count in (0, 1, 31, 32, 33, 127, 128, 129, 255, 256):
        raw = [rng.randrange(P) for _ in range(LEAVES)]
        for i in range(0, LEAVES, 37):
            raw[i] = 0
        active = [i < active_count for i in range(LEAVES)]
        internal, root, leaves = checkpoint_up(raw, active)
        got = checkpoint_down(internal, root, leaves)
        want = [pow(x, P - 2, P) for x in leaves]
        assert got == want
        cases += 1
    for _ in range(200):
        raw = [rng.randrange(P) for _ in range(LEAVES)]
        active = [rng.randrange(8) != 0 for _ in range(LEAVES)]
        for i in range(LEAVES):
            if rng.randrange(64) == 0:
                raw[i] = 0
        internal, root, leaves = checkpoint_up(raw, active)
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
