#!/usr/bin/env python3
"""CPU audit of the split root-inversion checkpoint and three-field recovery."""

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

    up_begin = source.index("void qsb_block_product_checkpoint(")
    up_end = source.index("void qsb_block_inverse_checkpoint(", up_begin)
    up = source[up_begin:up_end]
    assert "for(int count=N;count>1;count>>=1)" in up
    assert "if(node<2*N-2)" in up
    assert "node-N" in up
    assert "products[k][2*N-2]" in up

    down_begin = up_end
    down_end = source.index(
        "__global__ void __launch_bounds__(256,2) qsb_root_group_prepare",
        down_begin,
    )
    down = source[down_begin:down_end]
    assert "if(tid<N-2)" in down
    assert "products[k][N+tid]" in down
    assert "for(int count=2;count<N;count<<=1)" in down
    assert "inverses[k][N-2]" in down
    assert "qsb_field_normalize(value);" in down

    root_begin = source.index("__device__ __forceinline__ void qsb_block_inverse(")
    root_end = source.index("#define QSB_CHECKPOINT_NODES", root_begin)
    root = source[root_begin:root_end]
    assert root.index("qsb_field_normalize(root);") < root.index("_ModInv(root);")

    prep_begin = source.index(
        "__device__ __forceinline__ void qsb_xyzz_finish_prepare("
    )
    finish_begin = source.index(
        "__device__ __forceinline__ uint32_t qsb_xyzz_finish_precomputed("
    )
    finish_end = source.index("/* The canonical pinning tail", finish_begin)
    prep = source[prep_begin:finish_begin]
    finish = source[finish_begin:finish_end]
    assert "_ModSqr(W, ZZ);" in prep
    assert "_ModMult(W, X_D);" in prep
    assert "_ModSqr(qx,qx);" not in source
    assert "_ModMult(C, inv);" not in finish
    assert prep.count("_ModMult(") == 2
    assert prep.count("_ModSqr(") == 1
    assert finish.count("_ModMult(") == 8
    assert finish.count("_ModSqr(") == 2
    assert "pin_k_words" in source
    assert "Load256(qzz,prod);" in source
    assert "qsb_xyzz_finish_precomputed(" in source
    assert source.count("kernel_pinning_pipeline<FAST_TAIL,0>") == 1
    assert source.count("kernel_pinning_pipeline<FAST_TAIL,2>") == 1
    assert "GRDSZ*4u*QSB_CHECKPOINT_STRIDE*sizeof(uint64_t)" in source
    assert "BATCH*6u*sizeof(ulonglong2)" in source
    assert "qsb_block_product_checkpoint<QSB_TREE_N>" in source
    assert "_FixedBaseSignedXYZZScalar(qx,qy,qzz,qzzz,z,d_gt,qsb_prepare_scratch());" in source


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
    return products[256:510], products[510], products[:256]


def checkpoint_down(saved_internal, root, leaves):
    assert len(saved_internal) == 254
    products = list(leaves) + list(saved_internal) + [None]
    inverses = [0] * 255
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
    return [
        inverses[tid & 127] * products[tid ^ 128] % P
        for tid in range(LEAVES)
    ]


def inv(x):
    return pow(x % P, P - 2, P)


def on_curve_y(x):
    return pow((pow(x, 3, P) + 7) % P, (P + 1) // 4, P)


def is_on_curve(x, y):
    return pow(y, 2, P) == (pow(x, 3, P) + 7) % P


def affine_add(x1, y1, x2, y2):
    lam = ((y2 - y1) * inv(x2 - x1)) % P
    x3 = (lam * lam - x1 - x2) % P
    y3 = (lam * (x1 - x3) - y1) % P
    return x3, y3


def finish_symmetric(X, Y, U, V, a, b):
    d = (a * U - X) % P
    W = (U * U % P) * d % P
    I = inv(W)
    K = (3 * a * a) % P
    h = V * I % P
    t = V * h % P
    u = b * t % P
    v = Y * h % P
    F = (2 * u * u - K * t + a) % P
    H = (2 * u * v) % P
    x_plus = (F - H) % P
    x_minus = (F + H) % P
    y_plus = ((u - v) * (a - x_plus) - b) % P
    y_minus = (b - (u + v) * (a - x_minus)) % P
    return x_plus, x_minus, y_plus, y_minus, W


def compressed_prefix(x, y_odd):
    x0 = x & 0xFFFFFFFF
    x1 = (x >> 32) & 0xFFFFFFFF
    x2 = (x >> 64) & 0xFFFFFFFF
    x3 = (x >> 96) & 0xFFFFFFFF
    x4 = (x >> 128) & 0xFFFFFFFF
    x5 = (x >> 160) & 0xFFFFFFFF
    x6 = (x >> 192) & 0xFFFFFFFF
    x7 = (x >> 224) & 0xFFFFFFFF
    return (0x2 + int(y_odd), x7, x6, x5, x4, x3, x2, x1, x0)


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
    singular = 0
    rng = random.Random(0x3F13D5)
    while finish_cases < 4018:
        xP = rng.randrange(P)
        yP = on_curve_y(xP)
        if not is_on_curve(xP, yP):
            yP = (-yP) % P
            if not is_on_curve(xP, yP):
                continue
        xR = rng.randrange(P)
        yR = on_curve_y(xR)
        if not is_on_curve(xR, yR):
            yR = (-yR) % P
            if not is_on_curve(xR, yR):
                continue
        if xP == xR:
            sc = rng.randrange(1, P)
            U = pow(sc, 2, P)
            V = pow(sc, 3, P)
            X = xP * U % P
            Y = yP * V % P
            d = (xR * U - X) % P
            assert d == 0
            W = (U * U % P) * d % P
            assert W == 0
            singular += 1
            continue
        sc = rng.randrange(1, P)
        U = pow(sc, 2, P)
        V = pow(sc, 3, P)
        X = xP * U % P
        Y = yP * V % P
        xp, xm, yp, ym, W = finish_symmetric(X, Y, U, V, xR, yR)
        assert W != 0
        ap, aq = affine_add(xP, yP, xR, yR)
        am, an = affine_add(xP, yP, xR, (-yR) % P)
        assert (xp, yp) == (ap, aq)
        assert (xm, ym) == (am, an)
        assert (yp & 1) == (aq & 1)
        assert (ym & 1) == (an & 1)
        assert compressed_prefix(xp, yp & 1)[0] in (2, 3)
        finish_cases += 1

    while singular < 7:
        xP = rng.randrange(P)
        yP = on_curve_y(xP)
        if not is_on_curve(xP, yP):
            yP = (-yP) % P
            if not is_on_curve(xP, yP):
                continue
        xR = xP
        yR = yP
        sc = rng.randrange(1, P)
        U = pow(sc, 2, P)
        V = pow(sc, 3, P)
        X = xP * U % P
        Y = yP * V % P
        d = (xR * U - X) % P
        W = (U * U % P) * d % P
        assert W == 0
        singular += 1

    print(
        f"PASS: {cases} split trees and {finish_cases} on-curve "
        f"symmetric/affine/compressed-encoding cases, {singular} singular skips"
    )


if __name__ == "__main__":
    main()
