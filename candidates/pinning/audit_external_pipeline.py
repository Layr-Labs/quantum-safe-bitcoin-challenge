#!/usr/bin/env python3
"""CPU audit of the 128-leaf split tree and three-field Y/V/W recovery cut."""

import random
import re
from pathlib import Path

P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
LEAVES = 128


def audit_source():
    source = Path(__file__).with_name("pinning.cu").read_text()
    for name, value in (
        ("QSB_TREE_N", "128"),
        ("QSB_CHECKPOINT_NODES", "254"),
        ("QSB_CHECKPOINT_STRIDE", "256"),
    ):
        match = re.search(rf"^#define\s+{name}\s+(\d+)\b", source, re.MULTILINE)
        assert match and match.group(1) == value, (name, match)

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

    root_begin = source.index("__device__ __forceinline__ void qsb_block_inverse(")
    root_end = source.index("#define QSB_CHECKPOINT_NODES", root_begin)
    root = source[root_begin:root_end]
    assert root.index("qsb_field_normalize(root);") < root.index("_ModInv(root);")

    assert "_ModSqr(qx,qx);" not in source
    assert "_ModMult(qx,qzz);" not in source
    assert "Load256(qzz,prod);" in source
    assert "_ModMult(C, inv);" not in source
    assert "qsb_xyzz_finish_symmetric(" in source
    assert "pin_u2rk_words" in source
    prepare = source.split("void qsb_xyzz_finish_prepare(", 1)[1].split(
        "uint32_t qsb_xyzz_finish_symmetric(", 1
    )[0]
    finish = source.split("uint32_t qsb_xyzz_finish_symmetric(", 1)[1].split(
        "__device__ __constant__ uint32_t pin_tail_words", 1
    )[0]
    assert prepare.count("_ModMult(") == 2 and prepare.count("_ModSqr(") == 1
    assert finish.count("_ModMult(") == 8 and finish.count("_ModSqr(") == 1
    assert "qsb_field_normalize(u);" in finish
    assert "_ModNeg256(h, u);" in finish
    assert "bool usable = active && ((prod[0] | prod[1] | prod[2] | prod[3]) != 0);" in source
    assert "prod[0]=1; prod[1]=prod[2]=prod[3]=prod[4]=0;" in source
    assert "if (!usable) return;" in source
    assert source.count(
        "qsb_block_product_checkpoint<QSB_TREE_N>(prod,roots,tree,qsb_prepare_scratch());"
    ) == 1
    assert source.count(
        "qsb_block_inverse_checkpoint<QSB_TREE_N>(prod,roots,tree);"
    ) == 1
    assert "uint64_t sx0=ri ? q2x[0] : q1x[0];" in source
    assert "pb[0]=__byte_perm(x7,0x2+(uint8_t)((y_parities>>ri)&1u),0x4321);" in source
    assert "pb[1]=__byte_perm(x7,x6,0x0765);" in source
    assert "pb[8]=__byte_perm(x0,0x80,0x0456);" in source
    assert source.count("kernel_pinning_pipeline<FAST_TAIL,0>") == 1
    assert source.count("kernel_pinning_pipeline<FAST_TAIL,2>") == 1
    assert "GRDSZ*4u*QSB_CAND_STRIDE*sizeof(uint64_t)" in source


def checkpoint_up(raw, active):
    products = [0] * (2 * LEAVES - 1)
    products[:LEAVES] = [x % P if use and x % P else 1 for x, use in zip(raw, active)]
    offset = 0
    count = LEAVES
    while count > 1:
        half = count // 2
        for tid in range(half):
            products[offset + count + tid] = (
                products[offset + tid] * products[offset + half + tid] % P
            )
        offset += count
        count //= 2
    assert offset == 2 * LEAVES - 2
    # CUDA stores nodes N..2N-3; node 2N-2 uses the compact root array.
    return products[LEAVES : 2 * LEAVES - 2], products[-1], products[:LEAVES]


def checkpoint_down(saved_internal, root, leaves):
    assert len(saved_internal) == LEAVES - 2
    products = list(leaves) + list(saved_internal) + [None]
    inverses = [0] * (LEAVES - 1)
    # The root kernel is the only internal normalization boundary.
    inverses[LEAVES - 2] = pow(root % P, P - 2, P)
    offset = 2 * LEAVES - 4
    count = 2
    while count < LEAVES:
        half = count // 2
        for tid in range(count):
            parent = offset + count - LEAVES + (tid & (half - 1))
            inverses[offset - LEAVES + tid] = (
                inverses[parent] * products[offset + (tid ^ half)] % P
            )
        offset -= 2 * count
        count *= 2
    assert offset == 0
    # The CUDA leaf multiply is followed by the only per-leaf normalization.
    return [
        inverses[tid & (LEAVES // 2 - 1)] * products[tid ^ (LEAVES // 2)] % P
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
    return (x1, y1), (x2, (-s2) % P)


def finish_symmetric(X, Y, A, B, xR, yR):
    d = (xR * A - X) % P
    W = A * A % P * d % P
    if not W:
        return None
    inv = pow(W, P - 2, P)
    h = B * inv % P
    t = B * h % P
    u = yR * t % P
    v = Y * h % P
    K = 3 * xR * xR % P
    F = (2 * u * u - K * t + xR) % P
    H = 2 * u * v % P
    x_plus = (F - H) % P
    x_minus = (F + H) % P
    y_plus = ((u - v) * (xR - x_plus) - yR) % P
    y_minus = (yR - (u + v) * (xR - x_minus)) % P
    return (x_plus, y_plus), (x_minus, y_minus)


def affine_add(left, right):
    if left is None:
        return right
    if right is None:
        return left
    x1, y1 = left
    x2, y2 = right
    if x1 == x2:
        if (y1 + y2) % P == 0:
            return None
        slope = 3 * x1 * x1 * pow(2 * y1, P - 2, P) % P
    else:
        slope = (y2 - y1) * pow((x2 - x1) % P, P - 2, P) % P
    x3 = (slope * slope - x1 - x2) % P
    return x3, (slope * (x1 - x3) - y1) % P


def encode(point):
    x, y = point
    return bytes([2 | (y & 1)]) + x.to_bytes(32, "big")


def encode_kernel(point):
    """Emulate the production __byte_perm word construction through pb[8]."""
    x, y = point
    words = [(x >> (32 * i)) & 0xFFFFFFFF for i in range(8)]

    def byte_perm(a, b, selector):
        source = a.to_bytes(4, "little") + b.to_bytes(4, "little")
        return bytes(source[(selector >> (4 * i)) & 7] for i in range(4))

    pb = [byte_perm(words[7], 2 | (y & 1), 0x4321)]
    pb.extend(byte_perm(words[i], words[i - 1], 0x0765) for i in range(7, 0, -1))
    pb.append(byte_perm(words[0], 0x80, 0x0456))
    return b"".join(word[::-1] for word in pb)[:33]


def random_point(rng):
    while True:
        x = rng.randrange(P)
        y2 = (x * x * x + 7) % P
        if pow(y2, (P - 1) // 2, P) == 1:
            y = pow(y2, (P + 1) // 4, P)
            return x, y if rng.randrange(2) else (-y) % P


def check_recovery(point, R, z):
    x, y = point
    a, b = R
    A = z * z % P
    B = A * z % P
    X = x * A % P
    Y = y * B % P
    got = finish_symmetric(X, Y, A, B, a, b)
    if z == 0 or x == a:
        assert got is None
        return
    baseline = finish_original(X, Y, A, B, a, b)
    expected = affine_add(point, R), affine_add(point, (a, -b % P))
    assert all(q is not None for q in expected)
    assert got == baseline == expected, (point, R, z, got, baseline, expected)
    assert tuple(map(encode, got)) == tuple(map(encode, expected))
    assert tuple(map(encode_kernel, got)) == tuple(map(encode, expected))


def main():
    audit_source()
    rng = random.Random(0x455854524F4F54)
    cases = 0
    for active_count in (0, 1, 31, 32, 33, 63, 64, 65, 127, 128):
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

    R = (
        0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
        0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
    )
    recovery_cases = 0
    for _ in range(1000):
        point = random_point(rng)
        for z in (1, 2, P - 1, rng.randrange(1, P)):
            check_recovery(point, R, z)
            recovery_cases += 1

    # Force u=yR/(xR-xP) into the upper field half, including values close to
    # p, to bind the signed-square guard to valid curve points.
    for delta in (65537, 1 << 20, 1 << 32):
        u = P - delta
        x = (R[0] - R[1] * pow(u, P - 2, P)) % P
        y2 = (x * x * x + 7) % P
        assert pow(y2, (P - 1) // 2, P) == 1
        y = pow(y2, (P + 1) // 4, P)
        for point in ((x, y), (x, -y % P)):
            for z in (1, 2, P - 1):
                check_recovery(point, R, z)
                recovery_cases += 1

    for point in (R, (R[0], -R[1] % P)):
        for z in (1, 2, P - 1):
            check_recovery(point, R, z)
    check_recovery(random_point(rng), R, 0)

    print(
        f"PASS: {cases} 128-leaf split trees; {recovery_cases} on-curve "
        "baseline/affine/compressed recovery comparisons; 7 singular skips; "
        "10M+2S recovery"
    )


if __name__ == "__main__":
    main()
