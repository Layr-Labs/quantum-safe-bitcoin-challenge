#!/usr/bin/env python3
"""Independent congruence audit for fused XYZZ / recovery field helpers."""
import random
from pathlib import Path

M = 1 << 256
C = (1 << 32) + 977
P = M - C
MASK64 = (1 << 64) - 1


def to_limbs(x):
    return [(x >> (64 * i)) & MASK64 for i in range(4)]


def from_limbs(L):
    return sum(L[i] << (64 * i) for i in range(4))


def fold_negative(r, k):
    """r -= k*C, then if that wraps, subtract C again (≡ +p)."""
    if k == 0:
        return from_limbs(r)
    sub = k * C
    br = 0
    for i in range(4):
        diff = r[i] - (sub if i == 0 else 0) - br
        if diff < 0:
            r[i] = diff + (1 << 64)
            br = 1
        else:
            r[i] = diff
            br = 0
    if br:
        br2 = 0
        for i in range(4):
            diff = r[i] - (C if i == 0 else 0) - br2
            if diff < 0:
                r[i] = diff + (1 << 64)
                br2 = 1
            else:
                r[i] = diff
                br2 = 0
    return from_limbs(r)


def fold_positive(r, carry):
    """r += carry*C, then if that wraps, add C again."""
    s = r[0] + carry * C
    r[0] = s & MASK64
    cy = s >> 64
    for i in range(1, 4):
        s = r[i] + cy
        r[i] = s & MASK64
        cy = s >> 64
    if cy:
        s = r[0] + C
        r[0] = s & MASK64
        cy = s >> 64
        for i in range(1, 4):
            s = r[i] + cy
            r[i] = s & MASK64
            cy = s >> 64
    return from_limbs(r)


def limb_add_chain(terms):
    """terms is a list of (sign, limbs) with sign in {+1,-1}; plus running carry."""
    r = [0] * 4
    carry = 0
    for i in range(4):
        t = carry
        for sign, L in terms:
            t += sign * L[i]
        carry = t // (1 << 64)
        r[i] = t - carry * (1 << 64)
    return r, carry


def fused_apbcc(a, b, c):
    A, B, Cc = to_limbs(a), to_limbs(b), to_limbs(c)
    r, carry = limb_add_chain([(1, A), (1, B), (-1, Cc), (-1, Cc)])
    if carry > 0:
        return fold_positive(r, carry)
    return fold_negative(r, -carry)


def fused_abcc(a, b, c):
    A, B, Cc = to_limbs(a), to_limbs(b), to_limbs(c)
    r, carry = limb_add_chain([(1, A), (-1, B), (-1, Cc), (-1, Cc)])
    return fold_negative(r, -carry)


def fused_dblsub(a, b):
    A, B = to_limbs(a), to_limbs(b)
    r, carry = limb_add_chain([(1, A), (1, A), (-1, B)])
    if carry > 0:
        return fold_positive(r, carry)
    return fold_negative(r, -carry)


def main():
    src_math = Path(__file__).with_name('GPUMath.h').read_text()
    src_cu = Path(__file__).with_name('pinning.cu').read_text()
    assert '_ModAddSub256_abcc(T, T, PPP, Q)' in src_math
    assert '_ModSub256_abcc(T, T, ZZZ3, Q)' in src_math
    assert '_ModDblSub256(W, xR, C)' in src_cu
    assert src_math.count('_ModSub256(T, T, Q)') == 0
    assert '_ModAdd256(W, xR, xR)' not in src_cu

    rng = random.Random(0x33F15ED)
    N = 200_000
    for _ in range(N):
        a, b, c = (rng.randrange(M) for _ in range(3))
        r1 = fused_apbcc(a, b, c)
        r2 = fused_abcc(a, b, c)
        r3 = fused_dblsub(a, b)
        assert 0 <= r1 < M and 0 <= r2 < M and 0 <= r3 < M
        assert (r1 - (a + b - 2 * c)) % P == 0
        assert (r2 - (a - b - 2 * c)) % P == 0
        assert (r3 - (2 * a - b)) % P == 0

    edges = [0, 1, 2, P - 2, P - 1, P, P + 1, M - 2, M - 1, C - 1, C, C + 1, 2 * C, 3 * C]
    for a in edges:
        for b in edges:
            for c in edges:
                assert (fused_apbcc(a, b, c) - (a + b - 2 * c)) % P == 0
                assert (fused_abcc(a, b, c) - (a - b - 2 * c)) % P == 0
                assert (fused_dblsub(a, b) - (2 * a - b)) % P == 0

    print(
        f'PASS: fused field helpers; random={N}, edges={len(edges)**3}, '
        'source bindings ok'
    )


if __name__ == '__main__':
    main()
