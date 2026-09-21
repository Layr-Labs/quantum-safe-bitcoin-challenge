#!/usr/bin/env python3
"""Audit the uniform 16-chunk / 32 MiB subset table geometry.

Models tree.cu exactly: gt_recode_setup (subset's raw-SHA-scalar form),
gt_mixed_step, the ZLAB_DIRDIG gt_direct_digit extraction at
pos=gt_shift(c)+1 width gt_width(c), the flat table map, and the host ladder
construction, then checks a full reconstruction equals k*A on real secp256k1.

Host-only. No CUDA compilation, no device execution, no throughput claim.
SPDX-License-Identifier: GPL-3.0-only
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "harness"))
import crypto as C

N = C.N; M64 = (1 << 64) - 1; M256 = (1 << 256) - 1
CH, ENT, TOTAL = 16, 1 << 15, 1 << 19
GT_LO = GT_HI = 256
gt_shift = lambda c: 16 * c
gt_offset = lambda c: c << 15
gt_width = lambda c: (gt_shift(c) - gt_shift(c - 1)) if c == CH - 1 else (gt_shift(c + 1) - gt_shift(c))


def recode_setup(k):
    """tree.cu gt_recode_setup."""
    if k >= N:                              # exact: k>=n requires top limb == n3
        k -= N
    t = (k << 1) & M256; tc = k >> 255
    d = t - N
    ge = 1 if (tc or d >= 0) else 0
    m = (d & M256) if ge else t
    odd = m & 1
    return (m if odd else (N - m) & M256), (1 if odd else -1)


def mixed_step(M, sign, bits):
    digit = (M & ((1 << (bits + 1)) - 1)) - (1 << bits)
    M = ((M >> (bits + 1)) << 1) | 1
    return M, sign * digit


def recode_signed(k):
    M, sign = recode_setup(k); out = []
    for _ in range(CH - 1):
        M, d = mixed_step(M, sign, 16); out.append(d)
    out.append(sign * M)
    return out


def direct_digit(M, sign, c):
    """tree.cu gt_direct_digit, transcribed exactly:
         f = bits(M,pos) & ((1<<w)-1);  t = f >> (w-1)
         idx = last ? (f & mask) : ((f ^ (t-1)) & mask)
         neg = (last ? 0 : (t ^ 1)) ^ sflag
    """
    w = gt_width(c); pos = gt_shift(c) + 1
    last = (c == CH - 1)
    mask = (1 << (w - 1)) - 1
    f = (M >> pos) & ((1 << w) - 1)
    t = f >> (w - 1)
    idx = (f & mask) if last else ((f ^ ((t - 1) & 0xFFFFFFFF)) & mask)
    sflag = 1 if sign < 0 else 0
    neg = ((0 if last else (t ^ 1)) ^ sflag) & 1
    return idx, bool(neg)


def main():
    assert CH * ENT == TOTAL and TOTAL * 64 == 32 * 1024 * 1024

    for t in range(TOTAL):                      # exhaustive flat map
        c = t >> 15; d = t - gt_offset(c)
        assert 0 <= c < CH and 0 <= d < ENT and gt_offset(c) + d == t
        m = 2 * d + 1
        assert 0 <= (m & 255) < GT_LO and 0 <= (m >> 8) < GT_HI

    base = gt_offset(2)                          # rolled flat-base recurrence
    for c in range(2, CH):
        assert base == gt_offset(c); base += 1 << 15
    assert base == TOTAL

    cases = {1, 2, 3, N - 2, N - 1, N, N + 1, M256 - 1, M256}
    for b in range(256):
        for dd in (-2, -1, 0, 1, 2):
            v = (1 << b) + dd
            if 0 <= v <= M256: cases.add(v)
    for pv in (N, N // 2, (N + 1) // 2):
        for dd in range(-64, 65):
            if 0 <= pv + dd <= M256: cases.add(pv + dd)
    rng = random.Random(4242)
    for _ in range(20000): cases.add(rng.randrange(0, M256 + 1))

    for k in cases:                              # recode + DIRDIG agreement
        digits = recode_signed(k)
        M, sign = recode_setup(k)
        rep = 0
        for c, e in enumerate(digits):
            assert e and e & 1 and abs(e) < (1 << 16)
            d = (abs(e) - 1) >> 1
            assert 0 <= d < ENT
            di, dn = direct_digit(M, sign, c)
            assert di == d and dn == (e < 0), f"DIRDIG mismatch chunk {c}"
            rep += e << gt_shift(c)
        assert rep % N == (2 * (k % N)) % N
    print(f"PASS: {TOTAL} table slots; {len(cases)} scalars; "
          f"gt_direct_digit agrees with gt_recode_signed on every chunk")

    rng2 = random.Random(7)                      # end-to-end on real secp256k1
    nchk = 0
    for _ in range(3):
        A = C.point_mul(rng2.randrange(1, N), C.G)
        ah = C.point_mul(pow(2, N - 2, N), A)
        bases = [ah]
        for _ in range(1, CH): bases.append(C.point_mul(1 << 16, bases[-1]))
        for _ in range(3):
            k = rng2.randrange(1, N); acc = None
            for c, e in enumerate(recode_signed(k)):
                m = 2 * ((abs(e) - 1) >> 1) + 1
                hi, lo = m >> 8, m & 255
                hp = None if hi == 0 else C.point_mul(hi * 256, bases[c])
                lp = None if lo == 0 else C.point_mul(lo, bases[c])
                p = lp if hp is None else hp if lp is None else C.point_add(hp, lp)
                if e < 0: p = (p[0], (C.P - p[1]) % C.P)
                acc = p if acc is None else C.point_add(acc, p)
            assert acc == C.point_mul(k % N, A), "reconstruction != k*A"
            nchk += 1
    print(f"PASS: {nchk} end-to-end reconstructions equal k*A")


if __name__ == "__main__":
    main()
