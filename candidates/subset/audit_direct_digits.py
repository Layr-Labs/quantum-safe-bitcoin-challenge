"""Audit: direct bit-field digit extraction == streamed Joye-Tunstall recoder.

Reference mirrors pinning.cu: gt_recode_setup, gt_mixed_step<18>, 13x
gt_mixed_step<17>, last digit sign*M[0], then gt_digit_idx. Candidate reads
each digit's index/sign from fixed bit fields of the setup value M.
"""
import random

N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
MASK64 = (1 << 64) - 1


def recode_setup(k):
    if k >= N:
        k -= N
    t = 2 * k
    m = t - N if t >= N else t
    if m & 1:
        return m, 1
    return N - m, -1


def mixed_step(M, bits):
    digit = (M & ((1 << (bits + 1)) - 1)) - (1 << bits)
    M = ((M >> (bits + 1)) << 1) | 1
    return M, digit


def reference(k):
    M, sign = recode_setup(k)
    out = []
    M, d = mixed_step(M, 18)
    out.append(sign * d)
    for _ in range(13):
        M, d = mixed_step(M, 17)
        out.append(sign * d)
    out.append(sign * (M & MASK64))  # production reads M[0] as int32
    res = []
    for e in out:
        e32 = ((e + (1 << 31)) % (1 << 32)) - (1 << 31)  # int32 wrap like the kernel
        ae = abs(e32) & 0xFFFFFFFF
        res.append((((ae - 1) & 0xFFFFFFFF) >> 1, 1 if e32 < 0 else 0))
    return res


def field(M, pos):
    limbs = [(M >> (64 * i)) & MASK64 for i in range(4)] + [0]
    li, sh = pos >> 6, pos & 63
    v = (limbs[li] >> sh) | ((((limbs[li + 1] << 1) & MASK64) << (63 - sh)) & MASK64)
    return v & 0xFFFFFFFF


def direct(k):
    M, sign = recode_setup(k)
    sflag = 1 if sign < 0 else 0
    res = []
    f = field(M, 1) & ((1 << 18) - 1)
    t = f >> 17
    res.append(((f ^ ((t - 1) & 0xFFFFFFFF)) & ((1 << 17) - 1), (t ^ 1) ^ sflag))
    for c in range(1, 15):
        pos = 17 * c + 2
        f = field(M, pos) & 0x1FFFF
        t = f >> 16
        last = c == 14
        idx_reg = (f ^ ((t - 1) & 0xFFFFFFFF)) & 0xFFFF
        idx = (f & 0xFFFF) if last else idx_reg
        neg = (0 if last else (t ^ 1)) ^ sflag
        res.append((idx, neg))
    return res


def main():
    cases = [0, 1, 2, N - 1, N, N + 1, (1 << 256) - 1, N // 2, N // 2 + 1, (N + 1) // 2]
    for s in range(0, 256, 3):
        cases += [1 << s, (1 << s) - 1, (N - (1 << s)) % (1 << 256)]
    rnd = random.Random(20260917)
    cases += [rnd.getrandbits(256) for _ in range(200000)]
    bad = 0
    idx_max = [0] * 15
    for k in cases:
        k &= (1 << 256) - 1
        r, d = reference(k), direct(k)
        if r != d:
            bad += 1
            if bad < 4:
                print("MISMATCH", hex(k), r, d)
        for c, (i, _) in enumerate(d):
            idx_max[c] = max(idx_max[c], i)
    print(f"direct digit audit: {len(cases)} scalars, {bad} mismatches")
    print("max idx per chunk:", idx_max)
    assert bad == 0
    assert idx_max[0] < (1 << 17) and all(m < (1 << 16) for m in idx_max[1:])


main()
