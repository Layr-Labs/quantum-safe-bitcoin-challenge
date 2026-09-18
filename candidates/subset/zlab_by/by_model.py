#!/usr/bin/env python3
"""Bit-exact Python model of the C routine that will go into zinv32.cuh:
32-bit f,g registers, int32 (u,v,q,r) matrix, 7 table blocks + 2 branchless
single steps per 30-step batch, zi_row_ip on nine 32-bit signed limbs.
Validated against pow(x, p-2, p)."""
import random, sys
sys.path.insert(0, '/root/qsb/exp/sub-by/zlab_by')
from by_table import tab, clamp, block4, divstep   # noqa

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF
MM32 = 0xD2253531
MASK30 = 0x3FFFFFFF
PL = [0xFFFFFC2F,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0]
P = 2**256 - 2**32 - 977

def s32(x):
    x &= M32
    return x - (1 << 32) if x >> 31 else x
def s64(x):
    x &= M64
    return x - (1 << 64) if x >> 63 else x
def u32(x): return x & M32

# ------------------------------------------------- packed table (as shipped)
def pack(entry):
    (a,b,c,d), C, s, _ = entry
    for v in (a,b,c,d): assert -32 <= v <= 31
    assert -8 <= C <= 7
    return ((a & 63) | ((b & 63) << 6) | ((c & 63) << 12) | ((d & 63) << 18)
            | ((C & 15) << 24) | (s << 31))
LUT = [pack(t) for t in tab]

# --------------------------------------------------------- decision routine
def divstep30_by(delta, f, g, trace=None):
    """f,g uint32.  Returns (delta', u,v,q,r) exactly as the C code will."""
    u, v, q, r = 1, 0, 0, 1
    for _ in range(7):
        dc = clamp(delta)
        e = LUT[((dc + 4) << 7) + ((f & 14) << 3) + (g & 15)]
        a = s32((e << 26) & M32) >> 26
        b = s32((e << 20) & M32) >> 26
        c = s32((e << 14) & M32) >> 26
        d = s32((e <<  8) & M32) >> 26
        C = s32((e <<  4) & M32) >> 28
        sm = s32(e) >> 31                      # 0 or -1
        nf = u32(u32(a) * f + u32(b) * g) >> 4
        ng = u32(u32(c) * f + u32(d) * g) >> 4
        f, g = nf, ng
        nu, nv = s32(a*u + b*q), s32(a*v + b*r)
        q, r = s32(c*u + d*q), s32(c*v + d*r)
        u, v = nu, nv
        delta = s32(((delta ^ sm) - sm) + C)
        if trace is not None: trace.append(delta)
    for _ in range(2):
        mg = -(g & 1)                                   # 0 or -1
        sw = mg & (-1 if delta > 0 else 0)
        t = s32((s32(f) & mg))
        t = s32((t ^ sw) - sw)
        nf = u32(f ^ ((f ^ g) & u32(sw)))
        ng = u32(u32(g) + u32(t)) >> 1
        tu = s32((s32((u & mg) if False else s32(u) & mg) ^ sw) - sw)
        tv = s32((s32(s32(v) & mg) ^ sw) - sw)
        nu = s32(2 * s32((sw & (q ^ u)) ^ u))
        nv = s32(2 * s32((sw & (r ^ v)) ^ v))
        q, r = s32(q + tu), s32(r + tv)
        u, v = nu, nv
        f, g = nf, ng
        delta = s32(((delta ^ sw) - sw) + 1)
        if trace is not None: trace.append(delta)
    return delta, u, v, q, r

# ------------------------------------------------------- 9-limb row update
def row_ip(X, Y, a, b, modp):
    """In place: X = (a*X + b*Y [+ m*p]) >> 30.   X,Y are 9 uint32 limbs."""
    acc = s64((a * X[0] + b * Y[0]))
    m = (u32(acc) * MM32) & MASK30 & u32(0 - modp)
    acc = s64(acc - 977 * m)
    X[0] = u32(acc); acc = s64(acc) >> 32
    acc = s64(acc + a * X[1] + b * Y[1] - m)
    X[1] = u32(acc); acc = s64(acc) >> 32
    for i in range(2, 8):
        acc = s64(acc + a * X[i] + b * Y[i])
        X[i] = u32(acc); acc = s64(acc) >> 32
    acc = s64(acc + a * s32(X[8]) + b * s32(Y[8]) + m)
    X[8] = u32(acc)
    for i in range(8):
        X[i] = u32((X[i] >> 30) | (X[i+1] << 2))
    X[8] = u32(s32(X[8]) >> 30)

def condneg(X, neg):
    msk = u32(0 - neg); c = neg
    for i in range(9):
        c += (X[i] ^ msk); X[i] = u32(c); c >>= 32

def canon(X):
    hi = s32(X[8])
    acc = X[0] + hi * 977
    X[0] = u32(acc); acc >>= 32
    acc += X[1] + hi
    X[1] = u32(acc); acc >>= 32
    for i in range(2, 8):
        acc += X[i]; X[i] = u32(acc); acc >>= 32
    X[8] = u32(acc)
    mneg = u32(s32(X[8]) >> 31)
    c = 0
    for i in range(9):
        c += X[i] + (PL[i] & mneg); X[i] = u32(c); c >>= 32
    T = [0]*9; c = 1
    for i in range(9):
        c += X[i] + u32(~PL[i]); T[i] = u32(c); c >>= 32
    keep = u32(s32(T[8]) >> 31)
    for i in range(8):
        X[i] = (X[i] & keep) | (T[i] & u32(~keep))

def val9(X):
    v = 0
    for i in range(8): v |= X[i] << (32*i)
    return v + s32(X[8]) * (1 << 256)

# -------------------------------------------------------- the full inverse
def inverse(x, stats=None):
    F = [ (x >> (32*i)) & M32 for i in range(8) ] + [0]
    G = list(PL)            # careful: F = p, G = x below
    Fv = list(PL)           # f = p
    Gv = [ (x >> (32*i)) & M32 for i in range(8) ] + [0]
    R = [0]*9
    S = [0]*9; S[0] = 1
    delta = 1
    nb = 0
    maxabsd = 0
    clampev = 0
    while True:
        nb += 1
        assert nb <= 40, "no termination"
        d0, u, v, q, r = divstep30_by(delta, Fv[0], Gv[0])
        if abs(delta) > 4: clampev += 1
        maxabsd = max(maxabsd, abs(delta), abs(d0))
        delta = d0
        assert abs(u) + abs(v) <= 1 << 30 and abs(q) + abs(r) <= 1 << 30
        nF, nG, nR, nS = list(Fv), list(Gv), list(R), list(S)
        row_ip(nF, Gv, u, v, 0)
        row_ip(nG, Fv, r, q, 0)      # lane-1 form: ka=r (own=g), kb=q (partner=f)
        row_ip(nR, S,  u, v, 1)
        row_ip(nS, R,  r, q, 1)
        Fv, Gv, R, S = nF, nG, nR, nS
        if all(t == 0 for t in Gv): break
    fneg = 1 if s32(Fv[8]) < 0 else 0
    if stats is not None:
        stats['batches'] = nb; stats['fneg'] = fneg
        stats['maxabsd'] = maxabsd; stats['clampev'] = clampev
        stats['maxR'] = abs(val9(R))
        stats['fend'] = val9(Fv)
    condneg(R, fneg)
    canon(R)
    out = 0
    for i in range(8): out |= R[i] << (32*i)
    return out

if __name__ == '__main__':
    random.seed(7)
    hist = {}
    fneg_count = 0
    maxabsd = 0; maxR = 0; maxb = 0
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    bad = 0
    tests = [0, 1, 2, 3, P-1, P-2, P-3, 2**32, 2**32-1, 2**32+1, 2**64, 2**128, 2**192, 2**255]
    tests += [random.randrange(1, P) for _ in range(n)]
    for x in tests:
        st = {}
        inv = inverse(x, st)
        ref = pow(x, P-2, P) if x % P else 0
        if inv != ref:
            bad += 1
            if bad < 4: print("MISMATCH x=", hex(x), "got", hex(inv), "want", hex(ref))
        hist[st['batches']] = hist.get(st['batches'], 0) + 1
        fneg_count += st['fneg']
        maxabsd = max(maxabsd, st['maxabsd']); maxR = max(maxR, st['maxR'])
        maxb = max(maxb, st['batches'])
    print("cases", len(tests), "mismatches", bad)
    print("batch histogram", dict(sorted(hist.items())))
    print("mean batches %.3f  max %d" % (sum(k*v for k,v in hist.items())/len(tests), maxb))
    print("f ended at -1 in", fneg_count, "of", len(tests))
    print("max |delta| seen", maxabsd, " (clamp active whenever >4)")
    print("max |R| / p = %.2f   (zi_canon needs < 2^261/p = %.1f)" % (maxR/P, 2**261/P))
    sys.exit(1 if bad else 0)
