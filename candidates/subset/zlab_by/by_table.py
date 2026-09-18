#!/usr/bin/env python3
"""Bernstein-Yang 4-step divstep table generator + exactness proof.

divstep(delta,f,g):
  delta>0 and g odd : (1-delta, g, (g-f)/2)      [case A, "swap"]
  delta<=0 and g odd: (1+delta, f, (g+f)/2)      [case B]
  g even            : (1+delta, f, g/2)          [case C]

Over k=4 steps, (f4,g4) = ((a*f+b*g)>>4, (c*f+d*g)>>4) with integer a,b,c,d,
and delta4 = (-1)^s * delta + C where s = number of case-A steps and C is a
constant, BOTH determined by (clamp(delta,-4,4), f mod 16, g mod 16).
f is always odd, so f mod 16 has 8 values; delta clamp has 9; g mod 16 has 16.
"""
import random, sys

K = 4
DCLO, DCHI = -4, 4

def divstep(delta, f, g):
    if delta > 0 and (g & 1):
        return (1 - delta, g, (g - f) // 2, 'A')
    elif (g & 1):
        return (1 + delta, f, (g + f) // 2, 'B')
    else:
        return (1 + delta, f, g // 2, 'C')

def block4(delta, f, g):
    """Run K raw divsteps; return (delta_out, f_out, g_out, matrix, swaps)."""
    # matrix M: (f_i, g_i) = (M[0]*f0 + M[1]*g0, M[2]*f0 + M[3]*g0) / 2^i
    a, b, c, d = 1, 0, 0, 1
    s = 0
    for _ in range(K):
        if delta > 0 and (g & 1):
            a, b, c, d = 2*c, 2*d, c - a, d - b        # f'=g ; g'=(g-f)/2
            delta, f, g = 1 - delta, g, (g - f) // 2
            s += 1
        elif (g & 1):
            a, b, c, d = 2*a, 2*b, a + c, b + d        # f'=f ; g'=(g+f)/2
            delta, f, g = 1 + delta, f, (g + f) // 2
        else:
            a, b, c, d = 2*a, 2*b, c, d                # f'=f ; g'=g/2
            delta, f, g = 1 + delta, f, g // 2
    return delta, f, g, (a, b, c, d), s

def clamp(d):
    return DCLO if d < DCLO else (DCHI if d > DCHI else d)

# ---------------------------------------------------------------- build table
# entry[idx] with idx = (dc+4)*128 + ((f>>1)&7)*16 + (g&15)
tab = [None] * (9 * 8 * 16)
stats = {'a': set(), 'C': set(), 'dout': set()}
for dc in range(DCLO, DCHI + 1):
    for fl in range(1, 16, 2):
        for gl in range(16):
            dout, _f, _g, M, s = block4(dc, fl, gl)
            C = dout - ((-1) ** s) * dc
            idx = (dc + 4) * 128 + ((fl >> 1) & 7) * 16 + gl
            tab[idx] = (M, C, s & 1, dout)
            for v in M: stats['a'].add(v)
            stats['C'].add(C); stats['dout'].add(dout)
assert all(t is not None for t in tab)

# the (f mod 16, g mod 16) -> case-sequence must not depend on delta beyond clamp:
# verified below by the exhaustive small test + random big test.

if __name__ == '__main__':
    print("coeff range      :", min(stats['a']), "..", max(stats['a']))
    print("C range          :", min(stats['C']), "..", max(stats['C']))
    print("dout(clamped) rng:", min(stats['dout']), "..", max(stats['dout']))
    # ------------------------------------------------- exactness of the clamp
    def table_block(delta, f, g):
        dc = clamp(delta)
        idx = (dc + 4) * 128 + ((f >> 1) & 7) * 16 + (g & 15)
        (a, b, c, d), C, s, _ = tab[idx]
        f4 = (a * f + b * g) >> K
        g4 = (c * f + d * g) >> K
        d4 = (-delta if s else delta) + C
        return d4, f4, g4

    bad = 0
    random.seed(20240918)
    N1 = 0
    for delta in range(-40, 41):
        for fl in range(1, 256, 2):
            for gl in range(256):
                f = fl; g = gl
                r_d, r_f, r_g, _, _ = block4(delta, f, g)
                t_d, t_f, t_g = table_block(delta, f, g)
                N1 += 1
                if (r_d, r_f, r_g) != (t_d, t_f, t_g):
                    bad += 1
                    if bad < 5: print("MISMATCH small", delta, f, g, (r_d,r_f,r_g), (t_d,t_f,t_g))
    print("exhaustive delta[-40,40] x f,g in [0,256):", N1, "cases, mismatches:", bad)

    # (ii) random 256-bit signed f (odd), g, wide delta range
    N2 = 0
    for _ in range(400000):
        f = random.getrandbits(257) - (1 << 256)
        f |= 1
        g = random.getrandbits(257) - (1 << 256)
        delta = random.randint(-80, 80)
        r_d, r_f, r_g, _, _ = block4(delta, f, g)
        t_d, t_f, t_g = table_block(delta, f, g)
        N2 += 1
        if (r_d, r_f, r_g) != (t_d, t_f, t_g):
            bad += 1
            if bad < 5: print("MISMATCH big", delta, hex(f), hex(g))
    print("random 256-bit signed f,g x delta[-80,80]:", N2, "cases, total mismatches:", bad)
    sys.exit(1 if bad else 0)
