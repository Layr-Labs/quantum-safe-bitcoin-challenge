#!/usr/bin/env python3
"""One-level difference Karatsuba 256x256->512 product stage (PTX) for _ModMultCoreK.

Emits the product stage only: inputs %4..%11 (a0..a3, b0..b3 as u64 limbs), output the
exact 512-bit product in u32 registers x0..x15. The pseudo-Mersenne tail that follows is
the production text, unchanged. 48 mul.wide.u32 instead of 64.

  A = A1*2^128 + A0, B = B1*2^128 + B0
  L = A0*B0, H = A1*B1, P = |A0-A1| * |B1-B0|, s = sign((A0-A1)(B1-B0))
  mid = L + H + s*P = A0*B1 + A1*B0  (in [0, 2^257))
  A*B = L + mid*2^128 + H*2^256
"""
import sys

ZERO = 'kz'   # u32 zero from constant memory: keeps "x + 0 + carry" off IMAD.X (see QSB_CHAIN_ALU)


def mul128(a, b, out, p):
    """4x32 even/odd column accumulation -> out[0..7] (u32). Exact.
    Carry captures are 32-bit (SEL); the even chain's row-2 carry (bit 192) rides in the high
    half of the odd word o2 (bits 160..223) together with the odd row-1 carry (bit 160)."""
    e = [p + 'e%d' % i for i in range(4)]
    o = [p + 'o%d' % i for i in range(3)]
    y = [p + 'y%d' % i for i in range(6)]
    ce, co, oc, lc = p + 'ce', p + 'co', p + 'oc', p + 'lc'
    L = ['.reg .u64 %s;' % ','.join(e + o + [lc]), '.reg .u32 %s;' % ','.join(y + [ce, co, oc])]
    w = L.append
    # even rows 0..2
    w(f'mul.wide.u32 {e[0]}, {a[0]}, {b[0]};')
    w(f'mul.wide.u32 {e[1]}, {a[0]}, {b[2]};')
    w(f'mul.wide.u32 t, {a[1]}, {b[1]};')
    w(f'add.cc.u64 {e[1]}, {e[1]}, t;')
    w(f'mul.wide.u32 t, {a[1]}, {b[3]};')
    w(f'addc.u64 {e[2]}, t, 0;')
    w(f'mul.wide.u32 t, {a[2]}, {b[0]};')
    w(f'add.cc.u64 {e[1]}, {e[1]}, t;')
    w(f'mul.wide.u32 t, {a[2]}, {b[2]};')
    w(f'addc.cc.u64 {e[2]}, {e[2]}, t;')
    w(f'addc.u32 {ce}, 0, 0;')
    # odd rows 0..1
    w(f'mul.wide.u32 {o[0]}, {a[0]}, {b[1]};')
    w(f'mul.wide.u32 {o[1]}, {a[0]}, {b[3]};')
    w(f'mul.wide.u32 t, {a[1]}, {b[0]};')
    w(f'add.cc.u64 {o[0]}, {o[0]}, t;')
    w(f'mul.wide.u32 t, {a[1]}, {b[2]};')
    w(f'addc.cc.u64 {o[1]}, {o[1]}, t;')
    w(f'addc.u32 {co}, 0, 0;')
    # odd row 2: o2 = a2*b3 + {co, ce} + carry
    w(f'mov.b64 {lc}, {{{co}, {ce}}};')
    w(f'mul.wide.u32 t, {a[2]}, {b[1]};')
    w(f'add.cc.u64 {o[1]}, {o[1]}, t;')
    w(f'mul.wide.u32 t, {a[2]}, {b[3]};')
    w(f'addc.u64 {o[2]}, t, {lc};')
    # even row 3
    w(f'mul.wide.u32 t, {a[3]}, {b[1]};')
    w(f'add.cc.u64 {e[2]}, {e[2]}, t;')
    w(f'mul.wide.u32 t, {a[3]}, {b[3]};')
    w(f'addc.u64 {e[3]}, t, 0;')
    # odd row 3
    w(f'mul.wide.u32 t, {a[3]}, {b[0]};')
    w(f'add.cc.u64 {o[1]}, {o[1]}, t;')
    w(f'mul.wide.u32 t, {a[3]}, {b[2]};')
    w(f'addc.cc.u64 {o[2]}, {o[2]}, t;')
    w(f'addc.u32 {oc}, 0, 0;')
    # merge: out = E + O*2^32 (+ oc at bit 224)
    w(f'mov.b64 {{{out[0]},{out[1]}}}, {e[0]};')
    w(f'mov.b64 {{{out[2]},{out[3]}}}, {e[1]};')
    w(f'mov.b64 {{{out[4]},{out[5]}}}, {e[2]};')
    w(f'mov.b64 {{{out[6]},{out[7]}}}, {e[3]};')
    w(f'mov.b64 {{{y[0]},{y[1]}}}, {o[0]};')
    w(f'mov.b64 {{{y[2]},{y[3]}}}, {o[1]};')
    w(f'mov.b64 {{{y[4]},{y[5]}}}, {o[2]};')
    w(f'add.cc.u32 {out[1]}, {out[1]}, {y[0]};')
    for i in range(2, 7):
        w(f'addc.cc.u32 {out[i]}, {out[i]}, {y[i-1]};')
    w(f'addc.u32 {out[7]}, {out[7]}, {oc};')
    w(f'add.u32 {out[7]}, {out[7]}, {ZERO};')
    return L


def absdiff(x, y, d, m):
    """d = |X - Y| (4 limbs), m = 0xffffffff iff X < Y."""
    L = []
    w = L.append
    w(f'sub.cc.u32 {d[0]}, {x[0]}, {y[0]};')
    for i in range(1, 4):
        w(f'subc.cc.u32 {d[i]}, {x[i]}, {y[i]};')
    w(f'subc.u32 {m}, 0, 0;')
    w(f'add.u32 {m}, {m}, {ZERO};')
    for i in range(4):
        w(f'xor.b32 {d[i]}, {d[i]}, {m};')
    w(f'sub.cc.u32 {d[0]}, {d[0]}, {m};')
    for i in range(1, 3):
        w(f'subc.cc.u32 {d[i]}, {d[i]}, {m};')
    w(f'subc.u32 {d[3]}, {d[3]}, {m};')
    w(f'add.u32 {d[3]}, {d[3]}, {ZERO};')
    return L


def product_stage():
    a = ['a%d' % i for i in range(8)]
    b = ['b%d' % i for i in range(8)]
    x = ['x%d' % i for i in range(16)]
    da = ['kda%d' % i for i in range(4)]
    db = ['kdb%d' % i for i in range(4)]
    l = ['x0', 'x1', 'x2', 'x3'] + ['kl%d' % i for i in range(4, 8)]
    h = ['kh%d' % i for i in range(8)]
    pp = ['kp%d' % i for i in range(8)]
    m = ['km%d' % i for i in range(9)]
    L = []
    w = L.append
    w('.reg .u32 %s;' % ','.join(a + b))
    w('.reg .u64 t;')
    w('.reg .u32 %s;' % ','.join(x))
    w('.reg .u32 %s;' % ','.join(da + db + l[4:] + h + pp + m + ['kma', 'kmb', 'kms', 'kjunk', ZERO]))
    w(f'ld.const.u32 {ZERO}, [pin_zero_add];')
    for i in range(4):
        w(f'mov.b64 {{{a[2*i]},{a[2*i+1]}}}, %{4+i};')
    for i in range(4):
        w(f'mov.b64 {{{b[2*i]},{b[2*i+1]}}}, %{8+i};')
    L += absdiff(a[0:4], a[4:8], da, 'kma')          # |A0 - A1|
    L += absdiff(b[4:8], b[0:4], db, 'kmb')          # |B1 - B0|
    w('xor.b32 kms, kma, kmb;')                      # -1: (A0-A1)(B1-B0) < 0
    L += mul128(a[0:4], b[0:4], l, 'kL')
    L += mul128(a[4:8], b[4:8], h, 'kH')
    L += mul128(da, db, pp, 'kP')
    # m = L + H (257 bits)
    w(f'add.cc.u32 {m[0]}, {l[0]}, {h[0]};')
    for i in range(1, 8):
        w(f'addc.cc.u32 {m[i]}, {l[i]}, {h[i]};')
    w(f'addc.u32 {m[8]}, 0, 0;')
    # m += s*P: (P ^ ms) + (ms & 1), sign-extended by ms into limb 8
    for i in range(8):
        w(f'xor.b32 {pp[i]}, {pp[i]}, kms;')
    w('add.cc.u32 kjunk, kms, kms;')                 # carry = ms & 1
    for i in range(8):
        w(f'addc.cc.u32 {m[i]}, {m[i]}, {pp[i]};')
    w(f'addc.u32 {m[8]}, {m[8]}, kms;')
    w(f'add.u32 {m[8]}, {m[8]}, {ZERO};')
    # x = L + m*2^128 + H*2^256
    w(f'add.cc.u32 {x[4]}, {l[4]}, {m[0]};')
    for i in range(1, 4):
        w(f'addc.cc.u32 {x[4+i]}, {l[4+i]}, {m[i]};')
    for i in range(4):
        w(f'addc.cc.u32 {x[8+i]}, {h[i]}, {m[4+i]};')
    w(f'addc.cc.u32 {x[12]}, {h[4]}, {m[8]};')
    w(f'addc.cc.u32 {x[13]}, {h[5]}, {ZERO};')
    w(f'addc.cc.u32 {x[14]}, {h[6]}, {ZERO};')
    w(f'addc.u32 {x[15]}, {h[7]}, {ZERO};')
    w(f'add.u32 {x[15]}, {x[15]}, {ZERO};')
    return L


if __name__ == '__main__':
    lines = product_stage()
    if len(sys.argv) > 1 and sys.argv[1] == 'c':
        # C string literal body, one PTX statement per line
        print(''.join('"\\t%s\\n"\n' % s for s in lines), end='')
    else:
        print('\n'.join(lines))
