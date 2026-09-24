#!/usr/bin/env python3
"""QSB_CHAIN_PREFETCH_L1: prove the in-asm next-entry address equals the caller's.

Caller (tree.cu, 15-chunk DIGIT_SHIFT loop), next step: f=w0&0x1FFFF, t=f>>16,
idx=(f^(t-1))&0xFFFF, record offset (table_base+2^16+idx)*64.
Asm: m=bfe.s32(w0,16,1) (= -t), idx=(w0^~m)&0xFFFF, same offset.
Exhaustive over the low 17 bits of w0, with random upper bits, for every chunk
base the loop visits (gt_offset(2)..gt_offset(13)); also checks both loaded
words (+0, +32) stay inside the 64 MiB table.
"""
import random
M32 = 0xFFFFFFFF
TABLE_BYTES = 64 << 20
def gt_offset(c): return 0 if c == 0 else (c + 1) << 16
def ref(w0, base):
    f = w0 & 0x1FFFF; t = f >> 16
    return (base + 0x10000 + ((f ^ ((t - 1) & M32)) & 0xFFFF)) * 64
def asm(w0, base):
    m = M32 if (w0 >> 16) & 1 else 0            # bfe.s32 w0,16,1
    i = ((w0 ^ (~m & M32)) & 0xFFFF)
    i = (i + base) & M32; i = (i + 0x10000) & M32
    return i << 6                                # cvt.u64 + shl 6
rng = random.Random(1)
n = 0
for c in range(2, 14):                           # loop steps c=2..13 prefetch c+1
    base = gt_offset(c)
    for lo in range(1 << 17):
        w0 = (rng.getrandbits(15) << 17) | lo
        a, b = ref(w0, base), asm(w0, base)
        assert a == b, (c, hex(w0), a, b)
        assert 0 <= b and b + 36 <= TABLE_BYTES
        n += 1
print(f"ok: {n} (chunk, w0) cases, asm address == caller address, all in-table")
