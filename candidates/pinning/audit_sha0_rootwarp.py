#!/usr/bin/env python3
"""CPU binder + SHA0 bit-identity oracle for tip bad91ac lever."""
from __future__ import annotations
import re, sys, struct, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
cu = (ROOT / "pinning.cu").read_text()
gm = (ROOT / "GPUMath.h").read_text()
packed = (ROOT / "PackedRecovery.cuh").read_text()

def define_int(name: str, src: str = cu) -> int:
    m = re.search(rf"^#define\s+{name}\s+(\d+)\b", src, re.M)
    assert m, f"missing {name}"
    return int(m.group(1))

def ok(cond, msg):
    if not cond:
        raise AssertionError(msg)

ok(define_int("QSB_STREAM") == 1, "STREAM")
ok(define_int("QSB_SLOTPIPE") == 1, "SLOTPIPE")
ok(define_int("QSB_SLOTS") == 2, "SLOTS")
ok(define_int("QSB_EARLY_LOAD") == 0, "EARLY_LOAD must stay 0")
ok(define_int("QSB_SHA0") == 1, "SHA0")
ok(define_int("QSB_ROOT_WARP_BAR") == 1, "ROOT_WARP_BAR")
ok(define_int("QSB_FINISH_ROOT_SH") == 1, "FINISH_ROOT_SH")
ok(define_int("QSB_FUSE_SQRADDSUB2", gm) == 1, "tip fuse stays on")
ok("QSB_RESOLVE_LAST" not in cu or re.search(r"#define\s+QSB_RESOLVE_LAST\s+0\b", cu), "no RESOLVE_LAST")
ok(define_int("QSB_SLOTS") < 3, "no SLOTS>=3")
ok("qsb_st_v2(&saved[0" not in cu and "qsb_st_v2" not in packed, "no packed-plane STREAM")
ok("if(count>2){if(half>32)__syncthreads();else __syncwarp();}" in cu, "product warp")
ok("if((count<<1)>32)__syncthreads();else __syncwarp();" in cu, "inverse warp")
ok("sh_root_inv[4]" in cu and "sh_weighted_inv[4]" in cu, "finish SH")
ok("_SHA256TransformFastTail11_sha0" in cu, "sha0 device")
ok("qsb_sha0_fold_round0" in cu, "sha0 host")
ok("d_midstate + 8" in cu, "r0_work upload slot")

# --- SHA0 bit-identity oracle ---
def ror(x, n):
    x &= 0xFFFFFFFF
    return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

def s0(x):
    return ror(x, 2) ^ ror(x, 13) ^ ror(x, 22)

def s1(x):
    return ror(x, 6) ^ ror(x, 11) ^ ror(x, 25)

def Ch(x, y, z):
    return (z ^ (x & (y ^ z))) & 0xFFFFFFFF

def Maj(x, y, z):
    return ((x & y) | (z & (x | y))) & 0xFFFFFFFF

K = [
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
]

def sigma0(x):
    return ror(x, 7) ^ ror(x, 18) ^ (x >> 3)

def sigma1(x):
    return ror(x, 17) ^ ror(x, 19) ^ (x >> 10)

def round_step(a,b,c,d,e,f,g,h,k,w):
    t1 = (h + s1(e) + Ch(e,f,g) + k + w) & 0xFFFFFFFF
    t2 = (s0(a) + Maj(a,b,c)) & 0xFFFFFFFF
    return (t1+t2)&0xFFFFFFFF, a, b, c, (d+t1)&0xFFFFFFFF, e, f, g

def fast_tail11(state, w0, w1, w2):
    L = 79960
    a,b,c,d,e,f,g,h = state
    w = [0]*16
    w[0],w[1],w[2],w[15] = w0,w1,w2,L
    # R0..R15 specialized like device
    regs = [a,b,c,d,e,f,g,h]
    # easier: full 64 rounds with built W
    a,b,c,d,e,f,g,h = state
    # first 16 rounds
    for i in range(16):
        wi = w[i]
        t1 = (h + s1(e) + Ch(e,f,g) + K[i] + wi) & 0xFFFFFFFF
        t2 = (s0(a) + Maj(a,b,c)) & 0xFFFFFFFF
        h,g,f,e,d,c,b,a = g,f,e,(d+t1)&0xFFFFFFFF,c,b,a,(t1+t2)&0xFFFFFFFF
    # WMIX + rounds like reference SHA
    def wmix():
        for i in range(16):
            w[i] = (w[i] + sigma1(w[(i-2)&15]) + w[(i-7)&15] + sigma0(w[(i-15)&15])) & 0xFFFFFFFF
    # Device uses a specialized first WMIX then SHA256_RND/WMIX
    # Use hashlib for ground truth instead.
    return None

def sha256_compress(mid, block16):
    """One SHA-256 compression from midstate + 16 BE words -> new midstate."""
    # Build 64-byte block
    raw = b"".join(struct.pack(">I", x & 0xFFFFFFFF) for x in block16)
    # Use manual compress from mid
    a,b,c,d,e,f,g,h = [x & 0xFFFFFFFF for x in mid]
    w = list(block16) + [0]*48
    for i in range(16, 64):
        w[i] = (sigma1(w[i-2]) + w[i-7] + sigma0(w[i-15]) + w[i-16]) & 0xFFFFFFFF
    for i in range(64):
        t1 = (h + s1(e) + Ch(e,f,g) + K[i] + w[i]) & 0xFFFFFFFF
        t2 = (s0(a) + Maj(a,b,c)) & 0xFFFFFFFF
        h=g; g=f; f=e; e=(d+t1)&0xFFFFFFFF; d=c; c=b; b=a; a=(t1+t2)&0xFFFFFFFF
    out = [(mid[i] + [a,b,c,d,e,f,g,h][i]) & 0xFFFFFFFF for i in range(8)]
    return out

def fold_r0(mid, tail_base):
    a,b,c,d,e,f,g,h = mid
    t1 = (h + s1(e) + Ch(e,f,g) + K[0] + tail_base) & 0xFFFFFFFF
    t2 = (s0(a) + Maj(a,b,c)) & 0xFFFFFFFF
    return [(t1+t2)&0xFFFFFFFF, a, b, c, (d+t1)&0xFFFFFFFF, e, f, g]

def sha0_path(mid, r0_work, w0, w1, w2):
    """Emulate device sha0 path: start from r0_work+lt_lo, rounds 1..63, add mid."""
    lt_lo = w0 & 0xff
    L = 79960
    a = (r0_work[0] + lt_lo) & 0xFFFFFFFF
    b,c,d = r0_work[1], r0_work[2], r0_work[3]
    e = (r0_work[4] + lt_lo) & 0xFFFFFFFF
    f,g,h = r0_work[5], r0_work[6], r0_work[7]
    w = [0]*64
    w[0],w[1],w[2],w[15] = w0,w1,w2,L
    for i in range(16, 64):
        w[i] = (sigma1(w[i-2]) + w[i-7] + sigma0(w[i-15]) + w[i-16]) & 0xFFFFFFFF
    # Continue from round 1 (round 0 already done)
    for i in range(1, 64):
        t1 = (h + s1(e) + Ch(e,f,g) + K[i] + w[i]) & 0xFFFFFFFF
        t2 = (s0(a) + Maj(a,b,c)) & 0xFFFFFFFF
        h=g; g=f; f=e; e=(d+t1)&0xFFFFFFFF; d=c; c=b; b=a; a=(t1+t2)&0xFFFFFFFF
    return [(mid[i] + [a,b,c,d,e,f,g,h][i]) & 0xFFFFFFFF for i in range(8)]

# Random-ish midstates and locktimes
import random
random.seed(42)
mism = 0
for trial in range(200):
    mid = [random.getrandbits(32) for _ in range(8)]
    tail_base = random.getrandbits(32) & 0xFFFFFF00  # low byte clear
    lt = random.getrandbits(32)
    w0 = tail_base | (lt & 0xff)
    w1 = (((lt & 0xff00) << 16) | (lt & 0xff0000) | ((lt >> 16) & 0xff00) | (random.getrandbits(8))) & 0xFFFFFFFF
    w2 = random.getrandbits(32)
    block = [w0,w1,w2] + [0]*13
    block[15] = 79960
    ref = sha256_compress(mid, block)
    r0 = fold_r0(mid, tail_base)
    got = sha0_path(mid, r0, w0, w1, w2)
    if ref != got:
        mism += 1
        if mism <= 3:
            print("MISMATCH", trial, ref, got)

ok(mism == 0, f"SHA0 oracle mismatches: {mism}")
print("audit_sha0_rootwarp: OK (200 SHA0 cases + shape binders)")
print(f"  STREAM={define_int('QSB_STREAM')} SHA0={define_int('QSB_SHA0')} "
      f"ROOT_WARP={define_int('QSB_ROOT_WARP_BAR')} FINISH_SH={define_int('QSB_FINISH_ROOT_SH')} "
      f"EARLY_LOAD={define_int('QSB_EARLY_LOAD')} FUSE={define_int('QSB_FUSE_SQRADDSUB2', gm)}")
