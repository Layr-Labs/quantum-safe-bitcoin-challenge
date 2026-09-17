#!/usr/bin/env python3
"""Bit-exactness audit for the sparse-schedule SHA-256 transforms in pinning.cu.

Models, in pure Python:
  1. the generic _SHA256Transform (GPUHash.h) — reference;
  2. _SHA256TransformFastTail11  (W[0..2] live, W[3..14]=0, W[15]=9995*8,
     continues from an existing midstate);
  3. _SHA256TransformDigest32    (W[0..7] live, W[8]=0x80000000, W[9..14]=0,
     W[15]=256, starts from the SHA-256 IV);
  4. _SHA256TransformPubkey33    (W[0..8] live, W[9..14]=0, W[15]=0x108,
     starts from the SHA-256 IV).

Each sparse model replicates the source's exact in-place schedule order and
its round-rotation sequence. Every sparse output must equal the generic
transform on the same padded block, for random and boundary inputs.
Additionally, Digest32 is cross-checked against hashlib.sha256 over random
32-byte messages, and Pubkey33 against hashlib.sha256 over random 33-byte
messages with the kernel's exact byte_perm word packing.

Source-shape checks bind the audit to the production call sites.
"""

import hashlib
import random
import re
from pathlib import Path

MASK = 0xFFFFFFFF

K = [
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
]

IV = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]


def ror(x, n):
    return ((x >> n) | (x << (32 - n))) & MASK


def S0(x): return ror(x, 2) ^ ror(x, 13) ^ ror(x, 22)
def S1(x): return ror(x, 6) ^ ror(x, 11) ^ ror(x, 25)
def s0(x): return ror(x, 7) ^ ror(x, 18) ^ (x >> 3)
def s1(x): return ror(x, 17) ^ ror(x, 19) ^ (x >> 10)
def Maj(x, y, z): return (x & y) | (z & (x | y))
def Ch(x, y, z): return z ^ (x & (y ^ z))


def s2round(st, i0, k, w):
    """One S2Round on register list st rotated so st[0]=a; returns new list.

    st order is [a,b,c,d,e,f,g,h] as passed; updates d and h in place like the
    macro, then the caller rotates."""
    a, b, c, d, e, f, g, h = st
    t1 = (h + S1(e) + Ch(e, f, g) + k + w) & MASK
    t2 = (S0(a) + Maj(a, b, c)) & MASK
    d = (d + t1) & MASK
    h = (t1 + t2) & MASK
    return [a, b, c, d, e, f, g, h]


# The exact rotation sequence used by SHA256_RND: round r registers.
ROT = [
    (0, 1, 2, 3, 4, 5, 6, 7),
    (7, 0, 1, 2, 3, 4, 5, 6),
    (6, 7, 0, 1, 2, 3, 4, 5),
    (5, 6, 7, 0, 1, 2, 3, 4),
    (4, 5, 6, 7, 0, 1, 2, 3),
    (3, 4, 5, 6, 7, 0, 1, 2),
    (2, 3, 4, 5, 6, 7, 0, 1),
    (1, 2, 3, 4, 5, 6, 7, 0),
]


def run_rounds(regs, w, rounds, wsel):
    """Run `rounds` S2Rounds starting at the current rotation offset.

    regs: [a..h]; wsel(r) -> (K_index, w_value). Rotation continues mod 8.
    Returns updated regs and advances nothing else."""
    for r in rounds:
        rot = ROT[r % 8]
        cur = [regs[i] for i in rot]
        cur = s2round(cur, r, *wsel(r))
        for i, ri in enumerate(rot):
            regs[ri] = cur[i]
    return regs


def wmix(w):
    w[0] = (w[0] + s1(w[14]) + w[9] + s0(w[1])) & MASK
    w[1] = (w[1] + s1(w[15]) + w[10] + s0(w[2])) & MASK
    w[2] = (w[2] + s1(w[0]) + w[11] + s0(w[3])) & MASK
    w[3] = (w[3] + s1(w[1]) + w[12] + s0(w[4])) & MASK
    w[4] = (w[4] + s1(w[2]) + w[13] + s0(w[5])) & MASK
    w[5] = (w[5] + s1(w[3]) + w[14] + s0(w[6])) & MASK
    w[6] = (w[6] + s1(w[4]) + w[15] + s0(w[7])) & MASK
    w[7] = (w[7] + s1(w[5]) + w[0] + s0(w[8])) & MASK
    w[8] = (w[8] + s1(w[6]) + w[1] + s0(w[9])) & MASK
    w[9] = (w[9] + s1(w[7]) + w[2] + s0(w[10])) & MASK
    w[10] = (w[10] + s1(w[8]) + w[3] + s0(w[11])) & MASK
    w[11] = (w[11] + s1(w[9]) + w[4] + s0(w[12])) & MASK
    w[12] = (w[12] + s1(w[10]) + w[5] + s0(w[13])) & MASK
    w[13] = (w[13] + s1(w[11]) + w[6] + s0(w[14])) & MASK
    w[14] = (w[14] + s1(w[12]) + w[7] + s0(w[15])) & MASK
    w[15] = (w[15] + s1(w[13]) + w[8] + s0(w[0])) & MASK
    return w


def generic_transform(state, w):
    """Reference _SHA256Transform: RND(0) WMIX RND(16) WMIX RND(32) WMIX RND(48)."""
    w = list(w)
    assert len(w) == 16
    regs = list(state)
    run_rounds(regs, w, range(0, 16), lambda r: (K[r], w[r]))
    for base in (16, 32, 48):
        wmix(w)
        run_rounds(regs, w, range(base, base + 16), lambda r: (K[r], w[r - base]))
    return [(state[i] + regs[i]) & MASK for i in range(8)]


# ---------------------------------------------------------------- sparse models

def sparse_fast_tail11(state, w0, w1, w2):
    L = 9995 * 8
    w = [0] * 16
    w[0], w[1], w[2], w[15] = w0, w1, w2, L
    regs = list(state)
    vals = {0: w0, 1: w1, 2: w2, 15: L}
    run_rounds(regs, w, range(0, 16), lambda r: (K[r], vals.get(r, 0)))
    # first WMIX, sparse, exact source order
    w[0] = (w[0] + s0(w[1])) & MASK
    w[1] = (w[1] + s1(L) + s0(w[2])) & MASK
    w[2] = (w[2] + s1(w[0])) & MASK
    w[3] = s1(w[1])
    w[4] = s1(w[2])
    w[5] = s1(w[3])
    w[6] = (s1(w[4]) + L) & MASK
    w[7] = (s1(w[5]) + w[0]) & MASK
    w[8] = (s1(w[6]) + w[1]) & MASK
    w[9] = (s1(w[7]) + w[2]) & MASK
    w[10] = (s1(w[8]) + w[3]) & MASK
    w[11] = (s1(w[9]) + w[4]) & MASK
    w[12] = (s1(w[10]) + w[5]) & MASK
    w[13] = (s1(w[11]) + w[6]) & MASK
    w[14] = (s1(w[12]) + w[7] + s0(L)) & MASK
    w[15] = (w[15] + s1(w[13]) + w[8] + s0(w[0])) & MASK
    for base in (16, 32, 48):
        run_rounds(regs, w, range(base, base + 16), lambda r: (K[r], w[r - base]))
        if base != 48:
            wmix(w)
    return [(state[i] + regs[i]) & MASK for i in range(8)]


def sparse_digest32(m):
    w = [0] * 16
    for i in range(8):
        w[i] = m[i]
    w[8], w[15] = 0x80000000, 256
    regs = list(IV)
    vals = {i: m[i] for i in range(8)}
    vals[8] = 0x80000000
    vals[15] = 256
    run_rounds(regs, w, range(0, 16), lambda r: (K[r], vals.get(r, 0)))
    w[0] = (w[0] + s0(w[1])) & MASK
    w[1] = (w[1] + s1(256) + s0(w[2])) & MASK
    w[2] = (w[2] + s1(w[0]) + s0(w[3])) & MASK
    w[3] = (w[3] + s1(w[1]) + s0(w[4])) & MASK
    w[4] = (w[4] + s1(w[2]) + s0(w[5])) & MASK
    w[5] = (w[5] + s1(w[3]) + s0(w[6])) & MASK
    w[6] = (w[6] + s1(w[4]) + 256 + s0(w[7])) & MASK
    w[7] = (w[7] + s1(w[5]) + w[0] + s0(0x80000000)) & MASK
    w[8] = (0x80000000 + s1(w[6]) + w[1]) & MASK
    w[9] = (s1(w[7]) + w[2]) & MASK
    w[10] = (s1(w[8]) + w[3]) & MASK
    w[11] = (s1(w[9]) + w[4]) & MASK
    w[12] = (s1(w[10]) + w[5]) & MASK
    w[13] = (s1(w[11]) + w[6]) & MASK
    w[14] = (s1(w[12]) + w[7] + s0(256)) & MASK
    w[15] = (256 + s1(w[13]) + w[8] + s0(w[0])) & MASK
    for base in (16, 32, 48):
        run_rounds(regs, w, range(base, base + 16), lambda r: (K[r], w[r - base]))
        if base != 48:
            wmix(w)
    return [(IV[i] + regs[i]) & MASK for i in range(8)]


def sparse_pubkey33(m):
    w = [0] * 16
    for i in range(9):
        w[i] = m[i]
    w[15] = 0x108
    regs = list(IV)
    vals = {i: m[i] for i in range(9)}
    vals[15] = 0x108
    run_rounds(regs, w, range(0, 16), lambda r: (K[r], vals.get(r, 0)))
    w[0] = (w[0] + s0(w[1])) & MASK
    w[1] = (w[1] + s1(0x108) + s0(w[2])) & MASK
    w[2] = (w[2] + s1(w[0]) + s0(w[3])) & MASK
    w[3] = (w[3] + s1(w[1]) + s0(w[4])) & MASK
    w[4] = (w[4] + s1(w[2]) + s0(w[5])) & MASK
    w[5] = (w[5] + s1(w[3]) + s0(w[6])) & MASK
    w[6] = (w[6] + s1(w[4]) + 0x108 + s0(w[7])) & MASK
    w[7] = (w[7] + s1(w[5]) + w[0] + s0(w[8])) & MASK
    w[8] = (w[8] + s1(w[6]) + w[1]) & MASK
    w[9] = (s1(w[7]) + w[2]) & MASK
    w[10] = (s1(w[8]) + w[3]) & MASK
    w[11] = (s1(w[9]) + w[4]) & MASK
    w[12] = (s1(w[10]) + w[5]) & MASK
    w[13] = (s1(w[11]) + w[6]) & MASK
    w[14] = (s1(w[12]) + w[7] + s0(0x108)) & MASK
    w[15] = (0x108 + s1(w[13]) + w[8] + s0(w[0])) & MASK
    for base in (16, 32, 48):
        run_rounds(regs, w, range(base, base + 16), lambda r: (K[r], w[r - base]))
        if base != 48:
            wmix(w)
    return [(IV[i] + regs[i]) & MASK for i in range(8)]


# ---------------------------------------------------------------- test drivers

def byte_perm(a, b, sel):
    """NVIDIA __byte_perm: select 4 bytes from the 8-byte pair {a,b}."""
    src = [(a >> (8 * i)) & 0xFF for i in range(4)] + \
          [(b >> (8 * i)) & 0xFF for i in range(4)]
    r = 0
    for i in range(4):
        s = (sel >> (4 * i)) & 0xF
        r |= src[s] << (8 * i)
    return r


def words_be(data):
    assert len(data) % 4 == 0
    return [int.from_bytes(data[i:i + 4], "big") for i in range(0, len(data), 4)]


def main():
    rng = random.Random(0xC0FFEE)
    cases = 0

    # 1) FastTail11 vs generic, arbitrary midstate and locktime words
    boundary = [0, 1, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 500000000, 1744600000]
    for trial in range(4000):
        state = [rng.getrandbits(32) for _ in range(8)]
        if trial < len(boundary):
            lt = boundary[trial]
        else:
            lt = rng.getrandbits(32)
        t0 = rng.getrandbits(32) & 0xFFFFFF00  # pin_tail_words[0] low byte ORed with lt
        w0 = t0 | (lt & 0xFF)
        w1 = ((lt & 0xFF00) << 16) | (lt & 0xFF0000) | ((lt >> 16) & 0xFF00) | rng.getrandbits(8) | (rng.getrandbits(8) << 24)
        w1 &= 0xFF00FFFF  # keep the two middle bytes from lt as the kernel does
        w1 |= ((lt & 0xFF00) << 16) | (lt & 0xFF0000)
        w2 = rng.getrandbits(32)
        w = [w0, w1, w2] + [0] * 12 + [9995 * 8]
        ref = generic_transform(state, w)
        got = sparse_fast_tail11(state, w0, w1, w2)
        assert got == ref, f"FastTail11 mismatch trial {trial}"
        cases += 1

    # 2) Digest32 vs generic and vs hashlib
    for trial in range(4000):
        m = [rng.getrandbits(32) for _ in range(8)]
        w = m + [0x80000000] + [0] * 6 + [256]
        ref = generic_transform(IV, w)
        got = sparse_digest32(m)
        assert got == ref, f"Digest32 mismatch trial {trial}"
        raw = b"".join(x.to_bytes(4, "big") for x in m)
        lib = words_be(hashlib.sha256(raw).digest())
        assert got == lib, f"Digest32 vs hashlib mismatch trial {trial}"
        cases += 1

    # 3) Pubkey33 vs generic and vs hashlib, with kernel word packing
    for trial in range(4000):
        x = rng.getrandbits(256)
        parity = rng.getrandbits(1)
        xw = [(x >> (32 * i)) & MASK for i in range(4)]  # x0..x3 little-endian limbs
        x0, x1, x2, x3 = xw
        x4, x5 = (x2 & MASK), (x2 >> 32)  # alias for readability of packing below
        # kernel packing: x0..x7 are the 32-bit words, limb 0 = (x0, x1), ...
        W = []
        for limb in range(4):
            v = (x >> (64 * limb)) & ((1 << 64) - 1)
            W += [v & MASK, v >> 32]
        x0, x1, x2, x3, x4, x5, x6, x7 = W
        pb = [0] * 9
        pb[0] = byte_perm(x7, 0x2 + parity, 0x4321)
        pb[1] = byte_perm(x7, x6, 0x0765)
        pb[2] = byte_perm(x6, x5, 0x0765)
        pb[3] = byte_perm(x5, x4, 0x0765)
        pb[4] = byte_perm(x4, x3, 0x0765)
        pb[5] = byte_perm(x3, x2, 0x0765)
        pb[6] = byte_perm(x2, x1, 0x0765)
        pb[7] = byte_perm(x1, x0, 0x0765)
        pb[8] = byte_perm(x0, 0x80, 0x0456)
        w = pb + [0] * 6 + [0x108]
        ref = generic_transform(IV, w)
        got = sparse_pubkey33(pb)
        assert got == ref, f"Pubkey33 mismatch trial {trial}"
        # independent construction: 33-byte compressed key, big-endian x
        key = bytes([0x02 | parity]) + x.to_bytes(32, "big")
        lib = words_be(hashlib.sha256(key).digest())
        assert got == lib, f"Pubkey33 vs hashlib mismatch trial {trial}"
        cases += 1

    print(f"sparse-pad equivalence: {cases} cases passed "
          f"(FastTail11, Digest32+hashlib, Pubkey33+hashlib)")

    # 4) source-shape binding
    src = (Path(__file__).resolve().parent / "pinning.cu").read_text()
    assert src.count("_SHA256TransformDigest32(s2, state);") == 1
    assert src.count("_SHA256TransformPubkey33(hs,pb);") == 1
    assert "_SHA256TransformFastTail11(state, w0, w1, w2);" in src
    assert "uint32_t pb[9];" in src
    assert "pb[9]=0" not in src
    assert "uint32_t b2[16];" not in src
    # the sparse helpers must exist exactly once each
    assert src.count("__device__ __forceinline__ void _SHA256TransformDigest32(") == 1
    assert src.count("__device__ __forceinline__ void _SHA256TransformPubkey33(") == 1
    # the second-hash fallback path for non-fast modes must still exist
    assert src.count("_SHA256Transform(h2s,bb2);") == 1
    print("source-shape binding: OK")


if __name__ == "__main__":
    main()
