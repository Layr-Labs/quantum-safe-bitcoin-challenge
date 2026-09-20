#!/usr/bin/env python3
"""Host audit: digest word 0 of SHA-256(33-byte pubkey) equals the round-63
a-only formula used by QSB_PK_H0 in pair_shared.cuh.

No GPU. SPDX-License-Identifier: GPL-3.0-only
"""
from __future__ import annotations

import hashlib
import json
import random
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
K = [
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
]
IV = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]


def ror(x, n):
    return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF


def S0(x):
    return ror(x, 2) ^ ror(x, 13) ^ ror(x, 22)


def S1(x):
    return ror(x, 6) ^ ror(x, 11) ^ ror(x, 25)


def s0(x):
    return ror(x, 7) ^ ror(x, 18) ^ (x >> 3)


def s1(x):
    return ror(x, 17) ^ ror(x, 19) ^ (x >> 10)


def Ch(x, y, z):
    return z ^ (x & (y ^ z))


def Maj(x, y, z):
    return (x & y) | (z & (x | y))


def s2round(a, b, c, d, e, f, g, h, k, w):
    t1 = (h + S1(e) + Ch(e, f, g) + k + w) & 0xFFFFFFFF
    t2 = (S0(a) + Maj(a, b, c)) & 0xFFFFFFFF
    d = (d + t1) & 0xFFFFFFFF
    h = (t1 + t2) & 0xFFFFFFFF
    return a, b, c, d, e, f, g, h


def wmix(w):
    w = list(w)
    w[0] = (w[0] + s1(w[14]) + w[9] + s0(w[1])) & 0xFFFFFFFF
    w[1] = (w[1] + s1(w[15]) + w[10] + s0(w[2])) & 0xFFFFFFFF
    w[2] = (w[2] + s1(w[0]) + w[11] + s0(w[3])) & 0xFFFFFFFF
    w[3] = (w[3] + s1(w[1]) + w[12] + s0(w[4])) & 0xFFFFFFFF
    w[4] = (w[4] + s1(w[2]) + w[13] + s0(w[5])) & 0xFFFFFFFF
    w[5] = (w[5] + s1(w[3]) + w[14] + s0(w[6])) & 0xFFFFFFFF
    w[6] = (w[6] + s1(w[4]) + w[15] + s0(w[7])) & 0xFFFFFFFF
    w[7] = (w[7] + s1(w[5]) + w[0] + s0(w[8])) & 0xFFFFFFFF
    w[8] = (w[8] + s1(w[6]) + w[1] + s0(w[9])) & 0xFFFFFFFF
    w[9] = (w[9] + s1(w[7]) + w[2] + s0(w[10])) & 0xFFFFFFFF
    w[10] = (w[10] + s1(w[8]) + w[3] + s0(w[11])) & 0xFFFFFFFF
    w[11] = (w[11] + s1(w[9]) + w[4] + s0(w[12])) & 0xFFFFFFFF
    w[12] = (w[12] + s1(w[10]) + w[5] + s0(w[13])) & 0xFFFFFFFF
    w[13] = (w[13] + s1(w[11]) + w[6] + s0(w[14])) & 0xFFFFFFFF
    w[14] = (w[14] + s1(w[12]) + w[7] + s0(w[15])) & 0xFFFFFFFF
    w[15] = (w[15] + s1(w[13]) + w[8] + s0(w[0])) & 0xFFFFFFFF
    return w


def rnd16(st, kbase, w):
    names = ["a", "b", "c", "d", "e", "f", "g", "h"]
    regs = {n: st[i] for i, n in enumerate(names)}
    order = [
        ("a", "b", "c", "d", "e", "f", "g", "h"),
        ("h", "a", "b", "c", "d", "e", "f", "g"),
        ("g", "h", "a", "b", "c", "d", "e", "f"),
        ("f", "g", "h", "a", "b", "c", "d", "e"),
        ("e", "f", "g", "h", "a", "b", "c", "d"),
        ("d", "e", "f", "g", "h", "a", "b", "c"),
        ("c", "d", "e", "f", "g", "h", "a", "b"),
        ("b", "c", "d", "e", "f", "g", "h", "a"),
    ]
    order = order + order  # 16 rounds
    for i, (a, b, c, d, e, f, g, h) in enumerate(order):
        va, vb, vc, vd, ve, vf, vg, vh = [regs[x] for x in (a, b, c, d, e, f, g, h)]
        va, vb, vc, vd, ve, vf, vg, vh = s2round(va, vb, vc, vd, ve, vf, vg, vh, K[kbase + i], w[i])
        for n, v in zip((a, b, c, d, e, f, g, h), (va, vb, vc, vd, ve, vf, vg, vh)):
            regs[n] = v
    return [regs[n] for n in names]


def pubkey_block(pub: bytes):
    assert len(pub) == 33
    padded = pub + b"\x80" + b"\x00" * 22 + struct.pack(">Q", 33 * 8)
    assert len(padded) == 64
    return list(struct.unpack(">16I", padded))


def h0_from_state_after_15(a, b, c, d, e, f, g, h, w15):
    return (IV[0] + a + S1(f) + Ch(f, g, h) + K[63] + w15 + S0(b) + Maj(b, c, d)) & 0xFFFFFFFF


def digest_word0_h0only(pub: bytes) -> int:
    w = pubkey_block(pub)
    st = list(IV)
    st = rnd16(st, 0, w)
    w = wmix(w)
    st = rnd16(st, 16, w)
    w = wmix(w)
    st = rnd16(st, 32, w)
    w = wmix(w)
    # 15 rounds of group 48
    names = ["a", "b", "c", "d", "e", "f", "g", "h"]
    regs = {n: st[i] for i, n in enumerate(names)}
    order = [
        ("a", "b", "c", "d", "e", "f", "g", "h"),
        ("h", "a", "b", "c", "d", "e", "f", "g"),
        ("g", "h", "a", "b", "c", "d", "e", "f"),
        ("f", "g", "h", "a", "b", "c", "d", "e"),
        ("e", "f", "g", "h", "a", "b", "c", "d"),
        ("d", "e", "f", "g", "h", "a", "b", "c"),
        ("c", "d", "e", "f", "g", "h", "a", "b"),
        ("b", "c", "d", "e", "f", "g", "h", "a"),
    ]
    order15 = order + order[:7]  # 15
    for i, (a, b, c, d, e, f, g, h) in enumerate(order15):
        va, vb, vc, vd, ve, vf, vg, vh = [regs[x] for x in (a, b, c, d, e, f, g, h)]
        va, vb, vc, vd, ve, vf, vg, vh = s2round(va, vb, vc, vd, ve, vf, vg, vh, K[48 + i], w[i])
        for n, v in zip((a, b, c, d, e, f, g, h), (va, vb, vc, vd, ve, vf, vg, vh)):
            regs[n] = v
    return h0_from_state_after_15(regs["a"], regs["b"], regs["c"], regs["d"],
                                  regs["e"], regs["f"], regs["g"], regs["h"], w[15])


def audit(n=2000):
    rng = random.Random(0x10A11E24)
    mismatches = 0
    for i in range(n):
        prefix = 2 + (i & 1)
        x = rng.getrandbits(256).to_bytes(32, "big")
        pub = bytes([prefix]) + x
        expect = int.from_bytes(hashlib.sha256(pub).digest()[:4], "big")
        got = digest_word0_h0only(pub)
        if got != expect:
            mismatches += 1
            if mismatches <= 3:
                print("mismatch", i, hex(expect), hex(got), pub.hex())
    return n, mismatches


def audit_source():
    src = (HERE / "tests" / "gpu_epochs" / "pair_shared.cuh").read_text()
    assert "#define QSB_PK_H0 1" in src
    assert "QSB_GP_RND15(48)" in src
    assert "K[63] + w0[15]" in src
    return True


def main():
    n, bad = audit()
    src = audit_source()
    assert bad == 0
    print(json.dumps({"test": "QSB_PK_H0 last-round identity", "samples": n, "mismatches": bad, "source": src}))


if __name__ == "__main__":
    main()
