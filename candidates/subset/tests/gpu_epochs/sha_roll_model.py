#!/usr/bin/env python3
"""Round-14 exact model for QSB_POWER_SHA_ROLL4.

The million-vector pass checks the circular message schedule and exact K index
used by the two/three-iteration, two-eight-round-half device loops. Full SHA-256
checks run that rolled round stream for the 32-byte SHA256d block and 33-byte
compressed-pubkey block against hashlib. The FIRST8 audit independently proves
that the ranked 128-window selection has exactly eight first-block classes.
No GPU is required.
"""

import hashlib
import random

MASK = 0xFFFFFFFF
K = (
    0x428A2F98,0x71374491,0xB5C0FBCF,0xE9B5DBA5,0x3956C25B,0x59F111F1,0x923F82A4,0xAB1C5ED5,
    0xD807AA98,0x12835B01,0x243185BE,0x550C7DC3,0x72BE5D74,0x80DEB1FE,0x9BDC06A7,0xC19BF174,
    0xE49B69C1,0xEFBE4786,0x0FC19DC6,0x240CA1CC,0x2DE92C6F,0x4A7484AA,0x5CB0A9DC,0x76F988DA,
    0x983E5152,0xA831C66D,0xB00327C8,0xBF597FC7,0xC6E00BF3,0xD5A79147,0x06CA6351,0x14292967,
    0x27B70A85,0x2E1B2138,0x4D2C6DFC,0x53380D13,0x650A7354,0x766A0ABB,0x81C2C92E,0x92722C85,
    0xA2BFE8A1,0xA81A664B,0xC24B8B70,0xC76C51A3,0xD192E819,0xD6990624,0xF40E3585,0x106AA070,
    0x19A4C116,0x1E376C08,0x2748774C,0x34B0BCB5,0x391C0CB3,0x4ED8AA4A,0x5B9CCA4F,0x682E6FF3,
    0x748F82EE,0x78A5636F,0x84C87814,0x8CC70208,0x90BEFFFA,0xA4506CEB,0xBEF9A3F7,0xC67178F2,
)
IV = (0x6A09E667,0xBB67AE85,0x3C6EF372,0xA54FF53A,
      0x510E527F,0x9B05688C,0x1F83D9AB,0x5BE0CD19)

def rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & MASK

def s0(x):
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)

def s1(x):
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)

def expand(words):
    out = list(words)
    for i in range(16, 64):
        out.append((out[i-16] + s0(out[i-15]) + out[i-7] + s1(out[i-2])) & MASK)
    return out

def rolled_words(words):
    """Emit W16..W63 as the device's fixed 0..7/8..15 halves do."""
    ring = list(words)
    emitted = []
    for base in (16, 32, 48):
        for j in range(16):
            ring[j] = (ring[j] + s1(ring[(j+14)&15]) +
                       ring[(j+9)&15] + s0(ring[(j+1)&15])) & MASK
            emitted.append((base + j, ring[j], K[base + j]))
    return emitted

def compress(block, rolled=False):
    words = [int.from_bytes(block[i:i+4], "big") for i in range(0, 64, 4)]
    w = expand(words)
    stream = list(enumerate(zip(w, K)))
    if rolled:
        stream = list(enumerate(zip(words, K[:16])))
        stream += [(r, (word, constant)) for r, word, constant in rolled_words(words)]
    a,b,c,d,e,f,g,h = IV
    for r, (word, constant) in stream:
        big1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25)
        ch = (e & f) ^ ((~e) & g)
        t1 = (h + big1 + ch + constant + word) & MASK
        big0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        t2 = (big0 + maj) & MASK
        h,g,f,e,d,c,b,a = g,f,e,(d+t1)&MASK,c,b,a,(t1+t2)&MASK
    return b"".join(((x+y)&MASK).to_bytes(4,"big") for x,y in zip(IV,(a,b,c,d,e,f,g,h)))

def padded(message):
    bits = len(message) * 8
    block = message + b"\x80"
    block += b"\0" * ((56 - len(block)) % 64)
    block += bits.to_bytes(8, "big")
    assert len(block) == 64
    return block

def audit_schedules(count=1_000_000):
    x = 0x9E3779B9
    for vector in range(count):
        words = []
        for j in range(16):
            x ^= (x << 13) & MASK
            x ^= x >> 17
            x ^= (x << 5) & MASK
            words.append((x + vector + 0x7F4A7C15 * j) & MASK)
        ref = expand(words)
        got = rolled_words(words)
        for r, word, constant in got:
            if word != ref[r] or constant != K[r]:
                raise AssertionError((vector, r, word, ref[r], constant, K[r]))

def audit_hashes():
    rng = random.Random(0x514253524F4C4C34)
    messages = [b"\0"*32, b"\xff"*32, bytes(range(32)),
                b"\x02"+b"\0"*32, b"\x03"+b"\xff"*32]
    messages += [rng.randbytes(32) for _ in range(4096)]
    messages += [bytes((2 | rng.getrandbits(1),)) + rng.randbytes(32) for _ in range(4096)]
    for message in messages:
        block = padded(message)
        got = compress(block, rolled=True)
        if got != compress(block):
            raise AssertionError((message.hex(), "rolled/reference mismatch"))
        want = hashlib.sha256(message).digest()
        if got != want:
            raise AssertionError((message.hex(), got.hex(), want.hex()))

def audit_first8():
    triples = []
    for a in range(13):
        for b in range(a + 1, 13):
            for c in range(b + 1, 13):
                if (a >= 6 or (a <= 5 and b >= 7) or
                        (a == 0 and b == 1 and 8 <= c <= 10)):
                    triples.append((a, b, c))
    classes = {tuple(i for i in range(13) if i not in skip)[:6]
               for skip in triples}
    assert len(triples) == 128
    assert len(classes) == 8

if __name__ == "__main__":
    audit_schedules()
    audit_hashes()
    audit_first8()
    print("sha_roll_model: 1,000,000 schedules + 8,197 full hashes + FIRST8 exact")
