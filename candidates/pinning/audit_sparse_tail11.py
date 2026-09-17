#!/usr/bin/env python3
"""Audit the sparse SHA-256 compressor for the canonical 11-byte fast tail.

The exact d752 parent pinning.cu SHA256 was
d7269d78b625f084a8c0d3118611a365b0cfe5cd829df0266465175938ebebbf.
"""

import hashlib
import random
from pathlib import Path

MASK = (1 << 32) - 1
TOTAL_BYTES = 9995
PREFIX_BYTES = TOTAL_BYTES - 11
TOTAL_BITS = TOTAL_BYTES * 8
INITIAL = (
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
)
K = (
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
)


def ror(value, count):
    return ((value >> count) | (value << (32 - count))) & MASK


def small0(value):
    return ror(value, 7) ^ ror(value, 18) ^ (value >> 3)


def small1(value):
    return ror(value, 17) ^ ror(value, 19) ^ (value >> 10)


def big0(value):
    return ror(value, 2) ^ ror(value, 13) ^ ror(value, 22)


def big1(value):
    return ror(value, 6) ^ ror(value, 11) ^ ror(value, 25)


def ring_mix(words):
    words = list(words)
    for index in range(16):
        words[index] = (
            words[index]
            + small1(words[(index + 14) & 15])
            + words[(index + 9) & 15]
            + small0(words[(index + 1) & 15])
        ) & MASK
    return words


def sparse_first_mix(live):
    w = list(live) + [0] * 12 + [TOTAL_BITS]
    w[0] = (w[0] + small0(w[1])) & MASK
    w[1] = (w[1] + small1(TOTAL_BITS) + small0(w[2])) & MASK
    w[2] = (w[2] + small1(w[0])) & MASK
    w[3] = small1(w[1])
    w[4] = small1(w[2])
    w[5] = small1(w[3])
    w[6] = (small1(w[4]) + TOTAL_BITS) & MASK
    w[7] = (small1(w[5]) + w[0]) & MASK
    w[8] = (small1(w[6]) + w[1]) & MASK
    w[9] = (small1(w[7]) + w[2]) & MASK
    w[10] = (small1(w[8]) + w[3]) & MASK
    w[11] = (small1(w[9]) + w[4]) & MASK
    w[12] = (small1(w[10]) + w[5]) & MASK
    w[13] = (small1(w[11]) + w[6]) & MASK
    w[14] = (small1(w[12]) + w[7] + small0(TOTAL_BITS)) & MASK
    w[15] = (w[15] + small1(w[13]) + w[8] + small0(w[0])) & MASK
    return w


def schedule_generic(live):
    initial = list(live) + [0] * 12 + [TOTAL_BITS]
    windows = [initial]
    for _ in range(3):
        windows.append(ring_mix(windows[-1]))
    return [word for window in windows for word in window]


def schedule_sparse(live):
    initial = list(live) + [0] * 12 + [TOTAL_BITS]
    windows = [initial, sparse_first_mix(live)]
    windows.append(ring_mix(windows[-1]))
    windows.append(ring_mix(windows[-1]))
    return [word for window in windows for word in window]


def compress(state, schedule):
    a, b, c, d, e, f, g, h = state
    for index, word in enumerate(schedule):
        choose = (e & f) ^ ((~e) & g)
        majority = (a & b) ^ (a & c) ^ (b & c)
        t1 = (h + big1(e) + choose + K[index] + word) & MASK
        t2 = (big0(a) + majority) & MASK
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & MASK, c, b, a, (t1 + t2) & MASK
    return tuple((left + right) & MASK for left, right in zip(state, (a, b, c, d, e, f, g, h)))


def prefix_midstate(prefix):
    assert len(prefix) == PREFIX_BYTES and len(prefix) % 64 == 0
    state = INITIAL
    for offset in range(0, len(prefix), 64):
        block = prefix[offset:offset + 64]
        words = [int.from_bytes(block[i:i + 4], "big") for i in range(0, 64, 4)]
        schedule = list(words)
        for index in range(16, 64):
            schedule.append((small1(schedule[index - 2]) + schedule[index - 7]
                             + small0(schedule[index - 15]) + schedule[index - 16]) & MASK)
        state = compress(state, schedule)
    return state


def pack_tail(fixed_tail, locktime):
    assert len(fixed_tail) == 11
    w0 = (fixed_tail[0] << 24) | (fixed_tail[1] << 16) | (fixed_tail[2] << 8)
    w0 |= locktime & 0xFF
    w1 = ((locktime & 0xFF00) << 16) | (locktime & 0xFF0000)
    w1 |= ((locktime >> 16) & 0xFF00) | fixed_tail[7]
    w2 = (fixed_tail[8] << 24) | (fixed_tail[9] << 16) | (fixed_tail[10] << 8) | 0x80
    return [w0, w1, w2]


def audit_source():
    source = Path(__file__).with_name("pinning.cu").read_text()
    assert source.count("__device__ __forceinline__ void _SHA256TransformFastTail11") == 1
    assert source.count("_SHA256TransformFastTail11(state,blk);") == 1
    fast = source[source.index("if (FAST_TAIL) {"):source.index("} else {", source.index("if (FAST_TAIL) {"))]
    assert "uint32_t blk[3]" in fast
    assert "_SHA256Transform(state,blk)" not in fast.replace(" ", "")
    assert "w[15]=9995u*8u;" in source.replace(" ", "")
    assert "_SHA256TransformSha256d32" not in source
    assert source.count("uint32_t b2[16]") == 1
    assert source.replace(" ", "").count("_SHA256Transform(s2,b2);") == 1
    assert source.count("_SHA256TransformPk33(hs,pb)") == 1


def main():
    audit_source()
    rng = random.Random(0x5441494C3131)
    for _ in range(100_000):
        live = [rng.getrandbits(32) for _ in range(3)]
        initial = live + [0] * 12 + [TOTAL_BITS]
        assert sparse_first_mix(live) == ring_mix(initial)

    full_cases = 0
    boundaries = [0, 1, 255, 256, 65535, 65536, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]
    for seed in range(16):
        prefix = rng.randbytes(PREFIX_BYTES)
        state = prefix_midstate(prefix)
        fixed = bytearray(rng.randbytes(11))
        for case in range(625):
            locktime = boundaries[case] if case < len(boundaries) else rng.getrandbits(32)
            tail = bytearray(fixed)
            tail[3:7] = locktime.to_bytes(4, "little")
            live = pack_tail(fixed, locktime)
            block = bytes(tail) + b"\x80" + b"\0" * 44 + TOTAL_BITS.to_bytes(8, "big")
            packed = [int.from_bytes(block[i:i + 4], "big") for i in range(0, 64, 4)]
            assert packed == live + [0] * 12 + [TOTAL_BITS]
            generic = compress(state, schedule_generic(live))
            specialized = compress(state, schedule_sparse(live))
            assert specialized == generic
            digest = b"".join(word.to_bytes(4, "big") for word in specialized)
            assert digest == hashlib.sha256(prefix + tail).digest()
            full_cases += 1

    print("PASS: 100000 generic/sparse first-WMIX comparisons; "
          f"{full_cases} full 9995-byte SHA/hashlib and tail-packing comparisons")


if __name__ == "__main__":
    main()
