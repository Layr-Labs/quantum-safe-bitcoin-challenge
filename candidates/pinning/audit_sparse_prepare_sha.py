#!/usr/bin/env python3
"""Bit-exact audit of the sparse prepare-stage SHA message schedules."""

import random
from pathlib import Path


MASK = 0xFFFFFFFF


def ror(x, n):
    return ((x >> n) | (x << (32 - n))) & MASK


def s0(x):
    return ror(x, 7) ^ ror(x, 18) ^ (x >> 3)


def s1(x):
    return ror(x, 17) ^ ror(x, 19) ^ (x >> 10)


def generic_first_mix(words):
    w = list(words)
    for i in range(16):
        w[i] = (
            w[i]
            + s1(w[(i + 14) & 15])
            + w[(i + 9) & 15]
            + s0(w[(i + 1) & 15])
        ) & MASK
    return w


def tail11_first_mix(w0, w1, w2):
    w = [w0, w1, w2] + [0] * 12 + [79960]
    w[0] = (w[0] + s0(w[1])) & MASK
    w[1] = (w[1] + s1(79960) + s0(w[2])) & MASK
    w[2] = (w[2] + s1(w[0])) & MASK
    w[3] = s1(w[1])
    w[4] = s1(w[2])
    w[5] = s1(w[3])
    w[6] = (s1(w[4]) + 79960) & MASK
    w[7] = (s1(w[5]) + w[0]) & MASK
    w[8] = (s1(w[6]) + w[1]) & MASK
    w[9] = (s1(w[7]) + w[2]) & MASK
    w[10] = (s1(w[8]) + w[3]) & MASK
    w[11] = (s1(w[9]) + w[4]) & MASK
    w[12] = (s1(w[10]) + w[5]) & MASK
    w[13] = (s1(w[11]) + w[6]) & MASK
    w[14] = (s1(w[12]) + w[7] + s0(79960)) & MASK
    w[15] = (79960 + s1(w[13]) + w[8] + s0(w[0])) & MASK
    return w


def digest32_first_mix(first_eight):
    w = list(first_eight) + [0x80000000] + [0] * 6 + [256]
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
    return w


def audit_source():
    source = Path(__file__).with_name("pinning.cu").read_text()
    assert source.count("void _SHA256TransformTail11(") == 1
    assert source.count("void _SHA256TransformDigest32(") == 1
    assert source.count("_SHA256TransformTail11(state,blk);") == 1
    assert source.count("_SHA256TransformDigest32(s2,b2);") == 1
    tail = source.split("void _SHA256TransformTail11(", 1)[1].split(
        "void _SHA256TransformDigest32(", 1
    )[0]
    digest = source.split("void _SHA256TransformDigest32(", 1)[1].split(
        "template<bool FAST_TAIL", 1
    )[0]
    assert tail.count("S2Round(") == 16
    assert digest.count("S2Round(") == 16
    assert tail.count("SHA256_RND(") == digest.count("SHA256_RND(") == 3
    assert tail.count("WMIX();") == digest.count("WMIX();") == 2
    assert "K[15], 79960u" in tail
    assert "K[8], 0x80000000u" in digest
    assert "K[15], 256u" in digest


def main():
    audit_source()
    rng = random.Random(0x5350415253455348)
    cases = [0, 1, MASK, 0x80000000, 0xDEADBEEF]
    tail_cases = 0
    digest_cases = 0
    for _ in range(100_000):
        first = [rng.getrandbits(32) for _ in range(8)]
        if tail_cases < len(cases):
            first[0] = cases[tail_cases]
        tail_block = first[:3] + [0] * 12 + [79960]
        assert tail11_first_mix(*first[:3]) == generic_first_mix(tail_block)
        tail_cases += 1
        digest_block = first + [0x80000000] + [0] * 6 + [256]
        assert digest32_first_mix(first) == generic_first_mix(digest_block)
        digest_cases += 1
    print(
        f"PASS: sparse Tail11 and Digest32 schedules match generic WMIX in "
        f"{tail_cases + digest_cases} randomized/boundary cases; source calls bound"
    )


if __name__ == "__main__":
    main()
