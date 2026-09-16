#!/usr/bin/env python3
"""Audit the prepare-stage SHA256d specialization for 32-byte digests.

Provenance checked while preparing this research tree:
  exact 73/PR71 head: 8546da17354ff144b8158d0e0e4ead1d0678da9c
  exact 73 pinning.cu SHA256: 41819b769d97efa092c575cacc887eb2e7667481f5e578aebefef5a0e9c9c776
  abf/PR87 head: de4abe425f7f96ce7a2bc81ea0be2eadfd13f130
  abf full pinning.cu SHA256: 3299ee361a64b7e609d75ae02e8aae3075b006be6efa0fd52e298d319a2d8cb0
"""

import hashlib
import random
from pathlib import Path

MASK = (1 << 32) - 1
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


def generic_first_mix(words):
    words = list(words)
    for index in range(16):
        words[index] = (
            words[index]
            + small1(words[(index + 14) & 15])
            + words[(index + 9) & 15]
            + small0(words[(index + 1) & 15])
        ) & MASK
    return words


def sparse_first_mix(data_words):
    w = list(data_words) + [0x80000000] + [0] * 6 + [256]
    w[0] = (w[0] + small0(w[1])) & MASK
    w[1] = (w[1] + small1(256) + small0(w[2])) & MASK
    w[2] = (w[2] + small1(w[0]) + small0(w[3])) & MASK
    w[3] = (w[3] + small1(w[1]) + small0(w[4])) & MASK
    w[4] = (w[4] + small1(w[2]) + small0(w[5])) & MASK
    w[5] = (w[5] + small1(w[3]) + small0(w[6])) & MASK
    w[6] = (w[6] + small1(w[4]) + 256 + small0(w[7])) & MASK
    w[7] = (w[7] + small1(w[5]) + w[0] + small0(0x80000000)) & MASK
    w[8] = (w[8] + small1(w[6]) + w[1]) & MASK
    w[9] = (small1(w[7]) + w[2]) & MASK
    w[10] = (small1(w[8]) + w[3]) & MASK
    w[11] = (small1(w[9]) + w[4]) & MASK
    w[12] = (small1(w[10]) + w[5]) & MASK
    w[13] = (small1(w[11]) + w[6]) & MASK
    w[14] = (small1(w[12]) + w[7] + small0(256)) & MASK
    w[15] = (w[15] + small1(w[13]) + w[8] + small0(w[0])) & MASK
    return w


def sha256_block(message):
    padded = message + b"\x80" + b"\0" * 23 + (256).to_bytes(8, "big")
    words = [int.from_bytes(padded[i:i + 4], "big") for i in range(0, 64, 4)]
    assert words[8] == 0x80000000 and words[9:15] == [0] * 6 and words[15] == 256
    for index in range(16, 64):
        words.append((small1(words[index - 2]) + words[index - 7]
                      + small0(words[index - 15]) + words[index - 16]) & MASK)
    a, b, c, d, e, f, g, h = INITIAL
    for index in range(64):
        choose = (e & f) ^ ((~e) & g)
        majority = (a & b) ^ (a & c) ^ (b & c)
        t1 = (h + big1(e) + choose + K[index] + words[index]) & MASK
        t2 = (big0(a) + majority) & MASK
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & MASK, c, b, a, (t1 + t2) & MASK
    state = tuple((left + right) & MASK for left, right in zip(INITIAL, (a, b, c, d, e, f, g, h)))
    return b"".join(word.to_bytes(4, "big") for word in state)


def audit_source():
    source = Path(__file__).with_name("pinning.cu").read_text()
    assert source.count("__device__ __forceinline__ void _SHA256TransformSha256d32") == 1
    assert source.count("_SHA256TransformSha256d32(s2, state);") == 1
    assert "uint32_t b2[16]" not in source
    assert "_SHA256Transform(s2,b2)" not in source.replace(" ", "")
    assert "w[8] = 0x80000000u;" in source
    assert "w[15] = 256u;" in source
    # Preserve the independent finish-stage compressed-pubkey specialization.
    assert source.count("__device__ __forceinline__ void _SHA256TransformPk33") == 1
    assert source.count("_SHA256TransformPk33(hs,pb)") == 1


def main():
    audit_source()
    rng = random.Random(0x5348413235364432)
    for index in range(100_000):
        data_words = [rng.getrandbits(32) for _ in range(8)]
        expected_words = data_words + [0x80000000] + [0] * 6 + [256]
        assert sparse_first_mix(data_words) == generic_first_mix(expected_words)
        if index < 10_000:
            message = rng.randbytes(32)
            assert sha256_block(message) == hashlib.sha256(message).digest()
    print("PASS: 100000 in-place WMIX and 10000 full 32-byte SHA comparisons")


if __name__ == "__main__":
    main()
