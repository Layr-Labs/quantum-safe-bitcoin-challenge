#!/usr/bin/env python3
"""CPU-only contract and equivalence audit for the ranked paired SHA path."""

from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TREE = ROOT / "tests" / "gpu_epochs" / "tree.cu"
MASK = 0xFFFFFFFF
IV = (
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
EXPECTED_HELPER_SHA256 = "e8f0290217b78b887c55bd3bfcebdee95a2c88dd9341333e66c7dba269badf8b"
EXPECTED_9C_TREE_SHA256 = "a58c506a49aa24740c3e48846afd9f64f665275137299f1ac2b95cd35de268ec"

def ror(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & MASK


def schedule(block: bytes) -> list[int]:
    assert len(block) == 64
    w = [int.from_bytes(block[i:i + 4], "big") for i in range(0, 64, 4)]
    for i in range(16, 64):
        s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3)
        s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10)
        w.append((w[i - 16] + s0 + w[i - 7] + s1) & MASK)
    return w


def round_step(state: list[int], word: int, constant: int) -> list[int]:
    a, b, c, d, e, f, g, h = state
    big1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)
    choose = (e & f) ^ ((~e) & g)
    t1 = (h + big1 + choose + constant + word) & MASK
    big0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)
    majority = (a & b) ^ (a & c) ^ (b & c)
    t2 = (big0 + majority) & MASK
    return [(t1 + t2) & MASK, a, b, c, (d + t1) & MASK, e, f, g]


def compress(block: bytes) -> tuple[int, ...]:
    state = list(IV)
    for word, constant in zip(schedule(block), K):
        state = round_step(state, word, constant)
    return tuple((x + y) & MASK for x, y in zip(IV, state))


def compress_pair(block0: bytes, block1: bytes) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Independent loop model of the two interleaved, data-independent lanes."""
    states = [list(IV), list(IV)]
    words = [schedule(block0), schedule(block1)]
    for i, constant in enumerate(K):
        states[0] = round_step(states[0], words[0][i], constant)
        states[1] = round_step(states[1], words[1][i], constant)
    return tuple(
        tuple((x + y) & MASK for x, y in zip(IV, state))
        for state in states
    )


def byte_perm(x: int, y: int, selector: int) -> int:
    source = [(x >> (8 * i)) & 0xFF for i in range(4)]
    source += [(y >> (8 * i)) & 0xFF for i in range(4)]
    out = 0
    for i in range(4):
        out |= source[(selector >> (4 * i)) & 7] << (8 * i)
    return out


def packed_block(x: int, parity: int) -> bytes:
    limbs = [(x >> (64 * i)) & ((1 << 64) - 1) for i in range(4)]
    x32: list[int] = []
    for limb in limbs:
        x32.extend((limb & MASK, limb >> 32))
    words = [0] * 16
    words[0] = byte_perm(x32[7], 2 + parity, 0x4321)
    for j in range(1, 8):
        words[j] = byte_perm(x32[8 - j], x32[7 - j], 0x0765)
    words[8] = byte_perm(x32[0], 0x80, 0x0456)
    words[15] = 0x108
    return b"".join(word.to_bytes(4, "big") for word in words)


def digest_words(state: tuple[int, ...]) -> bytes:
    return b"".join(word.to_bytes(4, "big") for word in state)


def source_contract(source: str) -> None:
    start = source.index("/* Fused dual SHA-256")
    end = source.index("__device__ __constant__ uint64_t QSB_U2R[8];", start)
    helper = source[start:end].rstrip()
    assert hashlib.sha256(helper.encode()).hexdigest() == EXPECTED_HELPER_SHA256
    assert source.count("qsb_sha256_dual_recid(hs0,hs1,w0,w1);") == 1
    assert "const int easy_flag=0,single_hash_flag=1,calibrate_flag=0;" in source
    assert "if (single_hash_flag && !easy_flag && !calibrate_flag) {" in source
    assert "for(int ri=0;ri<2&&!v;ri++)" in source
    assert "_SHA256Transform(hs,pb);" in source
    assert source.count("QSB_PAIRED_RANKED_BEGIN") == 1
    assert source.count("QSB_PAIRED_GENERIC_FALLBACK_BEGIN") == 1
    assert "cudaStream" not in source

    # Mechanically remove the exact paired helper and unwrap the marked generic
    # fallback. The remaining production tree must be byte-identical to PR77.
    reconstructed = source[:start] + source[end:]
    pattern = re.compile(
        r"    /\* QSB_PAIRED_RANKED_BEGIN:.*?"
        r"    /\* QSB_PAIRED_GENERIC_FALLBACK_BEGIN \*/\n"
        r"(?P<fallback>.*?)"
        r"    /\* QSB_PAIRED_GENERIC_FALLBACK_END \*/\n"
        r"    }\n"
        r"    /\* QSB_PAIRED_RANKED_END \*/",
        re.S,
    )
    match = pattern.search(reconstructed)
    assert match is not None
    reconstructed = pattern.sub(match.group("fallback").rstrip("\n"), reconstructed, count=1)
    assert hashlib.sha256(reconstructed.encode()).hexdigest() == EXPECTED_9C_TREE_SHA256

def main() -> None:
    source = TREE.read_text()
    source_contract(source)
    rng = random.Random(0x5153425F50414952)
    cases = [
        (0, 0, 0, 1),
        (1, 1, (1 << 256) - 1, 0),
        ((1 << 255), 0, (1 << 128) + 17, 1),
    ]
    cases.extend((rng.getrandbits(256), rng.randrange(2),
                  rng.getrandbits(256), rng.randrange(2)) for _ in range(4096))
    for x0, p0, x1, p1 in cases:
        blocks = (packed_block(x0, p0), packed_block(x1, p1))
        messages = (
            bytes((2 + p0,)) + x0.to_bytes(32, "big"),
            bytes((2 + p1,)) + x1.to_bytes(32, "big"),
        )
        for block, message in zip(blocks, messages):
            expected = message + b"\x80" + b"\x00" * 22 + (264).to_bytes(8, "big")
            assert block == expected
        sequential = (compress(blocks[0]), compress(blocks[1]))
        assert compress_pair(*blocks) == sequential
        assert digest_words(sequential[0]) == hashlib.sha256(messages[0]).digest()
        assert digest_words(sequential[1]) == hashlib.sha256(messages[1]).digest()
    print({
        "status": "PASS",
        "source_helper_sha256": EXPECTED_HELPER_SHA256,
        "paired_cases": len(cases),
        "hashes_checked": 2 * len(cases),
        "ranked_constant_specialization": True,
        "generic_fallback_present": True,
        "reconstructed_9c_tree_sha256": EXPECTED_9C_TREE_SHA256,
        "double_stream_present": False,
    })


if __name__ == "__main__":
    main()
