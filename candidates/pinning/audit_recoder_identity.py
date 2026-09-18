#!/usr/bin/env python3
"""Audit the single-shift recoder state update in gt_mixed_step.

The change under test replaces

    r = M >> (BITS+1);  M = (r << 1) | 1        (four-limb shift right, then left)

with the arithmetic identity

    2*floor(M / 2^(BITS+1)) + 1 == (M >> BITS) | 1

applied limb-wise, least-significant limb first so every read of the next
limb sees its original value (may93182 ee23cca / bb9e6d6, here ported onto the
739,010,506 frontier aeadf37d).

Three checks, all against this directory's pinning.cu:

1. old_step(M) == new_step(M) on the emitted digit and the advanced 256-bit
   state for BITS in {17, 18}, over boundary limb values and randoms.
2. The full production recode (gt_recode_setup + the gt_recode_signed
   schedule: chunk 0 at 18 bits, chunks 1..13 at 17 bits, chunk 14 = the
   residual) reconstructs 2k mod n with odd digits inside the table geometry
   for every chunk, using the NEW step -- i.e. the semantics the table layout
   depends on are unchanged.
3. Source pins: the new body is present exactly once, the old body is gone,
   the two call sites (<18>, <17>) and the schedule are the crown's.
"""

from __future__ import annotations

import random
from pathlib import Path

N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
MASK64 = (1 << 64) - 1
CHUNKS = 15
TOTAL = 1 << 20


def limbs(x: int) -> list[int]:
    return [(x >> (64 * i)) & MASK64 for i in range(4)]


def pack(ls: list[int]) -> int:
    return ls[0] | (ls[1] << 64) | (ls[2] << 128) | (ls[3] << 192)


def old_step(m: list[int], bits: int) -> tuple[int, list[int]]:
    m0, m1, m2, m3 = m
    digit = (m0 & ((1 << (bits + 1)) - 1)) - (1 << bits)
    r0 = (m0 >> (bits + 1)) | ((m1 << (63 - bits)) & MASK64)
    r1 = (m1 >> (bits + 1)) | ((m2 << (63 - bits)) & MASK64)
    r2 = (m2 >> (bits + 1)) | ((m3 << (63 - bits)) & MASK64)
    r3 = m3 >> (bits + 1)
    n0 = ((r0 << 1) | 1) & MASK64
    n1 = ((r1 << 1) | (r0 >> 63)) & MASK64
    n2 = ((r2 << 1) | (r1 >> 63)) & MASK64
    n3 = ((r3 << 1) | (r2 >> 63)) & MASK64
    return digit, [n0, n1, n2, n3]


def new_step(m: list[int], bits: int) -> tuple[int, list[int]]:
    m0, m1, m2, m3 = m
    digit = (m0 & ((1 << (bits + 1)) - 1)) - (1 << bits)
    n0 = ((m0 >> bits) | ((m1 << (64 - bits)) & MASK64) | 1) & MASK64
    n1 = ((m1 >> bits) | ((m2 << (64 - bits)) & MASK64)) & MASK64
    n2 = ((m2 >> bits) | ((m3 << (64 - bits)) & MASK64)) & MASK64
    n3 = m3 >> bits
    return digit, [n0, n1, n2, n3]


def source_setup(k: int) -> tuple[list[int], int]:
    """gt_recode_setup: k -= n once if k >= n (k < 2n always, n > 2^255);
    M = 2k mod n; made odd by negation mod n."""
    if k >= N:
        k -= N
    t = (2 * k) & ((1 << 256) - 1)
    carry = (2 * k) >> 256
    d = t - N
    ge = carry or d >= 0
    m = (d % (1 << 256)) if ge else t
    odd = m & 1
    value = m if odd else (N - m)
    return limbs(value), (1 if odd else -1)


def recode(k: int, step) -> list[int]:
    words, sign = source_setup(k)
    e = []
    d, words = step(words, 18)
    e.append(sign * d)
    for _ in range(1, CHUNKS - 1):
        d, words = step(words, 17)
        e.append(sign * d)
    assert words[1:] == [0, 0, 0]
    e.append(sign * words[0])
    return e


def entries_for(chunk: int) -> int:
    return 1 << (17 if chunk == 0 else 16)


def offset_for(chunk: int) -> int:
    return 0 if chunk == 0 else (chunk + 1) << 16


def shift_for(chunk: int) -> int:
    return 0 if chunk == 0 else 17 * chunk + 1


def step_identity_audit() -> int:
    rng = random.Random(0xE23_CCA1)
    pool = [0, 1, MASK64, MASK64 - 1, 1 << 63, (1 << 63) - 1, (1 << 17) - 1,
            (1 << 18) - 1, 1 << 17, 1 << 18, MASK64 ^ ((1 << 18) - 1)]
    cases = 0
    for _ in range(120_000):
        m = [pool[rng.randrange(len(pool))] if rng.random() < 0.15
             else rng.getrandbits(64) for _ in range(4)]
        for bits in (17, 18):
            assert old_step(m, bits) == new_step(m, bits), (m, bits)
            cases += 1
    return cases


def production_audit() -> int:
    cases = {0, 1, 2, 3, N - 2, N - 1, N, N + 1, (1 << 256) - 2, (1 << 256) - 1}
    for bit in range(256):
        for delta in (-2, -1, 0, 1, 2):
            v = (1 << bit) + delta
            if 0 <= v < (1 << 256):
                cases.add(v)
    for pivot in (N, N // 2, (N + 1) // 2):
        for delta in range(-32, 33):
            cases.add(pivot + delta)
    rng = random.Random(0x21BDDA05A7E4)
    cases.update(rng.getrandbits(256) for _ in range(40_000))
    for k in cases:
        old = recode(k, old_step)
        new = recode(k, new_step)
        assert new == old, k
        represented = 0
        for chunk, digit in enumerate(new):
            assert digit and digit & 1, (k, chunk, digit)
            assert abs(digit) < (1 << (18 if chunk == 0 else 17)), (k, chunk)
            index = (abs(digit) - 1) >> 1
            assert index < entries_for(chunk)
            assert 0 <= offset_for(chunk) + index < TOTAL
            represented += digit << shift_for(chunk)
        assert represented % N == (2 * k) % N, k
    return len(cases)


def source_audit() -> None:
    src = Path(__file__).resolve().with_name("pinning.cu").read_text()
    new_body = (
        "    M[0]=(M[0]>>BITS)|(M[1]<<(64-BITS))|1ULL;\n"
        "    M[1]=(M[1]>>BITS)|(M[2]<<(64-BITS));\n"
        "    M[2]=(M[2]>>BITS)|(M[3]<<(64-BITS));\n"
        "    M[3]>>=BITS;\n"
        "    return sign*digit;\n"
    )
    assert src.count(new_body) == 1
    assert "uint64_t r0=(M[0]>>(BITS+1))" not in src
    assert "M[0]=(r0<<1)|1ULL;" not in src
    assert src.count("int32_t digit=(int32_t)(M[0]&((1u<<(BITS+1))-1))-(1<<BITS);") == 1
    assert src.count("template<int BITS>\n__device__ __forceinline__ int32_t gt_mixed_step(uint64_t M[4], int sign) {") == 1
    schedule = (
        "    e[0]=gt_mixed_step<18>(M,sign);\n"
        "    #pragma unroll\n"
        "    for(int c=1;c<GT_CHUNKS-1;c++)e[c]=gt_mixed_step<17>(M,sign);\n"
        "    e[GT_CHUNKS-1]=sign*(int32_t)M[0];\n"
    )
    assert src.count(schedule) == 1
    assert src.count("gt_mixed_step<") == 2
    for token in (
        "#define GT_CHUNKS 15",
        "#define GT_TOTAL_ENTRIES (1u << 20)",
        "return c == 0 ? (1u << 17) : (1u << 16);",
        "return c == 0 ? 0u : (unsigned)(c+1) << 16;",
        "return c == 0 ? 0 : 17*c+1;",
        "*idx = (ae - 1) >> 1;",
    ):
        assert token in src, token


def main() -> None:
    source_audit()
    steps = step_identity_audit()
    scalars = production_audit()
    print(f"PASS: recoder identity holds on {steps} step cases (BITS 17/18); "
          f"{scalars} scalars recode identically under old/new step and "
          f"reconstruct 2k mod n inside the table geometry; source pins bound")


if __name__ == "__main__":
    main()
