#!/usr/bin/env python3
"""Audit the streamed first-18/then-17-bit signed recoder and table map."""

from pathlib import Path
import random

N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
MASK64 = (1 << 64) - 1
MASK256 = (1 << 256) - 1
CHUNKS = 15
TOTAL = 1 << 20


def limbs(x: int) -> list[int]:
    return [(x >> (64 * i)) & MASK64 for i in range(4)]


def sub4(a: list[int], b: list[int]) -> tuple[list[int], int]:
    out, borrow = [], 0
    for x, y in zip(a, b):
        value = x - y - borrow
        out.append(value & MASK64)
        borrow = int(value < 0)
    return out, borrow


def source_setup(k: int) -> tuple[list[int], int]:
    kw, nw = limbs(k), limbs(N)
    diff, borrow = sub4(kw, nw)
    reduced = diff if borrow == 0 else kw
    doubled = [
        (reduced[0] << 1) & MASK64,
        ((reduced[1] << 1) | (reduced[0] >> 63)) & MASK64,
        ((reduced[2] << 1) | (reduced[1] >> 63)) & MASK64,
        ((reduced[3] << 1) | (reduced[2] >> 63)) & MASK64,
    ]
    carry = reduced[3] >> 63
    diff, borrow = sub4(doubled, nw)
    value = diff if carry or borrow == 0 else doubled
    odd = value[0] & 1
    negated, _ = sub4(nw, value)
    return (value if odd else negated), (1 if odd else -1)


def source_step(words: list[int], sign: int, bits: int) -> tuple[list[int], int]:
    digit = (words[0] & ((1 << (bits + 1)) - 1)) - (1 << bits)
    shift = bits + 1
    right = [
        ((words[0] >> shift) | (words[1] << (64 - shift))) & MASK64,
        ((words[1] >> shift) | (words[2] << (64 - shift))) & MASK64,
        ((words[2] >> shift) | (words[3] << (64 - shift))) & MASK64,
        words[3] >> shift,
    ]
    next_words = [
        ((right[0] << 1) | 1) & MASK64,
        ((right[1] << 1) | (right[0] >> 63)) & MASK64,
        ((right[2] << 1) | (right[1] >> 63)) & MASK64,
        ((right[3] << 1) | (right[2] >> 63)) & MASK64,
    ]
    return next_words, sign * digit


def materialized(k: int) -> list[int]:
    words, sign = source_setup(k)
    digits = []
    for bits in [18] + [17] * 13:
        words, digit = source_step(words, sign, bits)
        digits.append(digit)
    assert words[1:] == [0, 0, 0]
    digits.append(sign * words[0])
    return digits


def streamed(k: int) -> list[int]:
    # Mirrors the production seed (chunks 0/1), rolled body (2..13), and tail.
    words, sign = source_setup(k)
    out = []
    words, digit = source_step(words, sign, 18)
    out.append(digit)
    words, digit = source_step(words, sign, 17)
    out.append(digit)
    for chunk in range(2, CHUNKS):
        if chunk < CHUNKS - 1:
            words, digit = source_step(words, sign, 17)
        else:
            assert words[1:] == [0, 0, 0]
            digit = sign * words[0]
        out.append(digit)
    return out


def shift_for(chunk: int) -> int:
    return 0 if chunk == 0 else 17 * chunk + 1


def entries_for(chunk: int) -> int:
    return 1 << (17 if chunk == 0 else 16)


def offset_for(chunk: int) -> int:
    return 0 if chunk == 0 else (chunk + 1) << 16


def audit_scalar(k: int) -> None:
    old = materialized(k)
    new = streamed(k)
    assert new == old
    represented = 0
    for chunk, digit in enumerate(new):
        assert digit and digit & 1
        assert abs(digit) < (1 << (18 if chunk == 0 else 17))
        index = (abs(digit) - 1) >> 1
        assert index < entries_for(chunk)
        flat = offset_for(chunk) + index
        assert 0 <= flat < TOTAL
        represented += digit << shift_for(chunk)
    assert represented % N == 2 * (k % N) % N


def audit_table() -> None:
    for thread in range(TOTAL):
        chunk = 0 if thread < (1 << 17) else 1 + ((thread - (1 << 17)) >> 16)
        index = thread - offset_for(chunk)
        assert 0 <= chunk < CHUNKS
        assert 0 <= index < entries_for(chunk)
        assert offset_for(chunk) + index == thread
        multiplier = 2 * index + 1
        assert (multiplier & 255) & 1
        assert multiplier >> 8 < (1024 if chunk == 0 else 512)

    # Production starts its rolled loop at chunk 2 and carries a uniform flat
    # entry base instead of recomputing gt_offset(c) in every iteration.
    table_base = offset_for(2)
    for chunk in range(2, CHUNKS):
        assert table_base == offset_for(chunk)
        assert table_base + entries_for(chunk) - 1 < TOTAL
        table_base += 1 << 16
    assert table_base == TOTAL


def audit_source() -> None:
    source = Path(__file__).resolve().with_name("pinning.cu").read_text()
    required = (
        "#define GT_CHUNKS 15",
        "#define GT_TOTAL_ENTRIES (1u << 20)",
        "int32_t ec=gt_mixed_step<18>(M,sign);",
        "ec=gt_mixed_step<17>(M,sign);",
        "for (int c=2;c<GT_CHUNKS;c++)",
        "ec=(c<GT_CHUNKS-1)?gt_mixed_step<17>(M,sign):sign*(int32_t)M[0];",
        "_FixedBaseSignedXYZZScalar(qx,qy,qzz,qzzz,z,d_gt);",
        "return c == 0 ? 0 : 17*c+1;",
        "return c == 0 ? 0u : (unsigned)(c+1) << 16;",
        "uint32_t table_base=gt_offset(2);",
        "gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);",
        "table_base += 1u << 16;",
    )
    for token in required:
        assert token in source, token
    assert source.count("__device__ void _FixedBaseSignedXYZZScalar") == 1
    assert source.count("_FixedBaseSignedXYZZScalar(qx,qy,qzz,qzzz,z,d_gt);") == 1
    assert "int32_t gte[GT_CHUNKS]" not in source


def main() -> None:
    audit_source()
    audit_table()
    cases = {0, 1, 2, 3, N - 2, N - 1, N, N + 1, MASK256 - 1, MASK256}
    for bit in range(256):
        pivot = 1 << bit
        for delta in (-2, -1, 0, 1, 2):
            if 0 <= pivot + delta <= MASK256:
                cases.add(pivot + delta)
    for pivot in (N, N // 2, (N + 1) // 2):
        for delta in range(-32, 33):
            if 0 <= pivot + delta <= MASK256:
                cases.add(pivot + delta)
    rng = random.Random(0x21BDDA05A7E4)
    cases.update(rng.getrandbits(256) for _ in range(50_000))
    for scalar in cases:
        audit_scalar(scalar)
    print(
        f"PASS: streamed mixed recode equals materialized reference; "
        f"{len(cases)} scalars; {TOTAL} table slots; exact production schedule"
    )


if __name__ == "__main__":
    main()
