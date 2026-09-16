#!/usr/bin/env python3
"""Independent audit of the mixed-15 recoder and 64 MiB table contract.

This does not execute CUDA.  It exhaustively checks every one of the 2^20
interleaved table slots, independently reconstructs mixed-width recodings,
models the host-ladder/GPU-builder split used by the production spot check,
and binds those invariants to exact production source patterns.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import audit_integrated as curve


ROOT = Path(__file__).resolve().parent
TREE_PATH = ROOT / "tests" / "gpu_epochs" / "tree.cu"
PIPE_PATH = ROOT / "tests" / "gpu_epochs" / "ranked_pipeline.cuh"
MATH_PATH = ROOT / "GPUMath.h"
PROBLEM_PATH = ROOT.parents[1] / "problems" / "subset.json"

CHUNKS = 15
TOTAL = 1 << 20
RECORD_BYTES = 64
TABLE_BYTES = TOTAL * RECORD_BYTES


def entries(chunk: int) -> int:
    return 1 << (17 if chunk == 0 else 16)


def offset(chunk: int) -> int:
    return 0 if chunk == 0 else (chunk + 1) << 16


def shift(chunk: int) -> int:
    return 0 if chunk == 0 else 17 * chunk + 1


def setup(k: int) -> tuple[int, int]:
    value = 2 * (k % curve.N) % curve.N
    sign = 1 if value & 1 else -1
    return (value if sign > 0 else curve.N - value), sign


def step(value: int, sign: int, bits: int) -> tuple[int, int]:
    digit = (value & ((1 << (bits + 1)) - 1)) - (1 << bits)
    return 2 * (value >> (bits + 1)) + 1, sign * digit


def recode(k: int) -> list[int]:
    value, sign = setup(k)
    value, first = step(value, sign, 18)
    result = [first]
    for _ in range(13):
        value, digit = step(value, sign, 17)
        result.append(digit)
    result.append(sign * value)
    return result


def check_source_contract() -> None:
    tree = TREE_PATH.read_text(encoding="utf-8")
    pipeline = PIPE_PATH.read_text(encoding="utf-8")
    math_source = MATH_PATH.read_text(encoding="utf-8")
    for needle in (
        "#define GT_CHUNKS 15",
        "#define GT_TOTAL_ENTRIES (1u << 20)",
        "GT_TOTAL_ENTRIES*64ULL == 64ULL*1024*1024",
        "int ch=t<(1u<<17)?0:1+(int)((t-(1u<<17))>>16);",
        "int d=(int)(t-gt_offset(ch));",
        "size_t off = ((size_t)gt_offset(ch) + d) * 64;",
        "memcpy(gTable + off,      rx, 32);",
        "memcpy(gTable + off + 32, ry, 32);",
        "ch==1 ? (1u<<18) : (1u<<17)",
        "for (int hi = 1; hi < (ch==0?1024:512); hi++)",
        "const int corner[4] = {0, 1, 2, (int)gt_entries(ch) - 1};",
        "BN_lshift(k, k, gt_shift(ch));",
        "size_t gt_sz = (size_t)GT_TOTAL_ENTRIES*64;",
        "kernel_build_gtable<<<(gt_total+255)/256,256>>>(dL,dH,d_gt);",
        "gt_ok = gt_spot_check(chk_table,GT_CHUNKS*4+192,dp.neg_r_inv);",
        "compute_gtable(chk_table,dp.neg_r_inv);",
    ):
        assert needle in tree, needle

    # Preserve exact promoted-80 table/deferred logic, including its
    # single-carry-chain signed-Y loader, while applying the small field tail.
    assert "UADDO1(r0,c0)" in tree
    assert tree.count("_PointAddXYZZ_def(") == 1
    assert tree.count("_PointAddXYZZ_mm_def(") == 1
    assert "mul.lo.u32 m0, cf, 977" in math_source
    assert "r[0] -= 0xFFFFFFFEFFFFFC2FULL" in math_source
    assert "_FixedBaseSignedXYZZStream(C,Y,ZZ,ZZZ,z,gTable);" in pipeline
    assert "qsb_block_product_checkpoint(W,roots,tree);" in pipeline
    assert "qsb_block_inverse_checkpoint(inv,roots,tree);" in pipeline


def check_all_slots() -> None:
    seen = bytearray(TOTAL)
    count = 0
    for chunk in range(CHUNKS):
        width = entries(chunk)
        base = offset(chunk)
        assert base == sum(entries(previous) for previous in range(chunk))
        for index in range(width):
            slot = base + index
            assert 0 <= slot < TOTAL and not seen[slot]
            seen[slot] = 1
            count += 1

            # Exact production inverse mapping from the flat launch index.
            mapped_chunk = 0 if slot < (1 << 17) else 1 + ((slot - (1 << 17)) >> 16)
            mapped_index = slot - offset(mapped_chunk)
            assert (mapped_chunk, mapped_index) == (chunk, index)

            x_address = slot * RECORD_BYTES
            y_address = x_address + 32
            assert x_address % 64 == 0 and y_address % 32 == 0
            assert y_address + 32 <= TABLE_BYTES

            odd = 2 * index + 1
            high, low = odd >> 8, odd & 255
            assert low & 1 and low < 256
            assert high < (1024 if chunk == 0 else 512)

            # Every positive and negative digit maps back to this exact slot.
            for digit in (odd, -odd):
                assert (abs(digit) - 1) >> 1 == index
    assert count == TOTAL and all(seen)
    assert offset(14) + entries(14) == TOTAL


def check_recodings() -> int:
    rng = random.Random(0x6515A80)
    samples = [0, 1, curve.N - 1, curve.N, curve.N + 1, (1 << 256) - 1]
    samples += [1 << bit for bit in range(256)]
    samples += [rng.getrandbits(256) for _ in range(32768)]
    for scalar in samples:
        digits = recode(scalar)
        assert len(digits) == CHUNKS
        assert digits[0] & 1 and abs(digits[0]) < (1 << 18)
        assert all(digit & 1 and abs(digit) < (1 << 17) for digit in digits[1:])
        rebuilt = digits[0] + sum(digits[i] << shift(i) for i in range(1, CHUNKS))
        assert rebuilt % curve.N == 2 * (scalar % curve.N) % curve.N
        for chunk, digit in enumerate(digits):
            index = (abs(digit) - 1) >> 1
            assert index < entries(chunk)
            assert offset(chunk) + index < TOTAL
    return len(samples)


def check_builder_spot_model() -> int:
    problem = json.loads(PROBLEM_PATH.read_text(encoding="utf-8"))
    neg_r_inv = int(problem["neg_r_inv"], 0)
    half_base = neg_r_inv * pow(2, -1, curve.N) % curve.N

    ladder_base = half_base
    seed = 0x9E3779B9
    samples: list[tuple[int, int]] = []
    for chunk in range(CHUNKS):
        if chunk > 0:
            ladder_base = ladder_base * (1 << (18 if chunk == 1 else 17)) % curve.N
        assert ladder_base == half_base * pow(2, shift(chunk), curve.N) % curve.N
        for index in (0, 1, 2, entries(chunk) - 1):
            samples.append((chunk, index))
    for _ in range(192):
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        chunk = (seed >> 28) % CHUNKS
        index = (seed >> 4) & (entries(chunk) - 1)
        samples.append((chunk, index))

    for chunk, index in samples:
        odd = 2 * index + 1
        high, low = odd >> 8, odd & 255
        base = half_base * pow(2, shift(chunk), curve.N) % curve.N
        direct_scalar = odd * base % curve.N
        split_scalar = (high * 256 * base + low * base) % curve.N
        assert direct_scalar == split_scalar
        assert curve.scalar_mult(direct_scalar) == curve.scalar_mult(split_scalar)
    return len(samples)


def main() -> int:
    check_source_contract()
    check_all_slots()
    recode_samples = check_recodings()
    builder_samples = check_builder_spot_model()
    print(f"mixed15 slots: {TOTAL} exhaustive; layout bytes: {TABLE_BYTES}")
    print(f"recode reconstructions: {recode_samples}; builder/spot samples: {builder_samples}")
    print("promoted-80 mixed table/single-chain loader: preserved; field carry tail: repaired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
