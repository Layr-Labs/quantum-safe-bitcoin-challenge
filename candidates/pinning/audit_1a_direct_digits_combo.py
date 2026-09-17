#!/usr/bin/env python3
"""Bind exact 1a structure and prove direct digits equal serial recoding."""

import hashlib
import random
from pathlib import Path


ONE_A_HEAD = "5c85ae053bc0effa27db4df76fcbf09ee4aaa1b2"
DIRECT_SOURCE_HEAD = "eeaf084f9d28f69fb7772ec957c857aafe295507"
ONE_A_GPUMATH_SHA256 = "835b061d1a0b158778c0b576ebce23a9f1663f616a4a292ed7afe103ff6afecd"
FINAL_PINNING_SHA256 = "d6a115b7b07632bae9e66c3aac7cfc9ee12ea237453bb0727a5412ce9b002e9f"
MASK256 = (1 << 256) - 1
CHUNKS = 15


def digest(data):
    return hashlib.sha256(data).hexdigest()


def table_pair(digit):
    assert digit and digit & 1
    return ((abs(digit) - 1) // 2, int(digit < 0))


def serial_pairs(value, sign):
    """Mirror gt_mixed_step<18>, then thirteen <17> steps and the tail."""
    state = value
    out = []
    digit = (state & ((1 << 19) - 1)) - (1 << 18)
    out.append(table_pair(sign * digit))
    state = (state >> 18) | 1
    for _ in range(1, CHUNKS - 1):
        digit = (state & ((1 << 18) - 1)) - (1 << 17)
        out.append(table_pair(sign * digit))
        state = (state >> 17) | 1
    out.append(table_pair(sign * state))
    return out


def direct_pairs(value, sign):
    """Mirror the source's bit fields, unsigned masks and global sign xor."""
    sflag = int(sign < 0)
    out = []

    field = (value >> 1) & 0x3FFFF
    top = field >> 17
    index = (field ^ ((top - 1) & 0xFFFFFFFF)) & 0x1FFFF
    out.append((index, (top ^ 1) ^ sflag))

    field = (value >> 19) & 0x1FFFF
    top = field >> 16
    index = (field ^ ((top - 1) & 0xFFFFFFFF)) & 0xFFFF
    out.append((index, (top ^ 1) ^ sflag))

    pos = 36
    for chunk in range(2, CHUNKS):
        field = (value >> pos) & 0x1FFFF
        top = field >> 16
        last = chunk == CHUNKS - 1
        index = field & 0xFFFF if last else (
            field ^ ((top - 1) & 0xFFFFFFFF)
        ) & 0xFFFF
        negative = (0 if last else top ^ 1) ^ sflag
        out.append((index, negative))
        pos += 17
    return out


def audit_math():
    fixed = [
        1,
        3,
        MASK256,
        MASK256 - 2,
        (1 << 255) | 1,
        (1 << 254) | 1,
    ]
    for pos in (1, 18, 19, 35, 36, 52, 53, 63, 64, 69, 127, 128, 239, 240, 255):
        fixed.extend(((1 << pos) | 1, ((1 << pos) - 1) | 1))

    rng = random.Random(0x1A856D16175)
    values = fixed + [(rng.getrandbits(256) | 1) for _ in range(200_000)]
    checked = 0
    for value in values:
        value &= MASK256
        value |= 1
        for sign in (-1, 1):
            serial = serial_pairs(value, sign)
            direct = direct_pairs(value, sign)
            assert direct == serial, (hex(value), sign, direct, serial)
            assert len(direct) == CHUNKS
            assert direct[0][0] < (1 << 17)
            assert all(index < (1 << 16) for index, _ in direct[1:])
            checked += 1
    return checked


def audit_source():
    root = Path(__file__).resolve().parent
    gpu_math = (root / "GPUMath.h").read_bytes()
    source = (root / "pinning.cu").read_text()
    assert digest(gpu_math) == ONE_A_GPUMATH_SHA256
    assert digest(source.encode()) == FINAL_PINNING_SHA256

    required = (
        "#define QSB_TREE_N 128",
        "#define ZLAB_DIRECT_DIGITS 1",
        "zlab_field_bits",
        "uint64_t sflag = (uint64_t)(sign < 0);",
        "unsigned pos = 36u;",
        "idx = last ? (f & 0xffffu)",
        "gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);",
        "_PointAddXYZZ_mm(X,Y,ZZ,ZZZ, x0,y0, x1,y1);",
        "_PointAddXYZZ(X,Y,ZZ,ZZZ, cx,cy, y0, c != GT_CHUNKS-1);",
    )
    for token in required:
        assert token in source, token

    math_source = gpu_math.decode()
    for token in ("#define QSB_LAZY 1", "_ModAddLazy", "_ModX3Fused"):
        assert token in math_source, token

    # Direct extraction is the only imported 856 mechanism.
    forbidden = (
        "ZLAB_BATCH_LOG",
        "ZLAB_BIG_BATCH",
        "ZLAB_HOST_DRAIN",
        "ZLAB_FUSED_X3",
        "_ZModAddSub2",
        "cudaAccessPropertyPersisting",
        "_SHA256TransformFastTail11",
        "_SHA256TransformPk33",
    )
    for token in forbidden:
        assert token not in source and token not in math_source, token

    # Keep exact 1a launch geometry and allocation; do not smuggle 64M in.
    assert "#define QSB_BATCH 16777216" in source
    assert "#define QSB_PREFETCH 0" in source
    assert "int blocks=(batch_size+QSB_TREE_N-1)/QSB_TREE_N;" in source
    assert source.count("ZLAB_DIRECT_DIGITS") == 5


def main():
    audit_source()
    checked = audit_math()
    print(
        "PASS: exact 1a 128-thread/lazy representation preserved; "
        f"direct and serial table selections match for {checked:,} signed states"
    )


if __name__ == "__main__":
    main()
