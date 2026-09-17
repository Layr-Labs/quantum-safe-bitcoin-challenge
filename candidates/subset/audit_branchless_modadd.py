#!/usr/bin/env python3
"""Audit the exact public branchless _ModAdd256 on the subset frontier."""

import hashlib
import random
from pathlib import Path


FAB_HEAD = "4977db123b790574e65bcac7c80c8a057a44f407"
FAB_MODADD_SHA256 = "43207d95ca25e3f446f8e5c2521480ab75e1ca5cb19038f125777208c980908a"
FINAL_GPUMATH_SHA256 = "f695ecf14562fede0284d52851770b527ccc1ba276b7cc8ebf606d152ad017f8"
EXACT_D277_TREE_SHA256 = "a58c506a49aa24740c3e48846afd9f64f665275137299f1ac2b95cd35de268ec"

MASK64 = (1 << 64) - 1
MASK256 = (1 << 256) - 1
MASK320 = (1 << 320) - 1
P = (1 << 256) - 0x1000003D1


def words(value):
    return [(value >> (64 * i)) & MASK64 for i in range(4)]


def value(ws):
    return sum(w << (64 * i) for i, w in enumerate(ws))


def old_branch(a, b):
    total = a + b
    original = total & MASK256
    return ((total - P) & MASK256) if total >= P else original


def new_mask(a, b):
    total = a + b
    original = total & MASK256
    reduced_320 = (total - P) & MASK320
    high = (reduced_320 >> 256) & MASK64
    ge = 0 if high >> 63 else MASK64
    out = 0
    for i in range(4):
        reduced_limb = (reduced_320 >> (64 * i)) & MASK64
        original_limb = (original >> (64 * i)) & MASK64
        out |= ((reduced_limb & ge) | (original_limb & (~ge & MASK64))) << (64 * i)
    return out


def write_alias(a_words, b_words, out_words):
    selected = words(new_mask(value(a_words), value(b_words)))
    for i in range(4):
        out_words[i] = selected[i]


def main():
    root = Path(__file__).resolve().parent
    gpu_math = (root / "GPUMath.h").read_text()
    tree = (root / "tests/gpu_epochs/tree.cu").read_bytes()
    assert hashlib.sha256(gpu_math.encode()).hexdigest() == FINAL_GPUMATH_SHA256
    assert hashlib.sha256(tree).hexdigest() == EXACT_D277_TREE_SHA256

    begin = gpu_math.index("__device__ void _ModAdd256(uint64_t *r, uint64_t *a, uint64_t *b)")
    end = gpu_math.index("\n\n__device__ void _ModSub256", begin)
    function = gpu_math[begin:end] + "\n"
    assert hashlib.sha256(function.encode()).hexdigest() == FAB_MODADD_SHA256
    assert "if(_IsPositive(rr))" not in function
    assert "uint64_t ge = ~((uint64_t)((int64_t)rr[4] >> 63));" in function
    assert function.index("UADD(rr[4], 0UL, 0UL);") < function.index("SubP(rr);")
    assert function.index("SubP(rr);") < function.index("r[0] =")
    assert "_ModAdd256(T, T, PPP);" in gpu_math

    edge = [
        0, 1, 2, 0x1000003D0, 0x1000003D1,
        P - 2, P - 1, P, P + 1, P + 2,
        (1 << 255) - 1, 1 << 255, MASK256 - 1, MASK256,
    ]
    cases = [(a, b) for a in edge for b in edge]
    rng = random.Random(0x5AB5E7)
    cases.extend((rng.getrandbits(256), rng.getrandbits(256)) for _ in range(200_000))

    for index, (a, b) in enumerate(cases):
        expected = old_branch(a, b)
        got = new_mask(a, b)
        assert got == expected, (index, hex(a), hex(b), hex(expected), hex(got))

        aw, bw = words(a), words(b)
        separate = [0, 0, 0, 0]
        write_alias(aw[:], bw[:], separate)
        assert value(separate) == expected
        alias_a, b_copy = aw[:], bw[:]
        write_alias(alias_a, b_copy, alias_a)
        assert value(alias_a) == expected
        a_copy, alias_b = aw[:], bw[:]
        write_alias(a_copy, alias_b, alias_b)
        assert value(alias_b) == expected

    print(
        "PASS: exact fab branchless _ModAdd256 on exact d277 subset tree; "
        f"{len(cases)} identities; separate, r==a, and r==b alias modes"
    )


if __name__ == "__main__":
    main()
