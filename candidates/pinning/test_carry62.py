#!/usr/bin/env python3
"""Host-only audit of the two QSB_CARRY62 truncation boundaries.

This models the exact changed word operations.  It does not execute PTX and
does not make a GPU-throughput claim.
SPDX-License-Identifier: GPL-3.0-only
"""

import hashlib
import json
from pathlib import Path
import random


HERE = Path(__file__).resolve().parent
WORD_BITS = 32
WORD_BASE = 1 << WORD_BITS
WORD_MASK = WORD_BASE - 1
THREE_K = (3 << 32) + 0xB73


def fold_complete(z2, z3, z4, sfc, bits=WORD_BITS):
    """Carry-complete addc chain beginning at the last fold's z2 limb."""
    base = 1 << bits
    mask = base - 1
    total = z2 + sfc
    z2 = total & mask
    total = z3 + (total >> bits)
    z3 = total & mask
    z4 = (z4 + (total >> bits)) & mask
    return z2, z3, z4


def fold_carry62(z2, z3, z4, sfc, bits=WORD_BITS):
    """Candidate chain: consume the carry in z3 and deliberately stop there."""
    base = 1 << bits
    mask = base - 1
    total = z2 + sfc
    z2 = total & mask
    z3 = (z3 + (total >> bits)) & mask
    return z2, z3, z4


def split3p_complete(value, bits=WORD_BITS):
    """Subtract 3K through five limbs, as in the PR #739 control."""
    return (value - THREE_K) % (1 << (5 * bits))


def split3p_carry62(value, bits=WORD_BITS, threshold=THREE_K):
    """Subtract 3K modulo the low three limbs and leave high limbs unchanged."""
    low_width = 3 * bits
    low_mask = (1 << low_width) - 1
    low = ((value & low_mask) - threshold) & low_mask
    return (value & ~low_mask) | low


def audit_reduced_exhaustive():
    # Exhaust every state in a four-bit analogue.  sfc<=2 covers the largest
    # second-fold addend used by the three changed call sites.
    bits = 4
    base = 1 << bits
    fold_states = 0
    fold_differences = 0
    for sfc in range(3):
        for z2 in range(base):
            for z3 in range(base):
                for z4 in range(base):
                    full = fold_complete(z2, z3, z4, sfc, bits)
                    short = fold_carry62(z2, z3, z4, sfc, bits)
                    expected = z2 + sfc >= base and z3 == base - 1
                    assert (full != short) == expected
                    fold_states += 1
                    fold_differences += full != short

    # The split-3p proof is purely a borrow boundary.  Use a representable
    # analogue threshold and exhaust all three low limbs plus two high values.
    threshold = 3 * base + 3
    split_states = 0
    split_differences = 0
    for high in range(2):
        for low in range(base ** 3):
            value = (high << (3 * bits)) | low
            full = (value - threshold) % (1 << (4 * bits))
            short = split3p_carry62(value, bits, threshold)
            expected = low < threshold
            assert (full != short) == expected
            split_states += 1
            split_differences += full != short
    return fold_states, fold_differences, split_states, split_differences


def audit_32bit_boundaries():
    # Exercise every side of the exact carry predicate, including sfc=2.
    edges = (0, 1, 2, WORD_MASK - 2, WORD_MASK - 1, WORD_MASK)
    fold_cases = 0
    for sfc in range(3):
        for z2 in edges:
            for z3 in edges:
                for z4 in edges:
                    full = fold_complete(z2, z3, z4, sfc)
                    short = fold_carry62(z2, z3, z4, sfc)
                    expected = z2 + sfc >= WORD_BASE and z3 == WORD_MASK
                    assert (full != short) == expected
                    if expected:
                        assert (short[2] - full[2]) & WORD_MASK == WORD_MASK
                    fold_cases += 1

    width = 160
    modulus = 1 << width
    low96 = 1 << 96
    split_cases = 0
    for high in (0, 1, WORD_MASK, (1 << 64) - 1):
        for low in (
            0,
            1,
            THREE_K - 2,
            THREE_K - 1,
            THREE_K,
            THREE_K + 1,
            low96 - 2,
            low96 - 1,
        ):
            value = ((high << 96) | low) % modulus
            full = split3p_complete(value)
            short = split3p_carry62(value)
            expected = low < THREE_K
            assert (full != short) == expected
            if expected:
                assert (short - full) % modulus == low96
            split_cases += 1
    return fold_cases, split_cases


def audit_random(samples=1_000_000):
    rng = random.Random(0xC4625AFE)
    fold_differences = 0
    split_differences = 0
    for _ in range(samples):
        z2 = rng.getrandbits(32)
        z3 = rng.getrandbits(32)
        z4 = rng.getrandbits(32)
        sfc = rng.randrange(3)
        full = fold_complete(z2, z3, z4, sfc)
        short = fold_carry62(z2, z3, z4, sfc)
        expected = z2 + sfc >= WORD_BASE and z3 == WORD_MASK
        assert (full != short) == expected
        fold_differences += full != short

        value = rng.getrandbits(160)
        full = split3p_complete(value)
        short = split3p_carry62(value)
        expected = (value & ((1 << 96) - 1)) < THREE_K
        assert (full != short) == expected
        split_differences += full != short
    return fold_differences, split_differences


def audit_source():
    source = (HERE / "GPUMath.h").read_text()
    assert "#define QSB_CARRY62 1" in source
    assert source.count("QSB_SECOND_FOLD_TAIL") == 5  # two definitions, three uses
    assert "subc.u32 z2, z2, 0;\\n\"" in source
    assert "-DQSB_CARRY62=0 restores" in source
    return hashlib.sha256(source.encode()).hexdigest()


def main():
    reduced = audit_reduced_exhaustive()
    boundaries = audit_32bit_boundaries()
    random_differences = audit_random()
    source_sha256 = audit_source()
    result = {
        "test": "exact changed carry and borrow word operations",
        "reduced_fold_states": reduced[0],
        "reduced_fold_differences": reduced[1],
        "reduced_split3p_states": reduced[2],
        "reduced_split3p_differences": reduced[3],
        "32bit_fold_boundary_cases": boundaries[0],
        "32bit_split3p_boundary_cases": boundaries[1],
        "random_samples_per_site": 1_000_000,
        "random_fold_differences": random_differences[0],
        "random_split3p_differences": random_differences[1],
        "fold_difference_predicate": "z2+sfc >= 2^32 and z3 == 2^32-1",
        "fold_uniform_bound": "at most 2 / 2^64 = 2^-63 for sfc <= 2",
        "split3p_difference_predicate": "low96 < 3*(2^32+977)",
        "split3p_uniform_bound": f"{THREE_K}/2^96 < 2^-62",
        "GPUMath_sha256": source_sha256,
        "cuda_compiled": False,
        "gpu_executed": False,
        "official_score": None,
        "speedup": None,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
