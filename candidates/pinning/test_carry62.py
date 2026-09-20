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


def fold_c31(z2, z3, z4, sfc, bits=WORD_BITS):
    """C31 chain: consume the fold into z2 and stop before z3."""
    base = 1 << bits
    mask = base - 1
    z2 = (z2 + sfc) & mask
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


def split3p_c31(value, bits=WORD_BITS, threshold=THREE_K):
    """Subtract 3K modulo the low two limbs (64 bits at production width)."""
    low_width = 2 * bits
    low_mask = (1 << low_width) - 1
    low = ((value & low_mask) - threshold) & low_mask
    return (value & ~low_mask) | low


MASK64 = (1 << 64) - 1
K64 = (1 << 32) + 977


def klimb_sub_complete(t0, t1, k):
    borrow = 1 if t0 < k else 0
    return (t0 - k) & MASK64, (t1 - borrow) & MASK64


def klimb_sub_c31(t0, t1, k):
    return (t0 - k) & MASK64, t1


def klimb_add_complete(t0, t1, k):
    carry = 1 if t0 + k >= (1 << 64) else 0
    return (t0 + k) & MASK64, (t1 + carry) & MASK64


def klimb_add_c31(t0, t1, k):
    return (t0 + k) & MASK64, t1


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


def audit_c31_predicates():
    bits = 4
    base = 1 << bits
    fold_states = fold_diffs = 0
    for sfc in range(3):
        for z2 in range(base):
            for z3 in range(base):
                for z4 in range(base):
                    full = fold_complete(z2, z3, z4, sfc, bits)
                    short = fold_c31(z2, z3, z4, sfc, bits)
                    expected = z2 + sfc >= base
                    assert (full != short) == expected
                    fold_states += 1
                    fold_diffs += full != short
    threshold = 3 * base + 3
    split_states = split_diffs = 0
    for high in range(2):
        for low in range(base ** 2):
            value = (high << (2 * bits)) | low
            full = (value - threshold) % (1 << (4 * bits))
            short = split3p_c31(value, bits, threshold)
            expected = low < threshold
            assert (full != short) == expected
            split_states += 1
            split_diffs += full != short
    k_cases = k_sub_d = k_add_d = 0
    edges = (0, 1, K64 - 1, K64, K64 + 1, MASK64 - 1, MASK64)
    for t0 in edges:
        for t1 in edges:
            for k in (0, K64):
                fs, ss = klimb_sub_complete(t0, t1, k), klimb_sub_c31(t0, t1, k)
                fa, sa = klimb_add_complete(t0, t1, k), klimb_add_c31(t0, t1, k)
                assert (fs != ss) == (k != 0 and t0 < k)
                assert (fa != sa) == (k != 0 and t0 + k >= (1 << 64))
                k_cases += 1
                k_sub_d += fs != ss
                k_add_d += fa != sa
    rng = random.Random(0xC31A0D17)
    rand_sub = rand_add = 0
    for _ in range(200_000):
        t0 = rng.getrandbits(64)
        t1 = rng.getrandbits(64)
        k = K64 if rng.randrange(2) else 0
        if klimb_sub_complete(t0, t1, k) != klimb_sub_c31(t0, t1, k):
            rand_sub += 1
        if klimb_add_complete(t0, t1, k) != klimb_add_c31(t0, t1, k):
            rand_add += 1
    return {
        "c31_reduced_fold_states": fold_states,
        "c31_reduced_fold_differences": fold_diffs,
        "c31_reduced_split3p_states": split_states,
        "c31_reduced_split3p_differences": split_diffs,
        "c31_klimb_boundary_cases": k_cases,
        "c31_klimb_sub_boundary_differences": k_sub_d,
        "c31_klimb_add_boundary_differences": k_add_d,
        "c31_klimb_random_samples": 200_000,
        "c31_klimb_random_sub_differences": rand_sub,
        "c31_klimb_random_add_differences": rand_add,
    }


K32 = (1 << 32) + 977


def even_fold_overflow(r3, x14, cin=0):
    """f8 != 0 iff the even 977-chain overflows 64 bits."""
    return r3 + x14 * 977 + cin >= (1 << 64)


def odd_fold_overflow(x14, x15, cin=0):
    """g8 != 0 iff x15*(2^32+977)+x14+cin overflows 64 bits (~x15 >= 2^32-977)."""
    return x15 * K32 + x14 + cin >= (1 << 64)


def sqr_second_fold_sfq(z8, z9, rp):
    """Low 32 bits of the second-fold start. RP_SQR ignores z9."""
    if rp:
        return z8 & WORD_MASK
    return (z8 + z9 * 977) & WORD_MASK


def audit_rpsqr_predicates():
    """Host model of the QSB_RP_SQR overflow predicates. No PTX, no GPU."""
    # Even-fold f8: overflow iff r3 + 977*x14 + cin >= 2^64.
    f8_cases = 0
    f8_diffs = 0
    for x14 in list(range(8)) + [977, (1 << 32) - 1, (1 << 32) - 977]:
        for cin in (0, 1):
            # r3 just below / at / above the overflow threshold
            thresh = (1 << 64) - x14 * 977 - cin
            for r3 in [0, 1, max(0, thresh - 1), thresh % (1 << 64), (1 << 64) - 1]:
                f8_cases += 1
                overflow = even_fold_overflow(r3, x14, cin)
                # RP_SQR writes z8 without f8; complete writes z8+f8. Differ iff overflow.
                if overflow:
                    f8_diffs += 1
                    assert even_fold_overflow(r3, x14, cin)
                else:
                    assert not even_fold_overflow(r3, x14, cin)

    # Odd-fold g8: overflow when x15 is within 977 of 2^32.
    g8_cases = 0
    g8_diffs = 0
    for x15 in list(range(0, 4)) + list(range((1 << 32) - 980, 1 << 32)):
        for x14 in (0, 1, 977, (1 << 32) - 1):
            for cin in (0, 1):
                g8_cases += 1
                if odd_fold_overflow(x14, x15, cin):
                    g8_diffs += 1

    # Square second-fold start: mad.lo(z9*977+z8) vs z8.
    sf_cases = 0
    sf_diffs = 0
    for z8 in (0, 1, 977, WORD_MASK):
        for z9 in (0, 1, 2, 3):
            sf_cases += 1
            complete = sqr_second_fold_sfq(z8, z9, rp=False)
            rp = sqr_second_fold_sfq(z8, z9, rp=True)
            if complete != rp:
                sf_diffs += 1
                assert z9 != 0
            else:
                assert z9 == 0 or ((z8 + z9 * 977) & WORD_MASK) == (z8 & WORD_MASK)

    rng = random.Random(0x52505351)
    rand_f8 = rand_g8 = 0
    for _ in range(200_000):
        r3 = rng.randrange(1 << 64)
        x14 = rng.randrange(1 << 32)
        x15 = rng.randrange(1 << 32)
        if even_fold_overflow(r3, x14, 0):
            rand_f8 += 1
        if odd_fold_overflow(x14, x15, 0):
            rand_g8 += 1

    return {
        "rpsqr_f8_boundary_cases": f8_cases,
        "rpsqr_f8_boundary_overflows": f8_diffs,
        "rpsqr_g8_boundary_cases": g8_cases,
        "rpsqr_g8_boundary_overflows": g8_diffs,
        "rpsqr_sfq_cases": sf_cases,
        "rpsqr_sfq_differences": sf_diffs,
        "rpsqr_random_samples": 200_000,
        "rpsqr_random_f8_overflows": rand_f8,
        "rpsqr_random_g8_overflows": rand_g8,
        "rpsqr_f8_predicate": "r3 + 977*x14 + cin >= 2^64 (~2^-23)",
        "rpsqr_g8_predicate": "x15*(2^32+977)+x14+cin >= 2^64 (~x15 >= 2^32-977, ~2^-23)",
        "rpsqr_sfq_predicate": "mad.lo(z8, 977*z9) differs from z8 iff z9 != 0",
    }


def audit_source():
    source = (HERE / "GPUMath.h").read_text()
    assert "#define QSB_CARRY62 1" in source
    assert source.count("QSB_SECOND_FOLD_TAIL") == 6  # three definitions, three uses
    assert "subc.u32 z2, z2, 0;\\n\"" in source
    assert "-DQSB_CARRY62=0 restores" in source
    assert "#if QSB_C31 && QSB_SHORT_CARRY" in source
    assert "#define QSB_SECOND_FOLD_TAIL \"\"" in source
    assert "sub.u64 t0,t0,k;" in source
    assert "add.u64 t0,t0,k;" in source
    assert "#if QSB_RP_SQR && QSB_SHORT_CARRY" in source
    assert "-DQSB_RP_SQR=0 restores" in source
    assert 'QSB_MUL_Z8 "\\taddc.u32 z8, 0, w7;\\n"' in source
    assert 'QSB_SQR_SF_HEAD "mov.u32 sfq, z8;\\n"' in source
    assert "QSB_SAS_Z89" in source
    cu = (HERE / "pinning.cu").read_text()
    assert "#define QSB_HOST_GATE 1" in cu
    assert "#define QSB_C31 1" in cu
    assert "#define QSB_RP_SQR 0" in cu
    assert "#define QSB_FKIENE 0" in cu
    assert "#define QSB_RAW_X 1" in cu
    assert "#error \"QSB_RP_SQR requires QSB_HOST_GATE" in cu
    return hashlib.sha256(source.encode()).hexdigest()


def main():
    reduced = audit_reduced_exhaustive()
    boundaries = audit_32bit_boundaries()
    random_differences = audit_random()
    c31 = audit_c31_predicates()
    rpsqr = audit_rpsqr_predicates()
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
        "c31_fold_difference_predicate": "z2+sfc >= 2^32 (~2^-31)",
        "c31_split3p_difference_predicate": "low64 < 3K (~2^-30.4)",
        "c31_klimb_difference_predicate": "K add/sub carries out of t0 (~2^-33 with P(k=K)~1/2)",
        "rpsqr_f8_difference_predicate": "even-fold overflow f8 != 0 (~2^-23)",
        "rpsqr_g8_difference_predicate": "odd-fold overflow g8 != 0 on a square (~2^-23)",
        "GPUMath_sha256": source_sha256,
        "cuda_compiled": False,
        "gpu_executed": False,
        "official_score": None,
        "speedup": None,
    }
    result.update(c31)
    result.update(rpsqr)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
