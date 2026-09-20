#!/usr/bin/env python3
"""Host-only audit of the QSB_SAS_Z9SUB dead z9-lane removal.

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
# secp256k1 field prime; v (== Q, a field element) is always below it.
P_FIELD = (1 << 256) - (1 << 32) - 977


def sas_pair(g8, czc, z8e, carry_e, low, v, bits=WORD_BITS, z9sub=True):
    """The z9 lane of _ModSqrAddSub2: fold init, +3 split-3p bias, and the
    two consecutive v-subtractions across the (z9, z8, low) accumulator.

    g8      is the carry out of the g3 fold (`addc.u32 g8, 0, 0`).
    czc     is the carry out of `addc.cc.u32 z8, f8, w7` (z8 assembly wrap).
    z8e     is the fold's z8 value before the +3 bias lands.
    carry_e is the carry into z8 from the e-add chain.
    low     is the low eight limbs (256 bits at production width).
    With z9sub=False every producer and consumer of z9 is dropped, as under
    QSB_SAS_Z9SUB=1, and the fold reads z9 as literal 0.

    Returns (final z9, wrap, borrows).
    """
    base = 1 << bits
    low_bits = 8 * bits
    z8_raw = z8e + 3 + carry_e
    wrap = z8_raw >> bits                      # carry the bias pushes into z9
    z8 = z8_raw % base
    z9 = (g8 + czc + wrap) % base if z9sub else 0
    prefix = (z8 << low_bits) | low
    borrows = []
    for _ in range(2):
        borrow = 1 if prefix < v else 0
        prefix = (prefix - v) % (1 << (9 * bits))
        if z9sub:
            z9 = (z9 - borrow) % base
        borrows.append(borrow)
    return z9, wrap, borrows


def audit_reduced_exhaustive():
    # Exhaust every state in a four-bit analogue.  low and v use two limbs
    # each; the boundary depends only on (z8 == 0, low < v) per subtraction
    # and on z8e + carry_e + 3 >= base for the bias wrap.
    bits = 4
    base = 1 << bits
    low_span = base * base
    states = 0
    differences = 0
    for g8 in range(2):
        for czc in range(2):
            for z8e in range(base):
                for carry_e in range(2):
                    # A borrow can fire only when z8 after the bias is 0
                    # (z8_1 == 0 for sub 1; surviving z8 == 0 for sub 2), so
                    # only those rows exhaust v; the rest spot-check extremes.
                    z8_1 = (z8e + 3 + carry_e) % base
                    v_range = range(1, low_span) if z8_1 == 0 else \
                              (1, low_span - 1)
                    for low in range(low_span):
                        for v in v_range:
                            full, wrap, borrows = sas_pair(
                                g8, czc, z8e, carry_e, low, v, bits)
                            short, _, _ = sas_pair(
                                g8, czc, z8e, carry_e, low, v, bits,
                                z9sub=False)
                            # Net z9 delta is producers minus borrows; a wrap
                            # landing z8 exactly on zero can be cancelled by
                            # a sub borrow.
                            expected = full != 0
                            assert (full != short) == expected
                            # Any borrow implies the biased z8 is zero and the
                            # running low block is below v at that subtraction.
                            if any(borrows):
                                z8_1 = (z8e + 3 + carry_e) % base
                                z8_2 = (z8_1 - borrows[0]) % base
                                low2 = (low - v) % low_span
                                assert (z8_1 == 0 and low < v) or \
                                       (z8_2 == 0 and low2 < v)
                            states += 1
                            differences += full != short
    return states, differences


def audit_32bit_boundaries():
    cases = 0
    z8_edges = (0, 1, WORD_MASK - 4, WORD_MASK - 3, WORD_MASK - 2,
                WORD_MASK - 1, WORD_MASK)
    vs = (1, (1 << 128), P_FIELD - 1, P_FIELD, (1 << 256) - 1)
    for g8 in range(2):
        for czc in range(2):
            for z8e in z8_edges:
                for carry_e in range(2):
                    for v in vs:
                        lows = (0, 1, v - 1, v, v + 1, (1 << 256) - 1)
                        for low in lows:
                            if not (0 <= low < (1 << 256)):
                                continue
                            full, wrap, borrows = sas_pair(
                                g8, czc, z8e, carry_e, low, v)
                            short, _, _ = sas_pair(
                                g8, czc, z8e, carry_e, low, v, z9sub=False)
                            net = (g8 + czc + wrap - sum(borrows)) % WORD_BASE
                            assert (full != short) == (net != 0)
                            if full != short:
                                assert (full - short) % WORD_BASE == net
                            cases += 1
    return cases


def audit_random(samples=1_000_000):
    """Two cohorts.  'mechanism' samples g8/czc at 1/2 so every code path is
    exercised; 'production' draws g8 at the measured ~2^-23 square corner and
    czc at ~2^-32 so the reported rate is the shipped exposure."""
    rng = random.Random(0xC4625AFE)
    mech_diffs = 0
    for _ in range(samples):
        g8 = rng.randrange(2)
        czc = rng.randrange(2)
        z8e = rng.getrandbits(32)
        carry_e = rng.randrange(2)
        low = rng.getrandbits(256)
        v = rng.randrange(1, P_FIELD)
        full, wrap, borrows = sas_pair(g8, czc, z8e, carry_e, low, v)
        short, _, _ = sas_pair(g8, czc, z8e, carry_e, low, v, z9sub=False)
        net = (g8 + czc + wrap - sum(borrows)) % WORD_BASE
        assert (full != short) == (net != 0)
        mech_diffs += full != short
    prod_diffs = 0
    prod_g8 = prod_czc = prod_wraps = prod_b1 = prod_b2 = 0
    for _ in range(samples):
        g8 = 1 if rng.random() < 2.0 ** -23 else 0   # square fold corner
        czc = 1 if rng.random() < 2.0 ** -32 else 0  # z8 assembly wrap
        z8e = rng.getrandbits(32)
        carry_e = rng.randrange(2)
        low = rng.getrandbits(256)
        v = rng.randrange(1, P_FIELD)
        full, wrap, borrows = sas_pair(g8, czc, z8e, carry_e, low, v)
        short, _, _ = sas_pair(g8, czc, z8e, carry_e, low, v, z9sub=False)
        net = (g8 + czc + wrap - sum(borrows)) % WORD_BASE
        assert (full != short) == (net != 0)
        prod_diffs += full != short
        prod_g8 += g8
        prod_czc += czc
        prod_wraps += bool(wrap)
        prod_b1 += borrows[0]
        prod_b2 += borrows[1]
    return mech_diffs, (prod_diffs, prod_g8, prod_czc, prod_wraps,
                        prod_b1, prod_b2)


def audit_corner_targeted(samples=200_000):
    """Force z8e onto the wrap-corner residues to expose the mechanism and
    measure the conditional divergence rate.  g8 and czc are still free, so
    most sampled states diverge by construction; the assertion is the point."""
    rng = random.Random(0xC31A0D17)
    differences = 0
    for _ in range(samples):
        g8 = rng.randrange(2)
        czc = rng.randrange(2)
        z8e = rng.choice((WORD_MASK - 4, WORD_MASK - 3, WORD_MASK - 2,
                          WORD_MASK - 1))
        carry_e = rng.randrange(2)
        low = rng.getrandbits(256)
        v = rng.randrange(1, P_FIELD)
        full, wrap, borrows = sas_pair(g8, czc, z8e, carry_e, low, v)
        short, _, _ = sas_pair(g8, czc, z8e, carry_e, low, v, z9sub=False)
        net = (g8 + czc + wrap - sum(borrows)) % WORD_BASE
        assert (full != short) == (net != 0)
        differences += full != short
    return differences


def audit_source():
    source = (HERE / "GPUMath.h").read_text()
    assert "#define QSB_SAS_Z9SUB 1" in source
    assert "#if QSB_SAS_SPLIT3P && QSB_SAS_Z9SUB" in source
    # The only remaining textual occurrences are the byte-exact fallback tails.
    assert source.count('subc.u32 z9, z9, 0;\\n"') == 0  # no inline asm remains
    assert '#define QSB_SAS_SUB_TAIL " subc.u32 z9, z9, 0;"' in source
    assert '#define QSB_SAS_Z9ADD_TAIL " addc.u32 z9, z9, 0;"' in source
    assert '#define QSB_SAS_G8_TAIL "\\taddc.u32 g8, 0, 0;\\n"' in source
    assert '#define QSB_SAS_Z9INIT " addc.u32 z9, g8, 0;"' in source
    assert '#define QSB_SAS_SFQ "mov.u32 sfq, z8;"' in source
    assert '#define QSB_SAS_SFC "addc.u32 sfc, 0, 0;"' in source
    assert source.count('addc.u32 z9, z9, 0;\\n"') == 2  # #else tails untouched
    assert "subc.cc.u32 z8, z8, 0;\" QSB_SAS_SUB_TAIL" in source
    assert "addc.cc.u32 z8, z8, 3;\" QSB_SAS_Z9ADD_TAIL" in source
    assert "addc.cc.u32 z8, f8, w7;\" QSB_SAS_Z9INIT" in source
    cu = (HERE / "pinning.cu").read_text()
    assert "#define QSB_HOST_GATE 1" in cu
    assert "#define QSB_C31 1" in cu
    return hashlib.sha256(source.encode()).hexdigest()


def main():
    reduced = audit_reduced_exhaustive()
    boundary_cases = audit_32bit_boundaries()
    randoms = audit_random()
    corner_differences = audit_corner_targeted()
    source_sha256 = audit_source()
    result = {
        "test": "exact changed z9-lane carry and borrow word operations",
        "reduced_states": reduced[0],
        "reduced_differences": reduced[1],
        "32bit_boundary_cases": boundary_cases,
        "mechanism_random_samples": 1_000_000,
        "mechanism_random_differences": randoms[0],
        "production_random_samples": 1_000_000,
        "production_random_differences": randoms[1][0],
        "production_g8_ones": randoms[1][1],
        "production_z8_wraps": randoms[1][2],
        "production_bias_wraps": randoms[1][3],
        "production_first_sub_borrows": randoms[1][4],
        "production_second_sub_borrows": randoms[1][5],
        "corner_targeted_samples": 200_000,
        "corner_targeted_differences": corner_differences,
        "difference_predicate": "z9 lane nonzero: g8 (carry out of the g3 "
                                "fold, ~2^-23 on squares) or z8 assembly wrap "
                                "or +3 bias wrap or a borrow across bit 288 "
                                "(z8 == 0 and running low256 below v)",
        "uniform_bound": "per call ~2^-22-class dominated by g8 == 1; every "
                         "wrong z9 corrupts only that candidate's GPU point, "
                         "which the QSB_HOST_GATE exact check filters",
        "GPUMath_sha256": source_sha256,
        "cuda_compiled": False,
        "gpu_executed": False,
        "official_score": None,
        "speedup": None,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
