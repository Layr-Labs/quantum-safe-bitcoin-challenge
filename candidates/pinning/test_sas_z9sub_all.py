#!/usr/bin/env python3
"""Host-only audit of the QSB_SAS_Z9SUB_ALL dead z9-lane removal.

Covers all three remaining copies of the lane:
  * _ModSqr square body, QSB_SHORT_CARRY branch  (GPUMath.h)
  * _ModSqr square body, rollback branch         (GPUMath.h)
  * _ModSqrAddSub2 fused split-3p reduction      (GPUMath.h)

This models the exact changed 32-bit word operations.  It does not execute PTX
and does not make a GPU-throughput claim.
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
K_CONST = (1 << 32) + 977


# ---------------------------------------------------------------------------
# _ModSqr second-fold z9 lane
# ---------------------------------------------------------------------------
def sqr_fold(g8, czc, z8, z0, bits=WORD_BITS, z9sub=True):
    """The z9 lane of _ModSqr's second fold.

    g8   is the carry out of the g3 fold (`addc.u32 g8, 0, 0`).
    czc  is the carry out of `addc.cc.u32 z8, f8, w7` (z8 assembly wrap).
    z8   is the assembled z8 word before the fold.
    z0   is the low limb of the {z0, sfq} pair fed to the fold.
    With z9sub=False the lane is dropped (z9 read as literal 0).

    Returns (z9, sfq, sfc, out0) where out0 is the post-fold low limb z0.
    """
    base = 1 << bits
    z9 = (g8 + czc) % base if z9sub else 0
    sfq = (z8 + z9 * 977) % base
    sfz = (z0 << bits) | sfq
    acc = z8 * 977 + sfz
    sfc = (z9 + (acc >> (2 * bits))) % base
    out0 = acc % base
    return z9, sfq, sfc, out0


def sqr_diverge(g8, czc, z8, z0, bits=WORD_BITS):
    full = sqr_fold(g8, czc, z8, z0, bits=bits, z9sub=True)
    short = sqr_fold(g8, czc, z8, z0, bits=bits, z9sub=False)
    return full, short, full != short


# ---------------------------------------------------------------------------
# _ModSqrAddSub2 split-3p z9 lane (unchanged from the prior Z9SUB audit)
# ---------------------------------------------------------------------------
def sas_pair(g8, czc, z8e, carry_e, low, v, bits=WORD_BITS, z9sub=True):
    """fold init, +3 split-3p bias, and the two consecutive v-subtractions
    across the (z9, z8, low) accumulator."""
    base = 1 << bits
    low_bits = 8 * bits
    z8_raw = z8e + 3 + carry_e
    wrap = z8_raw >> bits
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


def audit_sqr_reduced_exhaustive():
    """Exhaust every 4-bit analogue state."""
    bits = 4
    base = 1 << bits
    states = differences = 0
    for g8 in range(2):
        for czc in range(2):
            for z8 in range(base):
                for z0 in range(base):
                    z9 = (g8 + czc) % base
                    full, short, diff = sqr_diverge(g8, czc, z8, z0, bits=bits)
                    assert diff == (z9 != 0)
                    states += 1
                    differences += diff
    return states, differences


def audit_sqr_random(samples=1_000_000):
    rng = random.Random(0x5A5A0D17)
    mech = 0
    for _ in range(samples):
        g8 = rng.randrange(2)
        czc = rng.randrange(2)
        z8 = rng.getrandbits(32)
        z0 = rng.getrandbits(32)
        _, _, diff = sqr_diverge(g8, czc, z8, z0)
        mech += diff
    prod = prod_g8 = prod_czc = 0
    for _ in range(samples):
        g8 = 1 if rng.random() < 2.0 ** -23 else 0   # square g3 fold corner
        czc = 1 if rng.random() < 2.0 ** -32 else 0  # z8 assembly wrap
        z8 = rng.getrandbits(32)
        z0 = rng.getrandbits(32)
        _, _, diff = sqr_diverge(g8, czc, z8, z0)
        prod += diff
        prod_g8 += g8
        prod_czc += czc
    return mech, (prod, prod_g8, prod_czc)


def audit_sas_reduced_exhaustive():
    bits = 4
    base = 1 << bits
    low_span = base * base
    states = differences = 0
    for g8 in range(2):
        for czc in range(2):
            for z8e in range(base):
                for carry_e in range(2):
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
                            assert (full != short) == (full != 0)
                            states += 1
                            differences += full != short
    return states, differences


def audit_sas_32bit_boundaries():
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
                            cases += 1
    return cases


def audit_sas_random(samples=1_000_000):
    rng = random.Random(0xC4625AFE)
    mech = 0
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
        mech += full != short
    prod = prod_g8 = prod_czc = 0
    for _ in range(samples):
        g8 = 1 if rng.random() < 2.0 ** -23 else 0
        czc = 1 if rng.random() < 2.0 ** -32 else 0
        z8e = rng.getrandbits(32)
        carry_e = rng.randrange(2)
        low = rng.getrandbits(256)
        v = rng.randrange(1, P_FIELD)
        full, wrap, borrows = sas_pair(g8, czc, z8e, carry_e, low, v)
        short, _, _ = sas_pair(g8, czc, z8e, carry_e, low, v, z9sub=False)
        net = (g8 + czc + wrap - sum(borrows)) % WORD_BASE
        assert (full != short) == (net != 0)
        prod += full != short
        prod_g8 += g8
        prod_czc += czc
    return mech, (prod, prod_g8, prod_czc)


def audit_source():
    source = (HERE / "GPUMath.h").read_text()
    assert "#define QSB_SAS_Z9SUB_ALL 1" in source
    assert "#if QSB_SAS_SPLIT3P && QSB_SAS_Z9SUB_ALL" in source
    # fallback tails exist byte-for-byte for flag-off
    assert '#define QSB_SAS_SUB_TAIL " subc.u32 z9, z9, 0;"' in source
    assert '#define QSB_SAS_Z9ADD_TAIL " addc.u32 z9, z9, 0;"' in source
    assert '#define QSB_SAS_G8_TAIL "\\taddc.u32 g8, 0, 0;\\n"' in source
    assert '#define QSB_SAS_Z9INIT " addc.u32 z9, g8, 0;"' in source
    assert '#define QSB_SAS_SFQ "mov.u32 sfq, z8;"' in source
    assert '#define QSB_SAS_SFC "addc.u32 sfc, 0, 0;"' in source
    # all three lane copies are macro-driven: no raw inline-asm tail remains
    for raw in ('addc.cc.u32 z8, f8, w7; addc.u32 z9, g8, 0;',
                'sfh;\\nmad.lo.u32 sfq, z9, 977, z8;',
                'add.cc.u64 sft, sft, sfz;\\naddc.u32 sfc, z9, 0;'):
        assert source.count(raw) == 0, raw
    # the only raw g8 tail left is the flag-off macro definition
    assert source.count('"\\taddc.u32 g8, 0, 0;\\n"') == 1
    # _ModMultCore (both PR #743 branches) already has the clean fold, plus
    # the one flag-off macro definition.
    assert source.count("mov.u32 sfq, z8;") == 3
    assert source.count("addc.u32 sfc, 0, 0;") == 3
    return hashlib.sha256(source.encode()).hexdigest()


def main():
    sqr_red = audit_sqr_reduced_exhaustive()
    sqr_rand = audit_sqr_random()
    sas_red = audit_sas_reduced_exhaustive()
    sas_bounds = audit_sas_32bit_boundaries()
    sas_rand = audit_sas_random()
    source_sha256 = audit_source()
    result = {
        "test": "exact changed z9-lane carry and borrow word operations",
        "sqr_reduced_states": sqr_red[0],
        "sqr_reduced_differences": sqr_red[1],
        "sqr_mechanism_random_differences": sqr_rand[0],
        "sqr_production_random_differences": sqr_rand[1][0],
        "sqr_production_g8_ones": sqr_rand[1][1],
        "sqr_production_czc_ones": sqr_rand[1][2],
        "sas_reduced_states": sas_red[0],
        "sas_reduced_differences": sas_red[1],
        "sas_32bit_boundary_cases": sas_bounds,
        "sas_mechanism_random_differences": sas_rand[0],
        "sas_production_random_differences": sas_rand[1][0],
        "sas_production_g8_ones": sas_rand[1][1],
        "sas_production_czc_ones": sas_rand[1][2],
        "difference_predicate": "z9 lane nonzero: g8 (carry out of the g3 "
                                "fold, ~2^-23 on squares) or z8 assembly wrap "
                                "or +3 bias wrap or a borrow across bit 288",
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
