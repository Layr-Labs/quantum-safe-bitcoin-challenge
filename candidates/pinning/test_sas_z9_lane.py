#!/usr/bin/env python3
"""Host audit for the fused square/add/sub carry-only z9 lane."""

from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
ADDENDS = range(5)


def add_tail_complete(low, high, addend, bits):
    base = 1 << bits
    total = low + addend
    return total % base, (high + total // base) % base


def add_tail_short(low, high, addend, bits):
    base = 1 << bits
    return (low + addend) % base, high


def sub_tail_complete(low, high, subtrahend, bits):
    base = 1 << bits
    borrow = low < subtrahend
    return (low - subtrahend) % base, (high - borrow) % base


def sub_tail_short(low, high, subtrahend, bits):
    base = 1 << bits
    return (low - subtrahend) % base, high


def audit_reduced_exhaustive():
    bits = 4
    base = 1 << bits
    states = differences = 0
    for high in range(base):
        for low in range(base):
            for addend in ADDENDS:
                full = add_tail_complete(low, high, addend, bits)
                short = add_tail_short(low, high, addend, bits)
                assert (full != short) == (low + addend >= base)
                states += 1
                differences += full != short
            for subtrahend in range(2):
                full = sub_tail_complete(low, high, subtrahend, bits)
                short = sub_tail_short(low, high, subtrahend, bits)
                assert (full != short) == (low < subtrahend)
                states += 1
                differences += full != short
    return states, differences


def audit_32bit_boundaries():
    mask = (1 << 32) - 1
    edges = (0, 1, 2, mask - 3, mask - 2, mask - 1, mask)
    cases = 0
    for high in edges:
        for low in edges:
            for addend in ADDENDS:
                full = add_tail_complete(low, high, addend, 32)
                short = add_tail_short(low, high, addend, 32)
                assert (full != short) == (low + addend > mask)
                cases += 1
            for subtrahend in range(2):
                full = sub_tail_complete(low, high, subtrahend, 32)
                short = sub_tail_short(low, high, subtrahend, 32)
                assert (full != short) == (low < subtrahend)
                cases += 1
    return cases


def audit_source():
    source = (HERE / "GPUMath.h").read_text()
    cu = (HERE / "pinning.cu").read_text()
    assert "#define QSB_SAS_Z9_LANE 1" in source
    assert '#define QSB_SAS_Z89 "\\taddc.u32 z8, 0, w7;\\n"' in source
    assert '#define QSB_SAS_Z9ADD_TAIL ""' in source
    assert '#define QSB_SAS_SUB_TAIL ""' in source
    assert '#define QSB_SAS_FOLD_HEAD "mov.u32 sfq, z8;"' in source
    assert '#define QSB_SAS_FOLD_CARRY "addc.u32 sfc, 0, 0;"' in source
    assert '#define QSB_SAS_Z9ADD_TAIL " addc.u32 z9, z9, 0;"' in source
    assert '#define QSB_SAS_SUB_TAIL " subc.u32 z9, z9, 0;"' in source
    assert '#define QSB_SAS_FOLD_HEAD "mad.lo.u32 sfq, z9, 977, z8;"' in source
    assert '#define QSB_SAS_FOLD_CARRY "addc.u32 sfc, z9, 0;"' in source
    assert (
        "#if QSB_SAS_Z9_LANE && QSB_RP_SQR && QSB_SHORT_CARRY && "
        "QSB_SAS_SPLIT3P" in source
    )

    body = re.search(
        r"void _ModSqrAddSub2.*?(?=\n__device__|\n// -----)", source, re.S
    )
    assert body
    fused = body.group(0)
    assert fused.count("QSB_SAS_Z9ADD_TAIL") == 3
    assert fused.count("QSB_SAS_SUB_TAIL") == 2
    assert "QSB_SAS_FOLD_HEAD" in fused
    assert "QSB_SAS_FOLD_CARRY" in fused
    assert "QSB_SAS_Z9_LANE requires QSB_HOST_GATE" in cu


def main():
    states, differences = audit_reduced_exhaustive()
    boundaries = audit_32bit_boundaries()
    audit_source()
    print(
        "PASS: SAS z9 lane predicates and source audit",
        {"reduced_states": states, "differences": differences, "boundaries": boundaries},
    )


if __name__ == "__main__":
    main()
