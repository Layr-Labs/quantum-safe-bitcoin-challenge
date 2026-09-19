#!/usr/bin/env python3
"""CPU-side source-shape binder for Scalar-wired QSB_EARLY_LOAD on tip 6288396.

No GPU / nvcc. Asserts the production fixed-base entry overlaps the next
table fill inside the T-scheduled madd, tip STREAM2 crown stays intact, and
known regressors stay off.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = (ROOT / "pinning.cu").read_text()
PACKED = (ROOT / "PackedRecovery.cuh").read_text()


def define_int(name: str, text: str = SRC) -> int:
    m = re.search(rf"^#define\s+{name}\s+(\d+)\b", text, re.M)
    assert m, f"missing #define {name}"
    return int(m.group(1))


def main() -> int:
    assert define_int("QSB_STREAM") == 1
    assert define_int("QSB_STREAM2") == 1
    assert define_int("QSB_SLOTPIPE") == 1
    assert define_int("QSB_SLOTS") == 2
    assert define_int("QSB_EARLY_LOAD") == 1
    assert "#define QSB_SLOTS 3" not in SRC

    # DIRECT_DIGITS must remain on with EARLY_LOAD (conflict only PREFETCH/S0_SHM).
    conflict = re.search(r"#if QSB_DIRECT_DIGITS && \(([^)]+)\)", SRC)
    assert conflict, "DIRECT_DIGITS conflict guard missing"
    guard = conflict.group(1)
    assert "QSB_PREFETCH" in guard and "QSB_S0_SHM" in guard
    assert "QSB_EARLY_LOAD" not in guard

    assert "_PointAddXYZZT_early" in SRC
    assert "gt_load_signed_flat(gTable, nbase, nidx, nneg, nx, ny)" in SRC
    # Early twin must follow tip T-schedule (S2 before U2) and keep fuse path.
    early_fn = SRC.split("__device__ __forceinline__ void _PointAddXYZZT_early(", 1)[1]
    early_fn = early_fn.split("\n}", 1)[0]
    assert "_ModMult(S2, ZZZ1)" in early_fn
    assert early_fn.index("_ModMult(S2, ZZZ1)") < early_fn.index("_ModMult(U2,")
    assert "_ModSqrAddSub2(T, R, PPP, Q)" in early_fn

    scalar_start = SRC.index("__device__ void _FixedBaseSignedXYZZScalar(")
    scalar_end = SRC.index("/* _FixedBaseSignedAffine:", scalar_start)
    scalar = SRC[scalar_start:scalar_end]
    assert "#if QSB_EARLY_LOAD" in scalar
    assert "_PointAddXYZZT_early(" in scalar
    assert "codes[(size_t)(c+1)*QSB_TREE_N+threadIdx.x]" in scalar
    early_arm = scalar.split("#if QSB_EARLY_LOAD", 1)[1].split("#else", 1)[0]
    assert "_PointAddXYZZT_early" in early_arm
    assert "qsb_load_decoded(table,2,base,x1,y1)" in early_arm

    assert "_FixedBaseSignedXYZZScalar(qx,qy,qzz,qzzz,z,d_gt,qsb_prepare_scratch())" in SRC

    # Known regressors stay off as primary levers.
    assert "QSB_RESOLVE_LAST" not in SRC or re.search(
        r"#define\s+QSB_RESOLVE_LAST\s+0\b", SRC
    )
    assert define_int("QSB_SLOTS") != 3

    # Tip STREAM2 crown: packed planes use qsb_st_v2 / finish uses qsb_ld_v2.
    assert "qsb_st_v2(&saved[0*s+i],vbar[0],vbar[1])" in PACKED
    assert "qsb_ld_v2(&saved[0*s+i])" in SRC

    print("audit_early_load_scalar: OK")
    print(
        f"  STREAM={define_int('QSB_STREAM')} STREAM2={define_int('QSB_STREAM2')} "
        f"SLOTPIPE={define_int('QSB_SLOTPIPE')} SLOTS={define_int('QSB_SLOTS')} "
        f"EARLY_LOAD={define_int('QSB_EARLY_LOAD')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
