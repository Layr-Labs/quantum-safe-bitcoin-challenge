#!/usr/bin/env python3
"""CPU-side source-shape binder for Scalar-wired QSB_EARLY_LOAD on tip bb5c9a0.

No GPU / nvcc. Asserts the production fixed-base entry overlaps the next
table fill inside the T-scheduled madd, and that tip host geometry is intact.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = (ROOT / "pinning.cu").read_text()


def define_int(name: str) -> int:
    m = re.search(rf"^#define\s+{name}\s+(\d+)\b", SRC, re.M)
    assert m, f"missing #define {name}"
    return int(m.group(1))


def main() -> int:
    assert define_int("QSB_STREAM") == 1
    assert define_int("QSB_SLOTPIPE") == 1
    assert define_int("QSB_SLOTS") == 2
    assert define_int("QSB_EARLY_LOAD") == 1
    assert "QSB_SLOTS 3" not in SRC
    assert "#define QSB_SLOTS 3" not in SRC

    # DIRECT_DIGITS must remain on with EARLY_LOAD (conflict only PREFETCH/S0_SHM).
    conflict = re.search(
        r"#if QSB_DIRECT_DIGITS && \(([^)]+)\)", SRC
    )
    assert conflict, "DIRECT_DIGITS conflict guard missing"
    guard = conflict.group(1)
    assert "QSB_PREFETCH" in guard and "QSB_S0_SHM" in guard
    assert "QSB_EARLY_LOAD" not in guard

    assert "_PointAddXYZZT_early" in SRC
    assert "gt_load_signed_flat(gTable, nbase, nidx, nneg, nx, ny)" in SRC

    # Production entry must call the early madd under EARLY_LOAD.
    scalar_start = SRC.index("__device__ void _FixedBaseSignedXYZZScalar(")
    scalar_end = SRC.index("/* _FixedBaseSignedAffine:", scalar_start)
    scalar = SRC[scalar_start:scalar_end]
    assert "#if QSB_EARLY_LOAD" in scalar
    assert "_PointAddXYZZT_early(" in scalar
    assert "codes[(size_t)(c+1)*QSB_TREE_N+threadIdx.x]" in scalar
    # Must not leave the tip hole: just-in-time load + plain T madd only.
    # Under the EARLY_LOAD arm the next fill is inside the madd.
    early_arm = scalar.split("#if QSB_EARLY_LOAD", 1)[1].split("#else", 1)[0]
    assert "_PointAddXYZZT_early" in early_arm
    assert "qsb_load_decoded(table,2,base,x1,y1)" in early_arm

    # Kernel still enters via Scalar (not the e[] path).
    assert "_FixedBaseSignedXYZZScalar(qx,qy,qzz,qzzz,z,d_gt,qsb_prepare_scratch())" in SRC

    # Rejected levers must stay off / unused as primary.
    assert "QSB_RESOLVE_LAST" not in SRC or re.search(
        r"#define\s+QSB_RESOLVE_LAST\s+0\b", SRC
    )
    assert define_int("QSB_SLOTS") != 3
    # Packed planes must still use ordinary assignments (not qsb_st_v2 on vbar/tbar).
    packed = (ROOT / "PackedRecovery.cuh").read_text()
    assert "saved[0*s+i]=make_ulonglong2(vbar[0],vbar[1])" in packed
    assert "qsb_st_v2" not in packed

    print("audit_early_load_scalar: OK")
    print(
        f"  STREAM={define_int('QSB_STREAM')} SLOTPIPE={define_int('QSB_SLOTPIPE')} "
        f"SLOTS={define_int('QSB_SLOTS')} EARLY_LOAD={define_int('QSB_EARLY_LOAD')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
