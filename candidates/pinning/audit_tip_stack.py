#!/usr/bin/env python3
"""Source-shape binder: Meganpark tip 99234b73 + fused XYZZ reductions."""
from pathlib import Path

root = Path(__file__).resolve().parent
cu = (root / "pinning.cu").read_text()
gm = (root / "GPUMath.h").read_text()

assert "#define QSB_L2_SKIP 1" in cu
assert "#define QSB_SPARSE_TAIL 1" in cu
assert "#define QSB_SPARSE_D 1" in cu
assert "#define QSB_FINAL_TEMPLATE 1" in cu
assert "#define QSB_SYM_FINISH 1" in cu
assert "#define QSB_DIRECT_DIGITS 1" in cu or "QSB_DIRECT_DIGITS" in cu
assert "QSB_SLOTS" not in cu

assert "#define QSB_FUSE_MULSUB 1" in gm
assert "#define QSB_FUSE_SQRADDSUB2 1" in gm
assert "_ModMulSubCore" in gm and "_ModSqrAddSub2" in gm
print("audit_tip_stack: PASS (Meganpark tip + fuse composition)")
