#!/usr/bin/env python3
"""Source-shape binder for tip ce0aff4e stack + composed fused reductions."""
from pathlib import Path

root = Path(__file__).resolve().parent
cu = (root / "pinning.cu").read_text()
gm = (root / "GPUMath.h").read_text()

assert "#define QSB_L2_SKIP 1" in cu
assert "#define QSB_COFACTOR 1" in cu
assert "#define QSB_DIRDIG 1" in cu
assert "#define QSB_SQFREE 1" in cu
assert "#define QSB_SPARSE_D 1" in cu
assert "#define QSB_EARLY_LOAD 0" in cu or "EARLY_LOAD 0" in cu
assert "#define QSB_PK_UNROLL 0" in cu  # tip default; not our lever
assert "QSB_SLOTS" not in cu  # do not resubmit failed slots port

assert "#define QSB_FUSE_MULSUB 1" in gm
assert "#define QSB_FUSE_SQRADDSUB2 1" in gm
assert "_ModMulSubCore" in gm and "_ModSqrAddSub2" in gm
print("audit_tip_stack: PASS (tip + fuse composition)")
