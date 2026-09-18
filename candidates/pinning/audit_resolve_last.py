#!/usr/bin/env python3
"""Binder for QSB_RESOLVE_LAST peel on tip 99234b73 / b89c5b1."""
from pathlib import Path
cu = Path(__file__).with_name("pinning.cu").read_text()
assert "#define QSB_RESOLVE_LAST 1" in cu
assert "for(int c=2;c<GT_CHUNKS-1;c++)" in cu
assert "qsb_load_decoded(table,GT_CHUNKS-1,base,x1,y1)" in cu
assert cu.count("_PointAddXYZZT<false>(X,Y,U,V,x1,y1,y0);") == 1
assert "for(int c=2;c<GT_CHUNKS;c++)" in cu
assert "_ModMult(x1,y0,V);_ModSub256(Y,Y,x1);" in cu
gm = Path(__file__).with_name("GPUMath.h").read_text()
assert "template<bool DEFER_Y>" in gm
assert "__device__ __forceinline__ void _PointAddXYZZT(" in gm
# tip markers retained
assert "#define QSB_L2_SKIP 1" in cu or "QSB_L2_SKIP" in cu
print("PASS: resolve-last peel binder")
