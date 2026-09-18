#!/usr/bin/env python3
"""Binder: tip-adapted resolve-last peel + streaming pipeline traffic on 2dc72281."""
from pathlib import Path

cu = Path(__file__).with_name("pinning.cu").read_text()
gm = Path(__file__).with_name("GPUMath.h").read_text()

assert "#define QSB_RESOLVE_LAST 1" in cu
assert "for(int c=2;c<GT_CHUNKS-1;c++)" in cu
assert "qsb_load_decoded(table,GT_CHUNKS-1,base,x1,y1)" in cu
assert cu.count("_PointAddXYZZT<false>(X,Y,U,V,x1,y1,y0);") == 1
assert "for(int c=2;c<GT_CHUNKS;c++)" in cu  # tip else-path retained
assert "_ModMult(x1,y0,V);_ModSub256(Y,Y,x1);" in cu

assert "#define QSB_STREAM 1" in cu
assert "st.global.cs.v2.u64" in cu
assert "ld.global.cs.v2.u64" in cu

# Tip stack retained
assert "#define QSB_SLOTPIPE 1" in cu
assert "#define QSB_SLOTS 2" in cu
assert "#define QSB_L2_SKIP 1" in cu
assert "#define QSB_FUSE_SQRADDSUB2 0" in gm  # do not resubmit rejected fuse-alone
assert "template<bool DEFER_Y>" in gm
assert "__device__ __forceinline__ void _PointAddXYZZT(" in gm

print("PASS: resolve-last + STREAM compose on slotted tip; fuse left off")
