#!/usr/bin/env python3
"""Source-shape binder: packed vbar/tbar planes use tip STREAM helpers; tip SLOTS=2."""
from pathlib import Path

root = Path(__file__).resolve().parent
cu = (root / "pinning.cu").read_text()
pr = (root / "PackedRecovery.cuh").read_text()
gm = (root / "GPUMath.h").read_text()

checks = [
    (cu, "#define QSB_STREAM 1", "STREAM retained on tip"),
    (cu, "#define QSB_SLOTPIPE 1", "SLOTPIPE retained"),
    (cu, "#define QSB_SLOTS 2", "tip two-slot default retained"),
    (cu, "#define QSB_STATE_PLANES 4u", "four production planes"),
    (cu, "#define QSB_TREE_OFFLOAD 0", "tree offload stays off"),
    (cu, "#define QSB_TREE_OFFLOAD2 0", "tree offload2 stays off"),
    (cu, "#define QSB_PREFETCH 0", "prefetch left off"),
    (cu, "#define QSB_EARLY_LOAD 0", "early-load left off"),
    (cu, "#define QSB_S0_SHM 0", "s0 shm left off"),
    (cu, "ulonglong2 y01=qsb_ld_v2(&saved[0*s+i]),y23=qsb_ld_v2(&saved[1*s+i]);",
     "finish loads plane0/1 via qsb_ld_v2"),
    (cu, "ulonglong2 v01=qsb_ld_v2(&saved[2*s+i]),v23=qsb_ld_v2(&saved[3*s+i]);",
     "finish loads plane2/3 via qsb_ld_v2"),
    (pr, "qsb_st_v2(&saved[0*s+i],vbar[0],vbar[1]);", "prepare stores plane0 via qsb_st_v2"),
    (pr, "qsb_st_v2(&saved[1*s+i],vbar[2],vbar[3]);", "prepare stores plane1 via qsb_st_v2"),
    (pr, "qsb_st_v2(&saved[2*s+i],tbar[0],tbar[1]);", "prepare stores plane2 via qsb_st_v2"),
    (pr, "qsb_st_v2(&saved[3*s+i],tbar[2],tbar[3]);", "prepare stores plane3 via qsb_st_v2"),
]
failed = []
for src, token, label in checks:
    if token not in src:
        failed.append(label)

for ban, label, src in [
    ("#define QSB_SLOTS 3", "must not retry SLOTS=3 deepen", cu),
    ("#define QSB_RESOLVE_LAST 1", "resolve-last must stay off", cu),
    ("#define QSB_FUSE_SQRADDSUB2 1", "fuse must not be primary/on", gm + cu),
    ("saved[0*s+i]=make_ulonglong2", "prepare must not use plain make_ulonglong2 plane0", pr),
    ("saved[1*s+i]=make_ulonglong2", "prepare must not use plain make_ulonglong2 plane1", pr),
    ("saved[2*s+i]=make_ulonglong2", "prepare must not use plain make_ulonglong2 plane2", pr),
    ("saved[3*s+i]=make_ulonglong2", "prepare must not use plain make_ulonglong2 plane3", pr),
    ("ulonglong2 y01=saved[0*s+i],y23=saved[1*s+i];", "finish must not plain-load plane0/1", cu),
    ("ulonglong2 v01=saved[2*s+i],v23=saved[3*s+i];", "finish must not plain-load plane2/3", cu),
]:
    if ban in src:
        failed.append(label)

if failed:
    raise SystemExit("FAIL: " + "; ".join(failed))
print("PASS: packed-plane STREAM shape; %d positive binders" % len(checks))
