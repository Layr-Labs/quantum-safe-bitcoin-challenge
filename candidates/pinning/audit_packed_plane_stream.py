#!/usr/bin/env python3
"""Source-shape binder: tip STREAM2 packed planes intact; EARLY_LOAD Scalar lever on."""
from pathlib import Path

root = Path(__file__).resolve().parent
cu = (root / "pinning.cu").read_text()
pr = (root / "PackedRecovery.cuh").read_text()

checks = [
    (cu, "#define QSB_STREAM 1", "STREAM retained on tip"),
    (cu, "#define QSB_STREAM2 1", "STREAM2 crown retained"),
    (cu, "#define QSB_SLOTPIPE 1", "SLOTPIPE retained"),
    (cu, "#define QSB_SLOTS 2", "tip two-slot default retained"),
    (cu, "#define QSB_TREE_OFFLOAD 0", "tree offload stays off"),
    (cu, "#define QSB_TREE_OFFLOAD2 0", "tree offload2 stays off"),
    (cu, "#define QSB_PREFETCH 0", "prefetch left off"),
    (cu, "#define QSB_EARLY_LOAD 1", "early-load Scalar lever on"),
    (cu, "#define QSB_S0_SHM 0", "s0 shm left off"),
    (cu, "ulonglong2 y01=qsb_ld_v2(&saved[0*s+i]),y23=qsb_ld_v2(&saved[1*s+i]);",
     "finish loads plane0/1 via qsb_ld_v2"),
    (cu, "ulonglong2 v01=qsb_ld_v2(&saved[2*s+i]),v23=qsb_ld_v2(&saved[3*s+i]);",
     "finish loads plane2/3 via qsb_ld_v2"),
    (pr, "qsb_st_v2(&saved[0*s+i],vbar[0],vbar[1]);", "prepare stores plane0 via qsb_st_v2"),
    (pr, "qsb_st_v2(&saved[1*s+i],vbar[2],vbar[3]);", "prepare stores plane1 via qsb_st_v2"),
    (pr, "qsb_st_v2(&saved[2*s+i],tbar[0],tbar[1]);", "prepare stores plane2 via qsb_st_v2"),
    (pr, "qsb_st_v2(&saved[3*s+i],tbar[2],tbar[3]);", "prepare stores plane3 via qsb_st_v2"),
    (cu, "_PointAddXYZZT_early(", "Scalar early madd present"),
]
failed = []
for src, token, label in checks:
    if token not in src:
        failed.append(label)

for ban, label, src in [
    ("#define QSB_SLOTS 3", "must not retry SLOTS=3 deepen", cu),
    ("#define QSB_RESOLVE_LAST 1", "resolve-last must stay off", cu),
]:
    if ban in src:
        failed.append(label)

if failed:
    raise SystemExit("FAIL: " + "; ".join(failed))
print("PASS: STREAM2 tip + EARLY_LOAD; %d positive binders" % len(checks))
