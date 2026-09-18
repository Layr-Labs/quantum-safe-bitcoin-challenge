#!/usr/bin/env python3
"""CPU-side shape binder for QSB_ROOT_WARP_BAR + QSB_FINISH_ROOT_SH on tip bb5c9a0."""
from pathlib import Path
import re, sys
root = Path(__file__).resolve().parent
cu = (root / "pinning.cu").read_text()
checks = []
def ok(name, cond, detail=""):
    checks.append((name, bool(cond), detail))

ok("tip STREAM=1", re.search(r"#define QSB_STREAM 1", cu))
ok("tip SLOTPIPE=1", re.search(r"#define QSB_SLOTPIPE 1", cu))
ok("tip SLOTS=2", re.search(r"#define QSB_SLOTS 2", cu))
ok("tip EARLY_LOAD=0", re.search(r"#define QSB_EARLY_LOAD 0", cu))
ok("tip L2_SKIP=1", re.search(r"#define QSB_L2_SKIP 1", cu))
ok("tip SPARSE_D=1", re.search(r"#define QSB_SPARSE_D 1", cu))
ok("ROOT_WARP_BAR default 1", re.search(r"#define QSB_ROOT_WARP_BAR 1", cu))
ok("FINISH_ROOT_SH default 1", re.search(r"#define QSB_FINISH_ROOT_SH 1", cu))
ok("product syncwarp branch", "if(count>2){if(half>32)__syncthreads();else __syncwarp();}" in cu)
ok("inverse syncwarp branch", "if((count<<1)>32)__syncthreads();else __syncwarp();" in cu)
ok("finish shared root arrays", "sh_root_inv[4]" in cu and "sh_weighted_inv[4]" in cu)
ok("finish __ldg root load", "__ldg(&roots[4ull*blockIdx.x+threadIdx.x])" in cu)
ok("finish barrier before early return",
   cu.find("sh_root_inv") < cu.find("if(!active)return;") and
   "__syncthreads();\n#endif\n    if(!active)return;" in cu.replace("\r",""))
ok("no packed-plane STREAM on saved[0]", "qsb_st_v2(&saved[0" not in cu)
ok("DIRECT_DIGITS still conflicts with EARLY_LOAD",
   "QSB_DIRECT_DIGITS && (QSB_PREFETCH || QSB_S0_SHM || QSB_EARLY_LOAD)" in cu)
ok("no TREE_OFFLOAD enabled", re.search(r"#define QSB_TREE_OFFLOAD 0", cu))
ok("cofactor geometry assert intact",
   "!QSB_TREE_OFFLOAD && !QSB_TREE_OFFLOAD2" in cu)

# Mirror leaf cofactor pattern already present
cof = (root / "cofactor_checkpoint.h").read_text()
ok("leaf cofactor already warp-scoped",
   "if(half>32)__syncthreads();else __syncwarp();" in cof)

failed = [n for n,c,d in checks if not c]
for n,c,d in checks:
    print(("PASS" if c else "FAIL"), n, d)
if failed:
    sys.exit(f"failed: {failed}")
print("OK", len(checks), "checks")
