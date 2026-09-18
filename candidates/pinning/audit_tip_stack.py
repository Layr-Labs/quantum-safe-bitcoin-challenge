#!/usr/bin/env python3
"""Source-shape binder for tip-adapted stack (slots + warp + PK_UNROLL + __ldg)."""
from pathlib import Path
cu = (Path(__file__).resolve().parent / 'pinning.cu').read_text()
assert '#define QSB_L2_SKIP 1' in cu
assert '#define QSB_COFACTOR 1' in cu
assert '#define QSB_DIRDIG 1' in cu
assert '#define QSB_SQFREE 1' in cu
assert '#define QSB_SPARSE_D 1' in cu
assert '_SHA256TransformDigest32' in cu and '_SHA256TransformPubkey33' in cu
assert '#define QSB_SLOTS 2' in cu
assert '#define QSB_PK_UNROLL 1' in cu
# EARLY_LOAD left off: tip lineage measured negative
assert '#define QSB_EARLY_LOAD 0' in cu or 'EARLY_LOAD 0' in cu
assert '__ldg(tx+0)' in cu and '__ldg(ty+0)' in cu
assert 'else if(count>2)__syncwarp();' in cu
assert 'if(count>=32)__syncthreads();' in cu and 'else __syncwarp();' in cu
assert 'drain_slot' in cu and 'slot_stream' in cu
print('audit_tip_stack: PASS')
