#!/usr/bin/env python3
"""Source-shape binder for tip-adapted multi-axis pinning stack on 713.2M."""
from pathlib import Path
root = Path(__file__).resolve().parent
cu = (root / 'pinning.cu').read_text()
# Tip-already-on — must remain, never double-apply as "new"
assert '#define QSB_L2_SKIP 1' in cu
assert '#define QSB_COFACTOR 1' in cu
assert '#define QSB_DIRDIG 1' in cu
assert '#define QSB_SQFREE 1' in cu
assert '#define QSB_SPARSE_D 1' in cu
assert '_SHA256TransformDigest32' in cu and '_SHA256TransformPubkey33' in cu
# This archive's levers
assert '#define QSB_PK_UNROLL 1' in cu
assert '#define QSB_EARLY_LOAD 1' in cu
assert '#define QSB_SLOTS 2' in cu
assert '__ldg(tx+0)' in cu and '__ldg(ty+0)' in cu
assert 'else if(count>2)__syncwarp();' in cu
assert 'if(count>=32)__syncthreads();' in cu and 'else __syncwarp();' in cu
assert 'cudaStream_t st' in cu
assert 'drain_slot' in cu
assert 'slot_stream' in cu
print('audit_tip_stack: PASS')
