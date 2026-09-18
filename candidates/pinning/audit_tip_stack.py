#!/usr/bin/env python3
"""Source-shape binder for the tip-adapted warp-barrier + switch + __ldg stack."""
from pathlib import Path
root = Path(__file__).resolve().parent
cu = (root / 'pinning.cu').read_text()
cof = (root / 'cofactor_checkpoint.h').read_text()
assert '#define QSB_L2_SKIP 1' in cu
assert '#define QSB_PK_UNROLL 1' in cu
assert '#define QSB_EARLY_LOAD 1' in cu
assert '__ldg(tx+0)' in cu and '__ldg(ty+0)' in cu
assert 'else if(count>2)__syncwarp();' in cu
assert 'if(count>=32)__syncthreads();' in cu and 'else __syncwarp();' in cu
assert 'else if(count>2)__syncwarp();' in cof
assert 'if(count>=32)__syncthreads();' in cof
# Digests/pubkeys still tip-on via SPARSE_D — do not double-apply
assert '#define QSB_SPARSE_D 1' in cu
assert '_SHA256TransformDigest32' in cu and '_SHA256TransformPubkey33' in cu
print('audit_tip_stack: PASS')
