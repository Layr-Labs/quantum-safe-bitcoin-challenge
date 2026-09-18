#!/usr/bin/env python3
"""Source-bound PR98 checkpoint transfer, preserving the pending PR74 closure."""
import hashlib,json,shutil,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent; PIN=HERE.parents[1]
sys.path.insert(0,str(PIN.parent/'subset'))
from check_candidate import function

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((PIN/'research/wide_windows/submitted-source.json').read_text())
public=HERE/'pr98-ranked_pipeline.cuh'
assert sha(public)=='0a868f6070b27602c6a944e7dc4f7f2f0d98ca63f42f0731d3b8c8e4c7737ed2'
original=(PIN/'pinning.cu').read_text();remote=public.read_text()
signatures=['__device__ __forceinline__ void qsb_block_product_checkpoint(',
            '__device__ __forceinline__ void qsb_block_inverse_checkpoint(']
provenance={'base_submission':receipt['submission_id'],'base_fingerprint':receipt['fingerprint'],
 'public_pr':98,'public_head':'2d5eba297f8d9823ca0f5a552d3d1788fbe3031c',
 'public_file_sha256':sha(public),'coauthor_if_submitted':'Meganpark980320','variants':{}}
for variant in ['warp_full','warp_compact','warp_compact_fused']:
    out=HERE/variant;out.mkdir(exist_ok=True)
    for n,h in receipt['production_sha256'].items():
        assert sha(PIN/n)==h,('Pending production changed',n)
        shutil.copyfile(PIN/n,out/n)
    src=original
    if variant!='warp_full':
        definitions='#define QSB_CHECKPOINT_FIRST 448\n#define QSB_CHECKPOINT_NODES 62\n#define QSB_CHECKPOINT_STRIDE 64\n'
        helpers='\n\n'.join(function(remote,sig) for sig in signatures)
    else:
        definitions='#define QSB_CHECKPOINT_NODES 254\n#define QSB_CHECKPOINT_STRIDE 256\n'
        product=function(original,signatures[0]).replace('if(count>2)__syncthreads();',
            'if(count>64)__syncthreads(); else if(count>2)__syncwarp();')
        inverse=function(original,signatures[1]).replace('        __syncthreads();',
            '        if(count>=32)__syncthreads(); else __syncwarp();')
        helpers=product+'\n\n'+inverse
    header='// GPL-3.0. Checkpoint/synchronization mechanism from Meganpark980320 PR98.\n// Exact source head 2d5eba297f8d9823ca0f5a552d3d1788fbe3031c.\n#pragma once\n'+definitions+helpers+'\n'
    (out/'warp_checkpoint.cuh').write_text(header)
    begin=src.index('#define QSB_CHECKPOINT_NODES 254')
    end=src.index('/* Batch the per-search-CTA roots one level further.',begin)
    src=src[:begin]+'#include "warp_checkpoint.cuh"\n\n'+src[end:]
    if variant=='warp_compact_fused':
        sig='__device__ __forceinline__ void qsb_block_inverse('
        before=function(src,sig)
        after=before.replace('if(count>2)__syncthreads();',
            'if(count>64)__syncthreads(); else if(count>2)__syncwarp();')
        after=after.replace('        __syncthreads();',
            '        if(count>=32)__syncthreads(); else __syncwarp();')
        assert after!=before
        src=src.replace(before,after,1)
    (out/'pinning.cu').write_text(src)
    provenance['variants'][variant]={'source_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()},
        'tree_nodes':254 if variant=='warp_full' else 62,
        'scope':'Search and root-group checkpoint helpers; coordinate state and field primitives unchanged.',
        'fused_tree_warp_boundaries':variant=='warp_compact_fused'}
(HERE/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Prepared three isolated checkpoint variants; pending production unchanged.')
