#!/usr/bin/env python3
"""Prototype three-field recovery state and rebuild-only search inverse trees."""
from pathlib import Path
import hashlib,json,shutil,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;PIN=HERE.parents[1];OUT=HERE/'candidate'
sys.path.insert(0,str(PIN))
from check_projective import function
receipt=json.loads((PIN/'research/wide_windows/submitted-source.json').read_text())
OUT.mkdir(exist_ok=True)
for name,h in receipt['production_sha256'].items():
    p=PIN/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
    shutil.copyfile(p,OUT/name)
s=(OUT/'pinning.cu').read_text()
product='__device__'+function(s,'__device__ __forceinline__ void qsb_block_product_checkpoint(')
product=product.replace('qsb_block_product_checkpoint(', 'qsb_block_product_only(',1)
product=product.replace('uint64_t *value, uint64_t *roots, uint64_t *checkpoint','uint64_t *value, uint64_t *roots, uint64_t *checkpoint')
product=product.replace('QSB_CHECKPOINT_STRIDE;','128u;')
product=product.replace('if(node<510)','if(node>=384 && node<510)')
product=product.replace('(size_t)k*QSB_CHECKPOINT_STRIDE+node-256','(size_t)k*128u+node-384')
rebuild='__device__'+function(s,'__device__ __forceinline__ void qsb_block_inverse(')
rebuild=rebuild.replace('qsb_block_inverse(uint64_t *value)','qsb_block_inverse_rebuild(uint64_t *value,const uint64_t *roots)',1)
a=rebuild.index('    if(tid==0){\n        uint64_t root[5];')
b=rebuild.index('    __syncthreads();',a)
rebuild=rebuild[:a]+'''    if(tid==0){
        #pragma unroll
        for(int k=0;k<4;k++)inverses[k][254]=roots[(size_t)blockIdx.x*4u+k];
    }
'''+rebuild[b:]
# Rebuild just the first 128 nodes; load the remaining 126 from a half-size checkpoint.
rebuild=rebuild.replace('const uint64_t *roots)', 'const uint64_t *roots,const uint64_t *checkpoint)',1)
a=rebuild.index('    int offset=0;')
b=rebuild.index('    if(tid==0){',a)
rebuild=rebuild[:a]+'''    if(tid<128){
        uint64_t left[5]={},right[5]={},out[5];
        #pragma unroll
        for(int k=0;k<4;k++){left[k]=products[k][tid];right[k]=products[k][tid+128];}
        qsb_field_mul(out,left,right);
        #pragma unroll
        for(int k=0;k<4;k++)products[k][256+tid]=out[k];
    }
    if(tid<126){
        #pragma unroll
        for(int k=0;k<4;k++)products[k][384+tid]=checkpoint[((size_t)blockIdx.x*4u+k)*128u+tid];
    }
    int offset=0;
'''+rebuild[b:]
assert '_ModInv' not in rebuild
old_finish='__device__'+function(s,'__device__ __forceinline__ uint32_t qsb_xyzz_finish_precomputed(')
finish=old_finish.replace('qsb_xyzz_finish_precomputed(', 'qsb_xyzz_finish_three(',1)
finish=finish.replace('uint64_t *C, uint64_t *Y, uint64_t *W, uint64_t *ZZZ,','uint64_t *Y, uint64_t *W, uint64_t *ZZZ,',1)
a=finish.index('    uint64_t yb[4], m[4], t[4], s[4];')
b=finish.index('    _ModSub256(m, yb, Y);',a)
finish=finish[:a]+'''    uint64_t delta[4],yb[4],m[4],t[4],s[4];
    // inv is 1/(W*B), with B=ZZZ. Since C*B^2=W^2,
    // C/W = W/B^2; h=B/W. No fourth checkpoint field is needed.
    _ModMult(yb,yR,ZZZ);
    _ModMult(delta,W,inv);      // 1/B
    _ModSqr(delta,delta);
    _ModMult(delta,W);          // W/B^2 = xR-xP
    _ModSqr(ZZZ,ZZZ);
    _ModMult(ZZZ,inv);          // B/W
    _ModAdd256(W,xR,xR);
    _ModSub256(W,delta);        // xP+xR

'''+finish[b:]
finish=finish.replace('(((s[0] & 1ULL) ^ 1ULL) << 1)','((((s[0]|s[1]|s[2]|s[3]) != 0) && !(s[0]&1ULL)) << 1)')
helper='// Isolated research helpers. GPL-3.0; inherited notices remain in candidate/COPYING.\n#pragma once\n'+product+'\n\n'+rebuild+'\n\n'+finish+'\n'
(OUT/'three_state.cuh').write_text(helper)
s=s.replace('template<bool FAST_TAIL, int STAGE>','#include "three_state.cuh"\n\ntemplate<bool FAST_TAIL, int STAGE>',1)
a=s.index('    /* Preserve exactly four fields across the kernel boundary.')
b=s.index('    if(active){',a)
s=s[:a]+'''    // Save Y, W and B=ZZZ. The inverse tree now consumes D=W*B.
    Load256(qzz,prod);
'''+s[b:]
a=s.index('        /* Eight vector planes retain SoA')
b=s.index('    qsb_block_product_checkpoint(prod,roots,tree);',a)
stores='''        // Six aligned vector planes; no per-candidate C checkpoint.
        size_t state_plane_stride=(size_t)batch_size;
        size_t state_idx=(size_t)idx;
'''
for i,name in enumerate(('qy','qzz','qzzz')):
    for pair in range(2):stores+=f'        saved[{2*i+pair}u*state_plane_stride+state_idx]=make_ulonglong2({name}[{2*pair}],{name}[{2*pair+1}]);\n'
s=s[:a]+stores+'    }\n'+s[b:]
s=s.replace('    qsb_block_product_checkpoint(prod,roots,tree);','    if(usable)_ModMult(prod,qzzz);\n    else {prod[0]=1;prod[1]=prod[2]=prod[3]=prod[4]=0;}\n    qsb_block_product_only(prod,roots,tree);',1)
a=s.index('        ulonglong2 qx01=saved[')
b=s.index('        usable = ',a)
loads=''
for i,name in enumerate(('qy','qzz','qzzz')):
    loads+=f'        ulonglong2 {name}01=saved[{2*i}u*state_plane_stride+state_idx];\n'
    loads+=f'        ulonglong2 {name}23=saved[{2*i+1}u*state_plane_stride+state_idx];\n'
    loads+=f'        {name}[0]={name}01.x;{name}[1]={name}01.y;{name}[2]={name}23.x;{name}[3]={name}23.y;\n'
s=s[:a]+loads+s[b:]
a=s.index('    /* qzz carries the original, pre-identity-substitution W,')
b=s.index('    if (!usable) return;',a)
s=s[:a]+'''    // Recreate the exact same leaf D. Unusable lanes remain identity factors.
    if(usable)_ModMult(prod,qzz,qzzz);
    else {prod[0]=1;prod[1]=prod[2]=prod[3]=0;}
    prod[4]=0;
    qsb_block_inverse_rebuild(prod,roots,tree);
'''+s[b:]
s=s.replace('qsb_xyzz_finish_precomputed(\n        qx,qy,qzz,qzzz,prod,u2rx,u2ry,','qsb_xyzz_finish_three(\n        qy,qzz,qzzz,prod,u2rx,u2ry,',1)
s=s.replace('size_t pipeline_state_bytes=(size_t)BATCH*8u*sizeof(ulonglong2);','size_t pipeline_state_bytes=(size_t)BATCH*6u*sizeof(ulonglong2);',1)
s=s.replace('size_t pipeline_tree_bytes=(size_t)GRDSZ*4u*QSB_CHECKPOINT_STRIDE*sizeof(uint64_t);','size_t pipeline_tree_bytes=(size_t)GRDSZ*4u*128u*sizeof(uint64_t); // Half the search nodes are reconstructed.',1)
# Keep allocation for the remaining half-size search checkpoints.
# Delay Y load until after inversion, reducing values held through the collective.
a=s.index('        ulonglong2 qy01=saved[');b=s.index('        ulonglong2 qzz01=saved[',a)
late_y=s[a:b]
s=s[:a]+s[b:]
a=s.index('    if (!usable) return;')+len('    if (!usable) return;')
s=s[:a]+'\n    {size_t state_plane_stride=(size_t)batch_size;size_t state_idx=(size_t)idx;\n'+late_y+'    }\n'+s[a:]
# Keep the conservative existing memory gate for this isolated compile/cost screen.
(OUT/'pinning.cu').write_text(s)
provenance={'base_submission':receipt['submission_id'],'base_fingerprint':receipt['fingerprint'],
 'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()},
 'production_modified':False,'change':'Search only: three fields Y,W,B; batch inverse D=W*B; rebuild128 search tree nodes, checkpoint126 upper nodes. Outer root hierarchy, table builder and fused path unchanged.'}
(HERE/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Prepared',OUT)
