#!/usr/bin/env python3
"""Stage a dual-table subset grinder; never overwrite submitted production."""
import json
from pathlib import Path
import re
import shutil
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SUBSET=HERE.parent.parent
sys.path.insert(0,str(SUBSET));sys.path.insert(0,str(HERE.parent/'wide_windows'))
from preflight import source_identity
from check_candidate import function

small=HERE.parent/'wide_windows/control'
wide=HERE.parent/'wide_windows/lookahead/prefetch'
assert source_identity(small)['source_fingerprint']=='44f9d5808add05e204b169194af8893dc91a9c01d31448362b0e01c667c9fd06'
assert source_identity(wide)['source_fingerprint']=='d0174fa761b852b0f30f71246dd3e43b3e1a053dc9c011b8a4508d246eb0af18'
out=HERE/'candidate'
if out.exists():raise SystemExit('Refusing to overwrite existing adaptive candidate')
shutil.copytree(wide,out)
dest=out/'tests/gpu_epochs'
old=(small/'tests/gpu_epochs/tree.cu').read_text()
s=(dest/'tree.cu').read_text()

def rename(text):
    text=re.sub(r'\bGT_', 'SMALL_',text)
    # Setup/digit-to-index are geometry-independent and remain shared.
    for before,after in [('gt_recode_step','small_recode_step'),('gt_recode_signed','small_recode_signed'),
        ('gt_load_signed','small_load_signed'),('_FixedBaseSignedXYZZ','small_fixed_xyzz'),
        ('kernel_build_gtable','small_build_kernel'),('gt_point_to_limbs','small_point_to_limbs'),
        ('gt_build_ladders','small_build_ladders'),('gt_spot_check','small_spot_check'),
        ('compute_gtable','small_compute_gtable')]:
        text=re.sub(r'\b'+before+r'\b',after,text)
    return text

geometry='''#pragma once
#define SMALL_CHUNKS 16
#define SMALL_ENTRIES (1u<<15)
#define SMALL_LO 256
#define SMALL_HI 256
'''
device_signatures=['__device__ __forceinline__ int32_t gt_recode_step(',
    '__device__ __forceinline__ void gt_recode_signed(',
    '__device__ __forceinline__ void gt_load_signed(', '__device__ void _FixedBaseSignedXYZZ(']
small_device=geometry+'\n'.join(rename(function(old,key)) for key in device_signatures)
(dest/'small_table_device.cuh').write_text(small_device+'\n')
marker='__device__ void _FixedBaseSignedProj('
s=s.replace(marker,'#include "small_table_device.cuh"\n\n'+marker)
wrapper=function(s,marker)
s=s.replace(wrapper,wrapper.replace('_FixedBaseSignedXYZZ(','small_fixed_xyzz('))
s=s.replace('#include "wide_table_kernels.cuh"',
    '#include "wide_table_kernels.cuh"\n\n'+rename(function(old,'__global__ void kernel_build_gtable(')))

host='\n'.join(rename(function(old,key)) for key in ['static void gt_point_to_limbs(',
    'static void gt_build_ladders(', 'static int gt_spot_check(', 'static void compute_gtable('])
host+='''
static void small_build_table(uint8_t **tableX,uint8_t **tableY,const uint8_t nri[32]){
    const size_t bytes=(size_t)SMALL_CHUNKS*SMALL_ENTRIES*32;
    wide_cuda_require(cudaMalloc(tableX,bytes),"allocate small table X");
    wide_cuda_require(cudaMalloc(tableY,bytes),"allocate small table Y");
    const size_t lb=(size_t)SMALL_CHUNKS*SMALL_LO*8*sizeof(uint64_t);
    const size_t hb=(size_t)SMALL_CHUNKS*SMALL_HI*8*sizeof(uint64_t);
    std::vector<uint64_t>L(lb/8),H(hb/8);
    small_build_ladders(L.data(),H.data(),nri);
    uint64_t *dL=nullptr,*dH=nullptr;
    wide_cuda_require(cudaMalloc(&dL,lb),"allocate small L");
    wide_cuda_require(cudaMalloc(&dH,hb),"allocate small H");
    wide_cuda_require(cudaMemcpy(dL,L.data(),lb,cudaMemcpyHostToDevice),"copy small L");
    wide_cuda_require(cudaMemcpy(dH,H.data(),hb,cudaMemcpyHostToDevice),"copy small H");
    small_build_kernel<<<(SMALL_CHUNKS*SMALL_ENTRIES+255)/256,256>>>(dL,dH,*tableX,*tableY);
    wide_cuda_require(cudaGetLastError(),"small table launch");
    wide_cuda_require(cudaDeviceSynchronize(),"small table execution");
    wide_cuda_require(cudaFree(dL),"free small L");wide_cuda_require(cudaFree(dH),"free small H");
    std::vector<uint8_t>x(bytes),y(bytes);
    wide_cuda_require(cudaMemcpy(x.data(),*tableX,bytes,cudaMemcpyDeviceToHost),"check small X");
    wide_cuda_require(cudaMemcpy(y.data(),*tableY,bytes,cudaMemcpyDeviceToHost),"check small Y");
    if(!small_spot_check(x.data(),y.data(),SMALL_CHUNKS*4+192,nri)){
        small_compute_gtable(x.data(),y.data(),nri);
        if(!small_spot_check(x.data(),y.data(),SMALL_CHUNKS*4+192,nri)){
            fprintf(stderr,"Small table OpenSSL fallback failed\\n");exit(2);
        }
        wide_cuda_require(cudaMemcpy(*tableX,x.data(),bytes,cudaMemcpyHostToDevice),"copy fallback X");
        wide_cuda_require(cudaMemcpy(*tableY,y.data(),bytes,cudaMemcpyHostToDevice),"copy fallback Y");
    }
}
'''
(dest/'small_table_host.cuh').write_text('#pragma once\n'+host)
s=s.replace('#include "wide_table_host.cuh"',
    '#include "wide_table_host.cuh"\n#include "small_table_host.cuh"\n#include "table_resources.cuh"\n#include "table_tuner.cuh"')
start=s.index('    // Bound the table plus ranked state')
end=s.index('    /* Upload params */',start)
s=s[:start]+'''    // The established32MiB path is always available. Generic modes use it.
    uint8_t *d_gtX=nullptr,*d_gtY=nullptr,*d_wide=nullptr;
    small_build_table(&d_gtX,&d_gtY,dp.neg_r_inv);
    d_wide=qsb_optional_wide_table(se_mode,dp.neg_r_inv);

'''+s[end:]
marker='        uint64_t epoch_base = 0;'
assert s.count(marker)==1
s=s.replace(marker,'        QsbTableTuner table_tuner(d_wide!=nullptr);\n'+marker)
marker='            uint32_t h_hit = 0;\n            cudaMemcpy(d_hit_cnt, &h_hit, 4, cudaMemcpyHostToDevice);\n            kernel_build_epochs'
assert s.count(marker)==1
s=s.replace(marker,marker.replace('            kernel_build_epochs',
    '            bool use_wide=table_tuner.wide();\n            table_tuner.begin();\n            kernel_build_epochs'))
old_call='''                    d_gtX,d_gtY,d_u2rx,d_u2ry,d_pipe_state,d_pipe_roots,d_pipe_tree,
                    d_pipe_super,d_pipe_root_tree,d_hit_cnt,d_hit_idx,d_hit_combos);'''
new_call='''                    use_wide?d_wide:d_gtX,use_wide?nullptr:d_gtY,d_u2rx,d_u2ry,
                    d_pipe_state,d_pipe_roots,d_pipe_tree,
                    d_pipe_super,d_pipe_root_tree,d_hit_cnt,d_hit_idx,d_hit_combos,use_wide);'''
assert old_call in s;s=s.replace(old_call,new_call)
marker='''            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) { printf("CUDA error: %s\\n", cudaGetErrorString(err)); return 1; }
            total_searched += batch_pos;'''
assert s.count(marker)==1
s=s.replace(marker,'''            table_tuner.end();
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) { printf("CUDA error: %s\\n", cudaGetErrorString(err)); return 1; }
            table_tuner.observe(batch_pos);
            if(d_wide&&!table_tuner.policy.pending()&&!table_tuner.wide()){
                wide_cuda_require(cudaFree(d_wide),"release unused wide table");
                d_wide=nullptr;
            }
            total_searched += batch_pos;''')
(dest/'tree.cu').write_text(s)

p=(dest/'ranked_pipeline.cuh').read_text()
prepare=function(p,'__global__ void __launch_bounds__(256,2) qsb_ranked_prepare(')
small_prepare=prepare.replace('qsb_ranked_prepare(', 'qsb_ranked_prepare_small(').replace('_FixedBaseSignedXYZZ(','small_fixed_xyzz(')
p=p.replace(prepare,prepare+'\n\n'+small_prepare)
p=p.replace('uint32_t *hit_count,uint32_t *hit_idx,uint8_t *hit_combos) {',
            'uint32_t *hit_count,uint32_t *hit_idx,uint8_t *hit_combos,bool use_wide) {')
old_launch='    qsb_ranked_prepare<<<blocks,256>>>(epochs,gTX,gTY,rx,state,roots,tree,count);'
assert old_launch in p
p=p.replace(old_launch,'''    if(use_wide)qsb_ranked_prepare<<<blocks,256>>>(epochs,gTX,gTY,rx,state,roots,tree,count);
    else qsb_ranked_prepare_small<<<blocks,256>>>(epochs,gTX,gTY,rx,state,roots,tree,count);''')
(dest/'ranked_pipeline.cuh').write_text(p)
for name in ['table_policy.cuh','table_tuner.cuh','table_resources.cuh']:shutil.copyfile(HERE/name,dest/name)
record={'status':'isolated adaptive-table prototype; not submitted',
    'small_control':source_identity(small),'wide_control':source_identity(wide),
    'candidate':source_identity(out),'geometry_invariant':'Both tables use the current runtime neg_r_inv base; all trial candidates/hits remain in the ordinary enumeration and output.',
    'evidence_limit':'No GPU execution or speed claim. Production remains unchanged.'}
(HERE/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'candidate':record['candidate']['source_fingerprint']}))
