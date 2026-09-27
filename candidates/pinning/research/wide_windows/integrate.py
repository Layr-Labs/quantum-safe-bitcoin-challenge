#!/usr/bin/env python3
"""Materialize the research ten-window candidate from the exact corrected PR64.

Only writes this directory's candidate/. Never selects a track or submits.
"""
import hashlib,json,shutil,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from ptx_field_model import function
from fused_screen import make_fused

def main():
    base=HERE.parent/'pr64_corrected';out=HERE/'candidate';out.mkdir(exist_ok=True)
    for name in ('GPUMath.h','GPUHash.h','COPYING'):shutil.copyfile(base/name,out/name)
    shutil.copyfile(HERE/'wide_geometry.cuh',out/'wide_geometry.cuh')
    s=(base/'pinning.cu').read_text().replace('#include <cuda_runtime.h>','#include <cuda_runtime.h>\n#include <vector>')
    start=s.index('/* Mixed regular odd digits:');end=s.index('/* n = secp256k1',start)
    s=s[:start]+'''/* Ten signed odd windows [26x6,25x4], 16 GiB runtime-specific table.
 * Keeps PR64 streamed recoding, deferred XYZZ and no-alias contracts. */
#include "wide_geometry.cuh"
#define GT_CHUNKS WIDE_CHUNKS
#define GT_TOTAL_ENTRIES ((uint32_t)WIDE_TOTAL_ENTRIES)
#define GT_LO 8192
#define GT_HI 8192
__host__ __device__ __forceinline__ unsigned gt_entries(int c){return wide_entries(c);}
__host__ __device__ __forceinline__ unsigned gt_offset(int c){return wide_offset(c);}
__host__ __device__ __forceinline__ int gt_shift(int c){return wide_shift(c);}

'''+s[end:]
    old=function(s,'__device__ __forceinline__ void gt_recode_signed(')
    s=s.replace(old,'''__device__ __forceinline__ void gt_recode_signed(const uint64_t k[4],int32_t e[GT_CHUNKS]){
    uint64_t M[4];int sign;gt_recode_setup(k,M,&sign);
    #pragma unroll
    for(int c=0;c<GT_CHUNKS-1;c++)e[c]=wide_step(M,sign,wide_bits(c));
    e[GT_CHUNKS-1]=sign*(int32_t)M[0];
}''')
    old=function(s,'__device__ void _FixedBaseSignedXYZZScalar(')
    new=function((HERE/'wide_probe.cu').read_text(),'__device__ void wide_fixed(')
    new=new.replace('__device__ void wide_fixed(uint64_t *X,uint64_t *Y,uint64_t *ZZ,uint64_t *ZZZ,const uint64_t k[4],const uint8_t *table)',
      '__device__ void _FixedBaseSignedXYZZScalar(uint64_t *__restrict__ X,uint64_t *__restrict__ Y,uint64_t *__restrict__ ZZ,uint64_t *__restrict__ ZZZ,const uint64_t *__restrict__ k,const uint8_t *__restrict__ table)')
    assert 'wide_fixed' not in new
    s=s.replace(old,new)
    s=s.replace('/* u1*G as raw XYZZ via the signed 64 MiB A-table. */',
                '/* u1*G as raw XYZZ via the signed 16 GiB runtime A-table. */')
    s=s.replace('3M+2S + 12*(7M+2S) + 8M+2S = 95M+28S across all\n * 15 points instead of 108M+28S.', '3M+2S + 7*(7M+2S) + 8M+2S = 60M+18S across all\n * 10 points; the former15-point chain cost95M+28S.')
    start=s.index('/* ============================================================\n * Fixed-base table construction')
    end=s.index('/* ============================================================\n * Host code',start)
    s=s[:start]+'#include "wide_table_kernels.cuh"\n\n'+s[end:]
    start=s.index('/* The two ladders the GPU builder needs:')
    end=s.index('/* Spot-check the built table against OpenSSL.',start)
    s=s[:start]+(HERE/'ladder_source.txt').read_text()+'\n\n'+s[end:]
    old=function(s,'static int gt_spot_check(');new=old
    new=new.replace('GT_CHUNKS * 4','GT_CHUNKS * 8').replace('ch = t / 4;','ch = t / 8;')
    new=new.replace('const int corner[4] = {0, 1, 2, (int)gt_entries(ch) - 1};','const int corner[8] = {0,1,2,4095,4096,4097,(int)gt_entries(ch)-2,(int)gt_entries(ch)-1};')
    new=new.replace('corner[t % 4]','corner[t % 8]')
    marker='        if (memcmp(gTable + off,      want,     32) != 0 ||\n            memcmp(gTable + off + 32, want + 4, 32) != 0) {'
    assert marker in new
    new=new.replace(marker,'''        uint8_t sample[64];
        if(cudaMemcpy(sample,gTable+off,64,cudaMemcpyDeviceToHost)!=cudaSuccess){ok=0;break;}
        if (memcmp(sample,want,32)!=0 || memcmp(sample+32,want+4,32)!=0) {''')
    s=s.replace(old,new)
    start=s.index('/* OpenSSL fallback builder');end=s.index('/* Params loader',start)
    s=s[:start]+'#include "wide_table_host.cuh"\n\n'+s[end:]
    start=s.index('    /* GTable */');end=s.index('    /* Upload midstate */',start)
    s=s[:start]+'''    /* Reserve the large runtime table only on the GPU. Builder scratch is
     * released before allocating the search checkpoints. */
    size_t free_bytes=0,total_bytes=0;
    wide_cuda_require(cudaMemGetInfo(&free_bytes,&total_bytes),"device memory query");
    const size_t gt_sz=(size_t)GT_TOTAL_ENTRIES*64;
    const size_t search_bytes=(5ull<<29)+(4ull<<20)+(8ull<<10);
    if(free_bytes<gt_sz+search_bytes+(64ull<<20)){
        fprintf(stderr,"Insufficient GPU memory for16GiB table and search checkpoints: %.2fGiB free\\n",(double)free_bytes/(1ull<<30));return 2;
    }
    uint8_t *d_gt=nullptr;
    wide_cuda_require(cudaMalloc(&d_gt,gt_sz),"allocate wide table");
    wide_build_table(d_gt,pp.neg_r_inv);

'''+s[end:]
    s=s.replace('    cudaDeviceSetLimit(cudaLimitStackSize, 4096);','')
    s=s.replace('    cudaSetDevice(gpu_index);','    wide_cuda_require(cudaSetDevice(gpu_index),"select device");\n    wide_cuda_require(cudaDeviceSetLimit(cudaLimitStackSize,4096),"set stack limit");')
    s=s.replace('cudaDeviceProp prop; cudaGetDeviceProperties(&prop, gpu_index);','cudaDeviceProp prop; wide_cuda_require(cudaGetDeviceProperties(&prop,gpu_index),"device properties");')
    # The imported grouped readback already checks its blocking counter copy;
    # retain the same check for the dependent hit-buffer transfer.
    s=s.replace('                cudaMemcpy(hits, d_hit_idx, nh*4, cudaMemcpyDeviceToHost);','                wide_cuda_require(cudaMemcpy(hits,d_hit_idx,nh*4,cudaMemcpyDeviceToHost),"hit records copy");')
    # Keep the exact search expressions, changing only the inverse schedule.
    (out/'wide_fused_kernel.cuh').write_text('// Generated by integrate.py from the split search expressions.\n#pragma once\n'+make_fused(s))
    marker='template<bool FAST_TAIL>\nstatic void launch_pinning_pipeline('
    assert marker in s
    s=s.replace(marker,'#include "wide_fused_kernel.cuh"\nstatic bool wide_use_fused=false;\n\n'+marker)
    marker='    int blocks=(batch_size+255)/256;\n    kernel_pinning_pipeline<FAST_TAIL,0>'
    assert marker in s
    s=s.replace(marker,'''    int blocks=(batch_size+255)/256;
    if(FAST_TAIL&&wide_use_fused){
        kernel_pinning_fused<FAST_TAIL><<<blocks,256>>>(
            d_midstate,d_suffix,suffix_len,seq_offset,lt_offset,total_preimage_len,
            seq_value,start_lt,d_neg_r_inv,d_u2rx,d_u2ry,d_neg2u2rx,d_neg2u2ry,
            d_gt,d_hit_cnt,d_hit_idx,batch_size,easy_mode,single_hash,group_slot,
            saved,roots,tree);
        cudaError_t err=cudaGetLastError();
        if(err!=cudaSuccess){fprintf(stderr,"Fused search launch failed: %s\\n",cudaGetErrorString(err));exit(2);}
        return;
    }
    kernel_pinning_pipeline<FAST_TAIL,0>''')
    s=s.replace('#include "wide_table_host.cuh"','#include "wide_table_host.cuh"\n#include "wide_search_tuner.cuh"')
    s=s.replace('    uint64_t total_searched = 0;','    uint64_t total_searched = 0;\n    WideSearchTuner schedule_tuner(fast_tail);')
    s=s.replace('            uint32_t h_hit = 0;','            readback_group=schedule_tuner.group_limit(fast_tail?MAX_READBACK_GROUP:64);\n            uint64_t group_start_count=total_searched;\n            schedule_tuner.begin();\n            uint32_t h_hit = 0;')
    s=s.replace('            err = cudaMemcpy(&h_hit, d_hit_cnt, 4, cudaMemcpyDeviceToHost);',
                '            schedule_tuner.end();\n            err = cudaMemcpy(&h_hit, d_hit_cnt, 4, cudaMemcpyDeviceToHost);')
    marker='            if (h_hit > 0) {'
    assert marker in s
    s=s.replace(marker,'            schedule_tuner.observe(total_searched-group_start_count);\n'+marker)
    (out/'pinning.cu').write_text(s)
    prov=json.loads((base/'PROVENANCE.json').read_text())
    prov['experiment']='Ten mixed signed windows; bounded chunked affine table builder; proven field repair; runtime comparison of external and CTA-local inverse schedules on distinct real ranges; not submitted.'
    prov['table_startup_prior_work']='MakiRH4 PR46 / jacklightChen PR53 batched host-ladder affine normalization; source inspected at9274883051636def6db5add0d3ba0e02314813f0.'
    prov['candidate_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.suffix in ('.cu','.cuh','.h') or p.name=='COPYING'}
    (out/'PROVENANCE.json').write_text(json.dumps(prov,indent=2)+'\n')
    print('Materialized complete research candidate; production remains untouched.')
if __name__=='__main__':main()
