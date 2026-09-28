#!/usr/bin/env python3
"""Prepare an isolated four-mode candidate from the frozen PR74 production files."""
from pathlib import Path
import hashlib, json, shutil

HERE=Path(__file__).resolve().parent
PIN=HERE.parents[1]
SUB=PIN.parent/'subset/tests/gpu_epochs'
OUT=HERE/'candidate'
receipt=json.loads((PIN/'research/wide_windows/submitted-source.json').read_text())
OUT.mkdir(exist_ok=True)
for name,want in receipt['production_sha256'].items():
    src=PIN/name
    assert hashlib.sha256(src.read_bytes()).hexdigest()==want, name+' changed from PR74'
    shutil.copyfile(src,OUT/name)
copied={}
for name in ('compact_geometry.cuh','compact_table_kernels.cuh','compact_table_host.cuh'):
    src=SUB/name;shutil.copyfile(src,OUT/name)
    copied[name]=hashlib.sha256(src.read_bytes()).hexdigest()
p=OUT/'compact_geometry.cuh'
p.write_text(p.read_text()+'''
#define COMPACT_CHUNKS MIXED_CHUNKS
#define COMPACT_TOTAL_ENTRIES ((uint32_t)MIXED_TOTAL_ENTRIES)
#define COMPACT_LO 256
#define COMPACT_HI 1024
__host__ __device__ __forceinline__ unsigned compact_entries(int c){return mixed_entries(c);}
__host__ __device__ __forceinline__ unsigned compact_offset(int c){return mixed_offset(c);}
__host__ __device__ __forceinline__ int compact_shift(int c){return mixed_shift(c);}
''')
p=OUT/'compact_table_host.cuh'
p.write_text(p.read_text().replace('Wide table','Compact table'))
s=(OUT/'pinning.cu').read_text()
def replace(old,new):
    global s
    assert s.count(old)==1,(old[:100],s.count(old))
    s=s.replace(old,new)
replace('#include "wide_geometry.cuh"','#include "wide_geometry.cuh"\n#include "compact_geometry.cuh"')
start=s.index('__device__ void _FixedBaseSignedXYZZScalar(')
end=s.index('\n}\n',start)+3
body=s[start:end]
body=body.replace('__device__ void','template<bool WIDE>\n__device__ void',1)
body=body.replace('wide_step(M,sign,26)','wide_step(M,sign,WIDE?26:18)',1)
body=body.replace('wide_step(M,sign,26)','wide_step(M,sign,WIDE?26:17)',1)
body=body.replace('wide_offset(0)','0u').replace('wide_offset(1)','(WIDE?wide_offset(1):mixed_offset(1))')
body=body.replace('wide_offset(2)','(WIDE?wide_offset(2):mixed_offset(2))')
body=body.replace('WIDE_CHUNKS-1','(WIDE?WIDE_CHUNKS:MIXED_CHUNKS)-1')
body=body.replace('wide_bits(c)','(WIDE?wide_bits(c):mixed_bits(c))')
body=body.replace('wide_entries(c)','(WIDE?wide_entries(c):mixed_entries(c))')
s=s[:start]+body+s[end:]
replace('template<bool FAST_TAIL, int STAGE>','template<bool FAST_TAIL, int STAGE, bool WIDE>')
replace('_FixedBaseSignedXYZZScalar(qx,qy,qzz,qzzz,z,d_gt);','_FixedBaseSignedXYZZScalar<WIDE>(qx,qy,qzz,qzzz,z,d_gt);')
replace('static bool wide_use_fused=false;','static bool wide_use_fused=false;\nstatic bool adaptive_use_wide=false;')
start=s.index('template<bool FAST_TAIL>\nstatic void launch_pinning_pipeline(')
end=s.index('\n}\n',start)+3
old=s[start:end]
impl=old.replace('template<bool FAST_TAIL>','template<bool FAST_TAIL, bool WIDE>',1).replace('launch_pinning_pipeline(','launch_pinning_geometry(',1)
impl=impl.replace('kernel_pinning_fused<FAST_TAIL>','kernel_pinning_fused<FAST_TAIL,WIDE>')
impl=impl.replace('kernel_pinning_pipeline<FAST_TAIL,0>','kernel_pinning_pipeline<FAST_TAIL,0,WIDE>')
impl=impl.replace('kernel_pinning_pipeline<FAST_TAIL,2>','kernel_pinning_pipeline<FAST_TAIL,2,false>')
args='''d_midstate,d_suffix,suffix_len,seq_offset,lt_offset,total_preimage_len,
            seq_value,start_lt,d_neg_r_inv,d_u2rx,d_u2ry,d_neg2u2rx,d_neg2u2ry,
            d_gt,d_hit_cnt,d_hit_idx,batch_size,easy_mode,single_hash,group_slot,
            saved,roots,tree,super_roots,root_checkpoint'''
wrapper=old[:old.index(') {')+3]+f'''
    if(FAST_TAIL && adaptive_use_wide)launch_pinning_geometry<FAST_TAIL,true>({args});
    else launch_pinning_geometry<FAST_TAIL,false>({args});
}}
'''
s=s[:start]+impl+'\n'+wrapper+s[end:]
replace('#include "wide_table_kernels.cuh"','#include "wide_table_kernels.cuh"\n#include "compact_table_kernels.cuh"')
replace('#include "wide_search_tuner.cuh"','#include "compact_table_host.cuh"\n#include "adaptive_resources.cuh"\n#include "adaptive_tuner.cuh"')
start=s.index('    /* Reserve the large runtime table only on the GPU.')
end=s.index('\n    /* Upload midstate */',start)
s=s[:start]+'''    // Compact geometry is mandatory and remains the timing baseline.
    uint8_t *d_compact=nullptr,*unused_y=nullptr,*d_wide=nullptr;
    compact_build_table(&d_compact,&unused_y,pp.neg_r_inv);
'''+s[end:]
replace('    /* Safe ranges */','''    // Search allocations already exist, so the optional wide table cannot
    // consume their reserve. Builder scratch is bounded and released before search.
    d_wide=adaptive_optional_wide(fast_tail,pp.neg_r_inv);

    /* Safe ranges */''')
replace('WideSearchTuner schedule_tuner(fast_tail);','AdaptiveSearchTuner schedule_tuner(fast_tail,d_wide!=nullptr);')
replace('            schedule_tuner.begin();','''            schedule_tuner.begin();
            uint8_t *d_gt=adaptive_use_wide?d_wide:d_compact;''')
(OUT/'pinning.cu').write_text(s)
p=OUT/'wide_fused_kernel.cuh'
s=p.read_text().replace('template<bool FAST_TAIL>','template<bool FAST_TAIL, bool WIDE>',1)
s=s.replace('_FixedBaseSignedXYZZScalar(qx,qy,qzz,qzzz,z,d_gt);','_FixedBaseSignedXYZZScalar<WIDE>(qx,qy,qzz,qzzz,z,d_gt);')
p.write_text(s)
(HERE/'provenance.json').write_text(json.dumps({'control_submission':receipt['submission_id'],'control_source':receipt['production_sha256'],'compact_builder_sources_from_our_PR86':copied,'new_files':['adaptive_policy.cuh','adaptive_tuner.cuh','adaptive_resources.cuh'],'production_modified':False},indent=2)+'\n')
for name in ('adaptive_policy.cuh','adaptive_tuner.cuh','adaptive_resources.cuh'):
    shutil.copyfile(HERE/name,OUT/name)
print('Prepared isolated candidate:',OUT)
