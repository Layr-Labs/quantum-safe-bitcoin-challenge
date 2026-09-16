#!/usr/bin/env python3
"""Combine checked64MiB and16GiB methods without altering promoted production."""
import json
from pathlib import Path
import re
import shutil
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent.parent))
from preflight import source_identity
from check_candidate import function

adaptive=HERE.parent/'adaptive_tables/candidate'
compact=HERE.parent/'mixed_windows/candidate'
assert source_identity(adaptive)['source_fingerprint']=='c8daa257dd1d3f3a1483eaa8047aa3c0af7a7b309103823a3ab959d43a46c9d6'
assert source_identity(compact)['source_fingerprint']=='3205a7177cb0a51fcb269b5ea1f06daedc7ef0b6c7000029095ad86c49d1bb0b'
out=HERE/'candidate'
if out.exists():raise SystemExit('Refusing to overwrite existing candidate')
shutil.copytree(adaptive,out)
dst=out/'tests/gpu_epochs';donor=compact/'tests/gpu_epochs'
old=(donor/'tree.cu').read_text()

def rename(text):
    text=re.sub(r'\bGT_', 'COMPACT_', text)
    text=re.sub(r'\bWIDE_', 'MIXED_', text)
    text=re.sub(r'\bwide_(?!cuda_require)', 'mixed_', text)
    for symbol in ['entries','offset','shift','recode_step','recode_signed',
                   'load_signed','prefetch_point','point_to_limbs','require',
                   'build_ladders','spot_check']:
        text=re.sub(r'\bgt_'+symbol+r'\b','compact_'+symbol,text)
    return text.replace('_FixedBaseSignedXYZZ','compact_fixed_xyzz')

(dst/'compact_geometry.cuh').write_text(rename((donor/'wide_geometry.cuh').read_text()))
declarations=old[old.index('#define GT_CHUNKS'):old.index('/* n = secp256k1')]
device='#pragma once\n#include "compact_geometry.cuh"\n'+rename(declarations)
for marker in ['__device__ __forceinline__ int32_t gt_recode_step(',
               '__device__ __forceinline__ void gt_recode_signed(',
               '__device__ __forceinline__ void gt_load_signed(',
               '__device__ __forceinline__ void gt_prefetch_point(',
               '__device__ void _FixedBaseSignedXYZZ(']:
    device+='\n'+rename(function(old,marker))
(dst/'compact_table_device.cuh').write_text(device+'\n')
(dst/'compact_table_kernels.cuh').write_text(rename((donor/'wide_table_kernels.cuh').read_text()))
host='#pragma once\n'
for marker in ['static void gt_point_to_limbs(', 'static void gt_require(',
               'static void gt_build_ladders(', 'static int gt_spot_check(']:
    host+=rename(function(old,marker))+'\n'
host+=rename(function((donor/'wide_table_host.cuh').read_text(),'static void wide_build_table('))
host+='''
static void compact_build_table(uint8_t **tableX,uint8_t **tableY,const uint8_t nri[32]){
    wide_cuda_require(cudaMalloc(tableX,(size_t)COMPACT_TOTAL_ENTRIES*64),"allocate compact64MiB table");
    *tableY=nullptr; // Both coordinates occupy the same interleaved allocation.
    mixed_build_table(*tableX,nri);
}
'''
(dst/'compact_table_host.cuh').write_text(host)
s=(dst/'tree.cu').read_text()
s=s.replace('#include "small_table_device.cuh"','#include "compact_table_device.cuh"')
s=s.replace('small_fixed_xyzz(','compact_fixed_xyzz(')
kernel=function(s,'__global__ void small_build_kernel(')
s=s.replace(kernel,'#include "compact_table_kernels.cuh"')
s=s.replace('#include "small_table_host.cuh"','#include "compact_table_host.cuh"')
s=s.replace('small_build_table(','compact_build_table(')
s=s.replace('The established32MiB path is always available.','The compact64MiB path is always available.')
(dst/'tree.cu').write_text(s)
p=dst/'ranked_pipeline.cuh'
s=p.read_text().replace('small_fixed_xyzz(','compact_fixed_xyzz(')
s=s.replace('qsb_ranked_prepare_small','qsb_ranked_prepare_compact')
p.write_text(s)
p=dst/'table_tuner.cuh';s=p.read_text().replace('small32MiB','compact64MiB').replace('small %.3f','compact %.3f');p.write_text(s)
# These two copied files are now unused, and are absent from the include closure.
(dst/'small_table_device.cuh').unlink();(dst/'small_table_host.cuh').unlink()
record={'status':'isolated adaptive64MiB/16GiB candidate; not submitted',
        'adaptive_parent':source_identity(adaptive),'compact_parent':source_identity(compact),
        'candidate':source_identity(out),'invariants':['Both branches use current runtime base and actual scalar','Only prepare/table builder geometry differs; inverse/recovery/hit flow is shared','Real comparison ranges and hits retained; optional wide allocation fallback and release retained'],
        'evidence_limit':'Requires source-bound CPU/native checks; no GPU timing claim.'}
(HERE/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
print(record['candidate']['source_fingerprint'])
