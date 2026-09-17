#!/usr/bin/env python3
"""Build one isolated hot/cold and streaming-checkpoint policy comparison."""
import json,re,shutil,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT))
from preflight import source_identity
parent=HERE.parent/'candidate';identity=source_identity(parent)
assert identity['source_fingerprint']=='230f651785e7bcd9affd8cb188e41287a057a06caa8d50749487c5f7a73dec0a'
dest=HERE/'candidate'
for name in [*identity['source_sha256'],'COPYING']:
 p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(parent/name,p)
h=dest/'tests/gpu_epochs'
shutil.copyfile(HERE/'cache_ops.cuh',h/'cache_ops.cuh')
p=h/'compact_table_device.cuh';s=p.read_text();s=s.replace('#pragma once','#pragma once\n#include "cache_ops.cuh"',1)
old='    ulonglong2 x0=tx[0],x1=tx[1],y0=ty[0],y1=ty[1];';assert s.count(old)==1
s=s.replace(old,'''    ulonglong2 x0,x1,y0,y1;
    if(c<12){x0=tx[0];x1=tx[1];y0=ty[0];y1=ty[1];}
    else{x0=qsb_cs_load128(tx);x1=qsb_cs_load128(tx+1);y0=qsb_cs_load128(ty);y1=qsb_cs_load128(ty+1);}''')
marker='__device__ __forceinline__ void compact_prefetch_point(const uint8_t *table,int c,uint32_t idx){';assert s.count(marker)==1
s=s.replace(marker,marker+'\n    if(c>=12)return; // Avoid ordinary L2 prefetch allocation of cold lines.')
p.write_text(s)
p=h/'ranked_pipeline.cuh';s=p.read_text().replace('#pragma once','#pragma once\n#include "cache_ops.cuh"',1)
old='checkpoint[block_base+(size_t)k*QSB_CHECKPOINT_STRIDE+node-256]=out[k];';assert s.count(old)==1
s=s.replace(old,'qsb_cs_store64(checkpoint+block_base+(size_t)k*QSB_CHECKPOINT_STRIDE+node-256,out[k]);')
old='products[k][256+tid]=checkpoint[block_base+(size_t)k*QSB_CHECKPOINT_STRIDE+tid];';assert s.count(old)==1
s=s.replace(old,'products[k][256+tid]=qsb_cs_load64(checkpoint+block_base+(size_t)k*QSB_CHECKPOINT_STRIDE+tid);')
s,n=re.subn(r'state\[([^\]\n]+)\]=make_ulonglong2\(([^;]+)\);',r'qsb_cs_store128(state+\1,make_ulonglong2(\2));',s);assert n==8,n
s,n=re.subn(r'=state\[([^\]\n]+)\]',r'=qsb_cs_load128(state+\1)',s);assert n==8,n
p.write_text(s)
report={'inherited_source':identity,'candidate_source':source_identity(dest),
 'changes':['First12table windows retain ordinary128-bit loads; last2use ld.global.cs.v2.u64.',
 'No normal L2prefetch for coldwindows; hotwindow lookahead preserved.',
 'State128-bit stores/loads and checkpoint64-bit stores/loads use .cs, including shared helpers in builder/root hierarchy.',
 'No L2setaside or persistence API; no unverified host reservation limits. No changes to roots, arithmetic, geometry or synchronization.'],
 'gpu_executed':False,'limits':'Evict-first allocates cache lines; it does not bypass L2. Host helper branches are value/address projections for CPU audits, not CUDA execution or cache simulation.'}
(HERE/'prepared-source.json').write_text(json.dumps(report,indent=2)+'\n')
print(report['candidate_source']['source_fingerprint'])
