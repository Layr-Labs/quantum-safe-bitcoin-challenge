#!/usr/bin/env python3
"""Prepare an isolated48MiB-small/4GiB-cold14-window guarded control."""
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from preflight import source_identity
parent=HERE.parent/'cold_seed/guarded_candidate';identity=source_identity(parent)
assert identity['source_fingerprint']=='d24ad6f6c020a36f84b8167613adb2e3fabd0ff0429b8e78da731b5800104be6'
dest=HERE/'candidate';assert not dest.exists(),'Preserve frozen source'
for name in [*identity['source_sha256'],'COPYING']:
 p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(parent/name,p)
h=dest/'tests/gpu_epochs';p=h/'compact_geometry.cuh';s=p.read_text()
start=s.index('// Twelve small windows');end=s.index('// Input M is odd.')
s=s[:start]+'''// Twelve17-bit windows occupy48MiB and cover204bits.
// Two26-bit cold windows occupy4GiB and cover the remaining52bits.
constexpr int MIXED_CHUNKS=14;
__host__ __device__ __forceinline__ int mixed_bits(int c){return c<12?17:26;}
__host__ __device__ __forceinline__ unsigned mixed_entries(int c){return 1u<<(mixed_bits(c)-1);}
__host__ __device__ __forceinline__ unsigned mixed_offset(int c){return c<13?((unsigned)c<<16):((12u<<16)+(1u<<25));}
__host__ __device__ __forceinline__ int mixed_shift(int c){return c<13?17*c:230;}
constexpr uint64_t MIXED_TOTAL_ENTRIES=(12ull<<16)+(2ull<<25);
static_assert(MIXED_TOTAL_ENTRIES*64==((48ull<<20)+(4ull<<30)),"small48 asymmetric table size");
''' +s[end:];p.write_text(s)
p=h/'compact_table_device.cuh';s=p.read_text()
assert s.count('M[3]>>13')==1 and s.count('M[3]>>39')==1
s=s.replace('bits205 and231','bits204 and230').replace('M[3]>>13','M[3]>>12').replace('M[3]>>39','M[3]>>38')
p.write_text(s)
p=h/'compact_table_kernels.cuh';s=p.read_text()
old='*ch=t<(1u<<17)?0:(t<(13u<<16)?1+(int)((t-(1u<<17))>>16):(t<((13u<<16)+(1u<<25))?12:13));'
assert s.count(old)==1
s=s.replace(old,'*ch=t<(12u<<16)?(int)(t>>16):(t<((12u<<16)+(1u<<25))?12:13);');p.write_text(s)
report={'inherited_source':identity,'candidate_source':source_identity(dest),
 'widths':[17]*12+[26]*2,'point_order':[12,13,*range(12)],
 'small_bytes':48<<20,'cold_bytes':4<<30,'total_bytes':(48<<20)+(4<<30),
 'normal_chain_M_S':[88,26],
 'changes':'Geometry, decoder and two cold-digit shifts only. Same guarded final helper and ordinary cache policy. No persistence API yet.',
 'hypothesis':'48MiB small working set fits within a newly author-reported50MiB persisting-cache capacity, unlike52MiBcontrol. Actual runtime capacity and cache effectiveness still require checking.',
 'gpu_executed':False,'limits':'Another1GiBofcoldtable; not a residency or speed guarantee; noPR77frontendintegration.'}
(HERE/'prepared-source.json').write_text(json.dumps(report,indent=2)+'\n')
print(report['candidate_source']['source_fingerprint'])
