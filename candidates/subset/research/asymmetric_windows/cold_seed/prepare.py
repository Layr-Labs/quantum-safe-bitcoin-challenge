#!/usr/bin/env python3
"""Move the two cold-window loads into the independent seed pair.

Keeps the checked asymmetric geometry, field arithmetic and builder unchanged.
No production edit, allocation change or cache-policy modification.
"""
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from preflight import source_identity

parent=HERE.parent/'candidate'
identity=source_identity(parent)
assert identity['source_fingerprint']=='230f651785e7bcd9affd8cb188e41287a057a06caa8d50749487c5f7a73dec0a'
dest=HERE/'candidate'
assert not dest.exists(),'Preserve an existing generated candidate'
for name in [*identity['source_sha256'],'COPYING']:
    p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(parent/name,p)
p=dest/'tests/gpu_epochs/compact_table_device.cuh';s=p.read_text()
start=s.index('__device__ void compact_fixed_xyzz(')
s=s[:start]+r'''
// For c>=1, regular recoding state is (original_M >> shift(c)) | 1.
// The two cold windows start at bits205 and231. Extract them independently
// before mutating M for the twelve small windows; all digits remain odd.
__device__ __forceinline__ void qsb_asym_cold_digits(
    const uint64_t M[4],int sign,int32_t *cold12,int32_t *cold13){
    int32_t d12=(int32_t)(((M[3]>>13)&((1u<<27)-1))|1ULL)-(1<<26);
    int32_t d13=(int32_t)((M[3]>>39)|1ULL);
    *cold12=sign*d12;*cold13=sign*d13;
}
__device__ void compact_fixed_xyzz(uint64_t *X, uint64_t *Y, uint64_t *ZZ, uint64_t *ZZZ,
                                      const uint64_t scalar[4], const uint8_t *gTX, const uint8_t *gTY) {
    uint64_t M[4]; int sign; gt_recode_setup(scalar,M,&sign);
    int32_t cold12,cold13;qsb_asym_cold_digits(M,sign,&cold12,&cold13);
    uint32_t idx;uint64_t neg;
    uint64_t x0[4],y0[4],x1[4],y1[4];
    gt_digit_idx(cold12,&idx,&neg);
    compact_load_signed(gTX,gTY,12,idx,neg,x0,y0);
    gt_digit_idx(cold13,&idx,&neg);
    compact_load_signed(gTX,gTY,13,idx,neg,x1,y1);

    gt_digit_idx(compact_recode_step(M,sign,0),&idx,&neg);
    compact_prefetch_point(gTX,0,idx);
    _PointAddXYZZ_mm(X,Y,ZZ,ZZZ, x0,y0, x1,y1);
    uint64_t cx[4],cy[4];
    #pragma unroll 1
    for(int c=0;c<11;++c){
        compact_load_signed(gTX,gTY,c,idx,neg,cx,cy);
        gt_digit_idx(compact_recode_step(M,sign,c+1),&idx,&neg);
        compact_prefetch_point(gTX,c+1,idx);
        _PointAddXYZZ<true>(X,Y,ZZ,ZZZ,cx,cy,y0);
        Load256(y0,cy);
    }
    compact_load_signed(gTX,gTY,11,idx,neg,cx,cy);
    _PointAddXYZZ<false>(X,Y,ZZ,ZZZ,cx,cy,y0);
}
'''
p.write_text(s)
report={'inherited_source':identity,'candidate_source':source_identity(dest),
 'point_order':[12,13,*range(12)],'point_lookups':14,'chain_M_S':[88,26],
 'changes':'Extract cold digits12/13 from immutable recode state, load them as seed pair; small windows0..11 follow with checked address-only lookahead. No new cache hints.',
 'hypothesis':'Two cold loads have no point-state dependency between them; a compiler may overlap them before the seed. Removes them from late serial additions, not their bytes or latency cost.',
 'gpu_executed':False,'limits':'Not integrated with PR77; full native scheduling, cache behavior and throughput must be examined. CPU checks alone do not prove device overlap.'}
(HERE/'prepared-source.json').write_text(json.dumps(report,indent=2)+'\n')
print(report['candidate_source']['source_fingerprint'])
