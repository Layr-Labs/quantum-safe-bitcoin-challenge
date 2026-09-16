#!/usr/bin/env python3
"""Create isolated lookahead variants from the checked wide-table prototype."""
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode = True
from audit_support import function
from preflight import source_identity

HERE = Path(__file__).resolve().parent
base = HERE / 'candidate'
identity = source_identity(base)
assert identity['source_fingerprint'] == '49ebdc07297b38908c1fd285e9563ec3c646ca8cd937a57976382b8debd36593'
source = (base / 'tests/gpu_epochs/tree.cu').read_text()
old = function(source, '__device__ void _FixedBaseSignedXYZZ(')
signature = old[:old.index('{') + 1]
prefix = '''
    uint64_t M[4]; int sign; gt_recode_setup(scalar,M,&sign);
    uint32_t idx; uint64_t neg;
    uint64_t x0[4],y0[4],x1[4],y1[4];
    gt_digit_idx(gt_recode_step(M,sign,0), &idx, &neg);
    gt_load_signed(gTX,gTY,0,idx,neg,x0,y0);
    gt_digit_idx(gt_recode_step(M,sign,1), &idx, &neg);
    gt_load_signed(gTX,gTY,1,idx,neg,x1,y1);
'''
loaded = signature + prefix + '''
    _PointAddXYZZ_mm(X,Y,ZZ,ZZZ, x0,y0, x1,y1);
    uint64_t cx[4],cy[4],nx[4],ny[4];
    gt_digit_idx(gt_recode_step(M,sign,2), &idx, &neg);
    gt_load_signed(gTX,gTY,2,idx,neg,cx,cy);
    #pragma unroll 1
    for(int c=2;c<GT_CHUNKS-1;++c){
        gt_digit_idx(gt_recode_step(M,sign,c+1), &idx, &neg);
        gt_load_signed(gTX,gTY,c+1,idx,neg,nx,ny);
        _PointAddXYZZ<true>(X,Y,ZZ,ZZZ,cx,cy,y0);
        Load256(y0,cy);
        Load256(cx,nx); Load256(cy,ny);
    }
    _PointAddXYZZ<false>(X,Y,ZZ,ZZZ,cx,cy,y0);
}'''
helper = '''// A cache hint only: correctness never depends on cache residency.
// Hint both 32-byte halves; do not assume a device's cache sector fetch width.
__device__ __forceinline__ void gt_prefetch_point(const uint8_t *table,int c,uint32_t idx){
    size_t off=((size_t)gt_offset(c)+idx)*64;
    asm volatile("prefetch.global.L2 [%0];" :: "l"(table+off));
    asm volatile("prefetch.global.L2 [%0];" :: "l"(table+off+32));
}
'''
prefetch = signature + prefix + '''
    gt_digit_idx(gt_recode_step(M,sign,2), &idx, &neg);
    gt_prefetch_point(gTX,2,idx);
    _PointAddXYZZ_mm(X,Y,ZZ,ZZZ, x0,y0, x1,y1);
    uint64_t cx[4],cy[4];
    #pragma unroll 1
    for(int c=2;c<GT_CHUNKS-1;++c){
        gt_load_signed(gTX,gTY,c,idx,neg,cx,cy);
        gt_digit_idx(gt_recode_step(M,sign,c+1), &idx, &neg);
        gt_prefetch_point(gTX,c+1,idx);
        _PointAddXYZZ<true>(X,Y,ZZ,ZZZ,cx,cy,y0);
        Load256(y0,cy);
    }
    gt_load_signed(gTX,gTY,GT_CHUNKS-1,idx,neg,cx,cy);
    _PointAddXYZZ<false>(X,Y,ZZ,ZZZ,cx,cy,y0);
}'''

record = {'base': identity, 'variants': {}, 'scope': 'Source scheduling experiments; no GPU throughput evidence'}
for name, replacement in [('loaded', loaded), ('prefetch', helper + prefetch)]:
    dest = HERE / 'lookahead' / name
    if dest.exists():
        raise SystemExit('Refusing to overwrite an existing experiment: ' + str(dest))
    shutil.copytree(base, dest)
    (dest / 'tests/gpu_epochs/tree.cu').write_text(source.replace(old, replacement))
    record['variants'][name] = source_identity(dest)
(HERE / 'lookahead' / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({key: value['source_fingerprint'] for key, value in record['variants'].items()}, indent=2))
