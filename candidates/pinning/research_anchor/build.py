#!/usr/bin/env python3
"""Generate full controlled sources inside this research directory only."""
from pathlib import Path
import hashlib,json,re,shutil

ROOT=Path(__file__).resolve().parent
SRC=ROOT.parent
def body(text,name,next_marker):
    start=text.index(name)
    end=text.index(next_marker,start)
    return text[start:end]

manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SRC.iterdir()
          if p.is_file() and p.suffix in ('.cu','.cuh','.h')}
for mode in ('control','anchor'):
    dest=ROOT/mode
    dest.mkdir(exist_ok=True)
    for name in manifest: shutil.copy2(SRC/name,dest/name)

dest=ROOT/'anchor'
math=(dest/'GPUMath.h').read_text()
sas=body(math,'__device__ __forceinline__ void _ModSqrAddSub2(',
         '#endif  /* QSB_FUSE_SQRADDSUB2 */')
sas=sas.replace('_ModSqrAddSub2','_ModSqrAddSub3')
# Retain the inherited approximate reduction, change the mathematical numerator
# from a*a+e+3p-2q to a*a+e+4p-3q. Only the default split-3p path is used.
qsub='''        "\\tsub.cc.u32 z0, z0, v0; subc.cc.u32 z1, z1, v1;\\n"
        "\\tsubc.cc.u32 z2, z2, v2; subc.cc.u32 z3, z3, v3;\\n"
        "\\tsubc.cc.u32 z4, z4, v4; subc.cc.u32 z5, z5, v5;\\n"
        "\\tsubc.cc.u32 z6, z6, v6; subc.cc.u32 z7, z7, v7;\\n"
        "\\tsubc.cc.u32 z8, z8, 0;" QSB_SAS_SUB_TAIL "\\n"
'''
assert sas.count(qsub)==2
sas=sas.replace(qsub+qsub,qsub*3)
sas=sas.replace('z8, z8, 3;','z8, z8, 4;')
sas=sas.replace('0xb73','0xf44').replace('z1, z1, 3;','z1, z1, 4;')
sas=sas.replace('0xfffff48d','0xfffff0bc').replace('0xfffffffc','0xfffffffb')
sas=sas.replace('z8, z8, 2;','z8, z8, 3;')
sas=sas.replace('z[8]+2+carry','z[8]+3+carry').replace('repeat<2','repeat<3')
math+='\n#if !QSB_SAS_SPLIT3P || !QSB_C31 || !QSB_YOFF || !QSB_NEG_Y_MAC\n#error "anchor experiment requires inherited default paths"\n#endif\n'+sas
(dest/'GPUMath.h').write_text(math)
shutil.copy2(ROOT/'anchor.cuh',dest/'anchor.cuh')

cu=(dest/'pinning.cu').read_text()
cu=cu.replace('#include "GPUMath.h"','#include "GPUMath.h"\n#include "anchor.cuh"')
seed=body(math,'__device__ void _PointAddXYZZ_mm(', '\n}',)+ '\n}'
seed=seed.replace('_PointAddXYZZ_mm','qsb_anchor_seed')
# The seed's Q already is X - x0*U in the active negative-Y path.
assert 'Load256(X3, T);' in seed
seed=seed.replace('Load256(X3, T);','Load256(X3, Q);')
cu=cu.replace('/* DER checks */',seed+'\n\n/* DER checks */')
# Forward declare because the production scalar function precedes this insertion.
decl='__device__ void qsb_anchor_seed(uint64_t*,uint64_t*,uint64_t*,uint64_t*,const uint64_t*,const uint64_t*,const uint64_t*,const uint64_t*);\n'
cu=cu.replace('__device__ void _FixedBaseSignedXYZZScalar(',decl+'__device__ void _FixedBaseSignedXYZZScalar(')
start=cu.index('__device__ void _FixedBaseSignedXYZZScalar(')
end=cu.index('/* _FixedBaseSignedAffine:',start)
fn=cu[start:end]
fn=fn.replace('_PointAddXYZZ_mm(X,Y,U,V,x0,y0,x1,y1);','qsb_anchor_seed(X,Y,U,V,x0,y0,x1,y1);')
fn=fn.replace('_PointAddXYZZT<true>(X,Y,U,V,x1,y1,y0);','qsb_anchor_add(X,Y,U,V,x1,y1,x0,y0,c==GT_CHUNKS-1);\n        Load256(x0,x1);')
cu=cu[:start]+fn+cu[end:]
(dest/'pinning.cu').write_text(cu)
(ROOT/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(ROOT/'seed_extracted.cuh').write_text(seed+'\n')
print('Generated control and anchor copies; production sources untouched.')
