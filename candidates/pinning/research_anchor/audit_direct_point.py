#!/usr/bin/env python3
"""Audit actual proposed mixed-add body with exact modular primitive shims."""
from pathlib import Path
import ctypes,hashlib,json,random,subprocess,sys
ROOT=Path(__file__).resolve().parent
source_path=(Path(sys.argv[1]) if len(sys.argv)>1 else
 ROOT.parent/'research_arithmetic/_build/subseed_bias/GPUMath.h')
if not source_path.exists():
 raise SystemExit('Generate the header first with research_arithmetic/generate.py subseed_bias, or supply a header path.')
source=source_path.read_text()
marker='template<bool DEFER_Y>\n__device__ __forceinline__ void _PointAddXYZZT('
pos=source.index(marker,source.index(marker)+len(marker))
point=source[pos:source.index('// Direct-three-affine prefix',pos)]
pos=source.index('__device__ void _PointAddXYZZ_mm(')
seed=source[pos:source.index('\n}',pos)+2].replace('_PointAddXYZZ_mm','qsb_direct_seed')
shim=(ROOT/'audit.cpp').read_text().split('#include "anchor.cuh"')[0]
shim+='''
#define QSB_FUSE_SQRADDSUB2 1
#define QSB_LAZY 1
static void _ModSqrAddSub2(uint64_t*r,const uint64_t*a,const uint64_t*e,const uint64_t*q){save(r,load(a)*load(a)+load(e)-2*load(q));}
static void qsb_mulsub_seed(uint64_t*r,const uint64_t*a,const uint64_t*b,const uint64_t*c){save(r,load(a)*load(b)-load(c));}
static void qsb_negate_residue(uint64_t*r){save(r,-load(r));}
'''+point+seed+'''
extern "C" void seed(uint64_t*s,const uint64_t*a,const uint64_t*b){qsb_direct_seed(s,s+4,s+8,s+12,a,a+4,b,b+4);}
extern "C" void step(uint64_t*s,const uint64_t*b,const uint64_t*a){_PointAddXYZZT<true>(s,s+4,s+8,s+12,b,b+4,a+4);}
'''
(ROOT/'audit_direct.cpp').write_text(shim)
subprocess.run(['g++','-O2','-shared','-fPIC','-I/tmp/qsb-boost-audit/usr/include',
 str(ROOT/'audit_direct.cpp'),'-o',str(ROOT/'audit_direct.so')],check=True)
lib=ctypes.CDLL(str(ROOT/'audit_direct.so'))
ptr=ctypes.POINTER(ctypes.c_uint64)
lib.seed.argtypes=[ptr,ptr,ptr];lib.step.argtypes=[ptr,ptr,ptr]
# Share only the independent affine/encoding helper definitions, without
# invoking the anchored-prototype experiment or reusing its equations.
helpers={'__file__':str(ROOT/'audit.py')}
exec((ROOT/'audit.py').read_text().split('subprocess.run([')[0],helpers)
P,G,plus,pack,val=[helpers[k] for k in ('P','G','plus','pack','val')]
rng=random.Random(0x445249454354)
pool=[G]
for i in range(1023):pool.append(plus(pool[-1],G))
checks=chains=singular=0
for trial in range(2048):
 points=[pool[rng.randrange(1024)] for _ in range(15)]
 points=[(x,(-y)%P) if rng.getrandbits(1) else (x,y) for x,y in points]
 a,b=points[:2]
 if a[0]==b[0]:singular+=1;continue
 expected=plus(a,b);state=(ctypes.c_uint64*16)();lib.seed(state,pack(a),pack(b));anchor=a
 for i in range(1,15):
  if i>=2:
   b=points[i]
   if expected is None or expected[0]==b[0]:singular+=1;break
   lib.step(state,pack(b),pack(anchor));expected=plus(expected,b);anchor=b
  X,N,U,V=[val(state,j) for j in (0,4,8,12)]
  assert pow(U,3,P)==pow(V,2,P)
  actual=(X*pow(U,-1,P)%P,(-N*pow(V,-1,P)-anchor[1])%P)
  assert actual==expected,(trial,i)
  checks+=1
 else:chains+=1
result={'source':str(source_path),'sha256':hashlib.sha256(source.encode()).hexdigest(),
 'exact_field_chains':chains,'checked_point_states':checks,'singular_exclusions':singular,
 'mismatches':0,'gpu_execution':False,
 'scope':'Actual proposed C++ point and seed bodies with exact field shims; approximate PTX primitive tested separately.'}
(ROOT/'direct_point_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
