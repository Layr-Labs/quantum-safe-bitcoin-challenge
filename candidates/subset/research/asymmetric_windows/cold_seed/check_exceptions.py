#!/usr/bin/env python3
"""Check the actual final-add helper, cold-prefix bounds and constructed scalars.

OpenSSL executes field operations; a separate Python affine law is the oracle.
The prefix argument assumes a valid nonzero order-n runtime base and the stated
regular odd recoding identity. This is not GPU execution or an exhaustive CUDA
verification of all input hashes.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'research/wide_windows'))
from audit_support import BACKEND,function
from preflight import source_identity
import audit_integrated as ref

N=ref.N;P=ref.P;MASK=(1<<64)-1
def words(values):return (C.c_uint64*(4*len(values)))(*(v>>(64*i)&MASK for v in values for i in range(4)))
def integers(words):return tuple(sum(words[4*j+i]<<(64*i) for i in range(4)) for j in range(len(words)//4))

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,default=HERE/'guarded_candidate')
parser.add_argument('--output',type=Path,default=HERE,help='directory for source-bound reports')
parser.add_argument('--first-width',type=int,choices=[17,18],default=18,help='independent expected first width; eleven further17-bit and two remaining cold windows follow')
args=parser.parse_args();base=args.source.resolve();outdir=args.output.resolve();outdir.mkdir(parents=True,exist_ok=True)
first=args.first_width;small_bits=first+11*17;last_shift=small_bits+26;last_width=256-last_shift;last_small_shift=small_bits-17
widths=[first]+[17]*11+[26,last_width]
identity=source_identity(base)
header=(base/'tests/gpu_epochs/compact_table_device.cuh').read_text()
math=(base/'GPUMath.h').read_text()
helper=function(header,'__device__ __forceinline__ void qsb_asym_last_add(')
code=BACKEND+r'''
#define __device__
#define __forceinline__ inline
static void Load256(uint64_t*r,const uint64_t*a){memcpy(r,a,32);}
'''+function(math,'template<bool DEFER_Y>')+helper+r'''
extern "C" void last_add(uint64_t *state,const uint64_t*point,const uint64_t*anchor,int guarded){
 ctx=BN_CTX_new();prime=nullptr;
 require(BN_hex2bn(&prime,"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F")!=0);
 if(guarded)qsb_asym_last_add(state,state+4,state+8,state+12,point,point+4,anchor);
 else _PointAddXYZZ<false>(state,state+4,state+8,state+12,point,point+4,anchor);
 BN_free(prime);BN_CTX_free(ctx);
}
'''
rng=random.Random(2026091744)
pool=[ref.scalar_mult(k) for k in [1,2,3,17,N-1,N-2]+[rng.randrange(1,N) for _ in range(16)]]
counts={'equal':0,'opposite':0,'ordinary':0,'unguarded_doubling_failures_detected':0}
with tempfile.TemporaryDirectory(prefix='qsb-cold-exception-') as tmp:
 cpp=Path(tmp)/'audit.cpp';libp=Path(tmp)/'audit.so';cpp.write_text(code)
 subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC','-I/opt/homebrew/opt/openssl@3/include',
                 '-L/opt/homebrew/opt/openssl@3/lib',str(cpp),'-lcrypto','-o',str(libp)],check=True)
 lib=C.CDLL(str(libp));u64=C.POINTER(C.c_uint64);lib.last_add.argtypes=[u64,u64,u64,C.c_int]
 for i in range(160):
  a=rng.choice(pool);b=rng.choice(pool)
  if b[0]==a[0]:b=ref.affine_add(a,ref.G)
  if b is None or b[0]==a[0]:continue
  for kind,q in [('equal',a),('opposite',(a[0],(-a[1])%P)),('ordinary',b)]:
   z=[1,P-1,rng.randrange(1,P)][i%3];anchor=rng.randrange(P)
   zz=z*z%P;zzz=zz*z%P
   initial=[a[0]*zz%P,(a[1]+anchor)*zzz%P,zz,zzz]
   out=words(initial);lib.last_add(out,words(q),words([anchor]),1)
   x,y,zz1,zzz1=integers(out);want=ref.affine_add(a,q)
   if want is None:assert zz1==0 and zzz1==0
   else:assert zz1 and zzz1 and (x*pow(zz1,-1,P)%P,y*pow(zzz1,-1,P)%P)==want,(kind,i)
   counts[kind]+=1
   if kind=='equal':
    old=words(initial);lib.last_add(old,words(q),words([anchor]),0)
    _,_,oldzz,oldzzz=integers(old)
    assert oldzz==oldzzz==0 and want is not None
    counts['unguarded_doubling_failures_detected']+=1

# Let M be odd, 1<=M<=n, B=small_bits. Cold sum C=((M>>B)|1)*2^B.
# Thus 2^B<=C<=2^256-2^B. Seed d12 +/- 2^26*d13 is odd,
# nonzero and has magnitude <2^(256-B)<n, so neither equal nor opposite.
assert (1<<(256-small_bits))<N
# After adding windows0..c, L=M-((M>>S)|1)*2^S, S=first+17*c.
# Before each c<=10, |L_previous|+|d_c*2^shift_c|<=2^S-1.
# Therefore both accumulator+Q and accumulator-Q are strictly in (0,n).
# Multiplication by the runtime base or global sign preserves non-equality.
bounds=[]
for c in range(11):
 end=first+17*c;previous=0 if c==0 else first+17*(c-1)
 before_max=0 if c==0 else (1<<previous)-1
 digit_max=((1<<(first if c==0 else 17))-1)<<previous
 assert before_max+digit_max==(1<<end)-1
 lo=(1<<small_bits)-((1<<end)-1);hi=(1<<256)-(1<<small_bits)+((1<<end)-1)
 assert 0<lo<=hi<N
 bounds.append({'window':c,'end_bit':end,'min_signed_difference':hex(lo),'max_signed_difference':hex(hi)})

cold_m=N-((1<<18)-2)*(1<<last_small_shift)
parent_m=((1<<last_width)-1)*(1<<(last_shift+1))-N
assert cold_m&1 and 0<cold_m<=N and parent_m&1 and 0<parent_m<=N
d11=(((cold_m>>last_small_shift)&((1<<18)-1))|1)-(1<<17)
assert d11==-(1<<17)+1 and cold_m-2*(d11<<last_small_shift)==N
d13=(parent_m>>last_shift)|1
assert parent_m-2*(d13<<last_shift)==-N
scalar_cases=[]
def append_m(m,repeats=1):
 if not (0<m<=N and m&1):return
 k=m*pow(2,-1,N)%N
 for _ in range(repeats):scalar_cases.extend([hex(k),hex((-k)%N)])
append_m(cold_m,3);append_m(parent_m,3)
for m in [1,3,N-2,N,cold_m,parent_m]:
 for delta in [-2,0,2,-(1<<last_small_shift),1<<last_small_shift]:append_m(m+delta)
for bit in [first,first+17,64,128,last_small_shift-17,last_small_shift,small_bits,small_bits+1,last_shift,last_shift+1,255]:
 for center in [1<<bit,N-(1<<bit)]:
  for delta in [-3,-2,-1,0,1,2,3]:append_m(center+delta)
(outdir/'exception-scalars.json').write_text(json.dumps(scalar_cases,indent=2)+'\n')
cmd=[sys.executable,'-B',str(ROOT/'research/wide_windows/check_wide.py'),'--compact',
 '--widths',','.join(map(str,widths)),
 '--prefetch-start','0','--prefetch-chunks','12',
 '--chain-order','12,13,0,1,2,3,4,5,6,7,8,9,10,11',
 '--extra-chain-scalars',str(outdir/'exception-scalars.json'),
 '--source',str(base),'--report',str(outdir/'guarded-curve-results.json')]
subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
assert source_identity(base)==identity
result={'status':'PASS','source_fingerprint':identity['source_fingerprint'],
 'actual_helper_cases':counts,'prefix_bounds':bounds,'added_chain_scalar_cases':len(scalar_cases),
 'whole_chain_report':'guarded-curve-results.json',
 'geometry_widths':widths,'small_bits':small_bits,
 'derivation':f'For odd1<=M<=n, C=((M>>{small_bits})|1)*2^{small_bits}; low prefix L=M-((M>>S)|1)*2^S. Bounds exclude equal/opposite/infinite prefixes through window10. Final helper handles both remaining cases.',
 'negative_controls':'Actual inherited incomplete final helper, tested above; old52MiB whole-chain controls are separately retained in cold_seed/.',
 'audit_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'gpu_executed':False,'limits':__doc__}
(outdir/'exception-domain-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='prefix_bounds'},indent=2))
