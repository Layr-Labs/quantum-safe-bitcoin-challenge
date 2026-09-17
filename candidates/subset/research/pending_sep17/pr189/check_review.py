#!/usr/bin/env python3
"""Source-bound finite arithmetic review; Python model, not actual CUDA helper execution."""
import hashlib,itertools,json,random,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'research'))
from ptx_field_model import function,extract_ptx,Program
P=2**256-2**32-977;C=2**32+977;W=2**64;MASK=W-1;MOD=2**384;MM=0xD838091DD2253531
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def limbs(x,n):return [(x>>(64*i))&MASK for i in range(n)]
def integer(a):return sum(v<<(64*i) for i,v in enumerate(a))
def signed(x,n):return x-2**n if x>>(n-1)&1 else x
def add_scan(a,b):
 sums=[(x+y)&MASK for x,y in zip(a,b)]
 g=sum(1<<i for i in range(32) if i%8<5 and sums[i]<a[i]);p=sum(1<<i for i in range(32) if i%8<6 and sums[i]==MASK)
 carry=(((g<<1)+p)^p)&0xffffffff
 return [(v+((carry>>i)&1))&MASK for i,v in enumerate(sums)]
def inverse(x):
 states=[P,x,0,1];batches=0
 while True:
  u,v,r,s=states;pos=max((u|v).bit_length()-1,0)//64
  u0=u&MASK;v0=v&MASK;uh=(u>>(64*pos))&MASK;vh=(v>>(64*pos))&MASK
  if pos:
   shift=64-(uh|vh).bit_length()
   if shift:uh=((u>>(64*(pos-1)))>>(64-shift))&MASK;vh=((v>>(64*(pos-1)))>>(64-shift))&MASK
  uu,uv,vu,vv=1,0,0,1;bits=62
  while True:
   sentinel=v0|(1<<bits);zeros=(sentinel&-sentinel).bit_length()-1
   v0>>=zeros;vh>>=zeros;uu<<=zeros;uv<<=zeros;bits-=zeros
   if not bits:break
   if vh<uh:uh,vh=vh,uh;u0,v0=v0,u0;uu,vu=vu,uu;uv,vv=vv,uv
   vh=(vh-uh)&MASK;v0=(v0-u0)&MASK;vv-=uv;vu-=uu
  assert all(-2**63<z<2**63 for z in [uu,uv,vu,vv])
  raw=[uu*u+uv*v,vu*u+vv*v,uu*r+uv*s,vu*r+vv*s]
  for i in range(2):
   # Author selects U/V sign from word4, requiring signed320-bit fit.
   assert -2**319<=raw[i]<2**319
   if raw[i]<0:raw[i]=-raw[i];raw[i+2]=-raw[i+2]
  for i in [2,3]:
   assert -2**383<=raw[i]<2**383
   m=(raw[i]*MM)&(2**62-1);raw[i]+=m*P
  assert all(z%2**62==0 for z in raw)
  states=[z//2**62 for z in raw];batches+=1
  assert all(-2**319<=z<2**319 for z in states) and batches<32
  if states[1]==0:break
 return states[2]%P if states[0]==1 else 0,batches

def main():
 h=HERE/'head';math=(h/'GPUMath.h').read_text();rng=random.Random(189)
 carry=0
 for pattern in itertools.product(range(3),repeat=6):
  a=[0]*32;b=[0]*32
  for base in range(0,32,8):
   for j,t in enumerate(pattern):a[base+j],b[base+j]=[(0,0),(MASK,0),(MASK,1)][t]
  out=add_scan(a,b)
  for base in range(0,32,8):assert integer(out[base:base+6])==(integer(a[base:base+6])+integer(b[base:base+6]))%MOD
  carry+=1
 for _ in range(10000):
  a=[rng.getrandbits(64) for i in range(32)];b=[rng.getrandbits(64) for i in range(32)];out=add_scan(a,b)
  for base in range(0,32,8):assert integer(out[base:base+6])==(integer(a[base:base+6])+integer(b[base:base+6]))%MOD
 sparse=0
 for m in [0,1,2,2**62-1]+[rng.getrandbits(62) for _ in range(10000)]:
  al=(m*C)&MASK;ah=m*C>>64;b=int(m!=0)
  words=[-al,-ah-b,-b,-b,m-b,0]
  assert integer([z&MASK for z in words])==m*P;sparse+=1
 inputs=[0,1,2,P-1,P,P+1,2**256-1]+[rng.randrange(1,P) for _ in range(512)]
 batches=[]
 for x in inputs:
  out,n=inverse(x);assert out==(pow(x%P,-1,P) if x%P else 0);batches.append(n)
 mul=function(math,'__device__ __forceinline__ void _ModMultCore(');prog=Program(extract_ptx(mul));witnesses=[]
 for a,b in [(P-1,P-1),(P-65537,P-65537),(P-1,P-(1<<224)),(P-1,P-(1<<64)),(2**256-1,2**256-1)]:
  actual=integer(prog.run(limbs(a,4)+limbs(b,4)));expected=a*b%P
  if actual%P!=expected:witnesses.append({'a':hex(a),'b':hex(b),'device_ptx_residue':hex(actual%P),'expected':hex(expected),'inputs_canonical':a<P and b<P})
 assert witnesses
 square=Program(extract_ptx(function(math,'__device__ __forceinline__ void _ModSqr(')))
 square_cases=[]
 for a in [P-65537,P-1,P-(1<<64)]:
  actual=integer(square.run(limbs(a,4)))%P;expected=a*a%P
  square_cases.append({'a':hex(a),'inputs_canonical':True,'device_ptx_residue':hex(actual),'expected':hex(expected),'matches':actual==expected})
 ours=ROOT/'research/weak_field/ordinary_region/candidate/GPUMath.h';om=ours.read_text()
 deps={}
 for sig in ['__device__ __forceinline__ uint32_t _CTZ(']:
  assert function(math,sig)==function(om,sig);deps[sig]=hashlib.sha256(function(math,sig).encode()).hexdigest()
 for name in ['MM64','MSK62','_IsNegative(x)','AddP(r)','SubP(r)']:
  prefix='#define '+name+' ';a=math.index(prefix);b=om.index(prefix)
  block=math[a:math.index('\n\n',a)];oursblock=om[b:om.index('\n\n',b)]
  if name=='_IsNegative(x)':block=block.splitlines()[0];oursblock=oursblock.splitlines()[0]
  assert block==oursblock,name
  deps[name]=hashlib.sha256(block.encode()).hexdigest()
 hashes={str(p.relative_to(HERE)):sha(p) for parent in ['head','base'] for p in (HERE/parent).rglob('*') if p.is_file()}
 assert sha(h/'GPUMath.h')==sha(HERE/'base/GPUMath.h')
 result={'status':'SOURCE_REVIEW_WITH_FINITE_ARITHMETIC_MODEL_PASS_NOT_FULL_HELPER_VALIDATION','public_head':'3c3d748e7e46bbf68fcd22d179c44b231eaa15de','source_sha256':hashes,'checker_sha256':sha(Path(__file__)),'carry_patterns_each_four_groups':carry,'random_four_group_additions':10000,'sparse_m_times_p_cases':sparse,'integer_inverse_model_cases':len(inputs),'matrix_batches_min_max':[min(batches),max(batches)],'hot_product_carry_failures':witnesses,'default_ZLAB_MODSQR_1_device_cases':square_cases,'public_hot_math_identical_to_base':True,'matched_dependencies':deps,'model_limit':'Carry/sparse-modulus/batched-inverse Python transcription plus actual-source straight-line hot-product PTX semantic execution. No actual cooperative C++ inverse, CUDA execution, native compilation or speed measurement.'}
 (HERE/'arithmetic-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','integer_inverse_model_cases','matrix_batches_min_max','hot_product_carry_failures']}))
if __name__=='__main__':main()
