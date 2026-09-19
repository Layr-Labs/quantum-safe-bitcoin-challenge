"""Audit source-bound recovery completion; Python only, no C++/CUDA compile."""
from pathlib import Path
import re,json,random,hashlib,argparse
from ptx_field_model import Program,function,extract_ptx,check_semantics
W=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,default=W/'candidate/candidates/pinning');ap.add_argument('--baseline',type=Path,default=W/'baseline/candidates/pinning');args=ap.parse_args();C=args.candidate;D=args.baseline
if not C.exists():C=W
B=1<<256;K=(1<<32)+977;p=B-K;mask=(1<<64)-1
check_semantics()
# PTX's subc consumes a borrow; instructions without .cc preserve it.
btest=Program('''{.reg .u64 t,k; sub.cc.u64 t,0,1; subc.u64 k,0,0;
mov.u64 %0,k; subc.u64 %1,5,1; sub.cc.u64 t,3,1; subc.u64 %2,5,1; mov.u64 %3,t;}''')
assert btest.run([])==[mask,3,4,2]
cu=(C/'pinning.cu').read_text();header=(C/'RecoverySquare.cuh').read_text();packed=(C/'PackedRecovery.cuh').read_text();host=(C/'RecoveryConstant.h').read_text()
def program(body):
 s=extract_ptx(body)
 if '.reg .pred take;' in s:assert s.count('take')==1;s=s.replace('.reg .pred take;','')
 return Program(s)
S=program(function(header,'void qsb_recovery_square_exact('));M=program(function(cu,'void qsb_field_mul('));SUB=program(function(header,'void qsb_recovery_sub_exact('))
limbs=lambda a:[(a>>(64*i))&mask for i in range(4)]
join=lambda a:sum(x<<(64*i) for i,x in enumerate(a))
def sq(a):return join(S.run(limbs(a)))%p
def mul(a,b):return join(M.run(limbs(a)+limbs(b)))%p
def sub(a,b):return join(SUB.run(limbs(a)+limbs(b)))
assert function(cu,'void qsb_field_mul(')==function((D/'pinning.cu').read_text(),'void qsb_field_mul(')
assert function(cu,'void qsb_field_mul_sc(')==function((D/'pinning.cu').read_text(),'void qsb_field_mul_sc(')
assert (C/'GPUMath.h').read_bytes()==(D/'GPUMath.h').read_bytes()
# Each new helper's normalization occurs after complete arithmetic.
assert function(header,'void qsb_recovery_square_exact(').rstrip().endswith('qsb_field_normalize(out);\n}')
prod=function(header,'void qsb_recovery_product_exact(')
assert re.search(r'uint64_t tmp\[5\];\s*qsb_field_mul\(tmp,const_cast<uint64_t\*>\(a\),const_cast<uint64_t\*>\(b\)\);\s*Load256\(out,tmp\);\s*qsb_field_normalize\(out\);',prod)
# Host equations bind to the actual BN sequence and explicit little-endian codec.
assert 'BN_mod_mul(h,c,half,p,ctx)' in host
assert 'BN_mod_sqr(t,c,p,ctx) && BN_mod_mul(t,t,half,p,ctx)' in host
assert 'BN_mod_sub(A,a,t,p,ctx)' in host
assert 'BN_copy(half,p) && BN_add_word(half,1) &&\n        BN_rshift1(half,half)' in host
assert 'cb[8*i+j]=(uint8_t)(cw[i]>>(8*j))' in host
for name in ['h','A']:
 assert f'cudaMemcpyToSymbol(pin_recovery_{name},recovery_{name},sizeof(recovery_{name}))!=cudaSuccess' in cu
 assert f'__device__ __constant__ uint64_t pin_recovery_{name}[4];' in cu
 assert all(f'pin_recovery_{name}[{i}]' in cu for i in range(4))
assert 'weighted_inv,u2rx,u2ry,recovery_h,recovery_A,q1x,q2x);' in cu
assert 'pin_recovery_c' not in cu
assert cu.index('#include "RecoverySquare.cuh"')<cu.index('#include "PackedRecovery.cuh"')
body=function(packed,'uint32_t qsb_packed_finish(')
# Interpret actual ordered call statements, including output aliases. The four
# inherited multiplies are modeled under their field contract; their existing
# approximate implementation is not certified by this test.
calls=re.findall(r'(qsb_recovery_mul|qsb_recovery_sub_exact|qsb_recovery_square_exact|qsb_recovery_product_exact|_ModAdd256|Load256|qsb_packed_raw_mul|qsb_parity_boundary)\(([^()]*)\);',body)
assert len(calls)==19,len(calls)
expected_names=['qsb_recovery_mul','qsb_recovery_mul','qsb_recovery_sub_exact','qsb_recovery_square_exact','_ModAdd256','_ModAdd256','qsb_recovery_product_exact','_ModAdd256','qsb_recovery_sub_exact','_ModAdd256','Load256','qsb_recovery_sub_exact','_ModAdd256','qsb_recovery_sub_exact','qsb_packed_raw_mul','qsb_parity_boundary','qsb_recovery_sub_exact','qsb_packed_raw_mul','qsb_parity_boundary']
assert [n for n,a in calls]==expected_names
rng=random.Random(2026092004)
edges=[0,1,2,K-1,K,K+1,p-K,p-65537,p-2,p-1,B-(1<<128)-1,B-(1<<192)-1]
rawedges=edges+[p,p+1,B-1]
subpairs=[(a,b) for a in edges for b in edges]+[(rng.randrange(p),rng.randrange(p)) for _ in range(2000)]
for a,b in subpairs:assert sub(a,b)==(a-b)%p
sqs=rawedges+[rng.getrandbits(256) for _ in range(2000)]
for a in sqs:assert sq(a)==a*a%p
rows=[(u,v,a,b,c) for u in edges for v in edges for a,b,c in [(0,0,0),(p-1,p-1,1),(1,2,p-1)]]
rows += [tuple(rng.randrange(p) for _ in range(5)) for _ in range(10000)]
def reference(u,v,a,b,c):
 l=(u-v)%p;m=(u+v)%p;ss=(l+m)%p
 x1=(a+ss*(l-c))%p;x2=(a+ss*(m-c))%p
 return x1,x2,((l*(a-x1)-b)%p)&1,((b-m*(a-x2))%p)&1
def evaluate(row,useptx):
 u,v,a,b,c=row
 # Reproduce explicit host byte<->word transfer before using either constant.
 cb=b''.join(x.to_bytes(8,'little') for x in limbs(c));assert int.from_bytes(cb,'little')==c
 half=(p+1)>>1;h=c*half%p;t=(c*c%p)*half%p;A=(a-t)%p
 for z in [h,A]: assert join(limbs(int.from_bytes(z.to_bytes(32,'little'),'little')))==z
 state={'a':a,'b':b,'h':h,'A':A,'tbar':u,'vbar':v,'weighted_inv':1,'root_inv':1}
 pars=[]
 for name,argstr in calls:
  args=[x.strip() for x in argstr.split(',')];out=args[0];vals=[state[x] for x in args[1:]]
  if name in ['qsb_recovery_mul','qsb_recovery_product_exact']:
   z=(mul(*vals) if useptx else vals[0]*vals[1]%p)
  elif name=='qsb_recovery_square_exact':z=sq(vals[0]) if useptx else vals[0]*vals[0]%p
  elif name=='qsb_recovery_sub_exact':z=sub(*vals) if useptx else (vals[0]-vals[1])%p
  elif name=='_ModAdd256':z=sum(vals)%p
  elif name=='Load256':z=vals[0]
  elif name=='qsb_packed_raw_mul':z=mul(*vals) if useptx else vals[0]*vals[1]%p
  elif name=='qsb_parity_boundary':
   # Exact canonical ordinate products suffice for the formula test. Existing
   # raw boundary proof is unchanged, separately source-checked below.
   s=state[out];b=vals[0];pars.append(((s-b)%p)&1 if len(pars)==0 else ((b-s)%p)&1);continue
  else:raise AssertionError(name)
  assert 0<=z<p,(name,z)
  state[out]=z
 return state['x1'],state['x2'],*pars
for row in rows:assert evaluate(row,False)==reference(*row)
for row in rows[:96]+rows[-160:]:assert evaluate(row,True)==reference(*row)
# The entire pre-finish recovery header is preserved, including raw/parity path.
old=(D/'PackedRecovery.cuh').read_text();marker='__device__ __forceinline__ uint32_t qsb_packed_finish('
prefix=packed[:packed.index('// Complete the square')];assert prefix==old[:old.index(marker)]
# Changed runtime surface excludes all original arithmetic/SHA/tree helpers.
unchanged=['GPUMath.h','GPUHash.h','LeafRecovery.cuh','cofactor_checkpoint.h','sha_schedule_interleaved.cuh','COPYING']
for name in unchanged:assert (C/name).read_bytes()==(D/name).read_bytes()
# Counterexample for omitted host correction.
assert (2*((-3*((p+1)//2))%p)**2+1)%p != reference(0,0,1,2,3)[0]
result={'status':'PASS','source_call_interpreter_cases':len(rows),'source_call_ptx_cases':256,'exact_square_ptx_cases':len(sqs),'exact_sub_ptx_cases':len(subpairs),'new_helper_instruction_counts':{'square':len(S.ops),'multiply':len(M.ops),'subtraction':len(SUB.ops)},'new_finish_temporaries':'six 4-limb local arrays vs seven in baseline, excluding compiler decisions','new_constants_bytes':64,'baseline_constants_bytes':32,'sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in C.iterdir() if x.name in ['pinning.cu','RecoverySquare.cuh','PackedRecovery.cuh','RecoveryConstant.h']},'limitations':'No native compilation/GPU execution. New square/sub helpers interpreted from actual PTX; source finish evaluated under field multiplication contract. Inherited approximate u/v/parity/tree helpers remain baseline; this is not universal bit equivalence to its rare errors. Host BN sequence checked structurally and modeled with Python integers, not executed as C++.'}
print(json.dumps(result,indent=2))
