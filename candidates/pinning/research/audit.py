from pathlib import Path
import re,ast,json,random,itertools,hashlib
from ptx_field_model import Program,function,extract_ptx,check_semantics
W=Path(__file__).resolve().parent;C=W.parent;D=W/'baseline'
B=1<<256;K=(1<<32)+977;P=B-K;M64=(1<<64)-1
check_semantics()
# Extend strict interpreter with 32-bit subtraction opcodes used by fused square.
# The arithmetic implementation already selects carry width from opcode.
limbs=lambda v:[v>>(64*i)&M64 for i in range(4)]
join=lambda a:sum(v<<(64*i) for i,v in enumerate(a))
def branches(s,short):
 # Only embedded QSB_SHORT_CARRY conditional has to be selected for fused PTX.
 return re.sub(r'#if QSB_SHORT_CARRY\n(.*?)#else\n(.*?)#endif',lambda m:m[1] if short else m[2],s,flags=re.S)
def blocks(s):
 s=re.sub(r'/\*.*?\*/','',s,flags=re.S);out=[]
 for m in re.finditer(r'asm\s*\(([^:]*?)\s*:\s*"=l"',s,re.S):
  p=''.join(ast.literal_eval(x) for x in re.findall(r'"(?:[^"\\]|\\.)*"',m[1]))
  if 'mul.wide.u32 e0, a0, b0;' in p or 'mul.wide.u32 e6, a2, a4;' in p:out.append(p)
 assert len(out)==5
 return out
base=(D/'GPUMath.h').read_text();new=(C/'GPUMath.h').read_text()
# First four asm bodies explicitly contain both compile-time variants.
ba=blocks(base);na=blocks(new);pairs=[(Program(x),Program(y),i<2) for i,(x,y) in enumerate(zip(ba[:4],na[:4]))]
for short in [True,False]:
 body0=branches(function(base,'void _ModSqrAddSub2('),short);body1=branches(function(new,'void _ModSqrAddSub2('),short)
 pairs.append((Program(extract_ptx(body0)),Program(extract_ptx(body1)),None))
for folder in [D,C]:
 src=function((folder/'pinning.cu').read_text(),'void qsb_field_mul(')
 p=Program(extract_ptx(src).replace('.reg .pred take;',''))
 if folder==D:oldraw=p
 else:pairs.append((oldraw,p,True))
rng=random.Random(2026092014);edges=[0,1,2,K-1,K,K+1,P-K,P-2,P-1,P,P+1,B-2,B-1]
rows=list(itertools.product(edges,repeat=2))+[(rng.getrandbits(256),rng.getrandbits(256)) for _ in range(1500)]
count=0
for a,b in rows:
 for old,nw,ismul in pairs:
  v=limbs(a)+(limbs(b) if ismul else [])
  if ismul is None:v+=limbs(b)+limbs(rng.choice(edges))
  assert old.run(v)==nw.run(v),(a,b,ismul)
  count+=1
# Derived first-fold bound for arbitrary raw e,q, also minimum nonnegative.
assert 3*P-2*(B-1)>0
assert ((B-1)*(K+1)+3*P+(B-1))//B < K+5
# Test second fold directly at entire mathematical high-word interval boundaries.
# Symbolic argument: if z9=0 sfq=z8; if z9=1 z8<K+5-2^32=982.
assert 977+(K+4-(1<<32))<1<<32
prefix=na[0];start=prefix.index('.reg .u32 sfq');end=prefix.index('addc.cc.u32 z3',start)
fold=prefix[start:end]
init='.reg .u32 z0,z1,z2,z8,z9; mov.b64 {z0,z1},%4; mov.b64 {z2,z8},%5; mov.u32 z9,%6;'
prog=Program('{'+init+fold+'mov.b64 %0,{z0,z1}; mov.b64 %1,{z2,0}; mov.b64 %2,0; mov.b64 %3,0;}')
fc=0
for c in [0,1,976,977,(1<<32)-2,(1<<32)-1,1<<32,K-1,K,K+3,K+4]:
 for low in [0,1,(1<<32)-1,(1<<64)-1,(1<<96)-1]+[rng.getrandbits(96) for _ in range(100)]:
  out=join(prog.run([low&M64,(low>>64)|((c&0xffffffff)<<32),c>>32]))
  assert out==(low+c*K)% (1<<96);fc+=1
# Source-bound call interpreter for finish. Existing approximate field helpers
# are tested separately above; formula proof uses their field contract.
names='qsb_recovery_mul|_ModSub256|_ModAdd256|qsb_packed_raw_mul|qsb_parity_boundary'
def finish(src,row):
 calls=re.findall(r'('+names+r')\(([^()]*)\);',function(src,'uint32_t qsb_packed_finish('))
 u,v,a,b,c=row;state=dict(tbar=u,vbar=v,weighted_inv=1,root_inv=1,a=a,b=b,c=c);par=[]
 for name,args in calls:
  out,*ins=[x.strip() for x in args.split(',')];v=[state[x] for x in ins]
  if name=='qsb_parity_boundary':
   par.append(((state[out]-v[0]) if not par else (v[0]-state[out]))%P&1);continue
  if name in ['qsb_recovery_mul','qsb_packed_raw_mul']:z=v[0]*v[1]%P
  elif name=='_ModSub256':z=(v[0]-v[1])%P
  else:z=sum(v)%P
  state[out]=z
 return state['x1'],state['x2'],par
bpack=(D/'PackedRecovery.cuh').read_text();npack=(C/'PackedRecovery.cuh').read_text()
finishrows=[tuple(rng.randrange(P) for _ in range(5)) for _ in range(12000)]
finishrows += [(u%P,v%P,a,b,c) for u,v in itertools.product(edges,repeat=2) for a,b,c in [(0,0,0),(P-1,P-1,P-1),(1,2,3)]]
for row in finishrows:assert finish(bpack,row)==finish(npack,row)
# Non-associative product oracle ensures tree topology / operand order are exact,
# not merely congruent. Also compare true field cofactor/root identities.
def tree(leaves,new,mul):
 n=len(leaves);prod=dict(enumerate(leaves));exc={};offset=0;count=n
 while count>(4 if new else 1):
  half=count//2
  for tid in range(half):prod[offset+count+tid]=mul(prod[offset+tid],prod[offset+half+tid])
  offset+=count;count//=2
 if new:
  for tid in range(2):prod[2*n-4+tid]=mul(prod[2*n-8+tid],prod[2*n-6+tid])
  for tid in range(5):
   left=2*n-4 if tid==4 else 2*n-4+((tid&1)^1)
   right=2*n-3 if tid==4 else 2*n-8+(tid^2)
   out=mul(prod[left],prod[right])
   if tid==4:root=out
   else:exc[n-8+tid]=out
  offset=2*n-16;count=8
 else:root=prod[2*n-2];exc[n-2]=1;offset=2*n-4;count=2
 while count<n:
  half=count//2
  for tid in range(count):
   parent=exc[offset+count-n+(tid&(half-1))];sib=prod[offset+(tid^half)]
   exc[offset-n+tid]=sib if count==2 else mul(parent,sib)
  offset-=count*2;count*=2
 return root,[mul(exc[tid&(n//2-1)],prod[tid^(n//2)]) for tid in range(n)]
treechecks=0
for n in [8,16,32,64,128,256]:
 for j in range(50):
  vals=[rng.randrange(1,P) if j else 1 for _ in range(n)]
  for mul in [lambda a,b:a*b%P, lambda a,b:(a*977+b*65537+11)%P]:
   assert tree(vals,False,mul)==tree(vals,True,mul);treechecks+=n
  root,excluded=tree(vals,True,lambda a,b:a*b%P)
  for i,v in enumerate(vals):assert excluded[i]*v%P==root
# Bind critical tree schedule and generic compile-time restriction to source.
treecode=(C/'cofactor_checkpoint.h').read_text()
for token in ['N>=8','count>4','count=8','offset=2*N-16','2*N-8+tid','2*N-6+tid','2*N-4+((tid&1)^1)','2*N-8+(tid^2)','N-8+tid']:
 assert token in treecode,token
assert treecode.count('if(tid<5)')==1 and treecode.count('if(tid==4)')==1
assert 'QSB_RECOVERY_N==128' in (C/'pinning.cu').read_text()
# All six edits end before carry propagation. No downstream arithmetic changes.
import sys
pat=re.compile(r'mul\.wide\.u32 t, z8, 977;\s*mov\.b64 \{m0,m1\}, t;\s*mad\.lo\.u32 m1, z9, 977, m1;\s*add\.cc\.u32 m1, m1, z8;\s*addc\.u32 m2, z9, 0;\s*add\.cc\.u32 z0, z0, m0;\s*addc\.cc\.u32 z1, z1, m1;\s*addc\.cc\.u32 z2, z2, m2;')
fused='''.reg .u32 sfq,sfc,sfl,sfh; .reg .u64 sfz,sft;
mad.lo.u32 sfq,z9,977,z8;
mov.b64 sfz,{z0,sfq};
mul.wide.u32 sft,z8,977;
add.cc.u64 sft,sft,sfz;
addc.u32 sfc,z9,0;
mov.b64 {sfl,sfh},sft;
mov.u32 z0,sfl;
add.cc.u32 z1,z1,sfh;
addc.cc.u32 z2,z2,sfc;'''

for name in ['GPUMath.h','pinning.cu']:
 old=(D/name).read_text();nw=(C/name).read_text()
 # String token normalization allows regenerated whitespace only.
 def normalized(s):
  s=re.sub(r'//[^\n]*','',s)
  def collapse(m):
   raw=''.join(ast.literal_eval(x) for x in re.findall(r'"(?:[^"\\]|\\.)*"',m[0]));return json.dumps(re.sub(r'\s+','',pat.sub(fused,raw)))
  return re.sub(r'\s+','',re.sub(r'"(?:[^"\\]|\\.)*"(?:\s*"(?:[^"\\]|\\.)*")*',collapse,s))
 assert normalized(old)==normalized(nw),name
result={'status':'PASS','primitive_pair_checks':count,'second_fold_boundary_checks':fc,'finish_field_contract_checks':len(finishrows),'tree_leaf_checks':treechecks,'PTX_counts':[(len(x.ops),len(y.ops)) for x,y,_ in pairs],'new_carry_omissions':0,'scope':'Python source/PTX semantics only; no native build or GPU measurements. Existing baseline short-carry remains. Finish proof is field-contract equivalence, not certification of inherited rare arithmetic errors. Tree uses non-associative oracle to certify unchanged grouping.','sha256':{n:hashlib.sha256((C/n).read_bytes()).hexdigest() for n in ['GPUMath.h','pinning.cu','PackedRecovery.cuh','cofactor_checkpoint.h']}}
(W/'audit-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
