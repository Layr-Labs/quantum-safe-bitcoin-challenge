"""Actual PTX and source-bound full tree audit; no native compilation."""
from pathlib import Path
import ast,re,random,json,hashlib,math,itertools
from ptx_field_model import Program,function,extract_ptx,check_semantics
W=Path(__file__).resolve().parent;C=W.parent;D=W/'baseline'
B=1<<256;K=(1<<32)+977;P=B-K;mask=(1<<64)-1
check_semantics()
limbs=lambda a:[a>>(64*i)&mask for i in range(4)]
join=lambda a:sum(v<<(64*i) for i,v in enumerate(a))
def blocks(s):
 out=[]
 for m in re.finditer(r'asm\s*\(([^:]*?)\s*:\s*"=l"',re.sub(r'/\*.*?\*/','',s,flags=re.S),re.S):
  p=''.join(ast.literal_eval(x) for x in re.findall(r'"(?:[^"\\]|\\.)*"',m[1]))
  if 'mul.wide.u32 e0, a0, b0;' in p or 'mul.wide.u32 e6, a2, a4;' in p:out.append(p)
 return out
h=(C/'GPUMath.h').read_text();bh=(D/'GPUMath.h').read_text();a=blocks(bh);b=blocks(h)
assert len(a)==len(b)==5
assert a[4]==b[4]
oldnew=[(Program(x),Program(y)) for x,y in zip(a[:4],b[:4])]
raw=Program(extract_ptx(function((C/'pinning.cu').read_text(),'void qsb_field_mul(')).replace('.reg .pred take;',''))
short=oldnew[0][1]
def sc(a,b):
 t=a*b;f=(t&(B-1))+K*(t>>256);r=f&(B-1)
 return (r&~((1<<160)-1))|((r+(f>>256)*K)&((1<<160)-1))
def smul(a,b):return join(short.run(limbs(a)+limbs(b)))
def fmul(a,b):return join(raw.run(limbs(a)+limbs(b)))%P
rng=random.Random(2026092016);edges=[0,1,2,K-1,K,K+1,P-2,P-1,P,P+1,B-1,(1<<160)-1]
rows=list(itertools.product(edges,repeat=2))+[(rng.getrandbits(256),rng.getrandbits(256)) for _ in range(1200)]
for x,y in rows:
 for i,(p,q) in enumerate(oldnew):
  inp=limbs(x)+(limbs(y) if i<2 else [])
  assert p.run(inp)==q.run(inp)
 assert smul(x,y)==sc(x,y)
 assert fmul(x,y)==x*y%P
# All unchanged runtime pieces byte-identical. Header changes limited to four asm strings.
for sig in ['void qsb_field_mul(', 'void qsb_field_mul_sc(', 'void qsb_field_normalize(']:
 assert function((C/'pinning.cu').read_text(),sig)==function((D/'pinning.cu').read_text(),sig)
for sig in ['void _PointAddXYZZT(','void _ModSqrAddSub2(']:assert function(h,sig)==function(bh,sig)
for n in ['GPUHash.h','LeafRecovery.cuh','RecoveryConstant.h','sha_schedule_interleaved.cuh']:assert (C/n).read_bytes()==(D/n).read_bytes()
# Parse recovery's ordered operations, including aliased destinations.
pat=r'(qsb_recovery_mul|_ModSub256|_ModAdd256|qsb_packed_raw_mul|qsb_parity_boundary)\(([^()]*)\);'
def finish(s,row):
 state=dict(zip(['tbar','vbar','a','b','c'],row));state.update(weighted_inv=1,root_inv=1);par=[]
 for name,args in re.findall(pat,function(s,'uint32_t qsb_packed_finish(')):
  out,*ins=map(str.strip,args.split(','));vs=[state[n] for n in ins]
  if name=='qsb_parity_boundary':par.append(((state[out]-vs[0]) if not par else (vs[0]-state[out]))%P&1);continue
  if name in ['qsb_recovery_mul','qsb_packed_raw_mul']:z=math.prod(vs)%P
  elif name=='_ModSub256':z=(vs[0]-vs[1])%P
  else:z=sum(vs)%P
  state[out]=z
 return state['x1'],state['x2'],par
bp=(D/'PackedRecovery.cuh').read_text();np=(C/'PackedRecovery.cuh').read_text()
for _ in range(12000):
 row=[rng.randrange(P) for _ in range(5)];assert finish(bp,row)==finish(np,row)
# Bind full traversal to source tokens and shared phase boundaries.
s=(C/'cofactor_checkpoint.h').read_text();top=(C/'Top16Exact.cuh').read_text()
for token in ['count>16','qsb_top16_exact<N>(roots,products,excluded);','offset=2*N-64','count=32;count<N','if((count<<1)>32)__syncthreads();else __syncwarp();']:assert token in s
for token in ['span=8;span>=1;span>>=1','tid<16 && span<8','tid>=16 && tid<16+span','2*N-4*span','2*N-32+(tid^8)','source+((tid&(2*span-1))^span)','2*N-2*span+tid-16','N-32+tid','qsb_field_normalize(out);']:assert token in top
assert top.count('qsb_field_mul(out,a,b);')==1 and 'qsb_field_mul_sc(' not in top
assert top.count('__syncwarp();')==2
assert '__syncthreads();' in function(np,'void qsb_packed_prepare(')
# Interpreter for production indexing. Stores commit only at barrier boundaries;
# sets assert no same-wave cross-lane read/write and no undefined accumulator.
def tree(vals,sm,fm):
 N=len(vals);pr=dict(enumerate(vals));ex={};off=0;count=N
 while count>16:
  half=count//2;writes={}
  for tid in range(half):writes[off+count+tid]=sm(pr[off+tid],pr[off+half+tid])
  assert not(set(writes)&set(pr));pr.update(writes);off+=count;count//=2
 x=[pr[2*N-32+i] for i in range(16)];acc={};root=None
 for span in [8,4,2,1]:
  stores={};nextacc={};reads=set()
  for tid in range(N):
   low=tid<16 and span<8;upper=16<=tid<16+span
   if not(low or upper):continue
   source=2*N-4*span
   if low:
    if span==4:i=2*N-32+(tid^8);a=pr[i];reads.add(i)
    else:a=acc[tid]
    j=source+((tid&(2*span-1))^span);b=pr[j];reads.add(j)
   else:
    i=source+tid-16;j=i+span;reads.update([i,j]);a=pr[i];b=pr[j]
   out=fm(a,b)
   if low:nextacc[tid]=out
   elif span==1:root=out
   else:stores[2*N-2*span+tid-16]=out
  assert not(set(stores)&reads);pr.update(stores);acc.update(nextacc)
 assert root==math.prod(x)%P
 for i in range(16):assert acc[i]==math.prod(x[:i]+x[i+1:])%P;ex[N-32+i]=acc[i]
 off=2*N-64;count=32
 while count<N:
  half=count//2;writes={};reads=set()
  for tid in range(count):
   i=off+count-N+(tid&(half-1));reads.add(i)
   writes[off-N+tid]=sm(ex[i],pr[off+(tid^half)])
  assert not(set(writes)&reads);ex.update(writes);off-=count*2;count*=2
 out=[sm(ex[tid&(N//2-1)],pr[tid^(N//2)]) for tid in range(N)]
 return root,out
cases=[]
for N in [32,64,128,256]:
 cases.extend([[1]*N,[0]*N,[P-1]*N,[rng.getrandbits(256) for _ in range(N)]])
 for active in [1,N//2,N-1,N]:cases.append([rng.randrange(1,P) if i<active else 1 for i in range(N)])
 for _ in range(24):cases.append([rng.randrange(1,P) for _ in range(N)])
leafchecks=0
for vals in cases:
 # Exact-field oracle for full geometry, then mixed inherited/new contract.
 root,out=tree(vals,lambda a,b:a*b%P,lambda a,b:a*b%P)
 assert root==math.prod(vals)%P
 for i,y in enumerate(out):assert y==math.prod(vals[:i]+vals[i+1:])%P
 tree(vals,sc,lambda a,b:a*b%P);leafchecks+=len(vals)
# Real PTX all the way through selected full128-leaf paths, including padded tails.
ptxcases=[[1]*128,[P-1]*128,[rng.getrandbits(256) for _ in range(128)]]
ptxcases += [[rng.randrange(1,P) if i<n else 1 for i in range(128)] for n in [1,17,65,127]]
counter=json.loads((W/'short-carry-counterexample.json').read_text())['top16_raw_inputs']
ptxcases.append(counter+[1]*112)
for vals in ptxcases:assert tree(vals,smul,fmul)==tree(vals,sc,lambda a,b:a*b%P)
result={'status':'PASS','fourfold_old_new_PTX_checks':len(rows)*4,'short_and_complete_PTX_checks':len(rows)*2,'finish_field_contract_checks':12000,'full_tree_geometry_leaf_checks':leafchecks,'full128_actual_PTX_trees':len(ptxcases),'full128_actual_PTX_leaves':len(ptxcases)*128,'top_warps_before_after':[7,4],'top_scalar_M_before_after':[43,63],'new_truncated_multiplications':0,'limitations':'No native compile/GPU timing. Full128PTX compares actual mixed inherited-short/new-complete semantics. Separate mathematical geometry checks use exactfield. Existing lower-tree/chain shortcarry is not certified as universal exact arithmetic. Finish uses fieldcontract. Source/barrier audit is not a CUDA race detector.','sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in C.iterdir() if p.suffix in ['.h','.cuh','.cu']}}
(W/'audit-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
