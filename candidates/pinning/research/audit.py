"""Source PTX and full-tree boundary audit; deliberately no native compilation."""
from pathlib import Path
import re,random,json,hashlib,math,itertools
from ptx_field_model import Program,function,extract_ptx,check_semantics
W=Path(__file__).resolve().parent;C=W.parent
B=1<<256;K=(1<<32)+977;P=B-K;mask=(1<<64)-1
check_semantics()
limbs=lambda a:[a>>(64*i)&mask for i in range(4)]
join=lambda a:sum(v<<(64*i) for i,v in enumerate(a))
def program(body):
 s=extract_ptx(body).replace('.reg .pred take;','')
 s=re.sub(r'\{\s*(?=\.reg)','',s[1:-1]);s=re.sub(r'(?<=;)\s*\}', '',s)
 return Program('{'+s+'}')
baseline=program(function((C/'pinning.cu').read_text(),'void qsb_field_mul('))
new=program(function((C/'Top16Mul.cuh').read_text(),'void qsb_top16_mul('))
f=function((C/'GPUMath.h').read_text(),'void _ModMultCore(')
f=re.sub(r'#if QSB_SHORT_CARRY\n(.*?)#else\n(.*?)#endif',lambda m:m[1],f,flags=re.S)
short=program(f)
def sc(a,b):
 t=a*b;f=(t&(B-1))+K*(t>>256);r=f&(B-1)
 return (r&~((1<<160)-1))|((r+(f>>256)*K)&((1<<160)-1))
def raw(a,b):
 t=a*b;f=(t&(B-1))+K*(t>>256);f=(f&(B-1))+K*(f>>256)
 return (f&(B-1))+(f>>256)*K
smul=lambda a,b:join(short.run(limbs(a)+limbs(b)))
fmul=lambda a,b:join(new.run(limbs(a)+limbs(b)))
rng=random.Random(202609202045)
edges=[0,1,2,K-1,K,K+1,P-2,P-1,P,P+1,B-1,(1<<160)-1]
for bit in [32,64,96,128,160,192,224,255]:
 edges.extend([(1<<bit)-1,1<<bit,(1<<bit)+1,B-(1<<bit)])
rows=list(itertools.product(edges,repeat=2))+[(rng.getrandbits(256),rng.getrandbits(256)) for _ in range(6000)]
# Also force both top words into the carry-active band rejected by truncation.
rows += [(B-1-rng.randrange(978<<224),B-1-rng.randrange(978<<224)) for _ in range(2000)]
for a,b in rows:
 v=limbs(a)+limbs(b);r=fmul(a,b)
 assert r==join(baseline.run(v))==raw(a,b)
 assert 0<=r<B and r%P==a*b%P
 assert smul(a,b)==sc(a,b)
# The fused high word cannot wrap: F=L+KH < (K+1)*B, so z9=1 implies z8<=977.
assert (B-1)+K*(B-2)<(K+1)*B
assert 977+977 < 1<<32
# Binding: production indices, complete multiplier, boundary normalize, and barriers.
s=(C/'cofactor_checkpoint.h').read_text();top=(C/'Top16Exact.cuh').read_text()
for t in ['count>16','qsb_top16_exact<N>(roots,products,excluded);','offset=2*N-64','count=32;count<N','if((count<<1)>32)__syncthreads();else __syncwarp();']:assert t in s
for t in ['span=8;span>=1;span>>=1','tid<16 && span<8','tid>=16 && tid<16+span','2*N-4*span','2*N-32+(tid^8)','source+((tid&(2*span-1))^span)','2*N-2*span+tid-16','N-32+tid','if(span==1)qsb_field_normalize(out);']:assert t in top
assert top.count('qsb_top16_mul(out,a,b);')==1 and 'qsb_field_mul_sc(' not in top
assert top.count('__syncwarp();')==2
assert '__syncthreads();' in function((C/'PackedRecovery.cuh').read_text(),'void qsb_packed_prepare(')
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
   if span==1:out%=P
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
leafchecks=0;casecount=0
for N in [32,64,128,256]:
 cases=[[1]*N,[0]*N,[P-1]*N,[P]*N,[B-1]*N]
 for active in [1,17,N//2,N-1,N]:cases.append([rng.randrange(1,P) if i<active else 1 for i in range(N)])
 cases += [[rng.getrandbits(256) for _ in range(N)] for _ in range(32)]
 for vals in cases:
  # Same final raw bits as full-canonical top16, including inherited lower-tree approximation.
  assert tree(vals,sc,raw)==tree(vals,sc,lambda a,b:a*b%P)
  root,out=tree(vals,lambda a,b:a*b%P,raw)
  assert root==math.prod(vals)%P
  for i,y in enumerate(out):assert y==math.prod(vals[:i]+vals[i+1:])%P
  leafchecks+=N;casecount+=1
ptxcases=[[1]*128,[0]*128,[P]*128,[P-1]*128,[B-1]*128,[rng.getrandbits(256) for _ in range(128)]]
ptxcases += [[rng.randrange(1,P) if i<n else 1 for i in range(128)] for n in [1,17,65,127]]
counter=json.loads((W/'short-carry-counterexample.json').read_text())['top16_raw_inputs']
ptxcases.append(counter+[1]*112)
for vals in ptxcases:
 assert tree(vals,smul,fmul)==tree(vals,sc,lambda a,b:a*b%P)
# Negative controls: lost last carry must be detected at extremes; raw p must be canonicalized.
assert (P-1)*(P-1)%P==1 and raw(P-1,P-1)==P+1
mutant=(B-1)*(B-1);mutant=(mutant&(B-1))+K*(mutant>>256);mutant=(mutant&(B-1))+K*(mutant>>256)
assert mutant>>256 and (mutant&(B-1))%P!=(B-1)*(B-1)%P
assert raw(P,1)==P and raw(P,1)%P==0
r={'status':'PASS','complete_fused_vs_frontier_PTX_pairs':len(rows),'short_PTX_vs_integer_model_pairs':len(rows),'full_tree_cases':casecount,'full_tree_leaf_checks':leafchecks,'actual128_leaf_PTX_trees':len(ptxcases),'actual128_leaf_PTX_leaves':128*len(ptxcases),'top_warp_multiply_waves_current_frontier_to_candidate':[6,4],'top_scalar_multiplications_current_frontier_to_candidate':[43,63],'normalizations_vs_previous_complete_top16':[63,17],'normalization_wave_count':[4,1],'negative_controls':2,'native_compile':False,'GPU_performance_measured':False,'limitations':'PTX semantic model and source-bound tree indexing are not native compilation, a race detector, or a GPU timing. Lower-tree and point-chain short-carry approximations are inherited. New top reassociation uses complete products; boundary outputs equal canonical top16 for all modeled inputs.','source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in C.iterdir() if p.suffix in ['.cu','.cuh','.h']}}
(W/'audit-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
