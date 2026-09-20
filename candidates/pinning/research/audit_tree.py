from pathlib import Path
import ast,hashlib,json,random,runpy,re
W=Path(__file__).resolve().parent;C=W.parent
B=1<<256;K=(1<<32)+977;P=B-K
from baseline_model import sm,original

def run(logical,m,G=2,N=128):
 values=[logical[(t%G)*N+t//G] for t in range(G*N)];pr=dict(enumerate(values));roots=[None]*G
 writer={t:(t,'block') for t in pr};reads=0;warplds=0;edges=0;conflicts=[]
 def phase(rd,wr,carry,newfence,root={}):
  nonlocal reads,warplds,edges
  # A None first operand means this thread's retained previous result.
  for dest,t in wr.items():
   if dest in pr:assert [u for u,(a,b) in rd.items() if dest in [a,b]]==[t]
  for t,addrs in rd.items():
   for addr in addrs:
    if addr is not None:
     reads+=1;p,f=writer[addr];assert f=='block' or p//32==t//32;edges+=1
  worst=1
  for operand in range(2):
   warplds+=len({t//32 for t,ab in rd.items() if ab[operand] is not None})
   for start in range(0,N*G,16):
    banks={}
    for t in range(start,start+16):
     if t not in rd or rd[t][operand] is None:continue
     addr=rd[t][operand]
     for word in (2*addr,2*addr+1):banks.setdefault(word%32,set()).add(word)
    worst=max([worst]+[len(x) for x in banks.values()])
  conflicts.append(worst)
  out={t:m(values[t] if aa is None else pr[aa],pr[bb]) for t,(aa,bb) in rd.items()}
  for addr,t in wr.items():pr[addr]=out[t];writer[addr]=(t,newfence)
  for t in carry:values[t]=out[t]
  for g,t in root.items():roots[g]=out[t]
  return out
 off=0
 for count in [128,64,32,16,8,4]:
  half=count//2;rd={t:(None,(off+half)*G+t) for t in range(G*half)}
  wr={(off+count)*G+t:t for t in rd};phase(rd,wr,rd,'block' if G*half>32 else 'warp');off+=count
 rd={};wr={};carry=[];rt={}
 for t in range(G*5):
  i,g=divmod(t,G);aa=(2*N-4+(((i&1)^1) if i<4 else 0))*G+g;bb=(2*N-8+(i^2) if i<4 else 2*N-3)*G+g
  rd[t]=(aa,bb)
  if i<4:wr[bb]=t;carry.append(t)
  else:rt[g]=t
 phase(rd,wr,carry,'warp',rt);off=2*N-16
 for count in [8,16,32,64]:
  half=count//2;rd={};wr={}
  for t in range(G*count):
   i,g=divmod(t,G);aa=None if i<half else (off+count+((i&(half-1))^(half//2)))*G+g;bb=(off+(i^half))*G+g
   rd[t]=(aa,bb);wr[bb]=t
  phase(rd,wr,rd,'block' if G*count*2>32 else 'warp');off-=count*2
 rd={}
 for t in range(G*N):
  i,g=divmod(t,G);rd[t]=(None if i<N//2 else (N+((i&(N//2-1))^(N//4)))*G+g,(i^(N//2))*G+g)
 out=phase(rd,{},rd,'none');logicalout=[None]*(G*N)
 for t,x in out.items():logicalout[(t%G)*N+t//G]=x
 return roots,logicalout,{'scalar_shared_field_reads':reads,'warp_shared_field_read_issues':warplds,'max_bank_conflict':max(conflicts),'read_edges':edges}
rng=random.Random(202609210225);checks=0;stats={};sym=lambda a,b:hashlib.sha256(a+b).digest()
for G in [2,4]:
 x=[hashlib.sha256(str(i).encode()).digest() for i in range(128*G)];r,o,_=run(x,sym,G)
 for g in range(G):assert original(x[g*128:(g+1)*128],sym)==(r[g],o[g*128:(g+1)*128])
 for active in sorted(set([0,1,31,127,128,129,255,256,257,128*G-1,128*G])):
  if active>128*G:continue
  for _ in range(6):
   x=[rng.getrandbits(256) if i<active else 1 for i in range(128*G)];r,o,st=run(x,sm,G)
   for g in range(G):assert original(x[g*128:(g+1)*128],sm)==(r[g],o[g*128:(g+1)*128])
   checks+=len(x)
 for v in [0,1,P-1,P,P+1,B-1]:
  x=[v]*(128*G);r,o,st=run(x,sm,G)
  for g in range(G):assert original(x[g*128:(g+1)*128],sm)==(r[g],o[g*128:(g+1)*128])
  checks+=len(x)
 stats[G]=st
q=runpy.run_path(str(W/'ptx_field_model.py'));f=q['function']((C/'GPUMath.h').read_text(),'void _ModMultCore(')
s=q['extract_ptx'](f.replace('QSB_SECOND_FOLD_TAIL','""'));s=re.sub(r'/\*.*?\*/','',s,flags=re.S);s=re.sub(r'\{\s*(?=\.reg)','',s[1:-1]);s=re.sub(r'(?<=;)\s*\}','',s)
p=q['Program']('{'+s+'}');limbs=lambda x:[x>>(64*k)&((1<<64)-1) for k in range(4)]
pm=lambda a,b:sum(v<<(64*k) for k,v in enumerate(p.run(limbs(a)+limbs(b))))
x=[rng.getrandbits(256) for _ in range(256)];assert run(x,pm)[:2]==run(x,sm)[:2]
h=(C/'InterleavedCarry.cuh').read_text()
for t in ['const int t=threadIdx.x,g=t%G,i=t/G','qsb_field_mul_sc(out,value,b)','(off+(i^half))*G+g','i<N/2?value[k]']:assert t in h
r={'status':'PASS','symbolic_ordered_leaves':768,'raw_leaves':checks,'actual_PTX_leaves':256,'stats':stats,'native_compile':False,'production_integrated':True,'limits':'Bank simulation assumes64bit16lane transactions. Source binding targeted, not full C++ interpreter. Candidate permutation and call sites are covered separately by audit_integration.py.'}
(W/'interleaved-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
