"""Research-only lane routing, compact shared layout and outer interleaving checks."""
from pathlib import Path
import json,hashlib,random
D=Path(__file__).resolve().parent;prior=D;P=2**256-2**32-977
ns={'__file__':str(prior/'independent_model.py')}
exec(compile((prior/'independent_model.py').read_text(),'independent_defs','exec'),ns)
R=random.Random(0x265632);mul=ns['mul'];stats={'cooperative_unique_calls':0,'actual_subgroup_calls':0,'shuffle_lanes':0,'shared_accesses':0,'negative_bad_route':0};slots=set()
def vec(values):return [(values[lane//8]>>(32*(lane&7)))&0xffffffff for lane in range(32)]
def fields(v):return [sum(v[8*g+d]<<(32*d) for d in range(8)) for g in range(4)]
def shfl(v,fn):
 stats['shuffle_lanes']+=32
 return [v[fn(l)] for l in range(32)]
def M(a,b):
 aa=fields(a);bb=fields(b);stats['cooperative_unique_calls']+=len(set(zip(aa,bb)));stats['actual_subgroup_calls']+=4
 return vec([mul(x,y) for x,y in zip(aa,bb)])
def upper(vals,scale,bad=False):
 u4=M(vec(vals[:4]),vec(vals[4:]))
 u2=M(shfl(u4,lambda l:8*((l//8)&1)+(l&7)),shfl(u4,lambda l:8*(((l//8)&1)+2)+(l&7)))
 u1=M(shfl(u2,lambda l:l&7),shfl(u2,lambda l:8+(l&7)))
 assert len(set(fields(u1)))==1
 inv=pow(fields(u1)[0]%P,-1,P)*scale%P
 v2=M(vec([inv]*4),shfl(u2,lambda l:8*(((l//8)&1)^1)+(l&7)))
 v4=M(shfl(v2,lambda l:8*((l//8)&1)+(l&7)),shfl(u4,lambda l:8*((l//8)^(1 if bad else 2))+(l&7)))
 got=fields(v4)
 expect=[scale*pow(vals[g]*vals[g+4]%P,-1,P)%P for g in range(4)]
 if bad:return got!=expect
 assert [v%P for v in got]==expect
 return got

def warp_inverse(values,scale,warp):
 # Actual flattened physical addresses; plane pitches derive from declarations.
 ps=228;ivs=116;pb=56*warp;ib=28*warp
 prod=[None]*(4*ps);inv=[None]*(4*ivs)
 def put(kind,i,x):
  nonlocal prod,inv
  base,span,pitch,arr=(pb,56,ps,prod) if kind=='p' else (ib,28,ivs,inv)
  assert base<=i<base+span
  for k in range(4):
   j=k*pitch+i;arr[j]=(x>>(64*k))&((1<<64)-1);slots.add((kind,j));stats['shared_accesses']+=1
 def get(kind,i):
  base,span,pitch,arr=(pb,56,ps,prod) if kind=='p' else (ib,28,ivs,inv)
  assert base<=i<base+span
  assert all(arr[k*pitch+i] is not None for k in range(4));stats['shared_accesses']+=4
  return sum(arr[k*pitch+i]<<(64*k) for k in range(4))
 for l,v in enumerate(values):put('p',pb+l,v)
 offset=0
 for count in [32,16]:
  half=count//2
  for l in range(half):put('p',pb+offset+count+l,get('p',pb+offset+l)*get('p',pb+offset+half+l)%P)
  offset+=count
 got=upper([get('p',pb+48+g) for g in range(8)],scale)
 for g,v in enumerate(got):put('i',ib+24+g,v)
 offset=48
 for count in [8,16]:
  half=count//2
  for l in range(count):put('i',ib+offset-32+l,get('i',ib+offset+count-32+(l&(half-1)))*get('p',pb+offset+(l^half))%P)
  offset-=2*count
 result=[get('i',ib+(l&15))*get('p',pb+(l^16))%P for l in range(32)]
 assert result==[scale*pow(x,-1,P)%P for x in values]
 return result
ns['warp_inverse']=warp_inverse
counts=[1,31,32,33,63,64,65,95,96,97,127,128,129,255,256,257,511,512,513,767,768,769,1023,1024]
cases=roots=zeros=events=0
for n in counts:
 for schedule in ['serial','reverse','random']:
  raw=[R.choice([0,P,P-1,P+1,2**256-1,R.randrange(2**256)]) for _ in range(n)]
  events+=ns['run'](raw,R.randrange(1,P),R.randrange(P),schedule);cases+=1;roots+=n;zeros+=sum(x%P==0 for x in raw)
for raw in [[0]*1024,[P]*1024,[0],[P],[1]*1024]:
 events+=ns['run'](raw,17,19,'random');cases+=1;roots+=len(raw);zeros+=sum(x%P==0 for x in raw)
for _ in range(128):stats['negative_bad_route']+=upper([R.randrange(1,P) for _ in range(8)],R.randrange(1,P),bad=True)
assert stats['negative_bad_route']==128
h=(D.parent/'RegisterRoots.cuh').read_text();old=(prior/'WarpIndependentRootsResearch.cuh').read_text()
assert h[h.index('// Launch exactly'):].replace('qsb_root_register','qsb_root_independent_research').replace('qsb_block_inverse_register_n','qsb_block_inverse_independent_n')==old[old.index('// Launch exactly'):]
for literal in ['products[4][4*56+4]','inverses[4][4*28+4]','pb=warp*56u,ib=warp*28u','8*((group&1u)^1u)+d','8*(group^2u)+d','pb+48+group','pb+52+group','ib+24+group']:
 assert literal in h,literal
assert '__syncthreads' not in h and h.count('__syncwarp(0xffffffffu)')==4
assert h.count('qsb_prefix_cyclic_research::multiply8')==5
assert 'for(unsigned count=32;count>8;count>>=1)' in h
assert 'for(unsigned count=8;count<32;count<<=1)' in h
result=dict(status='PASS_PYTHON_REGISTER_ROUTING_COMPACT_LAYOUT_INTERLEAVING',batches=cases,roots=roots,zero_or_p_roots=zeros,interleaving_events=events,shared_words_used=len(slots),shared_bytes=11008,warp_sync_rounds=6,CTA_barriers=0,extra_upper_words_retained_across_inverse=2,stats=stats,native_compile=False,GPU_execution=False,included_in_candidate=True,sha256=hashlib.sha256(h.encode()).hexdigest())
(D/'register-tree-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
