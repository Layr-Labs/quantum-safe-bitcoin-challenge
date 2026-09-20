"""Integrated source-bound audit; no native compile or GPU timing."""
from pathlib import Path
import re,random,json,hashlib,math
from ptx_field_model import Program,function,extract_ptx,check_semantics
W=Path(__file__).resolve().parent;C=W.parent;D=W/'baseline'
B=1<<256;K=(1<<32)+977;P=B-K;mask64=B64=(1<<64)-1;mask32=(1<<32)-1
NORDER=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
rng=random.Random(202609202250);check_semantics()
pin=(C/'pinning.cu').read_text();old=(D/'pinning.cu').read_text()
# Only source functions of the decoder and its seed consumers may differ.
a=pin;b=old
for sig,new_sig in [('void qsb_decode_to_shared(','uint2 qsb_decode_to_shared('),('void _FixedBaseSignedXYZZScalar(','void _FixedBaseSignedXYZZScalar(')]:
 a=a.replace(function(a,new_sig),'FUNCTION');b=b.replace(function(b,sig),'FUNCTION')
a=a.replace('__device__ __forceinline__ '+function(a,'void qsb_load_seed_code(')+'\n\n','')
assert a==b,'non-decoder change to pinning.cu'
assert '#define QSB_C31 1' in pin and '#define QSB_HOST_GATE 1' in pin
# Derive the actual production PTX. The C31 second-fold-tail macro expands to an empty string.
f=function((C/'GPUMath.h').read_text(),'void _ModMultCore(')
assert 'QSB_SECOND_FOLD_TAIL' in f
s=extract_ptx(f.replace('QSB_SECOND_FOLD_TAIL','""'))
s=re.sub(r'\{\s*(?=\.reg)','',s[1:-1]);s=re.sub(r'(?<=;)\s*\}', '',s)
s=re.sub(r'/\*.*?\*/','',s,flags=re.S)
prg=Program('{'+s+'}')
assert not any(op=='addc.u32' and args[0] in ['g8','z9'] for op,args in prg.ops)
assert not any(args[0] in ['z3','z4'] and op.startswith('addc') for op,args in prg.ops[-6:])
limbs=lambda a:[a>>(64*i)&mask64 for i in range(4)]
pm=lambda a,b:sum(v<<(64*i) for i,v in enumerate(prg.run(limbs(a)+limbs(b))))
def sm(a,b):
 f=a*b;f=(f&(B-1))+K*(f>>256);r=f&(B-1);h=(f>>256)&mask32
 return (r&~((1<<96)-1))|((r+h*K)&((1<<96)-1))
edges=[0,1,K-1,K,K+1,P-1,P,P+1,B-1,(1<<96)-1,(1<<160)-1]
rows=[(a,b) for a in edges for b in edges]+[(rng.getrandbits(256),rng.getrandbits(256)) for _ in range(1800)]
rows += [(B-1-rng.randrange(978<<224),B-1-rng.randrange(978<<224)) for _ in range(300)]
for x,y in rows:assert pm(x,y)==pm(y,x)==sm(x,y)
def baseline(x,m):
 pair=[m(x[i],x[i+8]) for i in range(8)]
 quad=[m(pair[i],pair[i+4]) for i in range(4)]
 octet=[m(quad[i],quad[i+2]) for i in range(2)]
 root=m(octet[0],octet[1])
 e4=[m(octet[(i&1)^1],quad[i^2]) for i in range(4)]
 e8=[m(e4[i&3],pair[i^4]) for i in range(8)]
 return root,[m(e8[i&7],x[i^8]) for i in range(16)]

def tree(vals,top,m):
 N=len(vals);pr=dict(enumerate(vals));ex={};off=0;count=N
 while count>16:
  half=count//2
  pr.update({off+count+t:m(pr[off+t],pr[off+half+t]) for t in range(half)})
  off+=count;count//=2
 root,outs=top([pr[2*N-32+i] for i in range(16)],m)
 ex.update({N-32+i:outs[i] for i in range(16)})
 off=2*N-64;count=32
 while count<N:
  half=count//2
  ex.update({off-N+t:m(ex[off+count-N+(t&(half-1))],pr[off+(t^half)]) for t in range(count)})
  off-=2*count;count*=2
 return root,[m(ex[t&(N//2-1)],pr[t^(N//2)]) for t in range(N)]
def duplicated(x,m):
 node=x+x;counts=[]
 for wave in range(6):
  old=node[:];cnt=0
  for tid in range(32):
   if wave<3:
    start=[16,24,28][wave];bit=[8,4,2][wave];work=tid>=start
    src=tid^bit
    a=old[tid];b=old[src]
   elif wave==3:
    work=24<=tid<28 or tid==31
    src=30 if tid==31 else 28+(((tid-24)&1)^1)
    a=old[src];b=old[tid]
   elif wave==4:
    work=16<=tid<24;src=24+(((tid-16)&3)^2)
    a=old[src];b=old[tid]
   else:
    work=tid<16;src=16+((tid&7)^4)
    a=old[src];b=old[tid]
   assert 0<=src<32
   if work:node[tid]=m(a,b);cnt+=1
  counts.append(cnt)
 assert counts==[16,8,4,5,8,16]
 return node[31],[node[i^8] for i in range(16)]

def expr(a,b):return '('+'*'.join(sorted([a,b]))+')'
def oldpack(f,bits,last,negative):
 tm=-negative if last else (f>>(bits-1))-1
 return ((f^tm)&((1<<(bits-1))-1))|((tm<0)<<31)

def pack(f,bits,last,negative):
 if last:
  tm=-negative;sign=negative<<31
 else:
  z=(~((f<<(32-bits))&mask32))&mask32
  tm=-(z>>31);sign=z&0x80000000
 return ((f^tm)&((1<<(bits-1))-1))|sign

def decode(M,negative,new):
 limbs=[M>>(64*i)&mask64 for i in range(4)];words=[M>>(32*i)&mask32 for i in range(8)];out=[]
 for c in range(15):
  pos=1 if c==0 else 17*c+2;bits=18 if c==0 else 17
  if new:
   wi,ws=divmod(pos,32);hi=words[(wi+1)&7] if wi<7 else 0
   f=((words[wi]>>ws)|(hi<<(32-ws)))&((1<<bits)-1)
  else:
   j,sh=divmod(pos,64);value=limbs[j]>>sh
   if j<3 and sh>46:value|=(limbs[j+1]<<(64-sh))&mask64
   f=value&((1<<bits)-1)
  out.append((pack if new else oldpack)(f,bits,c==14,negative))
 return out

# Static runtime mapping binding for the register helper and surrounding tree.
s=(C/'Top16Registers.cuh').read_text();co=(C/'cofactor_checkpoint.h').read_text()
for t in ['work=tid>=start;src=tid^bit;','src=tid==31?30:28+(((tid-24)&1)^1);','src=24+(((tid-16)&3)^2);','src=16+((tid&7)^4);','a[k]=wave<3?node[k]:other;b[k]=wave<3?other:node[k];','excluded[k][N-32+(tid^8)]=node[k];','else if(tid==31)','qsb_field_mul_sc(out,a,b);']:
 assert t in s,t
assert s.index('__shfl_sync')<s.index('if(work){') and s.count('__syncwarp();')==1
for t in ['count>16','qsb_top16_duplicate<N>(roots,products,excluded);','offset=2*N-64','count=32;count<N','if((count<<1)>32)__syncthreads();else __syncwarp();']:assert t in co
assert '#include "Top16Registers.cuh"' in co
assert 'qsb_top16_mul' not in s and 'normalize' not in s
assert pin.index('void qsb_field_mul_sc(')<pin.index('#include "cofactor_checkpoint.h"')
assert '__syncthreads();' in function((C/'PackedRecovery.cuh').read_text(),'void qsb_packed_prepare(')
# Ordered baseline topology is equivalent modulo ONLY commutative swaps.
assert duplicated([str(i) for i in range(16)],expr)==baseline([str(i) for i in range(16)],expr)
# Explicit baseline entire tree, including merged top2 and continuation offsets.
def original(vals,m):
 N=len(vals);pr=dict(enumerate(vals));ex={};off=0;count=N
 while count>2:
  half=count//2
  pr.update({off+count+t:m(pr[off+t],pr[off+half+t]) for t in range(half)})
  off+=count;count//=2
 root=m(pr[2*N-4],pr[2*N-3])
 ex.update({N-8+t:m(pr[2*N-4+((t&1)^1)],pr[2*N-8+(t^2)]) for t in range(4)})
 off=2*N-16;count=8
 while count<N:
  half=count//2
  ex.update({off-N+t:m(ex[off+count-N+(t&(half-1))],pr[off+(t^half)]) for t in range(count)})
  off-=2*count;count*=2
 return root,[m(ex[t&(N//2-1)],pr[t^(N//2)]) for t in range(N)]
leafchecks=0
for size in [32,64,128,256]:
 for active in [0,1,17,size//2,size-1,size]:
  for _ in range(16):
   vals=[rng.getrandbits(256) if i<active else 1 for i in range(size)]
   assert tree(vals,duplicated,sm)==original(vals,sm)
   leafchecks+=size
ptxcases=[]
for size in [128,256]:
 ptxcases += [[1]*size,[P-1]*size,[B-1]*size]
 ptxcases += [[rng.getrandbits(256) if i<active else 1 for i in range(size)] for active in [17,size-1]]
for vals in ptxcases:assert tree(vals,duplicated,pm)==original(vals,pm)==original(vals,sm)
# Source-bound decoder expressions and exact returned seed codes.
d=function(pin,'uint2 qsb_decode_to_shared(')
for t in ['pos=c==0?1u:17u*c+2u','wi=pos>>5,ws=pos&31u','bits=c==0?18u:17u','wi<7u?words[(wi+1u)&7u]:0u','__funnelshift_r(words[wi],hi,ws)&((1u<<bits)-1u)','((int32_t)(~ftop)>>31)','((~ftop)&0x80000000u)','if(c==0)seeds.x=code','else if(c==1)seeds.y=code','else codes[(size_t)c*QSB_TREE_N+threadIdx.x]=code']:
 assert t in d,t
fs=function(pin,'void _FixedBaseSignedXYZZScalar(')
assert 'qsb_load_seed_code(table,gt_offset(0),seeds.x,x0,y0);' in fs
assert 'qsb_load_seed_code(table,gt_offset(1),seeds.y,x1,y1);' in fs
assert 'qsb_load_decoded(table,c,base,x1,y1);' in fs
assert '((uint64_t)m32<<32)|m32' in function(pin,'void qsb_load_seed_code(')
fieldchecks=0
for bits,last in [(18,False),(17,False),(17,True)]:
 for f in range(1<<(16 if last else bits)):
  for neg in (range(2) if last else [0]):
   assert oldpack(f,bits,last,neg)==pack(f,bits,last,neg);fieldchecks+=1
scalars=[0,1,NORDER//2,NORDER//2+1,NORDER-1,NORDER,NORDER+1,B-1]+[rng.getrandbits(256) for _ in range(8000)]
for k in scalars:
 t=k%NORDER;D2=2*t-NORDER;M=D2%B;negative=int(D2<0)
 assert decode(M,negative,False)==decode(M,negative,True)
for M in [0,B-1]+[1<<i for i in range(256)]:
 for negative in [0,1]:assert decode(M,negative,False)==decode(M,negative,True)
# Negative control: the model must preserve new-base truncation semantics, not silently replace with exact field math.
assert any(sm(x,y)%P!=x*y%P for x,y in rows)
# Index mutant is caught by the nonassociative symbolic invariant.
x=[str(i) for i in range(16)];root,outs=duplicated(x,expr);assert outs!=outs[1:]+outs[:1]
r={'status':'PASS','current_frontier_PTX_symmetry_and_model_pairs':len(rows),'full_tree_bitwise_leaf_checks':leafchecks,'actual_full_PTX_trees':len(ptxcases),'actual_full_PTX_leaves':sum(map(len,ptxcases)),'decoder_exhaustive_field_checks':fieldchecks,'decoder_raw_scalars':len(scalars),'decoder_digit_comparisons':15*len(scalars),'basis_vectors_and_signs':516,'negative_controls':2,'new_carry_omissions':0,'native_compile':False,'GPU_timing':False,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in C.iterdir() if p.suffix in ['.cu','.cuh','.h']},'limitations':'Tests prove source-modeled equality to the latest promoted raw bit contract; inherited C31/truncations are not certified as exact field arithmetic. Host gate unchanged. Source/barrier audit is not a GPU race detector or compiler check.'}
(W/'audit-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
