from pathlib import Path
import random,json,hashlib
w=Path(__file__).resolve().parent;B=1<<256;MASK=B-1;K=(1<<32)+977;P=B-K;M64=(1<<64)-1
u32=lambda x:x&((1<<32)-1);u64=lambda x:x&M64
ns={'u32':u32,'u64':u64,'__umulhi':lambda a,b:(a*b)>>32}
exec((w/'header_semantics.py').read_text(),ns);fastfn=ns['model']
def limbs(v):return [u64(v>>(64*i)) for i in range(4)]
def raw(a,b):
 t=a*b;f=(t&MASK)+K*(t>>256);r=f&MASK;h=(f>>256)&((1<<32)-1)
 return (r>>96<<96)|((r+K*h)&((1<<96)-1))
def sub(a,b):
 t=(a-b)&MASK
 return (t>>64<<64)|((t-(K if a<b else 0))&M64)
def lazy(a,b):
 t=a+b;r=t&MASK
 return (r>>64<<64)|((r+(K if t>>256 else 0))&M64)
def add(a,b):
 t=a+b
 return (t-P if t>=P else t)&MASK

def sum_parity(v,b,n):
 # Literal promoted qsb_sum_parity behavior, b!=0 required by caller.
 t=v+b;s=t&MASK;c=t>>256;k=c;zero=False
 if s>>64==(1<<192)-1:
  lim=(B-2*K if c else P)&M64
  if (s&M64)>=lim:k+=1;zero=(s&M64)==lim
 return 0 if zero else ((v^b^k)&1)^n

paths={'fast':0,'fallback':0};traces=0;warp_fallback=0
rng=random.Random(1411860)
def compare(vbar,tbar,ri,wi,a,b,c):
 global traces
 assert 0<b<P
 u=raw(tbar,wi);v=raw(vbar,ri);l=sub(u,v);m=lazy(u,v);total=lazy(l,m)
 s1=raw(total,sub(l,c))%P;x1=add(s1,a)
 old0=sum_parity(raw(l,s1),b,1)
 s2=raw(total,sub(m,c))%P;x2=add(s2,a)
 old1=sum_parity(raw(m,s2),b,0)
 vals=[]
 for lhs,rhs,neg in [(l,s1,1),(m,s2,0)]:
  hit,bit=fastfn(limbs(lhs),limbs(rhs),limbs(b),neg)
  paths['fast' if hit else 'fallback']+=1
  vals.append(bit if hit else sum_parity(raw(lhs,rhs),b,neg))
 assert (old0|(old1<<1))==(vals[0]|(vals[1]<<1))
 # Coordinates and all retained ordered raw primitive operands are untouched by source reversal.
 traces+=1
for warp in range(512):
 before=paths['fallback']
 a,b,c=[rng.randrange(1,P) for _ in range(3)]
 for lane in range(32):compare(*[rng.randrange(B) for _ in range(4)],a,b,c)
 warp_fallback+=paths['fallback']!=before
edges=[0,1,K-1,K,(1<<64)-1,(1<<224)-1,P-1,P,B-1]
for v in edges:
 for t in edges:
  for b in [1,2,K,P-1]:compare(v,t,1,1,P-1,b,0)
# Common masked/dead-candidate inputs.
for _ in range(1024):compare(0,0,rng.randrange(B),rng.randrange(B),rng.randrange(P),rng.randrange(1,P),rng.randrange(P))
# Endpoint boundary corpus, including zero sum mod p.
endpoint=0
for a in edges:
 for b in edges:
  v=raw(a,b)
  for off in [1,P-1,(-v)%P,(-v+1)%P,(-v-1)%P]:
   if off==0:continue
   for n in [0,1]:
    hit,bit=fastfn(limbs(a),limbs(b),limbs(off),n)
    expected=sum_parity(v,off,n)
    assert (bit if hit else sum_parity(v,off,n))==expected
    assert expected==((-(v+off)%P if n else (v+off)%P)&1)
    endpoint+=1
r={'status':'PASS','finish_cases':traces,'finish_parity_paths':paths,'extra_endpoint_boundary_cases':endpoint,'uniform_synthetic_finish_warps':512,'uniform_synthetic_warps_with_fallback':warp_fallback,'scope':'Promoted C31 raw/lazy/sub semantic model and literal qsb_sum_parity; valid nonzero canonical offset. Synthetic saved values, not actual device nomination distribution; no timing or native compilation.'}
(w/'finish-model-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
