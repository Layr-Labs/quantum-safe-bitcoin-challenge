"""An exact guarded raw identity, not unconditional modular reassociation."""
from pathlib import Path
import json,random,collections,re
r=Path(__file__).resolve().parent;rng=random.Random(2300042)

def funcs(bits,lowbits,k):
    B=1<<bits;L=1<<lowbits;T=1<<(lowbits//2);mask=B-1
    def sub(u,v):
        z=(u-v)&mask
        return (z//L)*L+((z%L-(k if u<v else 0))%L)
    def lazy(u,v):
        z=(u+v)&mask
        return (z//L)*L+((z%L+(k if u+v>=B else 0))%L)
    def cert(u,v,d):
        uh=(u//T)%T;vh=(v//T)%T
        dh=(uh-vh)%T;sh=(uh+vh)%T;qh=(2*uh)%T
        top=d>>(bits-lowbits//2)
        return dh>=3 and sh<T-3 and (qh-2)%T<T-7 and (top-1)%T<T-2
    def broad_bounds(u,v,d):
        x=(u-v)%L;y=(u+v)%L;q=(2*u)%L
        return x>=k and y<L-k and k<=q<L-2*k and k<=d<B-k
    return B,L,T,sub,lazy,cert,broad_bounds

stats=collections.Counter();controls={}
# Exhaustive low words across eight high-word configurations. T=16, K=17.
B,L,T,sub,lazy,cert,bounds=funcs(12,8,17)
for hu,hv in [(0,1),(1,0),(7,8),(8,7),(14,1),(1,14),(15,15),(4,4)]:
 for ul in range(L):
  for vl in range(L):
    u=hu*L+ul;v=hv*L+vl;d=lazy(u,u)
    old=lazy(sub(u,v),lazy(u,v));ok=cert(u,v,d)
    if ok:
        assert bounds(u,v,d) and old==d
        stats['toy_accepted']+=1
    elif old!=d and len(controls)<8:
        controls[str(len(controls))]={'u':u,'v':v,'old':old,'double':d}
    stats['toy_pairs']+=1

B,L,T,sub,lazy,cert,bounds=funcs(256,64,(1<<32)+977)
k=T+977;P=B-k;wm=L-1
# Independent adversarial construction of low words at correction boundaries.
low=[0,1,976,977,k-1,k,k+1,2*k, L-2*k-1,L-2*k,L-k-1,L-k,L-1,
     (L-k)//2,(L-k)//2+1,1<<31,1<<32,1<<63]
high=[0,1,1<<95,1<<191,(1<<192)-2,(1<<192)-1]
cases=[((hu<<64)|ul,(hv<<64)|vl) for hu,hv in [(0,1),(1,0),(high[-1],high[-1]),(high[3],high[3]),(high[2],high[-2])]
       for ul in low for vl in low]
for _ in range(30000):cases.append((rng.randrange(B),rng.randrange(B)))
for _ in range(4096):
    u=rng.randrange(B);ul=u&wm;vh=rng.randrange(1<<192)
    for target in [k-1,k,k+1,L-k-1,L-k,L-1]:
        cases.append((u,(vh<<64)|((ul-target)&wm)))
        cases.append((u,(vh<<64)|((target-ul)&wm)))
for u,v in cases:
    d=lazy(u,u);old=lazy(sub(u,v),lazy(u,v));ok=cert(u,v,d)
    if ok:
        assert bounds(u,v,d)
        assert old==d
        stats['fullwidth_accepted']+=1
    else:stats['fullwidth_replays']+=1
    stats['fullwidth_pairs']+=1
assert lazy(sub(0,1),lazy(0,1))==P and lazy(0,0)==0
assert not cert(0,1,0)

# Source formula uses wrapping u32 comparisons; compare it independently.
s=(r/'GuardedSum2u.cuh').read_text()
for text in ['dh>=3u','sh<0xfffffffdu','(qh-2u)<0xfffffff9u','(top-1u)<0xfffffffeu']:
    assert text in s
for u,v in cases[-4096:]:
    d=lazy(u,u);uh=(u>>32)&(T-1);vh=(v>>32)&(T-1);top=d>>224
    source=((uh-vh)&(T-1))>=3 and ((uh+vh)&(T-1))<0xfffffffd and \
           (((uh<<1)-2)&(T-1))<0xfffffff9 and ((top-1)&(T-1))<0xfffffffe
    assert source==cert(u,v,d)
result={'passed':True,**stats,'toy_high_word_configurations':8,
        'unguarded_negative_control':{'u':0,'v':1,'old_sum':hex(P),'double':'0x0'},
        'toy_rejected_mismatch_examples':controls,
        'native_compile':False,'GPU_execution':False,
        'claim':'Under the explicit certificate, lazy(sub(u,v),lazy(u,v)) equals lazy(u,u) as a raw 256-bit value, for the active promoted low-word correction contract.',
        'limitations':'Directed test acceptance rates are not workload probabilities. A separate interval argument supplies the universal certificate reasoning.'}
print(json.dumps({k:v for k,v in result.items() if k!='toy_rejected_mismatch_examples'},indent=2))
