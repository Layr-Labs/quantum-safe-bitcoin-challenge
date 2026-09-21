from pathlib import Path
from fractions import Fraction
import random,json,runpy,re
W=Path(__file__).resolve().parent;root=W.parents[1]
B=1<<256;K=(1<<32)+977;P=B-K;W32=1<<32;S=1<<224;MASK=W32-1;rng=random.Random(202609211213)
# Source contract: dropped odd highcarry affectsbits>=288, truncatedsecondfoldonlylow96.
def raw(a,b):
 t=a*b;f=(t&(B-1))+K*(t>>256);r=f&(B-1);h=(f>>256)&MASK
 return (r>>96<<96)|((r+K*h)&((1<<96)-1))
# Partial floor(T / 2^224), 36completeproducts +7highhalves. Exact lowerbound.
def head(a,b):
 x=[a>>(32*i)&MASK for i in range(8)];y=[b>>(32*i)&MASK for i in range(8)]
 return sum((x[i]*y[j])<<(32*(i+j-7)) for i in range(8) for j in range(8) if i+j>=7)+sum((x[i]*y[6-i])>>32 for i in range(7))
bound=Fraction(7*(W32-1),W32)
for d in range(6):bound+=Fraction((d+1)*(W32-1)**2,W32**(7-d))
assert bound<13
stats={};fast=0;fallback=0;reasons={};negative_without_guard=0

def parity(a,b,offset,neg):
 global fast,fallback
 H=head(a,b);assert H<=Fraction(a*b,S)<H+13
 def full(reason):
  global fallback
  fallback+=1;reasons[reason]=reasons.get(reason,0)+1
  y=(raw(a,b)+offset)%P;return ((-y)%P if neg else y)&1
 if H>>32 != (H+12)>>32:return full('product_high_carry')
 hi=H>>32;assert hi==a*b>>256
 f=(H&MASK)+(K*hi>>224)
 if f>>32 != (f+13)>>32:return full('fold_carry')
 h=f>>32;actualF=((a*b)&(B-1))+K*hi;assert h==actualF>>256
 low=(f&MASK)*S;high=((f&MASK)+14)*S-1
 assert 0<=low<=raw(a,b)<=high<B
 qlo=(low+offset)//P;qhi=(high+offset)//P
 if qlo!=qhi or low+offset<=qlo*P:return full('parity_or_zero_boundary')
 bit=((a&1)*(b&1))^(hi&1)^(h&1)^(offset&1)^(qlo&1)^neg
 expect=(raw(a,b)+offset)%P;expect=(((-expect)%P) if neg else expect)&1
 assert bit==expect
 fast+=1;return bit
cases=[]
edges=[0,1,2,K-1,K,K+1,(1<<96)-1,(1<<224)-1,P-2,P-1,P,B-2,B-1]
for a in edges:
 for b in edges:
  w=raw(a,b)
  for c in [0,1,P-1,(-w)%P,(-w+1)%P,(-w-1)%P]:
   for n in (0,1):cases.append((a,b,c,n))
cases += [(rng.randrange(B),rng.randrange(B),rng.randrange(1,P),rng.randrange(2)) for _ in range(20000)]
for a,b,c,n in cases:parity(a,b,c,n)
# Bind rawcontract toactualcurrentpromotedPTX onboundaries/randoms usingintegersemantics.
q=runpy.run_path(str(W/'ptx_field_model.py'))
src=(W/'baseline/GPUMath.h').read_text();f=q['function'](src,'void _ModMultCore(');s=q['extract_ptx'](f.replace('QSB_SECOND_FOLD_TAIL','""'));s=re.sub(r'/\*.*?\*/','',s,flags=re.S);s=re.sub(r'\{\s*(?=\.reg)','',s[1:-1]);s=re.sub(r'(?<=;)\s*\}','',s);prog=q['Program']('{'+s+'}')
checks=0
for a,b in [(a,b) for a in edges for b in edges]+[(rng.randrange(B),rng.randrange(B)) for _ in range(256)]:
 vals=[v>>(64*i)&((1<<64)-1) for v in (a,b) for i in range(4)];actual=sum(v<<(64*i) for i,v in enumerate(prog.run(vals)));assert actual==raw(a,b),(a,b,actual,raw(a,b));checks+=1
r={'status':'PASS','parity_cases':len(cases),'fast':fast,'fallback':fallback,'fallback_reasons':reasons,'actual_promoted_PTX_contract_cases':checks,'partial_product_terms':43,'full_product_terms':64,'tail_bound':str(bound),'tail_integer_strict_bound':13,'scope':'Exact certificate preserves promotedrawproductparity,completefallbackonallambiguousboundaries;Pythonprototypeonly,nativecode/performanceunmeasured.'}
(W/'parity-window-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
