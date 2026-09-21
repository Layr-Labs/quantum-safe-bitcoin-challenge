from pathlib import Path
import sys,random,json
W=Path(__file__).resolve().parent
import reference as h
B=2**32;MASK=B-1;rng=random.Random(202609210804)
cs=[h.B2,h.B1];gs=[(b<<384)//h.N for b in cs]
def words(x):return [(x>>(32*i))&MASK for i in range(8)]
def partial(k,g):
 a,c=words(k),words(g)
 acc=sum((a[i]*c[10-i])>>32 for i in range(3,8));out=[]
 for s in range(11,15):
  high=0
  for i in range(s-7,8):
   v=a[i]*c[s-i];acc+=v&MASK;high+=v>>32
  assert acc<2**36 and high<2**35
  out.append(acc&MASK);acc=(acc>>32)+high
 assert acc<B;out.append(acc)
 return sum(v<<(32*i) for i,v in enumerate(out))
# Universal omitted-tail bound for all input words; no probability enters the certificate.
# Diagonal10 contributes five omitted fractions <1 each; lower diagonals contribute bounded full products.
from fractions import Fraction
bound=Fraction(5*(B-1),B)
for s in range(10):
 count=sum(i+j==s for i in range(8) for j in range(8))
 bound+=Fraction(count*(B-1)**2,B**(11-s))
assert bound<12
assert Fraction(h.N-1,2**352)<1
stats={'fast':0,'fallback':0,'negative_wrong_without_fallback':0};maxmissing=0;randomfallback=0

def certified(k,b,g,random_case=False):
 global maxmissing,randomfallback
 k%=h.N;H=partial(k,g);P=k*g
 assert H<=Fraction(P,2**352)<H+12
 # The true scaled quotient differs by k*(b/N-g/2^384)*2^32 <1.
 assert H<=Fraction(k*b*B,h.N)<H+13
 maxmissing=max(maxmissing,(P>>352)-H)
 low=(H+2**31)>>32;up=(H+13+2**31)>>32
 ref=(k*b+h.N//2)//h.N
 assert low<=ref<=up
 ambiguous=((H+2**31)&MASK)>=B-13
 assert ambiguous==(low!=up)
 if ambiguous:
  stats['fallback']+=1;randomfallback+=int(random_case);out=ref #existing full exact function
  if low!=ref:stats['negative_wrong_without_fallback']+=1
 else:stats['fast']+=1;out=low
 assert out==ref
 return out
ks=[0,1,2,h.N-1,h.N,h.N+1,2**256-1]
ks += [min(2**256-1,2**i+delta) for i in range(256) for delta in [-1,0,1] if 2**i+delta>=0]
for b in cs:
 inv=pow(b,-1,h.N)
 for delta in range(-128,129):ks.append(((h.N//2+delta)*inv)%h.N)
for k in ks:
 for b,g in zip(cs,gs):certified(k,b,g)
for _ in range(100000):
 k=rng.randrange(2**256)
 for b,g in zip(cs,gs):certified(k,b,g,True)
assert stats['fallback']>0 and stats['negative_wrong_without_fallback']>0
# Full corrected-coordinate equivalence, with the interval coefficients replacing both old divisions.
full=0
for k in ks+[rng.randrange(2**256) for _ in range(10000)]:
 kk=k%h.N;c1=certified(kk,cs[0],gs[0]);c2=certified(kk,cs[1],gs[1])
 x=kk-c1*h.A1-c2*h.A2;y=c1*h.B1-c2*h.B2
 if x>h.R:
  if y>=h.T:x-=h.A2;y-=h.B2
  else:x-=h.A1;y+=h.B1
 elif x<-h.R:
  if y<=-h.T:x+=h.A2;y+=h.B2
  else:x+=h.A1;y-=h.B1
 assert (x,y)==h.split_corr(k)[:2];full+=1
r={'status':'PASS','coefficient_checks':sum(stats[x] for x in ['fast','fallback']),'adversarial_scalar_inputs':len(ks),'random_scalar_inputs':100000,'full_split_checks':full,**stats,'random_fallback_coefficient_count':randomfallback,'universal_missing_tail_bound':str(bound),'missing_tail_integer_bound':12,'true_quotient_scaled_upper_addend':13,'max_observed_missing_integer_units':maxmissing,'fast_products_per_coefficient':15,'previous_source_products_per_coefficient':57,'fast_split_source_product_budget':86,'previous_split_source_product_budget':170,'reciprocals':[hex(g) for g in gs],'scope':'Exact interval certificate plus mandatory complete fallback. Python arithmetic proof/model only, no GPU execution, no frozen candidate edits.'}
(W/'interval-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
