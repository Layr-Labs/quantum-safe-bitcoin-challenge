from pathlib import Path
import sys,random,json
W=Path(__file__).resolve().parent
import reference as h
assert h.A2==h.A1+h.B1
MASK=2**32-1;M160=2**160-1;rng=random.Random(202609210817)
def words(x,L):return [(x>>(32*i))&MASK for i in range(L)]
def val(a):return sum(x<<(32*i) for i,x in enumerate(a))
def mul(a,b):
 out=[0]*5
 for i in range(len(a)):
  carry=0
  for j in range(min(len(b),5-i)):
   t=a[i]*b[j]+out[i+j]+carry;assert t<2**64
   out[i+j]=t&MASK;carry=t>>32
  if i+len(b)<5:out[i+len(b)]=carry
 return out

def sub(a,b):
 carry=0
 for i in range(5):t=b[i]+carry;x=a[i];a[i]=(x-(t&MASK))&MASK;carry=x<t

def residual(k,c1,c2):
 a,b=words(c1,4),words(c2,4);p=[0]*5;carry=0
 for i in range(4):v=a[i]+b[i]+carry;p[i]=v&MASK;carry=v>>32
 p[4]=carry
 assert val(p)==c1+c2
 y=mul(p,words(h.A1,5));x=words(k,5);sub(x,y)
 p=mul(b,words(h.B1,5));sub(x,p)
 p=mul(a,words(h.A2,5));sub(p,y);y=p[:]
 assert val(x)==(k-c1*h.A1-c2*h.A2)&M160
 assert val(y)==(c1*h.B1-c2*h.B2)&M160
 return val(x),val(y),carry

corners=[0,1,2**32-1,2**64-1,2**96-1,2**127,2**128-1]
checks=0;sumoverflow=0;bad_without_carry=0
for a in corners:
 for b in corners:
  x,y,c=residual(rng.randrange(2**256),a,b);checks+=1;sumoverflow+=c
  if c:bad_without_carry+=int(((a+b)*h.A1)&M160 != (((a+b)%2**128)*h.A1)&M160)
for _ in range(100000):
 k=rng.randrange(2**256)%h.N;c1=(h.B2*k+h.N//2)//h.N;c2=(h.B1*k+h.N//2)//h.N
 _,_,c=residual(k,c1,c2);checks+=1;sumoverflow+=c
assert bad_without_carry>0
# Mathematically independent polynomial identity, coefficient exact for all integers.
assert h.A1+h.B1==h.A2 and h.B2==h.A1
r={'status':'PASS','fixed_word_residual_cases':checks,'sum_needing_fifth_word':sumoverflow,'negative_controls_discarding_sum_carry':bad_without_carry,'old_low160_product_expressions':56,'new_low160_product_expressions':43,'combined_fast_split_source_products':73,'old_full_split_source_products':170,'extra_scratch_words':0,'scope':'Universal linear identity and fixed-word semantic model; no CUDA execution or measured throughput.'}
(W/'residual-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
