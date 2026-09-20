from pathlib import Path
import random,json
N=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

def recode(k):
 k%=N;D=2*k-N;s=1 if D>=0 else -1;m=abs(D);off=0;terms=[]
 for w in [18]+[17]*13:
  d=s*((m% (1<<(w+1)))-(1<<w));terms.append(d<<off);m=2*(m>>(w+1))+1;off+=w
 terms.append((s*m)<<off);assert sum(terms)==D;return terms
r=random.Random(910);cases=set([0,1,2,3,N-1,N-2,N//2,1<<128])
for bits in [18,35,69,103,137,171,205,239]:
 for t in range(-3,4):
  cases.add((N%(1<<bits)+t)%N);cases.add((N-N%(1<<bits)+t)%N)
B=1<<256
critical=(N%(1<<36))-(1<<35)
assert 0<critical<(1<<35)
for k in [N,N+1,N+critical,N-critical,B-1,B-2]:cases.add(k)
cases.update(r.randrange(B) for _ in range(20000))
found=[];pairs=0
for k in sorted(cases):
 terms=recode(k)
 for j in range(7):
  a,b=terms[2*j:2*j+2]
  assert (a-b)%N!=0 and (a+b)%N!=0
  # Each coefficient has distinct exact2-adic valuation and |a±b|<N.
  assert abs(a-b)<N and abs(a+b)<N
  pairs+=1
 current=terms[14]%N
 for j in range(6,-1,-1):
  q=(terms[2*j]+terms[2*j+1])%N
  if current==q or (current+q)%N==0:
   found.append({'k':hex(k),'pair_index':j,'kind':'double' if current==q else 'opposite'})
  current=(current+q)%N
 assert current==2*k%N
 for item in found:
  assert int(item['k'],16)%N in [0,critical,N-critical]
res={'status':'PASS','scalar_cases':len(cases),'pair_checks':pairs,'pair_nonzero_proof':'Pair(2j,2j+1) has unequal2-adic valuations. For j<=6, |a±b|<(2^17+1)*2^222<2^240<N (first pair separately<2^36). Thus a!=±b mod N. Since base A/2 is nonzero and curve has prime orderN, paired points have unequal x and nonzero affine denominator. This relies on valid nonzero fixed base, not random probability.',
 'chain_exceptions':found,'critical_scalar':hex(critical),'raw_domain':'0<=hash<2^256, reduced once moduloN as production setup does','decision':'Pair affine layer needs no probabilistic zero skip under proved regular-digit domain. Reordered eight-point chain DOES have exceptional additions, confined by coefficient valuation and magnitude bounds to the last pair. The critical scalar residues are0 and plus/minus0x4d0364141. Route these residues through a correct exceptional/reference path; do not silently skip them or assume random inputs exclude them.',
 'scope':'Integer group-coefficient proof/model; does not validate CUDA field code.'}
Path(__file__).with_name('pair-bounds-result.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
