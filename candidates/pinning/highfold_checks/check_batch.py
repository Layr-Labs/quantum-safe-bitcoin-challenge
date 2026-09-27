from pathlib import Path
import random,json
D=Path(__file__).resolve().parent
p=(1<<256)-(1<<32)-977
G=(55066263022277343669578718895168534326250603453777594175500187360389116729240,32670510020758816978083085130507043184471273380659243275938904335757337482424)
def add(a,b):
 if a is None:return b
 if b is None:return a
 x,y=a;u,v=b
 if x==u and (y!=v or y==0):return None
 lam=((3*x*x)*pow(2*y,-1,p) if x==u else (v-y)*pow(u-x,-1,p))%p
 z=(lam*lam-x-u)%p;return z,(lam*(x-z)-y)%p
def mul(k):
 r=None;a=G
 while k:
  if k&1:r=add(r,a)
  a=add(a,a);k>>=1
 return r
def batch(a,b):
 ns=[];ds=[];pre=[]
 for A,B in zip(a,b):
  x,y=A;u,v=B;den=(u-x)%p;num=(v-y)%p
  if den==0:
   if num!=0:return None
   den=2*y%p;num=3*x*x%p
   if den==0:return None
  ns.append(num);ds.append(den);pre.append(den*(pre[-1] if pre else 1)%p)
 u=pow(pre[-1],-1,p);out=[None]*len(a)
 for i in range(len(a)-1,-1,-1):
  ik=u*(pre[i-1] if i else 1)%p
  if i:u=u*ds[i]%p
  lam=ns[i]*ik%p;x,y=a[i];bx=b[i][0];z=(lam*lam-x-bx)%p
  out[i]=(z,(lam*(x-z)-y)%p)
 return out
# Source kg doubling tree, 4096 points, oracle recurrence with independent inverses.
kg=[G]
while len(kg)<4096:
 s=len(kg);n=batch(kg,[kg[s-1]]*s);assert n is not None;kg+=n
ref=[G]
for _ in range(4095):ref.append(add(ref[-1],G))
assert kg==ref
A=mul(8193);row=[A]+batch([A]*4095,kg[:4095])
r=A
for a in row:assert a==r;r=add(r,G)
# Bounded one-record tail must NOT evaluate extra points that may be opposite.
assert batch([G],[G])==[mul(2)]
assert batch([G],[(G[0],(-G[1])%p)]) is None
assert batch([(0,0)],[(0,0)]) is None # synthetic y=0 branch, not asserted on-curve
# Exact formula batch5n-3M+nS +15M+255S inversion (sourcechain count).
chain_sq=sum([1,1,3,3,2,11,22,44,88,44,3,23,5,3,2]);assert chain_sq==255
chain_mul=15
cost={}
for n in [256,4096]:cost[n]={'M_per_point':5+(chain_mul-3)/n,'S_per_point':1+chain_sq/n,'scratch_bytes_per_builder':9*n*32,'kg_bytes_10_windows':10*2*n*32}
ratio={str(s):4.25*(cost[4096]['M_per_point']+s*cost[4096]['S_per_point'])/(cost[256]['M_per_point']+s*cost[256]['S_per_point']) for s in [0.5,0.8,1]}
result={'kg_points':4096,'afold_row_points':4096,'exception_cases':3,'cost':cost,'highfold_vs_baseline_build_arithmetic_ratio':ratio,'scope':'independent Python curve+batch model; not native table construction, allocation or measured timing'}
(D/'batch-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
