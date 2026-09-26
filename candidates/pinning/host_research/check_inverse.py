from pathlib import Path
import random,json
P=2**256-2**32-977;U=2**64-1;M=2**62-1
# Integer reference of the copied 62-divstep transition; unsigned low-word arithmetic is wrapped.
def divstep(eta,f,g):
 u,v,q,r=1,0,0,1;i=62
 while True:
  z=g|((U<<i)&U);zeros=(z&-z).bit_length()-1
  g>>=zeros;u=u<<zeros&U;v=v<<zeros&U;eta-=zeros;i-=zeros
  if not i:break
  if eta<0:
   eta=-eta;f,g=g,(-f)&U;u,q=q,(-u)&U;v,r=r,(-v)&U
   limit=min(eta+1,i);m=((1<<limit)-1)&63;w=(f*g*(f*f-2))&m
  else:
   limit=min(eta+1,i);m=((1<<limit)-1)&15;w=(-(f+(((f+1)&4)<<1))*g)&m
  g=(g+f*w)&U;q=(q+u*w)&U;r=(r+v*w)&U
 def signed(x):return x-(1<<64) if x>>63 else x
 return eta,tuple(map(signed,(u,v,q,r)))
def inverse(a):
 if a%P==0:return 0,0
 eta=-1;f,g=P,a%P;d,e=0,1
 for n in range(1,25):
  eta,(u,v,q,r)=divstep(eta,f&U,g&U)
  nf=u*f+v*g;ng=q*f+r*g
  assert nf&M==ng&M==0
  f,g=nf>>62,ng>>62
  d,e=(u*d+v*e)*pow(2**62,-1,P)%P,(q*d+r*e)*pow(2**62,-1,P)%P
  assert (d*a-f)%P==0 and (e*a-g)%P==0
  if g==0:assert f in (-1,1);return (d if f==1 else -d)%P,n
 raise AssertionError('did not settle')
rng=random.Random(26926);cases=[0,1,2,P-2,P-1,P,P+1,2**256-1]+[1<<i for i in range(256)]+[rng.randrange(P) for _ in range(512)];mx=0
for a in cases:
 got,n=inverse(a);assert got==(pow(a,-1,P) if a%P else 0);mx=max(mx,n)
res={'scalar_inverse_transition_cases':len(cases),'max_batches':mx,'c_cpp_limb_execution':False}
print(res)
