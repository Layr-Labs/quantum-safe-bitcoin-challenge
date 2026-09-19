#!/usr/bin/env python3
"""Independent integer/curve/wave model. Does not compile or execute CUDA."""
from pathlib import Path
import random,json,hashlib,re
from functools import lru_cache
P=2**256-2**32-977
N=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
G=(0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
   0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)
HERE=Path(__file__).resolve().parent

def add(a,b):
 if a is None:return b
 if b is None:return a
 x,y=a;u,v=b
 if x==u:
  if (y+v)%P==0:return None
  s=3*x*x*pow(2*y,-1,P)%P
 else:s=(v-y)*pow(u-x,-1,P)%P
 xx=(s*s-x-u)%P
 return xx,(s*(x-xx)-y)%P

def mul(k,a=G):
 r=None;k%=N
 while k:
  if k&1:r=add(r,a)
  a=add(a,a);k>>=1
 return r

def decode(k):
 k%=N;d=2*k-N;raw=d%(1<<256);M=[(raw>>(64*j))&((1<<64)-1) for j in range(4)];negative=int(d<0)
 out=[]
 for c in range(16):
  pos=16*c+1;j=pos//64;sh=pos%64
  f=(M[j]>>sh)&65535
  if j<3 and sh==49:f|=(M[j+1]&1)<<15
  tm=-negative if c==15 else (f>>15)-1
  idx=(f^tm)&32767;neg=tm<0
  out.append(-(2*idx+1) if neg else 2*idx+1)
 return out

@lru_cache(None)
def term(c,d):return mul((d*pow(2,16*c,N)*pow(2,-1,N)*7)%N)

def leaves(k):return [term(c,d) for c,d in enumerate(decode(k))]

def prep(a,b):
 if a is None:return 1,b,0,0,1
 if b is None:return 1,a,0,0,1
 x,y=a;u,v=b;den=(u-x)%P;num=(v-y)%P;total=(x+u)%P
 if den==0:
  if num or not y:return 2,None,0,0,1
  num=3*x*x%P;den=2*y%P
 return 0,a,total,num,den

def finish(st,iv):
 kind,a,total,num,den=st
 if kind==2:return None
 if kind==1:return a
 x,y=a;s=num*iv%P;xx=(s*s-total)%P
 return xx,(s*(x-xx)-y)%P

R=mul(13);a,b=R;C=3*a*a*pow(2*b,-1,P)%P

def recover(pt,iv):
 if pt is None:return R,(a,-b%P)
 x,y=pt
 if x==a:
  twice=add(R,R)
  return (twice,None) if y==b else (None,(twice[0],-twice[1]%P))
 l=(b-y)*iv%P;m=(b+y)*iv%P;total=(l+m)%P
 xp=(a+total*(l-C))%P;xm=(a+total*(m-C))%P
 return (xp,(l*(a-xp)-b)%P),(xm,(b-m*(a-xm))%P)

def tree(arena,vals):
 arena[512:1024]=vals
 width=256
 while width:
  for tid in range(width):arena[width+tid]=arena[2*(width+tid)]*arena[2*(width+tid)+1]%P
  width//=2
 arena[1]=pow(arena[1],-1,P)
 width=1
 while width<256:
  for tid in range(width):
   i=width+tid;r=arena[i];l=arena[2*i];rr=arena[2*i+1]
   arena[2*i]=r*rr%P;arena[2*i+1]=r*l%P
  width*=2
 invs=[arena[256+i//2]*arena[512+(i^1)]%P for i in range(512)]
 assert all(x*y%P==1 for x,y in zip(vals,invs))
 return invs

def sha(pt):
 if pt is None:return None
 return hashlib.sha256(bytes([2+(pt[1]&1)])+pt[0].to_bytes(32,'big')).hexdigest()

def wave_run(ks,K):
 out={};arena=[None]*1024;steps=0
 for base in range(0,len(ks),32*K):
  cohorts=min(K,(len(ks)-base+31)//32)
  for wave in range(cohorts+4):
   jobs=[];den=[]
   for tid in range(512):
    lane=tid&31;stage=0 if tid<256 else 1 if tid<384 else 2 if tid<448 else 3 if tid<480 else 4
    cohort=wave-stage;idx=base+32*cohort+lane;active=0<=cohort<cohorts and idx<len(ks)
    st=(2,None,0,0,1);point=None;d=1
    def get(i):return None if arena[i]==arena[512+i]==0 else (arena[i],arena[512+i])
    if active:
     if stage<4:
      if stage==0:
       ll=leaves(ks[idx]);c=(tid>>5)*2;p,q=ll[c:c+2]
      else:
       first=((tid-256)>>5)*64+lane if stage==1 else 256+((tid-384)>>5)*64+lane if stage==2 else 384+lane
       p,q=get(first),get(first+32)
      st=prep(p,q);d=st[-1]
     else:
      point=get(448+lane);d=(a-point[0])%P if point else 1;d=d or 1
    jobs.append((active,stage,idx,st,point));den.append(d)
   # Queue read barrier; destroy all ready points and build the heap in-place.
   invs=tree(arena,den)
   # Final leaf-read barrier; all inverses retained before arbitrary arena clobber.
   arena[:]=[-123]*1024
   for tid,(active,stage,idx,st,point) in enumerate(jobs):
    if stage<4:
     r=finish(st,invs[tid]);arena[tid],arena[512+tid]=r or (0,0)
    elif active:
     assert idx not in out;out[idx]=recover(point,invs[tid])
   steps+=1
 assert set(out)==set(range(len(ks)))
 for i,k in enumerate(ks):
  pt=expected_point(k)
  ref=add(pt,R),add(pt,(a,-b%P))
  assert out[i]==ref,(i,k,out[i],ref)
  assert tuple(map(sha,out[i]))==tuple(map(sha,ref))
 return steps

@lru_cache(None)
def expected_point(k):return mul((k%N)*7)

def main():
 rng=random.Random(192019)
 edge={0,1,2,N-1,N,N+1,(1<<256)-1}
 for bit in range(256):
  for delta in (-1,0,1):
   if 0<=(1<<bit)+delta<1<<256:edge.add((1<<bit)+delta)
 vals=sorted(edge)+[rng.getrandbits(256) for _ in range(100000)]
 for k in vals:
  ds=decode(k);assert all(d&1 and abs(d)<=65535 for d in ds)
  assert (sum(d<<(16*c) for c,d in enumerate(ds))*pow(2,-1,N)-k)%N==0,k
 # Full scalar/point check and exceptional recovered points P=0,+R,-R.
 special=13*pow(7,-1,N)%N
 pool=[0,1,2,N-1,N,N+1,(1<<256)-1,special,N-special]+[rng.randrange(N) for _ in range(23)]
 for k in pool:
  total=None
  for p in leaves(k):total=add(total,p)
  assert total==expected_point(k)
 # Complete law: infinity, doubles, negatives, arbitrary pairs.
 pts=[None,G,add(G,G),(G[0],-G[1]%P),R,(a,-b%P)]
 for x in pts:
  for y in pts:
   st=prep(x,y);assert finish(st,pow(st[-1],-1,P))==add(x,y)
 cases=[]
 for count,K in [(1,16),(31,16),(32,16),(33,16),(127,16),(511,16),(512,16),(513,16),(1025,16),(67,1),(257,8),(4097,128)]:
  ks=[pool[i%len(pool)] for i in range(count)]
  cases.append(dict(count=count,K=K,waves=wave_run(ks,K)))
 src=(HERE/'W5R16.cuh').read_text();host=(HERE/'pinning.cu').read_text()
 assert '__shared__ uint64_t arena[4][1024]' in src
 assert 'w5_r16<<<(batch_size+span-1)/span,512' in host
 assert '#define GT_CHUNKS 16' in host and '#define QSB_STATE_PLANES 5u' in host
 assert 'BN_set_word(shift, (1u<<16))' in host
 assert 'for(unsigned width=1;width<256;width<<=1)' in src
 assert '__syncthreads();\n        if(stage<4)' in src
 out=dict(status='PASS',scalar_cases=len(vals),complete_scalar_points=len(pool),complete_law_pairs=len(pts)**2,
          schedule_cases=cases,model='Python big integers; source geometry checked, not CUDA execution',gpu_measured=False)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
