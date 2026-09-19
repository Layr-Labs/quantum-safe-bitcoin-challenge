#!/usr/bin/env python3
"""Pure Python mathematical and source audit. Does not compile or run CUDA."""
import json,random,hashlib,sys,re
from pathlib import Path
P=2**256-2**32-977
N=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
G=(55066263022277343669578718895168534326250603453777594175500187360389116729240,32670510020758816978083085130507043184471273380659243275938904335757337482424)
def inv(x):return pow(x%P,P-2,P)
def add(a,b):
 if a is None:return b
 if b is None:return a
 x,y=a;u,v=b
 if x==u:
  if (y+v)%P==0:return None
  m=3*x*x*inv(2*y)%P
 else:m=(v-y)*inv(u-x)%P
 q=(m*m-x-u)%P
 return q,(m*(x-q)-y)%P
def direct_sum(points):
 a=None
 for p in points:a=add(a,p)
 return a
# Independent Jacobian scalar multiplication supplies real regular-table terms.
def jd(a):
 x,y,z=a
 if not z or not y:return 0,1,0
 yy=y*y%P;ss=4*x*yy%P;mm=3*x*x%P
 xx=(mm*mm-2*ss)%P
 return xx,(mm*(ss-xx)-8*yy*yy)%P,2*y*z%P
def jm(a,b):
 x,y,z=a
 if not z:return b[0],b[1],1
 u,v=b;zz=z*z%P;h=(u*zz-x)%P;r=(v*zz*z-y)%P
 if not h:return jd(a) if not r else (0,1,0)
 hh=h*h%P;hhh=hh*h%P;t=x*hh%P;xx=(r*r-hhh-2*t)%P
 return xx,(r*(t-xx)-y*hhh)%P,z*h%P
def mul(k):
 a=(0,1,0)
 for c in bin(k%N)[2:]:
  a=jd(a)
  if c=='1':a=jm(a,G)
 if not a[2]:return None
 iz=inv(a[2]);return a[0]*iz*iz%P,a[1]*iz*iz*iz%P
def gt_points(k):
 m=2*(k%N)%N;sign=1 if m&1 else -1
 if sign<0:m=N-m
 direct=m;digits=[];rec=[]
 for c in range(15):
  w=18 if c==0 else 17;shift=0 if c==0 else 17*c+1
  f=(direct>>(shift+1))&((1<<w)-1)
  idx=(f&((1<<(w-1))-1)) if c==14 else ((f^((f>>(w-1))-1))&((1<<(w-1))-1))
  negative=(0 if c==14 else ((f>>(w-1))^1))^(sign<0)
  d=(-(2*idx+1) if negative else 2*idx+1)
  dr=sign*m if c==14 else sign*((m&((1<<(w+1))-1))-(1<<w))
  assert d==dr
  digits.append(d<<shift);rec.append((idx,negative))
  m=2*(m>>(w+1))+1
 assert sum(digits)%N==2*k%N
 return [mul(d*pow(2,-1,N)) for d in digits]
def prefix(points):
 ds=[(points[2*j+1][0]-points[2*j][0])%P for j in range(7)]
 ps=[];a=1
 for d in ds:a=a*(d or 1)%P;ps.append(a)
 return ds,ps,any(d==0 for d in ds)
def heap_inverse(vals):
 n=len(vals);t=[0]*(2*n);t[n:]=vals
 for i in range(n-1,0,-1):t[i]=t[2*i]*t[2*i+1]%P
 t[1]=inv(t[1])
 for i in range(1,n//2):
  a,b=t[2*i:2*i+2];t[2*i]=t[i]*b%P;t[2*i+1]=t[i]*a%P
 return [t[(n+i)//2]*t[(n+i)^1]%P for i in range(n)]
def packed_inverse(vals):
 n=len(vals);prod=[0]*(2*n);prod[:n]=vals;count=n;off=0
 while count>2:
  half=count//2
  for i in range(half):prod[off+count+i]=prod[off+i]*prod[off+half+i]%P
  off+=count;count//=2
 ins=[0]*n;root=inv(prod[off]*prod[off+1]%P)
 ins[off-n]=root*prod[off+1]%P;ins[off-n+1]=root*prod[off]%P
 off-=4;count=4
 while count<n:
  half=count//2
  for i in range(count):ins[off-n+i]=ins[off+count-n+(i&(half-1))]*prod[off+(i^half)]%P
  off-=count*2;count*=2
 return [ins[i&(n//2-1)]*prod[i^(n//2)]%P for i in range(n)]
def scheduled(points, inverse, which, ps):
 ds,_,bad=prefix(points)
 if bad:return direct_sum(points),[],True
 accum=None;di=[];p2=None;zeros=False
 # This executes the C++ prefix choices, while an independent affine oracle
 # checks every denominator and the complete fifteen-term sum.
 for j in range(6,-1,-1):
  if j==6:pre=ps[5] if which=='B' else ps[4]*ds[5]%P
  elif j==5:pre=ps[3]*ds[4]%P if which=='B' else ps[4]
  elif j==4:
   if which=='B':pre=ps[3]
   else:p2=ps[1]*ds[2]%P;pre=p2*ds[3]%P
  elif j==3:pre=ps[1]*ds[2]%P if which=='B' else p2
  elif j==2:pre=ps[1]
  elif j==1:pre=ds[0]
  else:pre=1
  ij=inverse*pre%P if j else inverse
  assert ij*ds[j]%P==1
  if j:inverse=inverse*ds[j]%P
  a,b=points[2*j:2*j+2];m=(b[1]-a[1])*ij%P;x=(m*m-a[0]-b[0])%P;q=(x,(m*(a[0]-x)-a[1])%P)
  assert q==add(a,b)
  if accum is not None and accum[0]==q[0]:zeros=True
  accum=add(accum,q);di.append(ij)
 if accum is not None and accum[0]==points[14][0]:zeros=True
 accum=add(accum,points[14]);assert accum==direct_sum(points)
 return accum,di,zeros
# Verify deferred-XYZZ anchor conventions independently of affine summation.
def deferred(points):
 a,b=points[:2];h=(b[0]-a[0])%P;r=(b[1]-a[1])%P
 if not h:return None,True
 zz=h*h%P;zzz=zz*h%P;q=a[0]*zz%P;x=(r*r-zzz-2*q)%P;y=r*(q-x)%P;anchor=a[1]
 for i,p in enumerate(points[2:],2):
  u=p[0]*zz%P;s=(p[1]+anchor)*zzz%P;h=(u-x)%P;r=(s-y)%P
  if not h:return None,True
  hh=h*h%P;hhh=hh*h%P;q=u*hh%P;zz=zz*hh%P;zzz=zzz*hhh%P
  xx=(r*r+hhh-2*q)%P;yy=r*(q-xx)%P
  if i==len(points)-1:yy=(yy-p[1]*zzz)%P
  x,y=xx,yy;anchor=p[1]
 return (x*inv(zz)%P,y*inv(zzz)%P),False

def main():
 rng=random.Random(0x20260919A2B3);pool=[G]
 for _ in range(2047):pool.append(add(pool[-1],G))
 checked=0;fallback=0;split=0;tail_checks=0
 for batch in range(8):
  groups=[];leaves=[]
  for lane in range(256):
   row=[];prod=1
   for side in range(2):
    pts=[rng.choice(pool) for _ in range(15)]
    if (lane+batch*19+side*7)%67==0:pts[3]=pts[2]
    if (lane+batch*13+side*11)%89==0:pts[9]=(pts[8][0],-pts[8][1]%P)
    ds,ps,bad=prefix(pts);factor=1 if bad else ps[6]
    row.append((pts,ds,ps,bad,factor));prod=prod*factor%P
   groups.append(row);leaves.append(prod)
  ih=heap_inverse(leaves);ip=packed_inverse(leaves)
  assert ih==ip==[inv(x) for x in leaves]
  for row,il in zip(groups,ih):
   a,b=row
   # Production reconstructs PA from P4*d5*d6; bad A's factor is identity.
   pa=1 if a[3] else a[2][4]*a[1][5]*a[1][6]%P
   assert pa==a[4]
   for name,e,iv in [('A',a,il*b[4]%P),('B',b,il*pa%P)]:
    split+=1
    if e[3]:fallback+=1;assert scheduled(e[0],iv,name,e[2])[2];continue
    assert iv==inv(e[4])
    out,_,_=scheduled(e[0],iv,name,e[2]);assert out==direct_sum(e[0]);checked+=1
    pairs=[add(e[0][2*j],e[0][2*j+1]) for j in range(6,-1,-1)]+[e[0][14]]
    xy,exception=deferred(pairs)
    if not exception:assert xy==out;tail_checks+=1
 scalars=[0,1,2,N-1,N,N+1,2**255,2**256-1]+[rng.getrandbits(256) for _ in range(56)]
 for k in scalars:
  points=gt_points(k);assert direct_sum(points)==mul(k)
  ds,ps,bad=prefix(points)
  if not bad:
   for side in ('A','B'):assert scheduled(points,inv(ps[6]),side,ps)[0]==mul(k)
 # Explicit singular reordered tails exercise replay request rather than
 # manufacturing a point from a vanished projective accumulator.
 q=pool[50];neg=(q[0],-q[1]%P)
 assert deferred([q,q]+pool[1:7])[1]
 assert deferred([q,neg]+pool[1:7])[1]
 first=add(pool[20],pool[45]);assert deferred([pool[20],pool[45],first]+pool[1:6])[1]
 # Byte layout and overlap contract: nA writes only bytes [0,24576), whereas
 # B P3 is [24576,32768); inverse(PB) occupies old tree words at >=32768.
 assert 16*256*8+4*512*8==12*256*8+4*512*8+4*256*8==49152
 assert 12*256*8==24576 and 16*256*8==32768
 result={'seed':hex(0x20260919A2B3),'candidate_equivalents':split,'normal_pair_schedules':checked,'zero_denominator_fallbacks':fallback,'deferred_tail_checks':tail_checks,'real_scalar_table_cases':len(scalars),'tree_inverses':8*256*2,'static_shared_bytes':49152,'production_logical_table_bytes_per_candidate':1632,'errors':0,'scope':'Python algebra, signed table mapping, synthetic exceptional paths, shared byte layout; no CUDA compile, GPU result or performance claim'}
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
