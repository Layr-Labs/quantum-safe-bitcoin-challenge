from pathlib import Path
import sys,ast,random,json,hashlib
W=Path(__file__).resolve().parent
import reference as h
fast=h.fast;MASK=2**32-1
rng=random.Random(202609210601)
def words(n,w):return [(n>>(32*i))&MASK for i in range(w)]
def value(a,w):
 n=sum(a[i]<<(32*i) for i in range(w));return n-(1<<(32*w)) if a[w-1]>>31 else n

def neg(a,w):
 c=1
 for i in range(w):c=(~a[i]&MASK)+c;a[i]=c&MASK;c>>=32

def inc(a,w,c):
 for i in range(w):t=a[i]+c;a[i]=t&MASK;c=t>>32

def sar(a,w,s):
 for i in range(w-1):a[i]=((a[i]>>s)|(a[i+1]<<(32-s)))&MASK
 a[w-1]=((a[w-1]>>s)|((-(a[w-1]>>31)&MASK)<<(32-s)))&MASK

def divdigit(x,r):
 h=x*226050910>>32;z=(x-19*h+6*r)&MASK;t=z*216>>12
 assert z<=145
 return (r*226050910+h+t)&MASK,z-19*t

def floor608(a,L):
 original=value(a,4);lo=a[0]&31;sar(a,4,5);negative=a[3]>>31
 if negative:neg(a,4)
 assert all(x==0 for x in a[L:4]),(L,original)
 q=[0]*4;r=0
 for i in range(L-1,-1,-1):q[i],r=divdigit(a[i],r)
 if negative:
  inc(q,4,r!=0);neg(q,4);r=19-r if r else 0
 a[:4]=q
 assert (value(a,4),32*r+lo)==divmod(original,608)
 return 32*r+lo

def final(x,y):
 a,b,t=x,y,0
 for xx,yy,tt in [(-x,-y,4),(-y,x-y,1),(y,y-x,5),(y-x,-x,2),(x-y,x,6)]:
  if xx<a or (xx==a and yy<b):a,b,t=xx,yy,tt
 u,v=-a,-b;rank=0 if u==0 else 1+(v-1)*237+(u-v)
 j=t&3;j=3-j if j else 0
 return rank|(j<<16)|((t>>2)<<18),(a,b),j,bool(t&4)

schedule=[512]*4+[608]*9
negative128=0;roundchecks=0;maxbits=[0]*14

def audit_pair(x,y):
 global roundchecks,negative128
 assert abs(x)<=h.R and abs(y)<=h.R
 orig=x,y;a=words(x,5);b=words(y,5);digits=[];scale=1
 if value(a,4)!=x or value(b,4)!=y:negative128+=1
 for i,m in enumerate(schedule):
  maxbits[i]=max(maxbits[i],abs(x).bit_length(),abs(y).bit_length())
  if m==512:
   ra=a[0]&511;rb=b[0]&511;w=5 if i==0 else 4
   sar(a,w,9);sar(b,w,9)
  else:
   L=3-(i-4)//3;ra=floor608(a,L);rb=floor608(b,L);w=4
  ref=h.descriptor_for_residue(x%m,y%m,m);d=fast(ra,rb,m);assert d==ref
  rank,j,n,dx,dy=d
  inc(a,w,dx<0);inc(b,w,dy<0)
  nx,ny=(x-dx)//m,(y-dy)//m
  assert (value(a,w),value(b,w))==(nx,ny)
  if i==0:assert (value(a,4),value(b,4))==(nx,ny)
  digits.append((dx,dy,scale));scale*=m;x,y=nx,ny;roundchecks+=1
 assert abs(x)<=236 and abs(y)<=236
 maxbits[13]=max(maxbits[13],abs(x).bit_length(),abs(y).bit_length())
 code,c,j,n=final(x,y);rc,rj,rn=h.canonical_Z(x,y)
 assert (c,j,n)==(rc,(-rj)%3,rn)
 assert h.unit_apply(*c,j,n)==(x,y)
 assert 0<=code&65535<55933 and code<2**19
 assert (sum(dx*s for dx,dy,s in digits)+x*scale,sum(dy*s for dx,dy,s in digits)+y*scale)==orig
 return code

ks=[0,1,2,3,h.N-1,h.N-2,h.N//2,2**128-1,2**128,2**255,2**256-1,
 int('afbd67f9619699cfe1988ad9f06c144a025b413f8a9a021ea648a7dd06839eb9',16)]
ks += [rng.randrange(2**256) for _ in range(10000)]
for k in ks:
 x,y,_=h.split_corr(k);audit_pair(x,y);assert (x+h.LAMBDA*y)%h.N==k%h.N
corners=[-h.R,-2**127,-2**127+1,-1,0,1,2**127-1,2**127,h.R]
for x in corners:
 for y in corners:audit_pair(x,y)
for i in range(2000):audit_pair(rng.randint(-h.R,h.R),rng.randint(-h.R,h.R))
# Every possible bounded final pair, all inverse orientations and final ranks.
ranks=set();bad_inverse=0
for x in range(-236,237):
 for y in range(-236,237):
  code,c,j,n=final(x,y);rc,rj,rn=h.canonical_Z(x,y)
  assert (c,j,n)==(rc,(-rj)%3,rn)
  assert h.unit_apply(*c,j,n)==(x,y)
  u,v=-c[0],-c[1];rank=code&65535
  assert 0<=rank<55933;ranks.add(rank)
  if h.unit_apply(*c,rj,n)!=(x,y):bad_inverse+=1
assert ranks==set(range(55933))
assert negative128>0 and bad_inverse>0
r={'status':'PASS','scalar_input_cases':len(ks),'additional_pair_cases':len(corners)**2+2000,'round_comparisons':roundchecks,'final_pair_exhaustive':473**2,'final_table_ranks':len(ranks),'signed128_negative_controls':negative128,'wrong_inverse_negative_controls':bad_inverse,'max_observed_abs_bits':maxbits,'scope':'Python fixed-u32 mirror vs independent big-int descriptor and quotient reference; integrated CUDA source is not compiled or executed.'}
(W/'recode-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
