"""Independent exact integer and secp256k1 oracle for GLV608 audits (no native code)."""
P=2**256-2**32-977
N=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
LAMBDA=0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72
BETA=0x7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE
A1=0x3086D221A7D46BCDE86C90E49284EB15
B1=0xE4437ED6010E88286F547FA90ABFE4C3
A2=0x114CA50F7A8E2F3F657C1108D9D44CFD8
B2=A1;R=A2//2;T=(B2-B1)//2
G=(0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
   0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)
def split_corr(k):
 k%=N;c1=(B2*k+N//2)//N;c2=(B1*k+N//2)//N
 x=k-c1*A1-c2*A2;y=c1*B1-c2*B2;which='0'
 if x>R:
  if y>=T:x-=A2;y-=B2;which='-v2'
  else:x-=A1;y+=B1;which='-v1'
 elif x<-R:
  if y<=-T:x+=A2;y+=B2;which='+v2'
  else:x+=A1;y-=B1;which='+v1'
 assert abs(x)<=R and abs(y)<=R and (x+LAMBDA*y)%N==k
 return x,y,which
def unit_apply(a,b,j,neg):
 for _ in range(j):a,b=-b,a-b
 return (-a,-b) if neg else (a,b)
def canonical(a,b,m):
 return min((x%m,y%m) for j in range(3) for n in [False,True] for x,y in [unit_apply(a,b,j,n)])
def rank_rep(a,b,m):return b if not a else m//2+1+(a-1)*m-3*(a-1)*a//2+b-2*a
def lift_rep(a,b,m):return a,b if 2*b<=m+a else b-m
def orbit_count(m):
 from math import gcd
 return (m*m+gcd(2,m)**2+2*gcd(3,m)+2)//6
def descriptor_for_residue(a,b,m):
 c=canonical(a,b,m)
 for j in range(3):
  for n in [False,True]:
   xx,yy=unit_apply(*c,j,n)
   if (xx%m,yy%m)==(a%m,b%m):
    dx,dy=unit_apply(*lift_rep(*c,m),j,n)
    return rank_rep(*c,m),j,n,dx,dy
 raise AssertionError('orbit')
def canonical_Z(a,b):
 x,y,j,n=min((*unit_apply(a,b,j,n),j,n) for j in range(3) for n in [False,True])
 return (x,y),j,n
def fast(a,b,m):
 na=m-a if a else 0;nb=m-b if b else 0;d=a-b if a>=b else a+m-b;nd=m-d if d else 0
 bx,by,tag=a,b,0
 for x,y,t in [(na,nb,4),(nd,na,1),(d,a,5),(nb,d,2),(b,nd,6)]:
  if x<bx or (x==bx and y<by):bx,by,tag=x,y,t
 rank=by if bx==0 else m//2+1+(bx-1)*m-3*(bx-1)*bx//2+(by-2*bx)
 x=bx;y=by if 2*by<=m+bx else by-m;dx,dy=x,y
 if tag&3==1:dx,dy=-y,x-y
 if tag&3==2:dx,dy=y-x,-x
 if tag&4:dx,dy=-dx,-dy
 return rank,tag&3,bool(tag&4),dx,dy

def jac_double(q):
 x,y,z=q
 if not z or not y:return (0,1,0)
 yy=y*y%P;s=4*x*yy%P;m=3*x*x%P
 xx=(m*m-2*s)%P
 return xx,(m*(s-xx)-8*yy*yy)%P,2*y*z%P
def jac_add(a,b):
 x1,y1,z1=a;x2,y2,z2=b
 if not z1:return b
 if not z2:return a
 zz1=z1*z1%P;zz2=z2*z2%P
 u1=x1*zz2%P;u2=x2*zz1%P;s1=y1*z2*zz2%P;s2=y2*z1*zz1%P
 h=(u2-u1)%P;r=(s2-s1)%P
 if not h:return jac_double(a) if not r else (0,1,0)
 hh=h*h%P;hhh=h*hh%P;v=u1*hh%P;xx=(r*r-hhh-2*v)%P
 return xx,(r*(v-xx)-s1*hhh)%P,z1*z2*h%P
def jac_mul(k,point=G):
 q=(0,1,0);a=(*point,1);k%=N
 while k:
  if k&1:q=jac_add(q,a)
  a=jac_double(a);k>>=1
 return q
def to_affine(q):
 x,y,z=q
 if not z:return None
 zi=pow(z,-1,P)
 return x*zi*zi%P,y*zi*zi*zi%P
assert (G[1]*G[1]-G[0]**3-7)%P==0
assert to_affine(jac_mul(LAMBDA))==(G[0]*BETA%P,G[1])
