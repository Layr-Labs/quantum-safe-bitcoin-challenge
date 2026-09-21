from pathlib import Path
import sys,ast,random,json,collections,time
W=Path(__file__).resolve().parent
import reference as e
import reference as h
P,N=e.P,e.N;C=0x800001e8;rng=random.Random(202609210658)
def mul(k):return e.to_affine(e.jac_mul(k%N))
def add(a,b):
 if a is None:return b
 if b is None:return a
 return e.to_affine(e.jac_add((*a,1),(*b,1)))
def neg(a):return None if a is None else (a[0],-a[1]%P)
# Matches C++ state machine, uses independent exact-field operators for the mathematical audit.
counts=collections.Counter()
def chain(points):
 state=0;X=Y=U=V=anchor=0
 def double(X,Y,U,V):
  counts['double']+=1
  if not Y:return (0,1,0,0,C,0)
  S=2*Y%P;A=S*S%P;B=A*S%P;Q=X*A%P;M=3*X*X%P
  T=(M*M-2*Q)%P;F=(Q-T)*M%P
  return T,(F-Y*B)%P,U*A%P,V*B%P,C,2
 for pt in points:
  if pt is None:counts['identity_skip']+=1;continue
  x,y=pt;yoff=y+C
  if state==0:X,Y=x,yoff;state=1;counts['first']+=1;continue
  if state==1:
   d=(x-X)%P;r=(yoff-Y)%P
   if not d:
    if r:state=0;counts['seed_opposite']+=1;continue
    X,Y,U,V,anchor,state=double(X,(Y-C)%P,1,1);continue
   anchor=Y;U=d*d%P;V=U*d%P;Q=X*U%P;T=(r*r-V-2*Q)%P
   Y=(Q-T)*r%P;X=T;state=2;counts['seed']+=1;continue
  S=(yoff+anchor-2*C)*V%P;r=(S-Y)%P;U2=x*U%P;d=(U2-X)%P
  if not d:
   if r:state=0;counts['madd_opposite']+=1;continue
   Y=(Y-(anchor-C)*V)%P
   X,Y,U,V,anchor,state=double(X,Y,U,V);continue
  pp=d*d%P;ppp=pp*d%P;Q=U2*pp%P;T=(r*r+ppp-2*Q)%P
  V=V*ppp%P;U=U*pp%P;Y=(Q-T)*r%P;X=T;anchor=yoff;counts['madd']+=1
 if state==0:return None
 if state==1:return X,(Y-C)%P
 return X*pow(U,-1,P)%P,((Y-(anchor-C)*V)*pow(V,-1,P))%P

# Independently enumerate host builder rows and compare each rank to residue/table reference.
rows=[];maps=[];scale=1;entries=0
for ch in range(14):
 table={}
 if ch<13:
  m=512 if ch<4 else 608
  rows0=[(1,m//2,0,1,0,1)]
  for a in range(1,(m-1)//3+1):
   low,high,cut=2*a,m-a-1,(m+a)//2;rank=m//2+1+(a-1)*m-3*(a-1)*a//2
   end=min(high,cut)
   if low<=end:rows0.append((rank,end-low+1,a,low,0,1))
   begin=max(low,cut+1)
   if begin<=high:rows0.append((rank+begin-low,high-begin+1,a,begin-m,0,1))
  n=h.orbit_count(m)
 else:rows0=[(1+(v-1)*237,237,-v,-v,-1,0) for v in range(1,237)];n=55933
 table[0]=(0,0)
 for start,count,a,b,da,db in rows0:
  assert 0<count<=608
  for i in range(count):
   rank=start+i;coord=a+i*da,b+i*db
   assert rank not in table and 0<rank<n
   if ch<13:
    ca,cb=h.canonical(coord[0]%m,coord[1]%m,m)
    assert h.rank_rep(ca,cb,m)==rank and h.lift_rep(ca,cb,m)==coord
   else:
    u,v=-coord[0],-coord[1];assert 1+(v-1)*237+u-v==rank
   assert (coord[0]+h.LAMBDA*coord[1])%N!=0
   table[rank]=coord
  rows.append((ch,start,count,a,b,da,db,scale))
 assert set(table)==set(range(n));entries+=n;maps.append(table)
 if ch<13:scale*=m
# Full descriptor decode on arbitrary instance bases; includes zero scalar and order boundary.
cases=[0,1,2,3,N-1,N,N+1,2**256-1,N//2,2**127,2**128]+[rng.randrange(2**256) for _ in range(69)]
full=0;pointchecks=0;rowchecks=0
for idx,k in enumerate(cases):
 nri=[1,2,N-1,0x123456789abcdef][idx%4];x,y,_=h.split_corr(k);scale=1;pts=[]
 for ch in range(14):
  if ch<13:
   m=512 if ch<4 else 608;r,j,sg,dx,dy=h.descriptor_for_residue(x%m,y%m,m)
   x,y=(x-dx)//m,(y-dy)//m
  else:
   canonical,j,sg=h.canonical_Z(x,y);j=(-j)%3;u,v=-canonical[0],-canonical[1]
   r=0 if not u else 1+(v-1)*237+u-v
  a,b=maps[ch][r];q=mul((a+h.LAMBDA*b)*scale*nri)
  if q is not None:
   xx,yy=q;bx=e.BETA*xx%P
   xx=xx if j==0 else bx if j==1 else (-xx-bx)%P
   q=(xx,(-yy)%P if sg else yy)
  pts.append(q);pointchecks+=1
  if ch<13:scale*=m
 assert chain(pts)==mul(k*nri),(k,nri);full+=1
# Construct every state transition, including reset then restart, double then normal add.
g=e.G;g2=mul(2);g3=mul(3)
exceptions=[[],[None]*14,[g],[None,g,None],[g,g],[g,neg(g)],[g,g2,neg(g3)],
 [g,g2,g3],[g,g2,g3,g],[g,neg(g),g2,g2],[g,g,None,neg(g2),g3],[g,g2,neg(g3),None,g,g2],
 [g,neg(g),None,g2,neg(g2),g3],[g2,g,neg(g3),g,g]]
for pts in exceptions:
 ref=None
 for q in pts:ref=add(ref,q)
 assert chain(pts)==ref
for _ in range(128):
 pts=[None if rng.randrange(4)==0 else mul(rng.randrange(-8,9)) for _ in range(14)]
 ref=None
 for q in pts:ref=add(ref,q)
 assert chain(pts)==ref
# A sampled host row ladder vs direct scalar products under nontrivial recovery base.
for ch,start,count,a,b,da,db,s in rows[::max(1,len(rows)//64)]:
 base=mul(s*0xdeadbeef);step=mul((da+h.LAMBDA*db)*s*0xdeadbeef)
 first=mul((a+h.LAMBDA*b)*s*0xdeadbeef);cur=first
 for i in range(min(count,4)):
  assert cur==mul((a+i*da+h.LAMBDA*(b+i*db))*s*0xdeadbeef);cur=add(cur,step);rowchecks+=1
assert all(counts[x]>0 for x in ['identity_skip','first','seed','seed_opposite','madd','madd_opposite','double'])
r={'status':'PASS','table_entries_ranked':entries,'host_rows':len(rows),'host_ladder_point_checks':rowchecks,'full_scalar_instance_checks':full,'oriented_table_point_checks':pointchecks,'exceptional_sequences':len(exceptions)+128,'state_events':dict(counts),'scope':'Exact-field source-shaped chain model and independent EC oracle. Production arithmetic retains promoted approximate field operations; this is not native CUDA execution.'}
(W/'chain-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

# Run after audit_chain.py definitions/checks in the same namespace.
def regular_chain(points):
 assert len(points)==14 and all(q is not None for q in points)
 (x0,y0),(x1,y1)=points[:2];d=(x1-x0)%P;r=(y1-y0)%P;assert d
 U=d*d%P;V=U*d%P;Q=x0*U%P;X=(r*r-V-2*Q)%P;Y=(Q-X)*r%P;anchor=y0
 for x,y in points[2:]:
  R=((y+anchor)*V-Y)%P;U2=x*U%P;d=(U2-X)%P;assert d
  PP=d*d%P;PPP=PP*d%P;Q=U2*PP%P;T=(R*R+PPP-2*Q)%P
  V=V*PPP%P;U=U*PP%P;Y=(Q-T)*R%P;X=T;anchor=y
 return X*pow(U,-1,P)%P,((Y-anchor*V)*pow(V,-1,P))%P
regular=rare=0
for k in [0,1,N-1,N,N+1]+[rng.randrange(N) for _ in range(96)]:
 x,y,_=h.split_corr(k);w=1;pts=[]
 for i in range(14):
  if i<13:
   m=512 if i<4 else 608;rank,j,sg,dx,dy=h.fast(x%m,y%m,m);x,y=(x-dx)//m,(y-dy)//m
  else:dx,dy=x,y
  pts.append(mul((dx+h.LAMBDA*dy)*w*0xdeadbeef))
  if i<13:w*=m
 if all(q is not None for q in pts):got=regular_chain(pts);regular+=1
 else:got=chain(pts);rare+=1
 assert got==mul(k*0xdeadbeef)
assert regular and rare
rr={'status':'PASS','fast_nonzero_sequences':regular,'cold_zero_digit_sequences':rare,'scope':'Exact field model, actual common seed/madd operand order, independent EC oracle; no native execution.'}
(W/'certified-chain-result.json').write_text(json.dumps(rr,indent=2)+'\n');print(json.dumps(rr,indent=2))
