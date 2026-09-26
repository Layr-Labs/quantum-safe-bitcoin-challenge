"""Pure Python model of the new signed batch, builder bounds and source integration.
No C++/CUDA compilation. Not a native runtime or throughput measurement.
"""
from pathlib import Path
import json,hashlib,random,re,sys
r=Path(__file__).resolve().parent
p=r.parent
main=(p/'cpu_cogrind.h').read_text();fold=(p/'cpu_cogrind_fold.h').read_text();table=(p/'cpu_fold_table.h').read_text();vi=(p/'cpu_cogrind_ifma.h').read_text();plan=(p/'cpu_signed_plan.h').read_text()
# Bind recoding and unchanged arithmetic primitives to the checked sources.
assert hashlib.sha256(plan.split('#pragma once',1)[1].encode()).hexdigest()=='6a6dea4b15319cce1f63a892be704202328c5e421443bc7a0f4afd9a81e6494f'
allowed={'cpu_cogrind.h','cpu_cogrind_ifma.h'}
for name,digest in {'cpu_ifma_field.h': 'aef88f6900256f0864a35fa96ac6928e70207bf8d25c4cd691600d4e07e912a4', 'cpu_sha4.h': '9b62e13b9a43b20be3b93749a929eef19b733002e319e671e8a0518d64e94e74', 'cpu_safegcd.h': 'e15f06184ceccebb01e1e6610cb3e8b2476ea734aba1455ef27cb8c0673fd195', 'cpu_cogrind_vec.h': '2d5f74e6fab668a705ab5bef82475eee7eb515f239b157d8749d03c367d9ed70'}.items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest,name
assert 'const unsigned rows = (count + 254) / 256;' in main
assert 'if (d < count) tentry_set' in main
assert 'S->signed_table && vsf && vsi && sf::selfcheck(w,vsi,vsf)' in main
assert 'const int cand[5] = {32, 16, 8, 4, 0};' in main
assert 'S->simd_env<0 && signed_memory_ok(nw)' in main
assert 'if(g_cg->signed_table)for(int j=a->f0;j<a->f1;++j)build_signed_window' in main
assert 'g_cg->ready.store(1, std::memory_order_release)' in main
assert 'return zero;' in vi
assert 'if(vi::chain_invert(inv,acc))return complete(w,0,capture);' in fold
assert 'if(vi::chain_invert(inv,acc))return complete(w,r,capture);' in fold
assert 'w->inf[i] || (g_cg->signed_infinity[r] && infinity_row(row[l]))' in fold
assert 'g_cg->signed_infinity[r]=true;' in table
assert 'if(!infinity)' in fold and 'if(lz_ok(h))publish(w,i,r);' in fold
assert 'if(!capture) { vi::hash_keys(w,b,r,x,y);return; }' in fold
assert 'for(int j=1;j<plan::windows-1;++j)' in fold
assert 'for(int r=0;r<2;++r)' in fold
W=[20]*10+[19]*3;S=[sum(W[:i]) for i in range(13)];H=[1<<(w-1) for w in W]
rows=sum(x+1 for x in H[:-1])+2*(H[-1]+1);assert rows*64==402654080
# Same unsigned word-extraction operations as the proposed C++ digits() routine.
def decode(z,widths=W):
 words=[z>>(64*k)&(2**64-1) for k in range(4)];out=[];carry=0;bit=0
 for j,w in enumerate(widths):
  word=bit>>6;s=bit&63;v=words[word]>>s
  if s+w>64 and word<3:v|=words[word+1]<<(64-s)
  d=(v&((1<<w)-1))+carry;half=1<<(w-1)
  if j+1<len(widths):carry=int(d>half);mag=(1<<w)-d if carry else d;out.append(-mag if carry else mag)
  else:out.append(d)
  bit+=w
 return out
R=random.Random(26092619);P=2**256-2**32-977;N=0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141
cases=[0,1,P-1,P,P+1,N-1,N,N+1,2**256-1]+[1<<i for i in range(256)]
for i,(w,sh) in enumerate(zip(W,S)):
 for a in [0,1,(1<<(w-1))-1,1<<(w-1),(1<<(w-1))+1,(1<<w)-1]:
  for tail in [0,(1<<sh)-1]:
   x=(a<<sh)|tail
   if x<2**256:cases.append(x)
cases += [R.getrandbits(256) for _ in range(12000)]
zero=0
for z in cases:
 ds=decode(z);assert z==sum(d<<sh for d,sh in zip(ds,S));assert all(abs(d)<=h for d,h in zip(ds,H));assert ds[-1]>=0
 zero+=any(d==0 for d in ds)
 assert ds[-1]<H[-1]+1

# All nonzero table coefficients below 2^20. On order n~2^256, x(uB)=x(vB)
# iff u=+-v mod n; only u=v=256 occurs in the batched builder and is handled.
count_checks=0
for count in [65536,262145,524289]:
 nd=0;last=0
 for block in range((count+254)//256):
  for k in range(256):
   d=256*block+k+1
   if d<count:nd+=1;last=d
   if block>=1:assert ((256*(block-1)+k+1)+256)==d
 assert nd==count-1 and last==count-1
 count_checks+=nd
# Parallel builder partitions cover all windows once, even with limited quotas.
for workers in range(1,9):
 got=[j for b in range(workers) for j in range(13*b//workers,13*(b+1)//workers)]
 assert got==list(range(13))
# Model the actual 2 interleaved x 8 lane prefix and reverse inverse schedule.
P=211;N=199
B=next((x,y) for x in range(P) for y in range(1,P) if (y*y-x*x*x-7)%P==0)
def add(a,b):
 if a is None:return b
 if b is None:return a
 x,y=a;u,v=b
 if x==u:
  if (y+v)%P==0:return None
  lam=3*x*x*pow(2*y,-1,P)%P
 else:lam=(v-y)*pow(u-x,-1,P)%P
 q=(lam*lam-x-u)%P;return q,(lam*(x-q)-y)%P
def neg(a):return None if a is None else (a[0],-a[1]%P)
points=[None]
for i in range(1,N):points.append(add(points[-1],B))
assert add(points[-1],B) is None
def mul(k):return points[k%N]
def digits(z):
 out=[];carry=0
 for j in range(3):
  d=((z>>(3*j))&7)+carry
  if j<2:carry=int(d>4);out.append(d-8 if carry else d)
  else:out.append(d)
 return out
stats={'regular_batches':0,'fallback_recid_0':0,'fallback_recid_1':0,'root_zero_fallbacks':0,'infinity_fallbacks':0,'verified_endpoints':0,'fold_rows':0,'fold_equal_x':0}
def inverses(ds):
 nb=len(ds);acc=[[1]*8 for _ in range(2)];c=[]
 for b,row in enumerate(ds):
  acc[b&1]=[a*d%P for a,d in zip(acc[b&1],row)];c.append(acc[b&1][:])
 if any(not acc[0][l]*acc[1][l]%P for l in range(8)):return None
 inv=[[pow(a,-1,P) for a in row] for row in acc];out=[[0]*8 for _ in ds]
 for b in range(nb-1,-1,-1):
  g=b&1
  for l in range(8):
   out[b][l]=inv[g][l]*(c[b-2][l] if b>=2 else 1)%P
   inv[g][l]=inv[g][l]*ds[b][l]%P
   assert out[b][l]*ds[b][l]%P==1
 return out
def do_fold(A):
 # Mirror in-place builder: both output banks consume saved base x/y, not
 # the just-written plus row. Equality denominators become one in prefix.
 X=[mul(d<<6) for d in range(5)];out=[[A]+[None]*4,[neg(A)]+[None]*4]
 ds=[(A[0]-X[d][0])%P for d in range(1,5)]
 prefix=[];acc=1
 for d in ds:acc=acc*(d or 1)%P;prefix.append(acc)
 inv=pow(prefix[-1],-1,P)
 for k in range(3,-1,-1):
  ik=inv*(prefix[k-1] if k else 1)%P;inv=inv*(ds[k] or 1)%P
  a=X[k+1]
  for recid,aa in enumerate([A,neg(A)]):
   if not ds[k]:q=add(a,aa);stats['fold_equal_x']+=1
   else:
    lam=(aa[1]-a[1])*ik%P;x=(lam*lam-a[0]-aa[0])%P;q=x,(lam*(a[0]-x)-a[1])%P
   assert q==add(a,aa);out[recid][k+1]=q;stats['fold_rows']+=1
 return out
def batch(zs,A,ft):
 n=len(zs);nb=(n+7)//8;d=[digits(z) for z in zs];cur=[mul(x[0]) for x in d];emitted=[]
 def retry(r,why):
  stats['fallback_recid_'+str(r)]+=1;stats[why]+=1
  for recid in range(r,2):
   for i,z in enumerate(zs):emitted.append((i,recid,add(mul(z),A if recid==0 else neg(A))))
  return emitted
 for j in range(1,2):
  dx=[[1]*8 for _ in range(nb)];active=[];target=[]
  for i in range(n):
   q=mul(d[i][j]<<(3*j));target.append(q)
   if q is None:continue
   if cur[i] is None:cur[i]=q;continue
   dx[i//8][i%8]=(q[0]-cur[i][0])%P;active.append(i)
  inv=inverses(dx)
  if inv is None:return retry(0,'root_zero_fallbacks')
  for i in active:
   a=cur[i];q=target[i];lam=(q[1]-a[1])*inv[i//8][i%8]%P;x=(lam*lam-a[0]-q[0])%P
   cur[i]=x,(lam*(a[0]-x)-a[1])%P
 for recid in range(2):
  dx=[[1]*8 for _ in range(nb)]
  for i in range(n):
   q=ft[recid][d[i][2]]
   if cur[i] is None or q is None:return retry(recid,'infinity_fallbacks')
   dx[i//8][i%8]=(q[0]-cur[i][0])%P
  inv=inverses(dx)
  if inv is None:return retry(recid,'root_zero_fallbacks')
  for b in range(nb-1,-1,-1):
   for l in range(8):
    i=b*8+l
    if i>=n:continue
    a=cur[i];q=ft[recid][d[i][2]];lam=(q[1]-a[1])*inv[b][l]%P;x=(lam*lam-a[0]-q[0])%P
    emitted.append((i,recid,(x,(lam*(a[0]-x)-a[1])%P)))
 stats['regular_batches']+=1
 return emitted
rng=random.Random(9262012)
for arank in range(1,N):
 A=mul(arank);ft=do_fold(A)
 for n in [1,2,7,8,9,17,33,65]:
  for case in range(3):
   zs=[rng.randrange(256) for _ in range(n)]
   if case==1:zs[0]=0
   if case==2:zs[-1]=255
   out=batch(zs,A,ft)
   assert len(out)==2*n and len({(i,recid) for i,recid,_ in out})==2*n
   for i,recid,q in out:assert q==add(mul(zs[i]),A if recid==0 else neg(A));stats['verified_endpoints']+=1
assert all(stats[k]>0 for k in stats),stats
# Canonical source shape of changed legacy inverse is exactly the old core
# plus a returned existing zero mask; GPU and field primitives byte-identical.
newvi=vi.replace('Q8T static __mmask8 chain_invert','Q8T static void chain_invert').replace('    return zero; // existing backend ignores it; folded backend retries complete recovery\n','')
assert hashlib.sha256(newvi.encode()).hexdigest()=='512fa9d759225b2c0fc01ba91cce4d60233e9bf33e1e34491e68d7c4bc6b191d'
res={'recoding_cases':len(cases),'table_nonzero_indices':count_checks,**stats,'source_invariants':True,'field_bytes_unchanged':True,'dense_point_additions':[17,13],'dense_multiplies':[82,65],'dense_squares':[17,13],'actual_root_stages':[16,13],'logical_table_bytes':402654080,'legacy_fallback_table_bytes':67108864,'native_compile':False,'native_execution':False,'submitted':False,'source_sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in p.iterdir() if x.is_file() and x.name in allowed|{'cpu_fold_table.h','cpu_cogrind_fold.h','cpu_signed_plan.h'}}}
print(json.dumps(res,indent=2))
