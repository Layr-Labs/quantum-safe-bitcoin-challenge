from pathlib import Path
import json,random
p=2**256-2**32-977;rng=random.Random(1010);N=128
checks=0
for n in [1,2,31,32,33,127,128,129,255,256,257]:
 saved=[None]*n;roots=[];ds=[]
 for i in range(n):
  row=[rng.randrange(1,p) for _ in range(7)];ds.append(row)
 for base in range(0,n,N):
  leaves=[]
  for t in range(N):
   prod=1
   if base+t<n:
    for d in ds[base+t]:prod=prod*d%p
   leaves.append(prod)
  # Model the actual flattened tree and exclusion index traversal.
  products=leaves+[0]*N;excluded=[0]*N;offset=0;count=N
  while count>1:
   half=count//2
   for t in range(half):products[offset+count+t]=products[offset+t]*products[offset+half+t]%p
   offset+=count;count//=2
  roots.append(products[2*N-2]);excluded[N-2]=1;offset=2*N-4;count=2
  while count<N:
   half=count//2
   for t in range(count):
    parent=excluded[offset+count-N+(t&(half-1))];sibling=products[offset+(t^half)]
    excluded[offset-N+t]=sibling if count==2 else parent*sibling%p
   offset-=count*2;count*=2
  for t in range(N):
   if base+t<n:
    c=excluded[t&(N//2-1)]*products[t^(N//2)]%p;saved[base+t]=(base+t,c)
 # First root hierarchy's semantic output.
 roots=[pow(x,-1,p) for x in roots]
 for base in range(0,n,N):
  I=roots[base//N]
  # All candidate lanes read old I before the collective can publish a new root.
  newroots=[]
  for t in range(N):
   i=base+t
   if i>=n:continue
   identity,c=saved[i];assert identity==i;r=c*I%p
   pref=[ds[i][0]]
   for d in ds[i][1:6]:pref.append(pref[-1]*d%p)
   inverses=[0]*7
   for j in range(6,0,-1):inverses[j]=r*pref[j-1]%p;r=r*ds[i][j]%p
   inverses[0]=r
   assert all(a*b%p==1 for a,b in zip(ds[i],inverses));checks+=7
   saved[i]=tuple(inverses)
   newroots.append(sum(inverses)%p or 1)
  # Reuse permitted only after all reads above, as enforced by packed_prepare entry barrier.
  prod=1
  for d in newroots:prod=prod*d%p
  roots[base//N]=prod
 assert len(saved)==n and all(len(x)==7 for x in saved)
result={'status':'PASS','batch_sizes':[1,2,31,32,33,127,128,129,255,256,257],'active_leaf_inverse_checks':checks,'inactive_leaves':'identity in first cofactor tree; no intermediate/final state writes outside n','scope':'Exact flattened tree indices and sequential kernel-phase model. Not GPU memory-race execution.'}
Path(__file__).with_name('tail-state-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
