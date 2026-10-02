#!/usr/bin/env python3
"""Independent integer audit of all ranked aligned epoch-group spans."""
from math import comb
from pathlib import Path
import json
N=137;T=6;total=comb(N,T)
def unrank(r):
 out=[];lo=0
 for i in range(T):
  for c in range(lo,N):
   count=comb(N-c-1,T-i-1)
   if r<count:break
   r-=count
  out.append(c);lo=c+1
 return out
def rank(c):
 lo=0;r=0
 for i,v in enumerate(c):
  for j in range(lo,v):r+=comb(N-j-1,len(c)-i-1)
  lo=v+1
 return r
expected={1048576:181498,2097152:362053,4194304:594292}
result={'total_epochs':total,'capacities':{},'ranked_VRAM_bytes':24*1024**3}
for cap,want in expected.items():
 maxgroups=0;at=None;count=0
 for base in range(0,total,cap):
  hi=min(base+cap,total)-1;first,last=unrank(base),unrank(hi)
  assert rank(first)==base and rank(last)==hi
  g=rank(last[:5])-rank(first[:5])+1
  if g>maxgroups:maxgroups,at=g,base
  count+=1
 assert maxgroups==want,(cap,maxgroups,want)
 table=22688113472;first_bytes=cap*16*8*4*2
 group_old=(2*cap+4)*128*2;group_tight=want*128*2
 result['capacities'][str(cap)]={'aligned_launches_checked':count,'max_groups':maxgroups,'at_base':at,'group_bytes_conservative':group_old,'group_bytes_tight':group_tight,'saved_bytes':group_old-group_tight,'ranked_table_plus_first_states':table+first_bytes,'old_lower_bound_bytes':table+first_bytes+group_old,'tight_lower_bound_bytes':table+first_bytes+group_tight}
print('GROUP_CAP_BOUND_AUDIT: PASS',json.dumps(result,indent=2))
Path('candidates/subset/lab/iter5-group-cap-bound-audit.json').write_text(json.dumps(result,indent=2))
