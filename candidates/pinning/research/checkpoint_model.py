from pathlib import Path
import random,json
P=2**256-2**32-977
rng=random.Random(910)
count=0
for size in [2,32,64,128]:
 for trial in range(20):
  ds=[[rng.randrange(1,P) for _ in range(7)] for _ in range(size)]
  ts=[]
  for row in ds:
   t=row[0]
   for d in row[1:]:t=t*d%P
   ts.append(t)
  root=1
  for t in ts:root=root*t%P
  I=pow(root,-1,P)
  # Reference exclusions stand in for same existing collective on only N leaves.
  Cs=[root*pow(t,-1,P)%P for t in ts]
  saved=[{'scalar':i,'C':Cs[i]} for i in range(size)]
  first_root=I
  # Simulate arbitrary lane order and early overwrites: each lane reads only its own state.
  order=list(range(size));rng.shuffle(order)
  for i in order:
   scalar,C=saved[i]['scalar'],saved[i]['C'];assert scalar==i
   inverse=C*first_root%P
   row=ds[i];prefix=[row[0]]
   for d in row[1:6]:prefix.append(prefix[-1]*d%P)
   out=[0]*7
   for j in range(6,0,-1):
    out[j]=inverse*prefix[j-1]%P;inverse=inverse*row[j]%P
   out[0]=inverse
   assert all(out[j]*row[j]%P==1 for j in range(7))
   saved[i]={'final':out};count+=7
  assert all('final' in x for x in saved)
  # Root must be read by every lane before root buffer reuse.
  # Kernel sequencing alone does not protect late lanes within the same kernel.
# A constructed unsafe in-place root overwrite: lane0 stores new root before lane1 loads old I.
ds=[[2+i*7+j for j in range(7)] for i in range(2)]
ts=[__import__('functools').reduce(lambda a,b:a*b%P,row,1) for row in ds]
oldI=pow(ts[0]*ts[1]%P,-1,P);newroot=3
assert (ts[0]*newroot%P)*ts[1]%P!=1
result={'status':'PASS','leaf_inverse_checks':count,'checkpoint_bytes_per_candidate':64,'old_checkpoint_bytes_per_candidate':256,'logical_extra_checkpoint_roundtrip_bytes':128,
 'arithmetic_per_candidate_excluding_O_1_over_N':{'first_local_product_M':6,'global_exclusion_tree_M':3,'local_inverse_expand_M':1,'recomputed_prefix_M':5,'reverse_local_inverse_M':12,'inverse_work_total_M':27,'affine_pair_finish_M':14,'affine_pair_finish_S':7,'eight_point_chain_M':46,'eight_point_chain_S':14,'total_M':87,'total_S':21,'baseline_M':95,'baseline_S':28},
 'prefix_shared_bytes':{str(n):6*32*n for n in [32,64,128]},
 'root_overwrite_negative_control':'Detected: lane0 must not overwrite first root until all lanes loaded it. Explicit block barrier or separate root arrays required before qsb_packed_prepare root publication.',
 'state_alias':'Per-lane64B checkpoint may be overwritten by final64B state after own reads; no other lane depends on those fields. Distinct slots per lane verified under shuffled execution.',
 'limits':'Assumes nonzero effective pair denominators; exceptional pairs require proof or fallback. Field model only, no native/GPU validation. Live registers, table traffic, and occupancy remain implementation obligations.'}
Path(__file__).with_name('checkpoint-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
