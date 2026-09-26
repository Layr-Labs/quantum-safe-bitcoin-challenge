"""Word-address demand model. Does not model emitted SASS or hardware wavefronts."""
from pathlib import Path
from collections import defaultdict
import json,math,hashlib
D=Path(__file__).resolve().parent

def request(entries):
 # entry=(lane,byte_address,width); identical words are broadcasts, not conflicts.
 bybank=defaultdict(set);words=set();byhalf=[]
 for lane,address,width in entries:
  assert address%4==0 and width in [4,8]
  for a in range(address//4,(address+width)//4):bybank[a%32].add(a);words.add(a)
 for half in range(2):
  b=defaultdict(set)
  for lane,address,width in entries:
   if lane//16==half:
    for a in range(address//4,(address+width)//4):b[a%32].add(a)
  byhalf.append(max(map(len,b.values()),default=0))
 return dict(unique_words=len(words),max_distinct_words_per_bank=max(map(len,bybank.values()),default=0),halfwarp_max=byhalf,bandwidth_min_rounds=math.ceil(len(words)/32))

def coop(pitch,base,width=8,store=False,groups=4):
 return [(l,8*((l%8//2)*pitch+base+l//8)+(4*(l&1) if width==4 else 0),width) for l in range(32) if l//8<groups and (not store or not l&1)]
rows=[]
for w in range(4):
 for label,width,store in [('upper_a_ld64',8,False),('upper_a_ld32',4,False),('v4_st64',8,True)]:
  st=label.startswith('v4');oldpitch=128 if st else 256;newpitch=116 if st else 228
  oldbase=(32 if st else 64)*w+(24 if st else 48);newbase=(28 if st else 56)*w+(24 if st else 48)
  old=request(coop(oldpitch,oldbase,width,store));new=request(coop(newpitch,newbase,width,store))
  assert old['max_distinct_words_per_bank']==4 and new['max_distinct_words_per_bank']==1
  assert new['max_distinct_words_per_bank']==new['bandwidth_min_rounds']
  rows.append(dict(warp=w,label=label,old=old,new=new))
# Scalar stages: no padding penalty under 32-bit or 64-bit instruction choices,
# both full-warp word demand and half-warp transaction proxy are kept separate.
dense_cases=0
for w in range(4):
 for kind in ['p','i']:
  oldpitch,newpitch=(256,228) if kind=='p' else (128,116)
  oldbase,newbase=(64*w,56*w) if kind=='p' else (32*w,28*w)
  for k in range(4):
   for width in [4,8]:
    for count in [4,8,16,32]:
     for route in ['contiguous','parent','sibling']:
      def req(pitch,base):
       return [(l,8*(k*pitch+base+(l if route=='contiguous' else l%(count//2) if route=='parent' else l^(count//2))),width) for l in range(count)]
      a=request(req(oldpitch,oldbase));b=request(req(newpitch,newbase))
      assert a==b,(w,kind,k,width,count,route,a,b);dense_cases+=1
# Prove all compact tree slots disjoint for all planes/warps, plus no plane overlap.
slots=set()
for kind,pitch,span in [('p',228,56),('i',116,28)]:
 for k in range(4):
  for w in range(4):
   for local in range(span):
    address=k*pitch+w*span+local
    assert (kind,address) not in slots;slots.add((kind,address))
    assert 0<=address<4*pitch
# Exact logical traffic saved by retaining u4/u2 instead of shared nodes.
# Counts are unique field objects, not transaction or performance claims.
saved_store=4+2+1+1+2
saved_read=4+2+1+1+2+2+4
result=dict(status='PASS_SOURCE_ADDRESS_BANK_DEMAND_MODEL',cooperative_cases=len(rows),dense_cases=dense_cases,compact_unique_64bit_slots=len(slots),shared_bytes_old=12288,shared_bytes_new=11008,padding_bytes=256,shared_unique_bytes_saved_per_warp=(saved_store+saved_read)*32,shared_unique_bytes_saved_per_CTA=4*(saved_store+saved_read)*32,warp_sync_rounds_old=11,warp_sync_rounds_new=6,CTA_barriers=0,extra_source_live_words_across_inverse=2,actual_SASS_width_unknown=True,hardware_wavefronts_measured=False,GPU_performance_predicted=False,rows=rows,primary_source='https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html')
(D/'shared-bank-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
